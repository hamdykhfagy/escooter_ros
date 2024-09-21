#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import cv_bridge
import numpy as np
import roslib.packages
from geometry_msgs.msg import Pose
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from escooter_ros.msg import EscooterResult
import cv2
import message_filters
from tf.transformations import euler_from_quaternion
from sbg_driver.msg import SbgEkfNav
from sbg_driver.msg import SbgEkfEuler
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import PointCloud2
import os
import sensor_msgs.point_cloud2 as pc2
import ros_numpy

import ScooterPoseEstimator

class EscooterNode():
    def __init__(self):
        self.initialize_parameters()
        self.model = YOLO(f"{self.pkg_path}/models/{self.yolo_model}")
        self.bridge = cv_bridge.CvBridge()
        self.spe = ScooterPoseEstimator.ScooterPoseEstimator(self.HFOV, self.VFOV, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self.sub_cam1 = message_filters.Subscriber(self.topic_name_cam, Image)
        self.sub_cam2 = message_filters.Subscriber(self.topic_name_cam2, Image)
        self.sub_gps = rospy.Subscriber("/sbg/ekf_nav",SbgEkfNav,callback=self.spe.GPS_callback)
        self.sub_euler = rospy.Subscriber("/sbg/ekf_euler",SbgEkfEuler,callback=self.spe.Euler_callback)
        if self.cam2_logic == "or":
            self.sub_cam1.registerCallback(self.image_callback)
            self.sub_cam2.registerCallback(self.image_callback)
        elif self.cam2_logic == "and":
            self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_cam1, self.sub_cam2], queue_size=10, slop=0.1)
            self.ts.registerCallback(self.image_sync_callback)
        else:
            if self.log_results:
                # Synchronize camera image and point cloud
                self.sub_lidar = message_filters.Subscriber("/velodyne_points", PointCloud2)
                self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_cam1, self.sub_lidar], queue_size=10, slop=0.1)
                self.ts.registerCallback(self.image_pc_callback)
            else:
                self.sub_cam1.registerCallback(self.image_callback)

        self.results_pub = rospy.Publisher(self.topic_name_result, EscooterResult, queue_size=1)
        self.results_pose_pub = rospy.Publisher("escooter/posess", Pose, queue_size=1)
        self.result_image_pub = rospy.Publisher(self.topic_name_result_image, Image, queue_size=1)
        self.result_image_pub2 = rospy.Publisher(self.topic_name_result_image2, Image, queue_size=1)

    def initialize_parameters(self):
        self.topic_name_cam = rospy.get_param("~topic_name_cam", "/right_cam_node/image_raw")
        self.topic_name_cam2 = rospy.get_param("~topic_name_cam2", "/left_cam_node/image_raw")
        self.topic_name_result = rospy.get_param("~topic_name_result", "/escooter/result")
        self.topic_name_result2 = rospy.get_param("~topic_name_result2", "/escooter/result2")
        self.topic_name_result_image = rospy.get_param("~topic_name_result_image", "/escooter/image")
        self.topic_name_result_image2 = rospy.get_param("~topic_name_result_image2", "/escooter/image2")
        self.IMAGE_WIDTH = rospy.get_param("~IMAGE_WIDTH", 2047) # TODO
        self.IMAGE_HEIGHT = rospy.get_param("~IMAGE_HEIGHT", 2464) # TODO
        self.HFOV = rospy.get_param("~HFOV", 101.662896) # TODO
        self.VFOV = rospy.get_param("~VFOV", 111.383104) # TODO
        self.yolo_conf_thresh = rospy.get_param("~yolo_conf_thresh", 0.5)
        self.pkg_path = roslib.packages.get_pkg_dir("escooter_ros")
        self.debug = rospy.get_param("~debug", "rviz")
        self.yolo_model = rospy.get_param("~yolo_model", "best_June.pt")
        self.cam2_logic = rospy.get_param("~cam2_logic", "or")
        self.camera = rospy.get_param("~CAMERA", 2)
        self.min_width = rospy.get_param("~min_width", 30)
        self.max_width = rospy.get_param("~max_width", 30)
        self.min_height = rospy.get_param("~min_height", 30)
        self.max_height = rospy.get_param("~max_height", 30)
        self.log_results = rospy.get_param("~log_results", False)

    def image_pc_callback(self, msg1, point_cloud_msg):
        cv_image1 = self.bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8")
        # rospy.loginfo(f"Image 1 timestamp: {msg1.header.stamp.to_sec()}, size = {cv_image1.shape}")        
        results = self.model.predict(source=cv_image1, conf=self.yolo_conf_thresh)
        result_msg = self.process_result(msg1, results[0], point_cloud_msg)
        self.publish_result(result_msg)

    def image_callback(self, msg1):
        cv_image1 = self.bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8")
        # rospy.loginfo(f"Image 1 timestamp: {msg1.header.stamp.to_sec()}, size = {cv_image1.shape}")        
        results = self.model.predict(source=cv_image1, conf=self.yolo_conf_thresh)
        result_msg = self.process_result(msg1, results[0], None)
        self.publish_result(result_msg)

    def image_sync_callback(self, msg1, msg2):
        cv_image1 = self.bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8")
        cv_image2 = self.bridge.imgmsg_to_cv2(msg2, desired_encoding="bgr8")
        results = self.model.predict(source=[cv_image1, cv_image2], conf=self.yolo_conf_thresh)        
        # Check if detections are present in both images
        if self.has_detections(results):
            result_msg = self.process_result(msg1, results[0])
            # result2_msg = self.process_result(msg2, results[1])
            combined_image = self.bridge.cv2_to_imgmsg(cv2.vconcat([results[0].plot(), results[0].plot()]), encoding="bgr8")
            self.publish_result(result_msg, image = combined_image)

    def has_detections(self, results):
        # Assuming results is a list containing results for cv_image1 and cv_image2
        if results[0] is not None and results[1] is not None:
            # Check if both images have detections
            if len(results[0].boxes) == len(results[1].boxes):
                return True
        return False

    def process_result(self, msg1, result, point_cloud_msg):
        result_msg = EscooterResult()
        result_msg.header = msg1.header
        result_msg.detections, result_msg.image, result_msg.locations = self.create_escooter_result_msg(result, msg1.header, point_cloud_msg)
        return result_msg

    def publish_result(self, result_msg, image = None):
        if image is not None:
            result_msg.image = image
        self.results_pub.publish(result_msg)
        for detection in result_msg.detections.detections:
            self.results_pose_pub.publish(detection.results[0].pose.pose)
        self.result_image_pub.publish(result_msg.image)

    def create_escooter_result_msg(self, result, header, point_cloud):
        detections_msg = Detection2DArray()
        bounding_box = result.boxes.xywh
        classes = result.boxes.cls
        confidence_score = result.boxes.conf
        class_names = result.names
        plotted_image = result.plot()
        nav_msg_array = []

        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            if not self.is_validate_bbox(bbox):
                continue
            detection = Detection2D()
            detection.bbox.center.x = float(bbox[0])
            detection.bbox.center.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)
            hypothesis.score = float(conf)
            hypothesis.pose.pose = self.spe.get_pose(bbox, class_names[int(cls)]) # pose is relative to the camera co-ordinate frame! https://stackoverflow.com/questions/17987465/how-is-the-camera-coordinate-system-in-opencv-oriented
            distance = hypothesis.pose.pose.position.z
            nav_msg_array.append(self.spe.get_nav_sat_location(bbox, distance, self.camera, header, class_names[int(cls)]))
            q = [hypothesis.pose.pose.orientation.x, hypothesis.pose.pose.orientation.y, hypothesis.pose.pose.orientation.z, hypothesis.pose.pose.orientation.w]
            rospy.loginfo(f"scooter type = {class_names[int(cls)]}, distance = {hypothesis.pose.pose.position.z}, pose = {euler_from_quaternion(q)}, seq = {header.seq}")
            plotted_image = self.create_result_image(plotted_image, bbox, hypothesis.pose.pose.position.z, euler_from_quaternion(q)[2], header.seq)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        
        # Directory structure
        if self.log_results and point_cloud is not None and len(bounding_box) > 0:
            sequence_number = header.seq
            directory_name = f"{self.pkg_path}/../../data/debug/{sequence_number}"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            
            # Save the point cloud
            point_cloud_filename = f"{directory_name}/{header.stamp.to_nsec()}_pointcloud.txt"
            self.save_pointcloud(point_cloud, point_cloud_filename)
            

            # Save the image frame
            image_filename = f"{directory_name}/{header.stamp.to_nsec()}_frame.png"
            cv2.imwrite(image_filename, plotted_image)
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return detections_msg, result_image_msg, nav_msg_array
    
    def save_pointcloud(self, point_cloud, filename):
        """Saves the point cloud to a TXT file."""
        cloud_arr = np.asarray(self.pointcloud2_to_xyz_array(point_cloud))
        np.savetxt(filename, cloud_arr, fmt='%f %f %f %f')

    def pointcloud2_to_xyz_array(self, point_cloud):
        """Converts a sensor_msgs/PointCloud2 to an Nx3 numpy array."""
        points_list = []
        for point in pc2.read_points(point_cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            points_list.append([point[0], point[1], point[2], point[3]])
        return np.array(points_list)

    def create_result_image(self, plotted_image, bbox, distance, angle, sequence):
        # Define the text to be added
        if angle < 45:
            text = f"d={distance:.2f}, stand, {sequence}"
        else:
            text = f"d={distance:.2f}, laying, {sequence}"
            
        # Define the position (bottom-left corner of the text string in the image)
        position = (int(bbox[0]), int(bbox[1]))
        # Define the font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Define font scale
        font_scale = 1
        # Define font color (B, G, R)
        color = (255, 255, 255)
        # Define line thickness
        thickness = 3
        # Define line type
        line_type = cv2.LINE_AA

        # Add the text to the image
        plotted_image = cv2.putText(plotted_image, text, position, font, font_scale, color, thickness, line_type)
        
        # Convert the plotted image with text to a ROS image message
        # result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return plotted_image

    def is_validate_bbox(self, bbox):
        if int(bbox[2]) > self.max_width or int(bbox[2]) < self.min_width:
            rospy.loginfo(f"width  = {int(bbox[2])}, height = {int(bbox[3])}")
            return False
        if int(bbox[3]) > self.max_width or int(bbox[3]) < self.min_width:
            rospy.loginfo(f"width  = {int(bbox[2])}, height = {int(bbox[3])}")
            return False
        
        return True

if __name__ == "__main__":
    rospy.init_node("escooter_node")
    node = EscooterNode()
    rospy.spin()