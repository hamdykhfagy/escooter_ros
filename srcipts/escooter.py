#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import cv_bridge
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from escooter_ros.msg import EscooterResult
import cv2


class EscooterNode():
    def __init__(self):
        self.topic_name_cam = rospy.get_param("~topic_name_cam", "/right_cam_node/image_raw")
        self.topic_name_result = rospy.get_param("~topic_name_result", "/escooter/result")
        self.topic_name_result_image = rospy.get_param("~topic_name_result_image", "/escooter/image")
        yolo_model = rospy.get_param("~yolo_model", "v3_best.pt")
        self.debug = rospy.get_param("~debug", False)
        path = roslib.packages.get_pkg_dir("escooter_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.sub = rospy.Subscriber(
            self.topic_name_cam,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        self.results_pub = rospy.Publisher(self.topic_name_result, EscooterResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(
                    self.topic_name_result_image, Image, queue_size=1
                )
        self.bridge = cv_bridge.CvBridge()
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.predict(source=cv_image)
        result_msg = EscooterResult()
        
        result_msg.header = msg.header
        result_msg.detections = self.create_detections_array(results)
        result_msg.image = self.create_result_image(results)
        self.results_pub.publish(result_msg)
        self.result_image_pub.publish(result_msg.image)
        if self.debug:
            cv2.destroyAllWindows()
            resized_image = cv2.resize(results[0].plot(), (1200, 800))
            cv2.imshow('Image', resized_image)
            cv2.waitKey(1000)

    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.x = float(bbox[0])
            detection.bbox.center.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)
            hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg

    def create_result_image(self, results):
        plotted_image = results[0].plot()
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return result_image_msg

if __name__ == "__main__":
    rospy.init_node("escooter_node")
    node = EscooterNode()
    rospy.spin()