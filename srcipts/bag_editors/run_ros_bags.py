import os
import rospy
import rosbag
from cv_bridge import CvBridge
import numpy as np
import cv2
import escooter_ros.srcipts.tutorials.undistort as undistort

def process_bagfiles(bags_path, result_path):
    # IMAGE_WIDTH = 2047
    # IMAGE_HEIGHT = 2464
    # # Your intrinsic parameter matrix
    # # Intrinsic parameter matrix (camera matrix)
    # intrinsic_matrix = np.array([[1262.387025, 0.000000, 1024.000000],
    #                                 [0.000000, 1262.387025, 1232.000000],
    #                                 [0.000000, 0.000000, 1.000000]])

    # Your radial distortion coefficients
    # radial_dist_coeffs = np.array([k1, k2, p1, p2, k3])
    # Distortion coefficients
    # dist_coeffs_camera = np.array([-0.242265, 0.042270, -0.002106, -0.001473, 0.000000])

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    files = os.listdir(bags_path)
    counter1 = 0
    for filename in files:
        if filename.endswith('.bag'):
            bagfile_path = os.path.join(bags_path, filename)
            print(f"Processing: {bagfile_path} {counter1}/{len(files)}")
            counter1 +=1
            bag = rosbag.Bag(bagfile_path)
            
            image_topic = bag.read_messages("/right_cam_node/image_raw")
            for b in image_topic:
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(b.message, desired_encoding="bgr8")
                resized_image = cv2.resize(cv_image, (1024, 1024))
                cv2.imwrite(f'{result_path}/right_cam_june__seq_{b.message.header.seq}.jpg', resized_image)
                # Calculate height of each cropped image
                # cropped_height = IMAGE_HEIGHT // 6
                # Crop and append each section
                # for i in range(6):
                #     upper = i * cropped_height
                #     lower = (i + 1) * cropped_height
                #     cropped_img = cv_image[upper:lower, :]
                #     undistort.undistort_image(cropped_img,intrinsic_matrix, dist_coeffs_camera)
                #     rotated_img = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
            bag.close()

if __name__ == "__main__":
    rospy.init_node('bagfile_player')  # Initialize ROS node
    result_path = "/media/khha/73695A0C0999281E//media/khha/73695A0C0999281E/runde_1_no_resizing"
    bags_path = "/media/khha/73695A0C0999281E/20230524_Messfahrt_ENQT_Runde_1_nicht_alles_enthalten"
    process_bagfiles(bags_path, result_path)
