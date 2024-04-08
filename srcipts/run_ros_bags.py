import os
import rospy
import rosbag
from cv_bridge import CvBridge
import numpy as np
import cv2

def process_bagfiles(bags_path, result_path):
    IMAGE_WIDTH = 2047
    IMAGE_HEIGHT = 2464
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
            image_topic = bag.read_messages("/camera/image_raw")
            for b in image_topic:
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(b.message, desired_encoding="bgr8")
                # Calculate height of each cropped image
                cropped_height = IMAGE_HEIGHT // 6
                # Crop and append each section
                for i in range(6):
                    upper = i * cropped_height
                    lower = (i + 1) * cropped_height
                    cropped_img = cv_image[upper:lower, :]
                    resized_image = cv2.resize(cropped_img, (800, 800))
                    rotated_img = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(f'{result_path}/camera_image_raw_seq_{b.message.header.seq}_camId_{i}.jpg', rotated_img)
            bag.close()

if __name__ == "__main__":
    rospy.init_node('bagfile_player')  # Initialize ROS node
    result_path = "/home/hamdy/catkin_ws/data//camera_image_raw"
    bags_path = "/media/hamdy/18E9C0676F799EF1/Messfahrt_DLR_Tag1_anon"
    process_bagfiles(bags_path, result_path)
