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
            counter = 0
            for b in image_topic:
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(b.message, desired_encoding="bgr8")
                # Calculate height of each cropped image
                cropped_height = IMAGE_HEIGHT // 6

                # List to store cropped images
                cropped_images = []

                # Crop and append each section
                for i in range(6):
                    upper = i * cropped_height
                    lower = (i + 1) * cropped_height
                    cropped_img = cv_image[upper:lower, :]
                    cropped_images.append(cropped_img)
                    resized_image = cv2.resize(cropped_img, (800, 800))
                    rotated_img = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
                    # cv2.imshow(f'Image_{i}', resized_image)
                    cv2.imwrite(f'{result_path}/camera_image_raw_{b.message.header.frame_id}_{b.message.header.seq}.jpg', rotated_img)
                # cv2.destroyAllWindows()
                # y_min, x_min, h, w = 0, 0 , int(IMAGE_HEIGHT/2), IMAGE_WIDTH
                # cropped_image = cv_image[y_min:y_min+h, x_min:x_min+w]
                # cv2.waitKey(0)
                counter += 1
            print(f"num of images saved = {counter}")
            bag.close()

if __name__ == "__main__":
    rospy.init_node('bagfile_player')  # Initialize ROS node
    result_path = "/home/hamdy/catkin_ws/data//camera_image_raw"
    bags_path = "/media/hamdy/18E9C0676F799EF1/Messfahrt_DLR_Tag1_anon"
    process_bagfiles(bags_path, result_path)
