from ultralytics import YOLO
import cv2
import numpy as np
from keypoints import get_matched_keypoints_inside_bbox, filter_matches_outside_bbox, visualize_matches


def calculate_distance(img_left, img_right, bbox_left, bbox_right, focal_length, baseline):
    # Convert bounding boxes to tuples (x, y, w, h)
    x1, y1, w1, h1 =  int(bbox_left[0]), int(bbox_left[1]), int(bbox_left[2]), int(bbox_left[3])
    x2, y2, w2, h2 =  int(bbox_right[0]), int(bbox_right[1]), int(bbox_right[2]), int(bbox_right[3])
    
    # Crop the bounding boxes from the images
    left_crop = img_left[y1:y1+h1, x1:x1+w1]
    right_crop = img_right[y2:y2+h2, x2:x2+w2]
    
    # Convert images to grayscale
    left_gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)
    
    # StereoBM for disparity calculation
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    
    # Compute the disparity map
    disparity_map = stereo.compute(left_gray, right_gray)
    
    # Calculate depth from disparity map
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    mask = disparity_map > 0
    depth_map[mask] = focal_length * baseline / disparity_map[mask]
    
    # Calculate the mean depth within the bounding box
    mean_depth = np.mean(depth_map[y1:y1+h1, x1:x1+w1])
    
    return mean_depth

model = YOLO(f"/home/hamdy/catkin_ws/src/escooter_roos/models/v3_best.pt")


image_path1 = '/home/hamdy/catkin_ws/data/HAW_car/_2023-07-24-14-10-14_30/right_cam_node_image_raw_pylon_camera_510.jpg'
image_path2 = '/home/hamdy/catkin_ws/data/escooter_training/left_cam_node_image_raw_pylon_camera_510.jpg'

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

results1 = model.predict(source=image1)
results2 = model.predict(source=image2)

bounding_box1 = results1[0].boxes.xywh
bounding_box2 = results2[0].boxes.xywh

matched_keypoints, good_matches = get_matched_keypoints_inside_bbox(results1[0].plot(),results2[0].plot(), bounding_box1[0],bounding_box2[0])

# filtered_keypoints1, filtered_keypoints2, filtered_matches = filter_matches_outside_bbox(matched_keypoints, good_matches, bounding_box1[0],bounding_box2[0])
# img = visualize_matches(image1, filtered_keypoints1, image2, filtered_keypoints2, filtered_matches)
# Show the images
# cv2.imshow("Matches", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
