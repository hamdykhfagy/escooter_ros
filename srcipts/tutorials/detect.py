from ultralytics import YOLO
import cv2
import numpy as np
from escooter_ros.srcipts.tutorials.keypoints import get_matched_keypoints_inside_bbox, filter_matches_outside_bbox, visualize_matches


def calculate_distance(img_left, img_right, bbox_left, bbox_right, focal_length, baseline):
    # Convert bounding boxes to tuples (x, y, w, h)
    x1, y1, w1, h1 =  int(bbox_left[0]), int(bbox_left[1]), int(bbox_left[2]), int(bbox_left[3])
    x2, y2, w2, h2 =  int(bbox_right[0]), int(bbox_right[1]), int(bbox_right[2]), int(bbox_right[3])
        
    # Crop the bounding boxes from the images
    left_crop = img_left[y1:y1+h1, x1:x1+w1]
    right_crop = img_right[y2:y2+h2, x2:x2+w2]
    
    # Resize images to have the same dimensions
    max_height = max(left_crop.shape[0], right_crop.shape[0])
    max_width = max(left_crop.shape[1], right_crop.shape[1])
    left_resized = cv2.resize(left_crop, (max_width, max_height))
    right_resized = cv2.resize(right_crop, (max_width, max_height))
    
    # Convert images to grayscale
    left_gray = cv2.cvtColor(left_resized, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_resized, cv2.COLOR_BGR2GRAY)
    
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

def calculate_mean_keypoint(keypoint_pairs):
    if not keypoint_pairs:
        return None
    
    num_pairs = len(keypoint_pairs)
    mean_keypoint1 = [0, 0]  # Initialize mean keypoint
    mean_keypoint2 = [0, 0]  # Initialize mean keypoint
    
    # Calculate sum of keypoints
    for keypoint1, keypoint2 in keypoint_pairs:
        mean_keypoint1[0] += keypoint1.pt[0]  # Sum of x coordinates
        mean_keypoint2[1] += keypoint2.pt[1]  # Sum of y coordinates

        mean_keypoint1[0] += keypoint1.pt[0]  # Sum of x coordinates
        mean_keypoint2[1] += keypoint2.pt[1]  # Sum of y coordinates
    
    # Calculate mean
    mean_keypoint1[0] /= num_pairs  # Mean of x coordinates
    mean_keypoint1[1] /= num_pairs  # Mean of y coordinates
    mean_keypoint2[0] /= num_pairs  # Mean of x coordinates
    mean_keypoint2[1] /= num_pairs  # Mean of y coordinates
    print(f"mean k_pt1 = {mean_keypoint1}, mean k_pt2 = {mean_keypoint2}")
    
    return [(mean_keypoint1[0], mean_keypoint1[1]), (mean_keypoint2[0], mean_keypoint2[1])]

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
focal_length = 2052.804457
baseline = 1000e-2
# distance = calculate_distance(image1, image2, bounding_box1[0], bounding_box2[0], focal_length, baseline)
mean_pt1, mean_pt2 = calculate_mean_keypoint(matched_keypoints) 
disparity_x =  mean_pt1[0] - mean_pt2[0]
distance = focal_length * baseline / disparity_x
print(f"distance = {distance}")

# filtered_keypoints1, filtered_keypoints2, filtered_matches = filter_matches_outside_bbox(matched_keypoints, good_matches, bounding_box1[0],bounding_box2[0])
# img = visualize_matches(image1, filtered_keypoints1, image2, filtered_keypoints2, filtered_matches)
# Show the images
# cv2.imshow("Matches", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
