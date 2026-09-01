import cv2

def detect_keypoints(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints
    keypoints = sift.detect(gray, None)
    
    return keypoints

def detect_keypoints_in_bbox(image, bbox):
    # Extract the bounding box coordinates
    bbox_x, bbox_y, bbox_width, bbox_height = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # Crop the image to the bounding box region
    bbox_image = image[bbox_y:bbox_y+bbox_height, bbox_x:bbox_x+bbox_width]
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints
    keypoints = sift.detect(gray, None)
    
    # Adjust the coordinates of keypoints to match the original image
    for kp in keypoints:
        kp.pt = (kp.pt[0] + bbox_x, kp.pt[1] + bbox_y)
    
    return keypoints

def match_keypoints(keypoints1, keypoints2, descriptors1, descriptors2):
    # Initialize a feature matcher (here we use BFMatcher with default parameters)
    matcher = cv2.BFMatcher()
    
    # Match descriptors of keypoints from both images
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)
    
    # Initialize list to store matching keypoints
    matched_keypoints = []
    
    # Store matching keypoints pairs
    for match in good_matches:
        matched_keypoints.append((keypoints1[match.queryIdx], keypoints2[match.trainIdx]))
    
    return matched_keypoints, good_matches

def compute_descriptors(image, keypoints):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Compute descriptors for keypoints
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    return descriptors

def visualize_matches(image1, keypoints1, image2, keypoints2, matched_keypoints):
    # Draw keypoints on images
    img1_kp = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Draw matches
    matched_img = cv2.drawMatches(img1_kp, keypoints1, img2_kp, keypoints2, matched_keypoints, None, matchesThickness=1)
    
    # Show the images
    cv2.imshow("Matches", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return matched_img

def filter_matches_outside_bbox(matched_keypoints, good_matches, bbox1, bbox2):
    filtered_keypoints1 = []
    filtered_keypoints2 = []
    filtered_matches = []
    bbox1_x, bbox1_y, bbox1_width, bbox1_height = bbox1
    bbox2_x, bbox2_y, bbox2_width, bbox2_height = bbox2
    
    for (kp1, kp2), d_match in zip(matched_keypoints, good_matches):
        # Get the coordinates of the keypoints
        x1, y1 = kp1.pt
        x2, y2 = kp2.pt
        
        # Check if the keypoints are within both bounding boxes
        if (bbox1_x <= x1 <= bbox1_x + bbox1_width and bbox1_y <= y1 <= bbox1_y + bbox1_height) and \
           (bbox2_x <= x2 <= bbox2_x + bbox2_width and bbox2_y <= y2 <= bbox2_y + bbox2_height):
            filtered_keypoints1.append(kp1)
            filtered_keypoints2.append(kp2)
            filtered_matches.append(d_match)
    
    return filtered_keypoints1, filtered_keypoints2, filtered_matches


def get_matched_keypoints_inside_bbox(image1, image2, bbox1, bbox2):
    keypoints1 = detect_keypoints_in_bbox(image1, bbox1)
    keypoints2 = detect_keypoints_in_bbox(image2, bbox2)
    descriptors1 = compute_descriptors(image1, keypoints1)
    descriptors2 = compute_descriptors(image2, keypoints2)
    matched_keypoints, good_matches = match_keypoints(keypoints1, keypoints2, descriptors1, descriptors2)
    visualize_matches(image1, keypoints1, image2, keypoints2, good_matches)
    return matched_keypoints, good_matches

def example_usage():
    # Example usage
    image_path1 = '/home/hamdy/catkin_ws/data/HAW_car/_2023-07-24-14-10-14_30/right_cam_node_image_raw_pylon_camera_510.jpg'
    image_path2 = '/home/hamdy/catkin_ws/data/escooter_training/left_cam_node_image_raw_pylon_camera_510.jpg'

    # Load the image
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    bbox1 = (100,100,300,100)
    bbox2 = (100,100,300,100)
    keypoints1 = detect_keypoints_in_bbox(image1, bbox1)
    keypoints2 = detect_keypoints_in_bbox(image2, bbox2)

    descriptors1 = compute_descriptors(image1, keypoints1)
    descriptors2 = compute_descriptors(image2, keypoints2)

    print("Number of keypoints1 detected:", len(keypoints1))
    print("Number of keypoints2 detected:", len(keypoints2))

    matched_keypoints, good_matches = match_keypoints(keypoints1, keypoints2, descriptors1, descriptors2)



    # Print the number of matched keypoints
    print("Number of matched keypoints:", len(matched_keypoints))
    # Print the matched keypoints pairs
    print("Matched Keypoints:", f"type= {type(matched_keypoints[0])}")

    for kp1, kp2 in matched_keypoints:
        print(kp1.pt, "<-->", kp2.pt)

    # Visualize matches
    visualize_matches(image1, keypoints1, image2, keypoints2, good_matches)

# example_usage()