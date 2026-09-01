import cv2

def detect_edges(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Detect edges using Canny edge detection algorithm
    edges = cv2.Canny(blurred_image, 150, 50)
    
    return edges

# Example usage:
image_path = '/home/hamdy/catkin_ws/data/escooter_training/left_cam_node_image_raw_pylon_camera_510.jpg'
edges = detect_edges(image_path)

# Display the original and edge-detected images
# cv2.imshow('Original Image', cv2.imread(image_path))
cv2.imshow('Edges Detected', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
