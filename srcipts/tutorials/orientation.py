import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/IBEO.AS/khha/scooter/new_car_images/new_car_images/scooters/right_cam_june__seq_2075.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Detect the largest contour, assuming it's the scooter
if contours:
    scooter_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(scooter_contour)

    # Determine if the scooter is lying down or standing
    if w > h:
        position = "Lying Down"
    else:
        position = "Standing"

    # Draw the bounding box and label
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, position, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Display the result
cv2.imshow('Scooter Position', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
