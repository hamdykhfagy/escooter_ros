import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = '/home/IBEO.AS/khha/scooter/train_day_1/images/left_cam_node_image_raw_pylon_camera_5493_jpg.rf.1f0e3963378040e77c938220bb02fd55.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Define the color range for filtering green color in HSV space
lower_bound = np.array([40, 40, 40])  # Lower bound of the green color range
upper_bound = np.array([80, 255, 255])  # Upper bound of the green color range

# Convert image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Create a mask based on the defined color range
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Apply the mask to the image
filtered_image = cv2.bitwise_and(image, image, mask=mask)

# Display the original and filtered images
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title('Filtered Image (Green)')
plt.imshow(filtered_image)
plt.show()
