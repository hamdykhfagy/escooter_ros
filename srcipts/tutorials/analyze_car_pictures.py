
import cv2

def analyze_image(image_path):
    image = cv2.imread(image_path)
    
    # Get resolution
    resolution = image.shape[:2]  # (height, width)
    
    # Convert to grayscale for histogram analysis
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    # Analyze sharpness
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    return resolution, histogram, laplacian_var

# Path to a sample car camera image
car_camera_image_path = "/home/IBEO.AS/khha/scooter/left_cam_node_seq_2061.jpg"

resolution, histogram, laplacian_var = analyze_image(car_camera_image_path)

print("Resolution:", resolution)
print("Resolution:", resolution)
print("Laplacian Variance (sharpness indicator):", laplacian_var)

# Optionally, plot the histogram
import matplotlib.pyplot as plt

plt.plot(histogram)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()