# Optionally, plot the histogram
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob



# Path to your images
# image_paths = glob.glob("path_to_your_iphone_images/*.jpeg")

# Desired resolution (replace with your car camera resolution)
# target_resolution = (640, 480)  # Example resolution

# for image_path in image_paths:
#     img = cv2.imread(image_path)
#     resized_img = cv2.resize(img, target_resolution)
#     cv2.imwrite(f"processed_images/{image_path.split('/')[-1]}", resized_img)

def equalize_histogram(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# for image_path in image_paths:
#     img = cv2.imread(image_path)
#     resized_img = cv2.resize(img, target_resolution)
#     equalized_img = equalize_histogram(resized_img)
#     cv2.imwrite(f"processed_images/{image_path.split('/')[-1]}", equalized_img)

def adjust_brightness(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# alpha = 1.0  # Simple contrast control
# beta = 0    # Simple brightness control

# for image_path in image_paths:
#     img = cv2.imread(image_path)
#     resized_img = cv2.resize(img, target_resolution)
#     equalized_img = equalize_histogram(resized_img)
#     adjusted_img = adjust_brightness(equalized_img, alpha, beta)
#     cv2.imwrite(f"processed_images/{image_path.split('/')[-1]}", adjusted_img)

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 0.05
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

def blur_image(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# for image_path in image_paths:
#     img = cv2.imread(image_path)
#     resized_img = cv2.resize(img, target_resolution)
#     equalized_img = equalize_histogram(resized_img)
#     adjusted_img = adjust_brightness(equalized_img, alpha, beta)
#     noisy_img = add_noise(adjusted_img)
#     blurred_img = blur_image(noisy_img)
#     cv2.imwrite(f"processed_images/{image_path.split('/')[-1]}", blurred_img)

def preprocess_image(image, target_resolution, alpha=1.0, beta=0):
    resized_img = cv2.resize(image, target_resolution)
    img_yuv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    equalized_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    adjusted_img = cv2.convertScaleAbs(equalized_img, alpha=alpha, beta=beta)
    row, col, ch = adjusted_img.shape
    mean = 0
    sigma = 0.05
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_img = adjusted_img + gauss
    blurred_img = cv2.GaussianBlur(noisy_img, (5, 5), 0)
    return blurred_img

# image_paths = glob.glob("path_to_your_iphone_images/*.jpeg")
target_resolution = (1200, 1920)  # Example resolution

# for image_path in image_paths:
# img = cv2.imread(image_path)
img = cv2.imread("/home/IBEO.AS/khha/scooter/Escooters.v22i.yolov8/train/images/IMG_0196_jpeg.rf.63df90bdc7ca4b39f55630508e3b1b0c.jpg")

cv2.imshow("BEFORE: Image", img)
cv2.waitKey(0)
processed_img = preprocess_image(img, target_resolution)
cv2.imshow("AFTER: Image", processed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
