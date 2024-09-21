from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageEnhance


def adjust_hue_brightness(img, hue_factor, brightness_factor, output_path):
    """
    Adjust the hue and brightness of an image and save the result.
    
    Parameters:
    - image_path: str, the path to the input image
    - hue_factor: float, the factor by which to adjust the hue (between -0.5 and 0.5, where 0 means no change)
    - brightness_factor: float, the factor by which to adjust the brightness (1.0 means no change)
    - output_path: str, the path to save the output image
    """

    # Convert image to HSV (hue, saturation, value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Adjust the hue
    hsv[..., 0] = (hsv[..., 0].astype(int) + int(hue_factor * 180)) % 180  # OpenCV hue range is [0, 180]

    # Convert back to BGR
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Adjust brightness
    img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

    # Save the image
    return img

model_path = "/home/IBEO.AS/khha/scooter/training_results/escooter-training600epochs/runs/detect/train/weights/best.pt"

model = YOLO(model_path)


image_path1 = '/home/IBEO.AS/khha/Downloads/escooter-training600epochs/Escooters-8/test/images/right_cam_node_image_raw_pylon_camera_336_jpg.rf.bfd343fe8441c7741e64522be7c4cedb.jpg'

img = cv2.imread(image_path1)
img = adjust_hue_brightness(img, 0.0, 1.7, 'output_image.jpg')
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
results1 = model.predict(source=img)

cv2.imshow("Image", results1[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
