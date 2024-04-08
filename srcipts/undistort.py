import numpy as np
import cv2

def undistort_image(image, intrinsic_matrix, radial_dist_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, radial_dist_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_image = cv2.undistort(image, intrinsic_matrix, radial_dist_coeffs, None, new_camera_matrix)

    return undistorted_image

def undistort_and_display_image(image_path, intrinsic_matrix, radial_dist_coeffs):
    # Load the image
    image = cv2.imread(image_path)

    # Undistort the image
    undistorted_image = undistort_image(image, intrinsic_matrix, radial_dist_coeffs)

    # Display the undistorted image
    cv2.imshow('Undistorted Image', undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Your intrinsic parameter matrix
    # Intrinsic parameter matrix (camera matrix)
    camera_matrix_camera = np.array([[1262.387025, 0.000000, 1024.000000],
                                    [0.000000, 1262.387025, 1232.000000],
                                    [0.000000, 0.000000, 1.000000]])

    # Your radial distortion coefficients
    # radial_dist_coeffs = np.array([k1, k2, p1, p2, k3])
    # Distortion coefficients
    dist_coeffs_camera = np.array([-0.242265, 0.042270, -0.002106, -0.001473, 0.000000])

    # Your tangential distortion coefficients
    # tangential_dist_coeffs = np.array([k4, k5])

    # Path to your image
    image_path = '/home/hamdy/catkin_ws/data/camera_image_raw/camera_image_raw_seq_288_camId_2.jpg'

    # Undistort and display the image
    undistort_and_display_image(image_path, camera_matrix_camera, dist_coeffs_camera)

if __name__ == "__main__":
    main()
