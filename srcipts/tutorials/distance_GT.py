import matplotlib.pyplot as plt

def plot_measurements1(lidar_data, camera_data):
    """
    Plots LiDAR and camera measurements on the same graph.

    Parameters:
    - lidar_data: list or numpy array of LiDAR measurements
    - camera_data: list or numpy array of camera measurements
    """
    # Ensure the data arrays are the same length
    if len(lidar_data) != len(camera_data):
        raise ValueError("LiDAR and camera data arrays must be the same length.")
    
    # Create a range for the x-axis
    x = range(len(lidar_data))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, lidar_data, label='LiDAR Measurements', marker='o')
    plt.plot(x, camera_data, label='Camera Measurements', marker='x')
    
    # Add titles and labels
    plt.title('LiDAR and Camera Measurements')
    plt.xlabel('Measurement Index')
    plt.ylabel('Measurement Value')
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.grid(True)
    plt.show()

def plot_measurements2(camera_measurements, lidar_measurements):
    """
    Plots camera_measurements on the x-axis and lidar_measurements on the y-axis.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(camera_measurements, lidar_measurements, color='blue')
    plt.title("Camera Measurements vs Lidar Measurements")
    plt.xlabel("Camera Measurements")
    plt.ylabel("Lidar Measurements")
    plt.grid(True)
    plt.show()

def plot_measurements3(camera_measurements, lidar_measurements):
    """
    Plots the difference between camera_measurements and lidar_measurements as a histogram.
    Also adds a vertical line representing the mean of the differences.
    """
    differences = [d1 - d2 for d1, d2 in zip(lidar_measurements, camera_measurements)]
    mean_difference = sum(differences) / len(differences)
    
    plt.figure(figsize=(8, 6))
    plt.hist(differences, bins=30, color='green', edgecolor='black')
    plt.axvline(mean_difference, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_difference:.2f}')
    
    plt.title("Histogram of Differences Between Lidar and Camera Measurements")
    plt.xlabel("Difference (Lidar - Camera)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
lidar_measurements = [9.0, 10.2, 4.8, 11.6, 8.36,17,6.63, 5.33,21.0, 8.67, 7.25, 10.64, 11.7, 10.44, 7.8,20, 9.98, 10.11, 5.85, 7.64]
camera_measurements = [6.259, 9.2169, 5.9, 7.9, 6.42,12.04,11.18, 2.68,14.21, 5.66, 4.58, 6.23, 6.81, 6.29, 5.26, 17.16, 6.91, 9.22, 4.43, 5.61]
average_error = 0
for d1,d2 in zip(lidar_measurements,camera_measurements):
    average_error += d1-d2
print(f"average_error = {average_error/len(lidar_measurements)}")
plot_measurements1(lidar_measurements, camera_measurements)
# Plotting the measurements
plot_measurements2(camera_measurements, lidar_measurements)
plot_measurements3(camera_measurements, lidar_measurements)
