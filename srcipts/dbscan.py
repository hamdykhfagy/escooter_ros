import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_point_cloud_from_txt(file_path):
    """
    Load a point cloud from a text file.
    Each line in the file should contain x, y, z coordinates separated by spaces or commas.

    :param file_path: Path to the text file containing the point cloud data.
    :return: Numpy array of shape (n_points, 3) with the point cloud data.
    """
    point_cloud = np.loadtxt(file_path, delimiter=' ')
    return point_cloud

def perform_dbscan_clustering(point_cloud, eps=0.5, min_samples=10):
    """
    Perform DBSCAN clustering on the point cloud data.

    :param point_cloud: Numpy array of shape (n_points, 3) with the point cloud data.
    :param eps: The maximum distance between two points for them to be considered as in the same neighborhood.
    :param min_samples: The number of points in a neighborhood for a point to be considered as a core point.
    :return: Cluster labels for each point in the point cloud.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(point_cloud)
    return labels

def visualize_clusters(point_cloud, labels, max_cluster_size=500):
    """
    Visualize the clustering result in 3D, skipping clusters larger than a given size.

    :param point_cloud: Numpy array of shape (n_points, 3) with the point cloud data.
    :param labels: Cluster labels for each point in the point cloud.
    :param max_cluster_size: The maximum number of points in a cluster to include in the visualization.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'  # Black for noise
        
        class_member_mask = (labels == k)
        cluster_size = np.sum(class_member_mask)
        
        # Skip clusters larger than max_cluster_size
        if cluster_size > max_cluster_size:
            continue
        
        xyz = point_cloud[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, marker='o', s=10)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title('DBSCAN Clustering of Point Cloud (Filtered by Cluster Size)')
    plt.show()

# Example usage:
if __name__ == "__main__":
    file_path = '/media/khha/INTENSO/20240606-Testfahrt_IR_CAM/has_scooter/debug/1073/1717682657649286773_pointcloud.txt'
    point_cloud = load_point_cloud_from_txt(file_path)
    
    labels = perform_dbscan_clustering(point_cloud, eps=0.3, min_samples=5)
    
    # Visualize the clustering result, skipping clusters larger than 500 points
    visualize_clusters(point_cloud, labels, max_cluster_size=500)
