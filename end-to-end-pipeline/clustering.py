import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import os
import string

"""Clustering Algorithm based on Union Find"""

NEAR_THRESHOLD = 0.25

cluster_labels = list(string.ascii_lowercase)  # Labels from 'a', 'b', 'c', ...

add_dot = False

def load_npy_data(npy_file_path):
    """Load the N x 14 feature vectors from the .npy file."""
    return np.load(npy_file_path)

def parse_features(feature_vector):
    """Parse visibility, type, and parameters from the 14-dim feature vector."""
    visibility_vector = feature_vector[0:2]
    type_vector = feature_vector[2:5]
    param_vector = feature_vector[5:]
    return visibility_vector, type_vector, param_vector

def calculate_bounding_box(type_vector, param_vector):
    """Calculate bounding box for each primitive based on type."""
    if np.array_equal(type_vector, [1, 0, 0]):  # LINE
        start = np.array(param_vector[0:2])
        end = np.array(param_vector[2:4])
        min_point = np.minimum(start, end)
        max_point = np.maximum(start, end)

    elif np.array_equal(type_vector, [0, 1, 0]):  # CIRCLE
        center = np.array(param_vector[4:6])
        radius = param_vector[14]
        min_point = center - radius
        max_point = center + radius

    elif np.array_equal(type_vector, [0, 0, 1]):  # ARC
        center = np.array(param_vector[4:6])
        radius = param_vector[14]
        start_angle = param_vector[15]
        end_angle = param_vector[16] 
        if start_angle > end_angle:
            end_angle += 360

        start_point = center + radius * np.array([np.cos(np.radians(start_angle)), np.sin(np.radians(start_angle))])
        end_point = center + radius * np.array([np.cos(np.radians(end_angle)), np.sin(np.radians(end_angle))])

        boundary_points = [start_point, end_point]
        for angle, point in [
            (0, center + [radius, 0]), (90, center + [0, radius]), (180, center - [radius, 0]), (270, center - [0, radius]),
            (360, center + [radius, 0]), (450, center + [0, radius]), (540, center - [radius, 0]), (630, center - [0, radius])
        ]:
            if start_angle <= angle <= end_angle:
                boundary_points.append(point)

        min_point = np.min(boundary_points, axis=0)
        max_point = np.max(boundary_points, axis=0)

    else:
        return None, None

    centroid = (min_point + max_point) / 2
    return min_point, max_point, centroid

def overlap_or_near(bbox1, bbox2):
    """Check if two bounding boxes overlap or are near each other."""
    min1, max1 = bbox1
    min2, max2 = bbox2

    overlap_x = not (max1[0] < min2[0] or max2[0] < min1[0])
    overlap_y = not (max1[1] < min2[1] or max2[1] < min1[1])
    if overlap_x and overlap_y:
        return True

    near_x = abs(min1[0] - max2[0]) < NEAR_THRESHOLD or abs(min2[0] - max1[0]) < NEAR_THRESHOLD
    near_y = abs(min1[1] - max2[1]) < NEAR_THRESHOLD or abs(min2[1] - max1[1]) < NEAR_THRESHOLD

    return (overlap_x and near_y) or (overlap_y and near_x)

class UnionFind:
    """Union-Find (Disjoint Set Union) data structure."""
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            # Union by rank
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def assign_clusters_with_union_find(bounding_boxes):
    """Assign each primitive to a cluster using Union-Find based on overlapping or near overlapping bounding boxes."""
    num_boxes = len(bounding_boxes)
    uf = UnionFind(num_boxes)

    # Iterate over all pairs of bounding boxes and connect them if they overlap or are near
    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            if overlap_or_near(bounding_boxes[i], bounding_boxes[j]):
                uf.union(i, j)

    # Assign clusters based on the connected components
    clusters = {}
    for i in range(num_boxes):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    return clusters


def draw_plot_with_boundaries(features_array, bounding_boxes, clusters, save_path):
    """Draw the entities of dxf as a plot using matplotlib with bounding boxes and boundary lines."""
    fig, ax = plt.subplots()

    for feature, (min_point, max_point) in zip(features_array, bounding_boxes):
        visibility = feature[0:2]
        entity_type = feature[2:5]
        params = feature[5:]
        color = 'blue' if np.array_equal(visibility, [1, 0]) else 'gray'

        if np.array_equal(entity_type, [1, 0, 0]):  # Line
            start = (params[0], params[1])
            end = (params[2], params[3])
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=0.5)

        elif np.array_equal(entity_type, [0, 1, 0]):  # Circle
            center = (params[4], params[5])
            radius = params[14]
            if radius > 0:
                circle_plot = Circle(center, radius, color='red', fill=False, linewidth=0.5)
                ax.add_patch(circle_plot)
                if add_dot:
                    ax.plot(center[0], center[1], 'ro', markersize=2)

        elif np.array_equal(entity_type, [0, 0, 1]):  # Arc
            center = (params[4], params[5])
            radius = params[14]
            start_angle = params[15] 
            end_angle = params[16]
            if start_angle > end_angle:
                end_angle += 360
            arc_plot = Arc(center, 2 * radius, 2 * radius, angle=0, theta1=start_angle, theta2=end_angle, color='green', fill=False, linewidth=0.5)
            ax.add_patch(arc_plot)
            if add_dot:
                ax.plot(center[0], center[1], 'go', markersize=2)

        width = max_point[0] - min_point[0]
        height = max_point[1] - min_point[1]
        rect = Rectangle(min_point, width, height, edgecolor='purple', facecolor='none', linestyle='--', linewidth=0.5)
        ax.add_patch(rect)

    for i, (cluster_id, indices) in enumerate(clusters.items()): 
        cluster_label = cluster_labels[i % len(cluster_labels)]  
        for idx in indices:
            min_point, max_point = bounding_boxes[idx]
            centroid = (min_point + max_point) / 2
            ax.text(centroid[0], centroid[1], cluster_label, fontsize=6, color='black', ha='center', va='center')
            #ax.text(centroid[0], centroid[1], str(cluster_id), fontsize=6, color='black', ha='center', va='center')

    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('DXF Feature Plot: Lines, Circles, and Arcs with Bounding Boxes and Clusters')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print(f"Figure saved as {save_path}")

def cluster_and_plot_with_union_find(npy_file_path, output_image_path):
    """Cluster primitives using Union-Find and plot with bounding boxes and boundary lines."""
    features_array = load_npy_data(npy_file_path)
    
    bounding_boxes = [
        calculate_bounding_box(parse_features(feature_vector)[1], parse_features(feature_vector)[2])[:2]
        for feature_vector in features_array
    ]
    bounding_boxes = [box for box in bounding_boxes if box[0] is not None]
    
    clusters = assign_clusters_with_union_find(bounding_boxes)
    # print("cluster: ", clusters)
    print(f"Total number of clusters: {len(clusters)}")

    # Visualization
    draw_plot_with_boundaries(features_array, bounding_boxes, clusters, output_image_path)

def process_all_npy_files_in_folder(folder_path, output_folder_path):
    """Process all .npy files in the specified folder."""
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for npy_file in os.listdir(folder_path):
        if npy_file.endswith(".npy"):
            npy_file_path = os.path.join(folder_path, npy_file)
            output_image_path = os.path.join(output_folder_path, f"{os.path.splitext(npy_file)[0]}_clustered_plot.png")
            cluster_and_plot_with_union_find(npy_file_path, output_image_path)
if __name__ == "__main__":
    npy_folder_path = "data/dxf_data/006"
    output_folder_path = "intermediate_files/clustered_img"
    process_all_npy_files_in_folder(npy_folder_path, output_folder_path)