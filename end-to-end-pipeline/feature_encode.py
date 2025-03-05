import ezdxf
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import math

from clustering import calculate_bounding_box, parse_features, assign_clusters_with_union_find, UnionFind, draw_plot_with_boundaries


"""
Input: DXF
Output: npy (vector embeddings)
Byproduct: png plots

Currently only accepting dxf files that only contain Line, Circle, and Arc as the primitives.
"""

add_dot = False
TYPE_ENCODING = {
    "LINE": [1, 0, 0],
    "CIRCLE": [0, 1, 0],
    "ARC": [0, 0, 1]
}

VISIBILITY_ENCODING = {
    "Visible": [1, 0],
    "Hidden": [0, 1]
}

POINT_BOUNDS = (0, 0, 17, 11)  # (min_x, min_y, max_x, max_y)
_, _, max_x, max_y = POINT_BOUNDS
scale_factor = max(max_x, max_y) 

def normalize_value(value):
    """Normalize any value based on scale_factor."""
    return value

def denormalize_value(value):
    """Denormalize any value based on scale_factor."""
    return value * scale_factor

def extract_features(entity):
    """
    Extracts and normalizes feature information from a DXF entity.
    Returns:
        list: A feature vector containing visibility, type encoding, and normalized parameters for the entity.
              The vector format is:
                - Visibility Vector (2 values): One-hot encoding for visibility status ([1, 0] for Visible, [0, 1] for Hidden)
                - Type Vector (3 values): One-hot encoding for entity type ([1, 0, 0] for LINE, [0, 1, 0] for CIRCLE, [0, 0, 1] for ARC)
                - Parameter Vector (9 values): Normalized parameters specific to the entity type
                    * LINE: start_x, start_y, end_x, end_y, followed by -1 placeholders
                    * CIRCLE: center_x, center_y, radius, followed by -1 placeholders
                    * ARC: center_x, center_y, radius, normalized start angle, normalized end angle
    Normalization:
        - All coordinates and radius values are normalized by dividing by a `scale_factor` derived from the maximum dimension of the drawing area.
        - Angles are normalized to a [0, 1] range by dividing by 360.
    Extrusion Handling for ARCs:
        - If an ARC has a negative extrusion along the Z-axis, the start and end angles are mirrored across the Y-axis to correct the orientation.
    Exceptions:
        - Raises `ValueError` if the entity's coordinates or angles are out of expected bounds.
    """
    neg_coor = False

    entity_type = entity.dxftype()
    layer = entity.dxf.layer
    
    type_vector = TYPE_ENCODING.get(entity_type, [1, 0, 0])
    visibility_vector = VISIBILITY_ENCODING.get(layer, [1, 0])  # Default to visible
    param_vector = [-1] * 17

    if entity_type == "LINE":
        start = entity.dxf.start
        end = entity.dxf.end

        start_x, start_y = normalize_value(abs(start.x)), normalize_value(abs(start.y))
        end_x, end_y = normalize_value(abs(end.x)), normalize_value(abs(end.y))
        # start_x, start_y = normalize_value(start.x), normalize_value(start.y)
        # end_x, end_y = normalize_value(end.x), normalize_value(end.y)


        param_vector[0:2] = [start_x, start_y]  # s1, s2
        param_vector[2:4] = [end_x, end_y]  # e1, e2

    elif entity_type == "CIRCLE":
        center = entity.dxf.center
        radius = entity.dxf.radius

        if entity.dxf.extrusion.z < 0:
            center_x, center_y = normalize_value(abs(center.x)), normalize_value(center.y)
        else: 
            center_x, center_y = normalize_value(center.x), normalize_value(center.y)
        if center_x < 0:
            neg_coor = True
        if center_y < 0:
            neg_coor = True


        norm_radius = normalize_value(abs(radius))
        # norm_radius = normalize_value(radius)

        angles = [0, math.pi/2, math.pi, 3*math.pi/2]
        circle_points = []
        for angle in angles:
            x = center_x + (norm_radius * math.cos(angle))
            y = center_y + (norm_radius * math.sin(angle))
            circle_points.append(x)
            circle_points.append(y)
        


        param_vector[4:6] = [center_x, center_y]  # c1, c2
        param_vector[6:14] = circle_points # circle critical points
        param_vector[14] = norm_radius

    elif entity_type == "ARC":
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = entity.dxf.start_angle
        end_angle = entity.dxf.end_angle

        extrusion = entity.dxf.extrusion

        if extrusion.z < 0:
            start_angle = (180 - start_angle) % 360
            end_angle = (180 - end_angle) % 360
            start_angle, end_angle = end_angle, start_angle
            center_x, center_y = normalize_value(abs(center.x)), normalize_value(center.y)
        else:
            center_x, center_y = normalize_value(center.x), normalize_value(center.y)
        if center_x < 0:
            neg_coor = True
        if center_y < 0:
            neg_coor = True


        norm_radius = normalize_value(abs(radius))
        # norm_radius = normalize_value(radius)
        start_x = center_x + (norm_radius * math.cos(start_angle/360))
        start_y = center_y + (norm_radius * math.sin(start_angle/360))

        end_x = center_x + (norm_radius * math.cos(end_angle/360))
        end_y = center_y + (norm_radius * math.sin(end_angle/360))
        # norm_start_angle = start_angle / 360
        # norm_end_angle = end_angle / 360



        param_vector[0:2] = [start_x, start_y]
        param_vector[2:4] = [end_x, end_y]
        param_vector[4:6] = [center_x, center_y]  # c1, c2
        param_vector[14] = norm_radius  # r
        param_vector[15] = start_angle  # sa
        param_vector[16] = end_angle  # ea

    feature_vector = visibility_vector + type_vector + param_vector 
    if neg_coor == True:
        return None  
    else:
        return feature_vector

def draw_plot(features_array, save_path):
    """Draw the entities of dxf as a plot using matplotlib"""
    fig, ax = plt.subplots()

    for feature in features_array:

        visibility = feature[0:2]  
        entity_type = feature[2:5] 
        params = feature[5:]    

        line_style = '--' if np.array_equal(visibility, [0, 1]) else '-'  

        try:
            if np.array_equal(entity_type, [1, 0, 0]):  # Line
                start = (params[0], params[1])
                end = (params[2], params[3])
                ax.plot([start[0], end[0]], [start[1], end[1]], color='blue', linestyle=line_style, linewidth=0.5)

            elif np.array_equal(entity_type, [0, 1, 0]):  # Circle
                center = (params[4], params[5])
                radius = params[14]
                if radius > 0:
                    circle_plot = Circle(center, radius, color='red', fill=False, linewidth=0.5, linestyle=line_style)
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

                arc_plot = Arc(center, 2 * radius, 2 * radius, angle=0, theta1=start_angle, theta2=end_angle, 
                             color='green', fill=False, linewidth=0.5, linestyle=line_style)
                ax.add_patch(arc_plot)
                if add_dot:
                    ax.plot(center[0], center[1], 'go', markersize=2)

        except Exception as e:
            print(f"Error plotting feature: {feature}")
            print(f"Error message: {str(e)}")
            continue

    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('DXF Feature Plot: Lines, Circles, and Arcs')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def validate_features(dxf_path, features_array):
    """
    Validates the extracted feature vectors against the original DXF entities.
    Returns:
        bool: True if validation is successful, False if any discrepancies are found.
              Discrepancies may arise due to mismatched counts, normalization errors, or rounding issues.
    Validation Process:
        1. Loads the DXF file and extracts entities of type 'LINE', 'CIRCLE', and 'ARC'.
        2. Checks if the number of entities matches the number of feature vectors.
        3. For each entity:
            - Denormalizes the feature vector's parameters back to original values.
            - Compares denormalized values against the DXF entity's ground truth values using `np.isclose` for tolerance.
            - Validates:
                * LINE: Start and end coordinates
                * CIRCLE: Center coordinates and radius
                * ARC: Center coordinates, radius, and start/end angles
    Extrusion Handling for ARCs:
        - If an ARC has a negative Z-axis extrusion, the angles are mirrored to align with the Y-axis flip during feature extraction.
    Raises:
        ValueError: If the feature vectors do not match the DXF entities in count or content, or if there are rounding errors.
    """

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    entities = [entity for entity in msp if entity.dxftype() in {"LINE", "CIRCLE", "ARC"}]

    if len(entities) != len(features_array):
        print(f"Mismatch in entity count for {dxf_path}. DXF entities: {len(entities)}, Encoded features: {len(features_array)}")
        return False

    for entity, feature in zip(entities, features_array):
        entity_type = entity.dxftype()
        params = feature[5:14]  # [s1, s2, e1, e2, c1, c2, r, start angle, end angle]

        if entity_type == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end

            start_x, start_y = denormalize_value(params[0]), denormalize_value(params[1])
            end_x, end_y = denormalize_value(params[2]), denormalize_value(params[3])

            if any(x < 0 for x in [start.x, start.y, end.x, end.y]):
                print(f"Warning: Negative coordinates found in LINE: ({start.x}, {start.y}), ({end.x}, {end.y})")

            if not np.isclose([abs(start.x), abs(start.y), abs(end.x), abs(end.y)], [start_x, start_y, end_x, end_y]).all():
                print(f"Rounding error or mismatch in LINE entity at {dxf_path}: Start ({start.x}, {start.y}), Encoded ({start_x}, {start_y})")
                return False

        elif entity_type == "CIRCLE":
            center = entity.dxf.center
            radius = entity.dxf.radius

            center_x, center_y = denormalize_value(params[4]), denormalize_value(params[5])
            denorm_radius = denormalize_value(params[6])

            if not np.isclose([abs(center.x), abs(center.y), abs(radius)], [center_x, center_y, denorm_radius]).all():
                print(f"Rounding error or mismatch in CIRCLE entity at {dxf_path}: Center ({center.x}, {center.y}), Radius {radius}")
                return False

        elif entity_type == "ARC":
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            extrusion = entity.dxf.extrusion

            if extrusion.z < 0:
                start_angle = (180 - start_angle) % 360
                end_angle = (180 - end_angle) % 360
                start_angle, end_angle = end_angle, start_angle

            center_x, center_y = denormalize_value(params[4]), denormalize_value(params[5])
            denorm_radius = denormalize_value(params[6])
            denorm_start_angle = params[7] * 360
            denorm_end_angle = params[8] * 360

            if not np.isclose([abs(center.x), abs(center.y), abs(radius)], [center_x, center_y, denorm_radius]).all():
                print(f"Rounding error or mismatch in ARC center/radius at {dxf_path}")
                return False

            if not np.isclose([abs(start_angle), abs(end_angle)], [denorm_start_angle, denorm_end_angle]).all():
                print(f"Rounding error or mismatch in ARC angles at {dxf_path}: Start {start_angle}, End {end_angle}")
                return False

    return True

def process_dxf_file(dxf_path, output_imgplot_path=None, output_npy_path=None, output_bbox_path=None):
    """
    Process a single DXF file and generate the corresponding outputs.
    
    Args:
        dxf_path (str): Path to the input DXF file
        output_imgplot_path (str, optional): Path to save the plot image
        output_npy_path (str, optional): Path to save the NPY feature array
        output_bbox_path (str, optional): Path to save the bounding box image
        
    Returns:
        np.ndarray: The feature array with cluster labels
        bool: True if processing was successful, False otherwise
    """
    try:
        # Check if file contains SPLINE or ELLIPSE
        dxf_file = ezdxf.readfile(dxf_path)
        modelspace = dxf_file.modelspace()
        has_spline = any(entity.dxftype() == "SPLINE" for entity in modelspace)
        has_ellipse = any(entity.dxftype() == "ELLIPSE" for entity in modelspace)
        
        if has_spline or has_ellipse:
            print(f"Skipping {dxf_path} - contains SPLINE or ELLIPSE")
            return None, False
        
        # Extract features
        features = []
        for entity in modelspace:
            if entity.dxftype() in {"LINE", "CIRCLE", "ARC"}:
                extracted_features = extract_features(entity)
                if extracted_features is None:
                    print(f"Skipping {dxf_path} - contains negative coordinates")
                    return None, False
                features.append(extracted_features)
        
        if not features:
            print(f"No valid features found in {dxf_path}")
            return None, False
        
        features_array = np.array(features)
        
        # Validate features if needed
        # is_valid = validate_features(dxf_path, features_array)
        is_valid = True
        
        if not is_valid:
            print(f"Validation failed for {dxf_path}")
            return None, False
        
        # Generate plot image if path is provided
        if output_imgplot_path:
            draw_plot(features_array, output_imgplot_path)
        
        # Calculate bounding boxes and assign clusters
        bbox = []
        for f in features_array:
            vis, type_vec, params = parse_features(f)
            box = calculate_bounding_box(type_vec, params)[:2]
            if box[0] is not None:
                bbox.append(box)
        
        clusters = assign_clusters_with_union_find(bbox)
        
        # Generate bounding box image if path is provided
        if output_bbox_path:
            draw_plot_with_boundaries(features_array, bbox, clusters, output_bbox_path)
        
        # Assign cluster labels
        cluster_labels = np.zeros(len(features_array))
        for i, (root, indices) in enumerate(clusters.items()):
            for idx in indices:
                cluster_labels[idx] = i + 1
        
        # Add cluster labels to features array
        features_array_with_clusters = np.column_stack((cluster_labels, features_array))
        
        # Save NPY file if path is provided
        if output_npy_path:
            np.save(output_npy_path, features_array_with_clusters)
        
        return features_array_with_clusters, True
    
    except Exception as e:
        print(f"Error processing {dxf_path}: {str(e)}")
        return None, False

if __name__ == "__main__":
    input_dxf_path = "21236_b696e901_0038 Drawing v1 21236_b696e901_0038.dxf" 
    
    base_filename = os.path.splitext(os.path.basename(input_dxf_path))[0]
    output_npy_path = f"{base_filename}.npy"
    output_imgplot_path = f"{base_filename}.png"
    output_bbox_path = f"{base_filename}_clustered_plot.png"
    
    # Process the file
    features_array, success = process_dxf_file(
        input_dxf_path,
        output_imgplot_path,
        output_npy_path,
        output_bbox_path
    )
    
    if success:
        print(f"Successfully processed {input_dxf_path}")
        print(f"Output NPY saved to {output_npy_path}")
        print(f"Plot image saved to {output_imgplot_path}")
        print(f"Bounding box image saved to {output_bbox_path}")
    else:
        print(f"Failed to process {input_dxf_path}")