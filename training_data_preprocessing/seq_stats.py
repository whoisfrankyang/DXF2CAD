# Author: Bohan Yang (Jan 3, 2025)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Given processed 3D CAD json files, perform some basic statistics and analysis. 
"""
def analyze_taper_angles(folder_path):
    """Analyze taper angles in sequences"""
    total_files = 0
    files_with_taper = 0
    nonzero_taper_angles = []
    
    # Process all JSON files
    for filename in os.listdir(folder_path):
        if filename.endswith('_extracted.json'):
            file_path = os.path.join(folder_path, filename)
            total_files += 1
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check each command for taper angles
                for cmd in data:
                    if cmd["command"] == "Extrusion":
                        taper_angle = cmd.get("entity_one_taper_angle")
                        if taper_angle is not None and abs(taper_angle) > 1e-10:
                            files_with_taper += 1
                            nonzero_taper_angles.append({
                                'file': filename,
                                'angle': taper_angle
                            })
                            break  # Only count each file once
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Print statistics
    print(f"\nTaper Angle Statistics:")
    print(f"Total files processed: {total_files}")
    print(f"Files with non-zero taper angles: {files_with_taper}")
    print(f"Percentage: {(files_with_taper/total_files)*100:.2f}%")
    
    if nonzero_taper_angles:
        print("\nFiles with non-zero taper angles:")
        for entry in nonzero_taper_angles:
            print(f"File: {entry['file']}, Angle: {entry['angle']}")

def analyze_value_distribution(folder_path):
    """Analyze distribution of all numerical values"""
    x_coords = []
    y_coords = []
    z_coords = []
    extrusion_distances = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('_extracted.json'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                for cmd in data:
                    if cmd["command"] == "Line":
                        x_coords.extend([cmd["start"][0], cmd["end"][0]])
                        y_coords.extend([cmd["start"][1], cmd["end"][1]])
                        z_coords.extend([cmd["start"][2], cmd["end"][2]])
                    
                    elif cmd["command"] == "Extrusion":
                        dist1 = cmd.get("entity_one_distance")
                        dist2 = cmd.get("entity_two_distance")
                        if dist1 is not None:
                            extrusion_distances.append(abs(dist1))
                        if dist2 is not None:
                            extrusion_distances.append(abs(dist2))
                            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    values = {
        'x': np.array(x_coords),
        'y': np.array(y_coords),
        'z': np.array(z_coords),
        'extrusion': np.array(extrusion_distances)
    }
    
    # Print statistics for each type
    for name in values:
        data = values[name]
        print(f"\n{name.upper()} Coordinate Statistics:")
        print(f"Count: {len(data)}")
        print(f"Min: {np.min(data):.3f}")
        print(f"Max: {np.max(data):.3f}")
        print(f"Mean: {np.mean(data):.3f}")
        print(f"Std: {np.std(data):.3f}")
        
        # Value distribution
        ranges = [0, 1, 5, 10, 20, 50, 100, 500, 1000, float('inf')]
        labels = ['0-1', '1-5', '5-10', '10-20', '20-50', '50-100', '100-500', '500-1000', '>1000']
        
        counts = np.zeros(len(labels))
        for val in np.abs(data):
            for i in range(len(ranges)-1):
                if ranges[i] <= val < ranges[i+1]:
                    counts[i] += 1
                    break
        
        print("\nValue Distribution:")
        print("Range        Count     Percentage")
        print("-" * 40)
        for label, count in zip(labels, counts):
            percentage = (count/len(data))*100
            print(f"{label:<12} {int(count):<10} {percentage:.2f}%")
        
        # Print outliers
        threshold = np.percentile(np.abs(data), 99)
        outliers = data[np.abs(data) > threshold]
        if len(outliers) > 0:
            print(f"\nPotential outliers (>99th percentile, {threshold:.2f}):")
            print(outliers)
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of Sequence Values')
    
    sns.histplot(values['x'], ax=axes[0,0], bins=50)
    axes[0,0].set_title('X Coordinates')
    
    sns.histplot(values['y'], ax=axes[0,1], bins=50)
    axes[0,1].set_title('Y Coordinates')
    
    sns.histplot(values['z'], ax=axes[1,0], bins=50)
    axes[1,0].set_title('Z Coordinates')
    
    sns.histplot(values['extrusion'], ax=axes[1,1], bins=50)
    axes[1,1].set_title('Extrusion Distances')
    
    plt.tight_layout()
    plt.show()
    
    # Box plots
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([values['x'], values['y'], values['z'], values['extrusion']], 
               labels=['X', 'Y', 'Z', 'Extrusion'])
    ax.set_title('Value Distributions by Type')
    plt.show()

def check_zero_coordinates(folder_path):
    """Check if each file has at least one zero in any dimension (x, y, or z)"""
    total_files = 0
    files_without_zero = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('_extracted.json'):
            file_path = os.path.join(folder_path, filename)
            total_files += 1
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Collect all points for this file
                has_zero = False
                
                for cmd in data:
                    if cmd["command"] == "Line":
                        # Check start point
                        if any(abs(coord) < 1e-10 for coord in cmd["start"]):
                            has_zero = True
                            break
                        # Check end point
                        if any(abs(coord) < 1e-10 for coord in cmd["end"]):
                            has_zero = True
                            break
                
                if not has_zero:
                    # If no zero found, collect minimum values for reporting
                    x_coords = []
                    y_coords = []
                    z_coords = []
                    for cmd in data:
                        if cmd["command"] == "Line":
                            x_coords.extend([cmd["start"][0], cmd["end"][0]])
                            y_coords.extend([cmd["start"][1], cmd["end"][1]])
                            z_coords.extend([cmd["start"][2], cmd["end"][2]])
                    
                    files_without_zero.append({
                        'file': filename,
                        'min_x': min(x_coords),
                        'min_y': min(y_coords),
                        'min_z': min(z_coords)
                    })
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Print results
    print("\nZero Coordinate Check:")
    print(f"Total files processed: {total_files}")
    print(f"Files without any zero coordinates: {len(files_without_zero)}")
    print(f"Percentage: {(len(files_without_zero)/total_files)*100:.2f}%")
    
    if files_without_zero:
        print("\nFiles with no zero coordinates in any dimension:")
        for entry in files_without_zero:
            print(f"\nFile: {entry['file']}")
            print(f"Minimum coordinates:")
            print(f"  X: {entry['min_x']:.3f}")
            print(f"  Y: {entry['min_y']:.3f}")
            print(f"  Z: {entry['min_z']:.3f}")

def analyze_extrusion_sides(folder_path):
    """Analyze which sides have extrusion distances"""
    total_files = 0
    side1_only = 0
    side2_only = 0
    both_sides = 0
    files_by_type = {
        'side1_only': [],
        'side2_only': [],
        'both_sides': []
    }
    
    for filename in os.listdir(folder_path):
        if filename.endswith('_extracted.json'):
            file_path = os.path.join(folder_path, filename)
            total_files += 1
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                for cmd in data:
                    if cmd["command"] == "Extrusion":
                        dist1 = cmd.get("entity_one_distance")
                        dist2 = cmd.get("entity_two_distance")
                        
                        if dist1 is not None and dist2 is None:
                            side1_only += 1
                            files_by_type['side1_only'].append(filename)
                            break
                        elif dist1 is None and dist2 is not None:
                            side2_only += 1
                            files_by_type['side2_only'].append(filename)
                            break
                        elif dist1 is not None and dist2 is not None:
                            both_sides += 1
                            files_by_type['both_sides'].append(filename)
                            break
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Print statistics
    print("\nExtrusion Side Statistics:")
    print(f"Total files processed: {total_files}")
    print(f"\nDistribution:")
    print(f"Side 1 only: {side1_only} files ({(side1_only/total_files)*100:.2f}%)")
    print(f"Side 2 only: {side2_only} files ({(side2_only/total_files)*100:.2f}%)")
    print(f"Both sides: {both_sides} files ({(both_sides/total_files)*100:.2f}%)")
    
    # Print example files for each case
    if files_by_type['side1_only']:
        print("\nExample files with Side 1 only:")
        for f in files_by_type['side1_only'][:5]:  # Show first 5 examples
            print(f"  {f}")
    
    if files_by_type['side2_only']:
        print("\nExample files with Side 2 only:")
        for f in files_by_type['side2_only'][:5]:
            print(f"  {f}")
    
    if files_by_type['both_sides']:
        print("\nExample files with Both sides:")
        for f in files_by_type['both_sides'][:5]:
            print(f"  {f}")

def analyze_lengths(folder_path):
    """Analyze line lengths and extrusion distances"""
    line_lengths = []
    extrusion_lengths = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('_extracted.json'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                for cmd in data:
                    if cmd["command"] == "Line":
                        # Calculate Euclidean distance between start and end points
                        start = np.array(cmd["start"])
                        end = np.array(cmd["end"])
                        length = np.linalg.norm(end - start)
                        line_lengths.append(length)
                    
                    elif cmd["command"] == "Extrusion":
                        # Collect all non-None extrusion distances
                        if cmd.get("entity_one_distance") is not None:
                            extrusion_lengths.append(abs(cmd["entity_one_distance"]))
                        if cmd.get("entity_two_distance") is not None:
                            extrusion_lengths.append(abs(cmd["entity_two_distance"]))
                            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Convert to numpy arrays
    line_lengths = np.array(line_lengths)
    extrusion_lengths = np.array(extrusion_lengths)
    
    # Print statistics for both types
    for name, data in [("Line", line_lengths), ("Extrusion", extrusion_lengths)]:
        print(f"\n{name} Length Statistics:")
        print(f"Count: {len(data)}")
        print(f"Min: {np.min(data):.3f}")
        print(f"Max: {np.max(data):.3f}")
        print(f"Mean: {np.mean(data):.3f}")
        print(f"Median: {np.median(data):.3f}")
        print(f"Std: {np.std(data):.3f}")
        
        # Value distribution
        ranges = [0, 1, 5, 10, 20, 50, 100, 500, 1000, float('inf')]
        labels = ['0-1', '1-5', '5-10', '10-20', '20-50', '50-100', '100-500', '500-1000', '>1000']
        
        counts = np.zeros(len(labels))
        for val in data:
            for i in range(len(ranges)-1):
                if ranges[i] <= val < ranges[i+1]:
                    counts[i] += 1
                    break
        
        print("\nLength Distribution:")
        print("Range        Count     Percentage")
        print("-" * 40)
        for label, count in zip(labels, counts):
            percentage = (count/len(data))*100
            print(f"{label:<12} {int(count):<10} {percentage:.2f}%")
    
    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Distribution of Lengths')
    
    sns.histplot(line_lengths, ax=ax1, bins=50)
    ax1.set_title('Line Lengths')
    ax1.set_xlabel('Length')
    
    sns.histplot(extrusion_lengths, ax=ax2, bins=50)
    ax2.set_title('Extrusion Distances')
    ax2.set_xlabel('Distance')
    
    plt.tight_layout()
    plt.show()
    
    # Box plots
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([line_lengths, extrusion_lengths], 
               labels=['Line Lengths', 'Extrusion Distances'])
    ax.set_title('Length Distributions by Type')
    plt.show()

def analyze_file_length_ranges(folder_path):
    """Analyze files based on line and extrusion length ranges"""
    total_files = 0
    files_with_small = 0  # Files with lengths between 0-1 (but not >500)
    files_valid = 0       # Files with all lengths between 1-500
    files_with_large = 0  # Files with lengths >500 (but not <1)
    files_with_both = 0   # Files with both small and large values
    
    small_files = []  # Files with lengths < 1
    large_files = []  # Files with lengths > 500
    
    for filename in os.listdir(folder_path):
        if filename.endswith('_extracted.json'):
            file_path = os.path.join(folder_path, filename)
            total_files += 1
            
            has_small = False  # 0-1
            has_large = False  # >500
            has_valid = True   # all between 1-500
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check each command in file
                for cmd in data:
                    if cmd["command"] == "Line":
                        start = np.array(cmd["start"])
                        end = np.array(cmd["end"])
                        length = np.linalg.norm(end - start)
                        
                        if length < 1:
                            has_small = True
                            has_valid = False
                        elif length > 500:
                            has_large = True
                            has_valid = False
                    
                    elif cmd["command"] == "Extrusion":
                        dist1 = cmd.get("entity_one_distance")
                        dist2 = cmd.get("entity_two_distance")
                        
                        if dist1 is not None:
                            if abs(dist1) < 1:
                                has_small = True
                                has_valid = False
                            elif abs(dist1) > 500:
                                has_large = True
                                has_valid = False
                                
                        if dist2 is not None:
                            if abs(dist2) < 1:
                                has_small = True
                                has_valid = False
                            elif abs(dist2) > 500:
                                has_large = True
                                has_valid = False
                
                # Categorize file
                if has_small and has_large:
                    files_with_both += 1
                elif has_small:
                    files_with_small += 1
                    small_files.append(filename)
                elif has_large:
                    files_with_large += 1
                    large_files.append(filename)
                elif has_valid:
                    files_valid += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Print statistics
    print("\nLength Range Analysis:")
    print(f"Total files processed: {total_files}")
    print("\nBreakdown:")
    print(f"Files with all lengths between 1-500: {files_valid} ({(files_valid/total_files)*100:.2f}%)")
    print(f"Files with lengths 0-1 only: {files_with_small} ({(files_with_small/total_files)*100:.2f}%)")
    print(f"Files with lengths >500 only: {files_with_large} ({(files_with_large/total_files)*100:.2f}%)")
    print(f"Files with both small and large values: {files_with_both} ({(files_with_both/total_files)*100:.2f}%)")
    
    # Print example files
    if small_files:
        print("\nExample files with small lengths (0-1):")
        for f in small_files[:5]:  # Show first 5 examples
            print(f"  {f}")
    
    if large_files:
        print("\nExample files with large lengths (>500):")
        for f in large_files[:5]:
            print(f"  {f}")

def analyze_operation_types(folder_path):
    """Analyze distribution of extrusion operation types, start extent types, and extent types"""
    # Operation type tracking
    operation_counts = {
        'NewBodyFeatureOperation': 0,
        'JoinFeatureOperation': 0,
        'CutFeatureOperation': 0,
        'IntersectFeatureOperation': 0,
        'Other': 0
    }
    
    # Start extent type tracking
    extent_counts = {
        'ProfilePlaneStartDefinition': 0,
        'OffsetStartDefinition': 0,
        'Other': 0
    }
    
    # Feature extent type tracking
    feature_extent_counts = {
        'OneSideFeatureExtentType': 0,      # Extrude in one direction
        'SymmetricFeatureExtentType': 0,     # Equal distance both sides
        'TwoSidesFeatureExtentType': 0,      # Different distances each side
        'Other': 0
    }
    
    total_files = 0
    files_by_operation = {op: [] for op in operation_counts}
    files_by_extent = {ext: [] for ext in extent_counts}
    files_by_feature_extent = {ext: [] for ext in feature_extent_counts}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('_extracted.json'):
            file_path = os.path.join(folder_path, filename)
            total_files += 1
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Track types in this file
                file_operations = set()
                file_extents = set()
                file_feature_extents = set()
                
                for cmd in data:
                    if cmd["command"] == "Extrusion":
                        # Track operation type
                        operation = cmd.get("operation")
                        if operation in operation_counts:
                            operation_counts[operation] += 1
                            file_operations.add(operation)
                        else:
                            operation_counts['Other'] += 1
                            file_operations.add('Other')
                        
                        # Track start extent type
                        start_extent = cmd.get("start_extent", {}).get("type")
                        if start_extent in extent_counts:
                            extent_counts[start_extent] += 1
                            file_extents.add(start_extent)
                        else:
                            extent_counts['Other'] += 1
                            file_extents.add('Other')
                            
                        # Track feature extent type
                        feature_extent = cmd.get("extent_type")
                        if feature_extent in feature_extent_counts:
                            feature_extent_counts[feature_extent] += 1
                            file_feature_extents.add(feature_extent)
                        else:
                            feature_extent_counts['Other'] += 1
                            file_feature_extents.add('Other')
                
                # Add filename to each type found in this file
                for op in file_operations:
                    files_by_operation[op].append(filename)
                for ext in file_extents:
                    files_by_extent[ext].append(filename)
                for ext in file_feature_extents:
                    files_by_feature_extent[ext].append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Print operation type statistics
    print("\nExtrusion Operation Type Statistics:")
    print(f"Total files processed: {total_files}")
    print("\nOperation counts:")
    total_ops = sum(operation_counts.values())
    for op, count in operation_counts.items():
        if count > 0:
            print(f"{op}: {count} instances ({(count/total_ops)*100:.2f}%)")
    
    # Print start extent type statistics
    print("\nStart Extent Type Statistics:")
    total_exts = sum(extent_counts.values())
    for ext, count in extent_counts.items():
        if count > 0:
            print(f"{ext}: {count} instances ({(count/total_exts)*100:.2f}%)")
    
    # Print feature extent type statistics
    print("\nFeature Extent Type Statistics:")
    total_feature_exts = sum(feature_extent_counts.values())
    for ext, count in feature_extent_counts.items():
        if count > 0:
            print(f"{ext}: {count} instances ({(count/total_feature_exts)*100:.2f}%)")
            if ext == 'OneSideFeatureExtentType':
                print("  (extrude in one direction only)")
            elif ext == 'SymmetricFeatureExtentType':
                print("  (equal distance in both directions)")
            elif ext == 'TwoSidesFeatureExtentType':
                print("  (different distances in each direction)")
    
    # Print example files for each type
    print("\nExample files by feature extent type:")
    for ext, files in files_by_feature_extent.items():
        if files:
            print(f"\n{ext}:")
            for f in files[:5]:  # Show first 5 examples
                print(f"  {f}")

def analyze_sequence_lengths(folder_path):
    """Analyze the number of operations (Line and Extrusion commands) in sequences"""
    line_counts = []
    extrusion_counts = []
    total_op_counts = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('_extracted.json'):
            try:
                with open(os.path.join(folder_path, filename), 'r') as f:
                    data = json.load(f)
                
                # Count operations in this sequence
                n_lines = sum(1 for cmd in data if cmd["command"] == "Line")
                n_extrusions = sum(1 for cmd in data if cmd["command"] == "Extrusion")
                
                line_counts.append(n_lines)
                extrusion_counts.append(n_extrusions)
                total_op_counts.append(n_lines + n_extrusions)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Convert to numpy arrays for statistics
    line_counts = np.array(line_counts)
    extrusion_counts = np.array(extrusion_counts)
    total_op_counts = np.array(total_op_counts)
    
    print("\nSequence Length Statistics:")
    print(f"Number of sequences analyzed: {len(total_op_counts)}")
    
    print("\nTotal Operations (Lines + Extrusions):")
    print(f"Mean: {np.mean(total_op_counts):.2f}")
    print(f"Median: {np.median(total_op_counts):.2f}")
    print(f"Min: {np.min(total_op_counts)}")
    print(f"Max: {np.max(total_op_counts)}")
    print(f"Std: {np.std(total_op_counts):.2f}")
    
    print("\nLine Commands:")
    print(f"Mean: {np.mean(line_counts):.2f}")
    print(f"Median: {np.median(line_counts):.2f}")
    print(f"Min: {np.min(line_counts)}")
    print(f"Max: {np.max(line_counts)}")
    print(f"Std: {np.std(line_counts):.2f}")
    
    print("\nExtrusion Commands:")
    print(f"Mean: {np.mean(extrusion_counts):.2f}")
    print(f"Median: {np.median(extrusion_counts):.2f}")
    print(f"Min: {np.min(extrusion_counts)}")
    print(f"Max: {np.max(extrusion_counts)}")
    print(f"Std: {np.std(extrusion_counts):.2f}")
    
    # Plot distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.hist(total_op_counts, bins=50)
    plt.title('Total Operations')
    plt.xlabel('Number of Operations')
    plt.ylabel('Frequency')
    
    plt.subplot(132)
    plt.hist(line_counts, bins=50)
    plt.title('Line Commands')
    plt.xlabel('Number of Lines')
    
    plt.subplot(133)
    plt.hist(extrusion_counts, bins=50)
    plt.title('Extrusion Commands')
    plt.xlabel('Number of Extrusions')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder_path = "2D_data_preprocess/data/line_only_3D_seq_processed_clipped"

    analyze_operation_types(folder_path)