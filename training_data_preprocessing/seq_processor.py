# Author: Bohan Yang (Jan 3, 2025)

"""
Dec 29, 2024: Bohan Yang
1. Skip files with non-zero taper angles (all remaining files have zero taper angles)
    and skip files with OffsetStartDefinition as the start extent type
2. Convert negative extrusion distances to positive
3. Shift coordinates to ensure all coordinates are positive
4. Determine extrusion direction by checking which coordinate remains constant
5. Remove extra curves between last extrusion and END
6. Convert units from cm to mm (multiply by 10)
7. Clipping any files with small line length or (large coordinate values)
"""

import os
import json
import numpy as np

def determine_extrusion_direction(curves):
    """
    Determine extrusion direction by checking which coordinate remains constant
    in the profile drawing.
    """
    # Collect all coordinates
    all_coords = []
    for curve in curves:
        if curve["command"] == "Line":
            all_coords.append(curve["start"])
            all_coords.append(curve["end"])
    
    # Convert to numpy array for easier analysis
    coords = np.array(all_coords)
    
    # Check variation in each dimension
    variations = np.std(coords, axis=0)
    
    # The dimension with smallest variation is likely the extrusion direction
    min_var_idx = np.argmin(variations)
    
    # Map index to direction
    directions = ['x', 'y', 'z']
    return directions[min_var_idx]

def clean_small_numbers(value, threshold=1e-10):
    """Convert very small numbers to 0"""
    if isinstance(value, list):
        return [clean_small_numbers(x) for x in value]
    elif isinstance(value, (int, float)):
        return 0.0 if abs(value) < threshold else value
    return value

def has_nonzero_taper(data):
    """Check if sequence has any non-zero taper angles"""
    for cmd in data:
        if cmd["command"] == "Extrusion":
            taper_angle = cmd.get("entity_one_taper_angle")
            if taper_angle is not None and abs(taper_angle) > 1e-10:
                return True
    return False

def has_offset_start(data):
    """Check if sequence has any OffsetStartDefinition"""
    for cmd in data:
        if cmd["command"] == "Extrusion":
            start_extent = cmd.get("start_extent", {}).get("type")
            if start_extent == "OffsetStartDefinition":
                return True
    return False

def process_sequence(data, filename="", clip_length_range=True, clip_large_coord=True):
    """Process a single JSON sequence"""
    # 1. Skip files with non-zero taper angles or offset start definition
    if has_nonzero_taper(data):
        return None, "taper"
    if has_offset_start(data):
        return None, "offset"
    
    # 2. Handle negative extrusion distances
    for cmd in data:
        if cmd["command"] == "Extrusion":
            # Get distances, defaulting to None
            dist1 = cmd.get("entity_one_distance")
            dist2 = cmd.get("entity_two_distance")
            
            # Case 1: One negative, one None
            if dist1 is not None and dist1 < 0 and dist2 is None:
                cmd["entity_one_distance"] = None
                cmd["entity_one_taper_angle"] = None
                cmd["entity_two_distance"] = -dist1
                cmd["entity_two_taper_angle"] = 0.0
            elif dist2 is not None and dist2 < 0 and dist1 is None:
                cmd["entity_two_distance"] = None
                cmd["entity_two_taper_angle"] = None
                cmd["entity_one_distance"] = -dist2
                cmd["entity_one_taper_angle"] = 0.0
            
            # Case 2: Both negative
            elif dist1 is not None and dist2 is not None:
                if dist1 < 0 and dist2 < 0:
                    raise ValueError(f"File {filename}: Both entity_one_distance and entity_two_distance are negative")
                elif dist1 < 0 and dist2 > 0:
                    raise ValueError(f"File {filename}: entity_one_distance is negative and entity_two_distance is positive")
                elif dist1 > 0 and dist2 < 0:
                    raise ValueError(f"File {filename}: entity_one_distance is positive and entity_two_distance is negative")

    # 3. Find min values and shift coordinates
    x_min = float('inf')
    y_min = float('inf')
    z_min = float('inf')
    
    for cmd in data:
        if cmd["command"] == "Line":
            x_min = min(x_min, cmd["start"][0], cmd["end"][0])
            y_min = min(y_min, cmd["start"][1], cmd["end"][1])
            z_min = min(z_min, cmd["start"][2], cmd["end"][2])
    
    # Calculate offsets to ensure all coordinates are positive
    x_offset = x_min  # This will be subtracted, so using min directly
    y_offset = y_min
    z_offset = z_min
    
    # Apply shifts to ALL points
    shifted_data = []
    current_profile = []
    for cmd in data:
        shifted_cmd = cmd.copy()
        if cmd["command"] == "Line":
            shifted_cmd["start"] = [
                cmd["start"][0] - x_offset,  # Subtracting min shifts everything up by min value
                cmd["start"][1] - y_offset,
                cmd["start"][2] - z_offset
            ]
            shifted_cmd["end"] = [
                cmd["end"][0] - x_offset,
                cmd["end"][1] - y_offset,
                cmd["end"][2] - z_offset
            ]
            current_profile.append(shifted_cmd)
            shifted_data.append(shifted_cmd)
        else:
            shifted_data.append(shifted_cmd)
    
    # 4. Process extrusion directions and remove extra curves
    processed_data = []
    current_profile = []
    last_extrusion_index = -1
    
    # Find last extrusion index first
    for i, cmd in enumerate(shifted_data):
        if cmd["command"] == "Extrusion":
            last_extrusion_index = i
    
    # Process commands
    for i, cmd in enumerate(shifted_data):
        processed_cmd = cmd.copy()
        if cmd["command"] == "Line":
            if i < last_extrusion_index:  # Only keep lines before last extrusion
                current_profile.append(processed_cmd)
                processed_data.append(processed_cmd)
        elif cmd["command"] == "Extrusion":
            if not "direction" in cmd:
                processed_cmd["direction"] = determine_extrusion_direction(current_profile)
            processed_data.append(processed_cmd)
            current_profile = []
        elif cmd["command"] in ["START", "END"]:  # Always keep START and END
            processed_data.append(processed_cmd)
    
    # 5. Clean small numbers
    cleaned_data = []
    for cmd in processed_data:
        cleaned_cmd = cmd.copy()
        if cmd["command"] == "Line":
            cleaned_cmd["start"] = clean_small_numbers(cmd["start"])
            cleaned_cmd["end"] = clean_small_numbers(cmd["end"])
        elif cmd["command"] == "Extrusion":
            cleaned_cmd = clean_small_numbers(cleaned_cmd)
        cleaned_data.append(cleaned_cmd)
    
    # 6. Convert units from cm to mm (multiply by 10)
    converted_data = []
    for cmd in cleaned_data:
        converted_cmd = cmd.copy()
        if cmd["command"] == "Line":
            converted_cmd["start"] = [x * 10 for x in cmd["start"]]
            converted_cmd["end"] = [x * 10 for x in cmd["end"]]
        elif cmd["command"] == "Extrusion":
            if cmd.get("entity_one_distance") is not None:
                converted_cmd["entity_one_distance"] *= 10
            if cmd.get("entity_two_distance") is not None:
                converted_cmd["entity_two_distance"] *= 10
        converted_data.append(converted_cmd)
    
    # 7. Check lengths and coordinates (do this AFTER all processing)
    for cmd in converted_data:
        if cmd["command"] == "Line":
            start = np.array(cmd["start"])
            end = np.array(cmd["end"])
            
            # Check line length (must be between 1 and 1024)
            length = np.linalg.norm(end - start)
            if clip_length_range and (length < 1 or length > 1024):
                return None, "length"
            
            # Check coordinate values (must be ≤ 1024)
            if clip_large_coord:
                if any(abs(x) > 1024 for x in start) or any(abs(x) > 1024 for x in end):
                    return None, "coordinate"
        
        elif cmd["command"] == "Extrusion":
            dist1 = cmd.get("entity_one_distance")
            dist2 = cmd.get("entity_two_distance")
            
            # Check extrusion distances (must be between 10 and 1024)
            if clip_length_range:
                if dist1 is not None:
                    if abs(dist1) < 1 or abs(dist1) > 1024:
                        return None, "length"
                if dist2 is not None:
                    if abs(dist2) < 1 or abs(dist2) > 1024:
                        return None, "length"
    
    return converted_data, "success"

def process_json_files(input_folder, output_folder, clip_length_range=True, clip_large_coord=True):
    """Process all JSON files in the input folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    count = 0
    skipped_taper = 0
    skipped_offset = 0
    skipped_length = 0
    skipped_coord = 0
    errors = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith('_extracted.json'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                with open(input_path, 'r') as f:
                    data = json.load(f)
                
                # Process sequence with clipping flags
                processed_data, status = process_sequence(data, filename, clip_length_range, clip_large_coord)
                
                if processed_data is None:
                    if status == "taper":
                        print(f"Skipped (non-zero taper): {filename}")
                        skipped_taper += 1
                    elif status == "offset":
                        print(f"Skipped (offset start): {filename}")
                        skipped_offset += 1
                    elif status == "length":
                        print(f"Skipped (length out of range): {filename}")
                        skipped_length += 1
                    elif status == "coordinate":
                        print(f"Skipped (coordinate too large): {filename}")
                        skipped_coord += 1
                    continue
                
                # Save processed data
                with open(output_path, 'w') as f:
                    json.dump(processed_data, f, indent=4)
                
                count += 1
                print(f"Processed: {filename}")
                
            except ValueError as e:
                print(f"Error: {str(e)}")
                errors += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                errors += 1
    
    print(f"\nTotal files processed: {count}")
    print(f"Files skipped (non-zero taper): {skipped_taper}")
    print(f"Files skipped (offset start): {skipped_offset}")
    if clip_length_range:
        print(f"Files skipped (length out of range): {skipped_length}")
    if clip_large_coord:
        print(f"Files skipped (coordinate > 1024): {skipped_coord}")
    print(f"Files with errors: {errors}")

if __name__ == "__main__":
    input_folder = "2D_data_preprocess/data/line_only_3D_seq_matched"
    output_folder = "2D_data_preprocess/data/line_only_3D_seq_processed_clipped"
    
    # Set clipping flags
    clip_length_range = True  # Skip files with any length < 1
    clip_large_coord = False   # Skip files with any coordinate > 1024
    
    process_json_files(input_folder, output_folder, 
                      clip_length_range=clip_length_range, 
                      clip_large_coord=clip_large_coord)