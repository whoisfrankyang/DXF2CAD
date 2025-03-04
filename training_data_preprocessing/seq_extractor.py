# Copyright (c) 2024 Pixelate Inc. All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Author: Bohan Yang (Jan 3, 2025)

import json
import os

"""
1. Process raw json file to extract clear construction sequence and store in a new json file
2. Filter out the files with unsupported operation types (currently only support line and extrusion)
"""

def process_single_json(input_path):
    """
    Processes a single JSON file to extract the construction sequence.

    Args:
        input_path (str): The path to the input JSON file.

    Returns:
        list: The construction sequence extracted from the JSON file.
    """
    with open(input_path) as f:
        data = json.load(f)

    construction_sequence = []
    construction_sequence.append({
        "command": "START"
    })
    # Iterate through timeline to maintain sequence order
    for item in data["timeline"]:
        entity_id = item["entity"]
        entity = data["entities"].get(entity_id, {})
        
        # Process Sketch entities
        if entity.get("type") == "Sketch":
            start_point = None  # Track the start point of the loop
            loop_closed = False  # Track if the loop is closed
            for curve_id, curve in entity.get("curves", {}).items():
                # Line: (start, end)
                if curve["type"] == "SketchLine":
                    start_point_data = entity["points"][curve["start_point"]]
                    end_point_data = entity["points"][curve["end_point"]]
                    start_point = start_point or (start_point_data["x"], start_point_data["y"], start_point_data["z"])
                    end_point = (end_point_data["x"], end_point_data["y"], end_point_data["z"])
                    construction_sequence.append({
                        "command": "Line",
                        "start": (start_point_data["x"], start_point_data["y"], start_point_data["z"]),
                        "end": end_point
                    })
                # # Arc: (center, radius, start angle, end angle)
                # elif curve["type"] == "SketchArc":
                #     start_point_data = entity["points"][curve["start_point"]]
                #     end_point_data = entity["points"][curve["end_point"]]
                #     center_point_data = entity["points"][curve["center_point"]]
                #     radius = curve["radius"]
                #     start_angle = curve["start_angle"]
                #     end_angle = curve["end_angle"]
                #     start_point = start_point or (start_point_data["x"], start_point_data["y"], start_point_data["z"])
                #     end_point = (end_point_data["x"], end_point_data["y"], end_point_data["z"])
                #     construction_sequence.append({
                #         "command": "Arc",
                #         "center": (center_point_data["x"], center_point_data["y"], center_point_data["z"]),
                #         "radius": radius,
                #         "start angle": start_angle,
                #         "end angle": end_angle
                #     })
                # # Circle
                # elif curve["type"] == "SketchCircle":
                #     center_point_data = entity["points"][curve["center_point"]]
                #     radius = curve["radius"]
                #     construction_sequence.append({
                #         "command": "Circle",
                #         "center": (center_point_data["x"], center_point_data["y"], center_point_data["z"]),
                #         "radius": radius
                #     })
                # add an else statement. If other types, skip over the entire file 
                else:
                    print(f"Skipping file due to unsupported curve type: {curve['type']}")
                    return None  # Return None to indicate file should be skipped
        
        # Process ExtrudeFeature entities
        elif entity.get("type") == "ExtrudeFeature":

            entity_operation = entity["operation"]
            extent_type = entity["extent_type"]
            start_extent = entity["start_extent"]
            entity_one_distance = entity["extent_one"]["distance"]["value"]
            entity_one_taper_angle = entity["extent_one"]["taper_angle"]["value"]
            entity_two = entity.get("extent_two", {}).get("entity", "N/A")  # Gets entity_two or "N/A"
            if entity_two != "N/A":
                entity_two_distance = entity["extent_two"]["distance"]["value"]
                entity_two_taper_angle = entity["extent_two"]["taper_angle"]["value"]

            construction_sequence.append({
                "command": "Extrusion",
                "operation": entity_operation,
                "start_extent": start_extent,
                "extent_type": extent_type,
                "entity_one_distance": entity_one_distance,
                "entity_one_taper_angle": entity_one_taper_angle,
                "entity_two_distance": entity_two_distance if entity_two != "N/A" else None,
                "entity_two_taper_angle": entity_two_taper_angle if entity_two != "N/A" else None
            })

    # Add END command
    construction_sequence.append({
        "command": "END"
    })
    return construction_sequence

def process_json_files(input_folder, output_folder):
    """
    Processes all JSON files in the input folder and saves the results in the output folder.
    Skips files with unsupported curve types.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed = 0
    skipped = 0

    # Iterate over all JSON files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_extracted.json"
            output_path = os.path.join(output_folder, output_filename)
            
            construction_sequence = process_single_json(input_path)
            if construction_sequence is not None:
                with open(output_path, 'w') as outfile:
                    json.dump(construction_sequence, outfile, indent=4)
                print(f"Construction sequence saved to {output_path}")
                processed += 1
            else:
                print(f"Skipped {filename} due to unsupported curve types")
                skipped += 1

    print(f"\nProcessing complete:")
    print(f"Files processed: {processed}")
    print(f"Files skipped: {skipped}")

if __name__ == "__main__":
    input_folder = '2D_data_preprocess/data/3D_meta_json'
    output_folder = '2D_data_preprocess/data/line_only_3D_seq'
    process_json_files(input_folder, output_folder)
