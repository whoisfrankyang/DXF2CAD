# Copyright (c) 2024 Pixelate Inc. All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Author: Bohan Yang (Jan 3, 2025)

import os
import numpy as np
from orthographic_projection import (
    load_step_file, 
    get_projected_edges,
    get_projected_edges_with_visibility,
    check_body_count
)

"""
Project the STEP file into 6 orthographic views. Produce 6 .npy files and 6 .png images.
"""
def process_step_file(step_path, image_output_dir, npy_output_dir):
    """
    Process a single STEP file to create:
    1. Six orthographic projection images
    2. NPY file containing line data with view and visibility information
    
    NPY format per line:
    [view_index, visibility, x1, y1, z1, x2, y2, z2]
    where:
    - view_index: 0-5 (left, front, right, top, bottom, back)
    - visibility: 0 (hidden) or 1 (visible)
    - x1,y1,z1: start point coordinates
    - x2,y2,z2: end point coordinates
    """
    try:
        # Load STEP file and check body count
        shape = load_step_file(step_path)
        if not check_body_count(shape):
            print(f"Skipped {step_path} - multiple bodies found")
            return False
        
        # Create output directories if they don't exist
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(npy_output_dir, exist_ok=True)
        
        # Base filename without extension
        base_name = os.path.splitext(os.path.basename(step_path))[0]
        
        # Get projections for all views
        views = ['left', 'front', 'right', 'top', 'bottom', 'back']
        all_lines = []
        
        for view_idx, view in enumerate(views):
            # Get projected edges with visibility information
            visible_edges, hidden_edges = get_projected_edges_with_visibility(shape, view)
            
            # Save image
            img_path = os.path.join(image_output_dir, f"{base_name}_{view}.png")
            get_projected_edges(shape, view, save_path=img_path)
            
            # Process visible edges
            for edge in visible_edges:
                start = edge[0]
                end = edge[1]
                line_data = np.array([view_idx, 1, *start, *end])
                all_lines.append(line_data)
            
            # Process hidden edges
            for edge in hidden_edges:
                start = edge[0]
                end = edge[1]
                line_data = np.array([view_idx, 0, *start, *end])
                all_lines.append(line_data)
        
        # Save all lines to NPY file
        if all_lines:
            npy_path = os.path.join(npy_output_dir, f"{base_name}.npy")
            all_lines = np.stack(all_lines)
            np.save(npy_path, all_lines)
            
        return True
        
    except Exception as e:
        print(f"Error processing {step_path}: {str(e)}")
        return False

def process_step_folder(input_folder, image_output_dir, npy_output_dir):
    """Process all STEP files in a folder"""
    processed = 0
    skipped = 0
    errors = 0
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.step', '.stp')):
            step_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            
            success = process_step_file(step_path, image_output_dir, npy_output_dir)
            
            if success:
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed} files...")
            else:
                skipped += 1
    
    print("\nProcessing complete:")
    print(f"Files processed: {processed}")
    print(f"Files skipped: {skipped}")
    print(f"Files with errors: {errors}")

if __name__ == "__main__":
    input_folder = "2D_data_preprocess/data/line_only_step"
    image_output_dir = "2D_data_preprocess/data/temp/project_img"
    npy_output_dir = "2D_data_preprocess/data/step_project_npy"
    
    process_step_folder(input_folder, image_output_dir, npy_output_dir) 