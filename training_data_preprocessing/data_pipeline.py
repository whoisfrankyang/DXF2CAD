# Author: Bohan Yang (Mar 18, 2025)

import os
import shutil
from typing import Tuple
from file_matcher import match_files, MatchMode
import seq_extractor
import seq_processor
import seq2npy
import step_project

def create_pipeline_directories():
    """Create all necessary directories for the pipeline"""
    directories = [
        "intermediate_data/line_only_3D_seq",
        "intermediate_data/line_only_3D_seq_matched",
        "intermediate_data/line_only_3D_seq_processed_clipped",
        "intermediate_data/tokenized_3D_sequences",
        "intermediate_data/line_only_step",
        "intermediate_data/temp/project_img",
        "intermediate_data/step_project_npy",
        "train/input",
        "train/output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def extract_line_only_json():
    """Step 1: Extract line-only sequences from 3D meta JSON"""
    print("\nStep 1: Extracting line-only sequences...")
    input_folder = '3D_meta_json'
    output_folder = 'intermediate_data/line_only_3D_seq'
    seq_extractor.process_json_files(input_folder, output_folder)

def match_step_files():
    """Step 2: Match line-only JSON with STEP files"""
    print("\nStep 2: Matching STEP files...")
    input_folder = "intermediate_data/line_only_3D_seq"
    step_folder = "step_files"
    output_folder = "intermediate_data/line_only_step"
    
    # Match files and copy matching STEP files
    count1, count2 = match_files(
        input_folder, step_folder,
        None, output_folder,
        ['.json'], ['.step', '.stp'],
        MatchMode.MATCH_2_TO_1,
        ['_extracted'], []
    )
    
    print(f"Matched {count2} STEP files")

def process_sequences():
    """Step 3: Process sequences and convert to NPY"""
    print("\nStep 3: Processing sequences...")
    
    # First process the sequences
    input_folder = "intermediate_data/line_only_3D_seq"
    output_folder = "intermediate_data/line_only_3D_seq_processed_clipped"
    seq_processor.process_json_files(
        input_folder, output_folder,
        clip_length_range=True,
        clip_large_coord=False
    )
    
    # Then convert to NPY
    input_folder = "intermediate_data/line_only_3D_seq_processed_clipped"
    output_folder = "intermediate_data/tokenized_3D_sequences"
    seq2npy.process_folder(input_folder, output_folder)

def project_step_files():
    """Step 4: Project STEP files to create projected NPY files"""
    print("\nStep 4: Projecting STEP files...")
    input_folder = "intermediate_data/line_only_step"
    image_output_dir = "intermediate_data/temp/project_img"
    npy_output_dir = "intermediate_data/step_project_npy"
    
    step_project.process_step_folder(input_folder, image_output_dir, npy_output_dir)

def create_training_data():
    """Step 5: Create final training data by matching tokenized and projected NPY files"""
    print("\nStep 5: Creating training data...")
    folder1 = "intermediate_data/step_project_npy"
    folder2 = "intermediate_data/tokenized_3D_sequences"
    output_dir1 = "train/input"
    output_dir2 = "train/output"
    
    count1, count2 = match_files(
        folder1, folder2,
        output_dir1, output_dir2,
        ['.npy'], ['.npy'],
        MatchMode.MATCH_BOTH
    )
    
    print(f"Created training data with {count1} input files and {count2} output files")

def cleanup_intermediate_directories():
    """Clean up all intermediate directories after pipeline completion"""
    print("\nCleaning up intermediate directories...")
    intermediate_dirs = [
        "intermediate_data/line_only_3D_seq",
        "intermediate_data/line_only_3D_seq_matched",
        "intermediate_data/line_only_3D_seq_processed_clipped",
        "intermediate_data/tokenized_3D_sequences",
        "intermediate_data/line_only_step",
        "intermediate_data/temp/project_img",
        "intermediate_data/step_project_npy",
        "intermediate_data/temp"  
    ]
    
    for directory in intermediate_dirs:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Removed: {directory}")
            except Exception as e:
                print(f"Error removing {directory}: {str(e)}")

def run_pipeline():
    """Run the complete data processing pipeline"""
    print("Starting data processing pipeline...")
    
    try:
        # Create all necessary directories
        create_pipeline_directories()
        
        # Run each step
        extract_line_only_json()
        match_step_files()
        process_sequences()
        project_step_files()
        create_training_data()
        
        print("\nPipeline completed successfully!")
        
        # Clean up intermediate directories
        # cleanup_intermediate_directories()
        
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        print("Intermediate directories will not be cleaned up due to pipeline failure.")

if __name__ == "__main__":
    run_pipeline() 