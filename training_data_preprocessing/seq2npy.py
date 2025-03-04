# Copyright (c) 2024 Pixelate Inc. All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Author: Bohan Yang (Jan 3, 2025)

import os
import json
import numpy as np
import matplotlib.pyplot as plt

"""
Convert the processed line-only 3D json construction sequence into 3D seq tokens.
"""
# Token definitions
START_TOKEN = 1025
LINE_TOKEN = 1026
EXTRUSION_TOKEN = 1027
END_TOKEN = 1028

# Extrusion operation type tokens (beyond parameter space)
EXTRUSION_TYPE_TOKENS = {
    'NewBodyFeatureOperation': 1029,  # Type 1
    'JoinFeatureOperation': 1030,     # Type 2
    'CutFeatureOperation': 1031,      # Type 3
    'IntersectFeatureOperation': 1032  # Type 4
}

EXTRUSION_EXTENT_TYPE_TOKENS = {
    'OneSideFeatureExtentType': 1033,
    'SymmetricFeatureExtentType': 1034,
    'TwoSidesFeatureExtentType': 1035,
}



def quantize_values(arr, n_bits=10):
#     """
#     Quantizes an array of values in the range [1, 1024] to n_bits discrete levels.
    
#     Args:
#       arr (np.array or list): Array of continuous values to quantize.
#       n_bits (int): Number of bits for quantization (e.g., 10 bits for 1024 levels).
      
#     Returns:
#       np.array: Discrete tokens in the range [1, 2^n_bits].
#     """
#     min_range = 0
#     max_range = 1024
#     n_levels = 2**n_bits  # Number of quantization bins
    
#     # Scale and quantize to discrete levels
#     quantized_arr = (arr - min_range) * (n_levels - 1) / (max_range - min_range)
    
#     # Round to nearest level 
#     return np.round(quantized_arr).astype(int) 
      return arr   
 

def tokenize_sequence(data):
    """Convert a JSON sequence into tokens"""
    tokens = [START_TOKEN]
    
    for cmd in data:
        if cmd["command"] == "Line":
            tokens.append(LINE_TOKEN)
            # Quantize and add start coordinates
            start_coords = np.array(cmd["start"])
            tokens.extend(quantize_values(start_coords).tolist())
            # Quantize and add end coordinates
            end_coords = np.array(cmd["end"])
            tokens.extend(quantize_values(end_coords).tolist())
        
        elif cmd["command"] == "Extrusion":
            tokens.append(EXTRUSION_TOKEN)
            
            # Add operation type token (1029-1032)
            op_type_token = EXTRUSION_TYPE_TOKENS.get(
                cmd["operation"], 
                EXTRUSION_TYPE_TOKENS['NewBodyFeatureOperation']
            )
            tokens.append(op_type_token)
            
            # Add extent type token (1033-1035)
            extent_type_token = EXTRUSION_EXTENT_TYPE_TOKENS.get(
                cmd["extent_type"],
                EXTRUSION_EXTENT_TYPE_TOKENS['OneSideFeatureExtentType']
            )
            tokens.append(extent_type_token)
            
            # Add distances and taper angles
            dist1 = cmd.get("entity_one_distance")
            dist2 = cmd.get("entity_two_distance")
            
            # Distance 1 and taper angle 1
            if dist1 is not None:
                tokens.append(quantize_values(np.array([dist1]))[0])
            else:
                tokens.append(0)
            tokens.append(0)  # taper_angle_1 = 0
            
            # Distance 2 and taper angle 2
            if dist2 is not None:
                tokens.append(quantize_values(np.array([dist2]))[0])
            else:
                tokens.append(0)
            tokens.append(0)  # taper_angle_2 = 0
    
    tokens.append(END_TOKEN)
    return tokens

def process_folder(input_folder, output_folder):
    """Process all JSON files in folder to tokenized numpy arrays"""
    os.makedirs(output_folder, exist_ok=True)
    
    skipped = 0
    processed = 0
    max_length = 0
    
    # Process and save files
    for filename in os.listdir(input_folder):
        if filename.endswith('_extracted.json'):
            try:
                with open(os.path.join(input_folder, filename), 'r') as f:
                    data = json.load(f)
                
                # Tokenize sequence
                tokens = tokenize_sequence(data)
                max_length = max(max_length, len(tokens))
                
                # Save as numpy array
                output_filename = filename.replace('_extracted.json', '.npy')
                output_path = os.path.join(output_folder, output_filename)
                np.save(output_path, np.array(tokens))
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed} files...")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                skipped += 1
    
    print(f"\nProcessing complete:")
    print(f"Files processed: {processed}")
    print(f"Files skipped: {skipped}")
    print(f"Maximum sequence length: {max_length}")

def analyze_token_sequence_lengths(folder_path):
    """Analyze the lengths of tokenized sequences in .npy files"""
    sequence_lengths = []
    filename_lengths = []  # Store (filename, length) pairs
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            try:
                # Load the token sequence
                tokens = np.load(os.path.join(folder_path, filename))
                length = len(tokens)
                sequence_lengths.append(length)
                filename_lengths.append((filename, length))
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Convert to numpy array for statistics
    sequence_lengths = np.array(sequence_lengths)
    
    # Find file with maximum length
    max_file, max_len = max(filename_lengths, key=lambda x: x[1])
    
    print("\nToken Sequence Length Statistics:")
    print(f"Number of sequences analyzed: {len(sequence_lengths)}")
    print(f"Mean length: {np.mean(sequence_lengths):.2f}")
    print(f"Median length: {np.median(sequence_lengths):.2f}")
    print(f"Min length: {np.min(sequence_lengths)}")
    print(f"Max length: {np.max(sequence_lengths)} (File: {max_file})")
    print(f"Std: {np.std(sequence_lengths):.2f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 5))
    plt.hist(sequence_lengths, bins=50)
    plt.title('Distribution of Token Sequence Lengths')
    plt.xlabel('Sequence Length (number of tokens)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    plt.axvline(np.mean(sequence_lengths), color='r', linestyle='--', label=f'Mean: {np.mean(sequence_lengths):.1f}')
    plt.axvline(np.median(sequence_lengths), color='g', linestyle='--', label=f'Median: {np.median(sequence_lengths):.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print percentile information
    percentiles = [50, 75, 90, 95, 99]
    print("\nPercentile Information:")
    for p in percentiles:
        value = np.percentile(sequence_lengths, p)
        print(f"{p}th percentile: {value:.1f}")

if __name__ == "__main__":
    input_folder = "2D_data_preprocess/data/line_only_3D_seq_processed_clipped"
    output_folder = "2D_data_preprocess/data/tokenized_3D_sequences"
    
    # Process JSON files to NPY
    process_folder(input_folder, output_folder)
    
    # Analyze token sequence lengths
    analyze_token_sequence_lengths(output_folder)
