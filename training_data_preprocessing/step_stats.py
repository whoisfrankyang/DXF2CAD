# Author: Bohan Yang (Jan 3, 2025)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

"""
Given 3D CAD step files, perform some basic statistics and analysis. 
"""
def get_bbox_dimensions(step_file_path):
    """Get bounding box dimensions from STEP file"""
    # Load STEP file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file_path)
    
    if status != IFSelect_RetDone:
        raise Exception("Could not read STEP file")
    
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    if shape.IsNull():
        raise Exception("Null shape")
    
    # Get bounding box
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    # Calculate dimensions
    x_range = abs(xmax - xmin)
    y_range = abs(ymax - ymin)
    z_range = abs(zmax - zmin)
    
    return {
        'x_range': x_range,
        'y_range': y_range,
        'z_range': z_range
    }

def analyze_step_files(folder_path):
    """Analyze dimensions of all STEP files in folder"""
    x_ranges = []
    y_ranges = []
    z_ranges = []
    
    # Process all STEP files
    for filename in os.listdir(folder_path):
        if filename.endswith(('.step', '.stp')):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            
            try:
                dims = get_bbox_dimensions(file_path)
                x_ranges.append(dims['x_range'])
                y_ranges.append(dims['y_range'])
                z_ranges.append(dims['z_range'])
                print(f"Dimensions - X: {dims['x_range']:.3f}, Y: {dims['y_range']:.3f}, Z: {dims['z_range']:.3f}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return x_ranges, y_ranges, z_ranges

def print_range_counts(x_ranges, y_ranges, z_ranges):
    """Print distribution of dimensions in ranges"""
    # Define bins
    bins = [0, 1, 5, 10, 20, 50, 100, float('inf')]
    bin_labels = ['0-1', '1-5', '5-10', '10-20', '20-50', '50-100', '>100']
    
    def count_in_ranges(values):
        counts = [0] * len(bin_labels)
        for v in values:
            for i in range(len(bins)-1):
                if bins[i] <= v < bins[i+1]:
                    counts[i] += 1
                    break
        return counts
    
    x_counts = count_in_ranges(x_ranges)
    y_counts = count_in_ranges(y_ranges)
    z_counts = count_in_ranges(z_ranges)
    
    print("\nDimension Range Distribution:")
    print("\nRange (mm)    X Width    Y Width    Z Height")
    print("-" * 45)
    for i, label in enumerate(bin_labels):
        print(f"{label:<12} {x_counts[i]:<10} {y_counts[i]:<10} {z_counts[i]:<10}")

def plot_distributions(x_ranges, y_ranges, z_ranges):
    """Create distribution plots"""
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of STEP File Dimensions')
    
    # X range distribution
    sns.histplot(x_ranges, ax=axes[0,1], bins=50)
    axes[0,1].set_title('X Width Distribution')
    axes[0,1].set_xlabel('X Width (mm)')
    
    # Y range distribution
    sns.histplot(y_ranges, ax=axes[1,0], bins=50)
    axes[1,0].set_title('Y Width Distribution')
    axes[1,0].set_xlabel('Y Width (mm)')
    
    # Z range distribution
    sns.histplot(z_ranges, ax=axes[1,1], bins=50)
    axes[1,1].set_title('Z Height Distribution')
    axes[1,1].set_xlabel('Z Height (mm)')
    
    plt.tight_layout()
    plt.show()
    
    # Box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [x_ranges, y_ranges, z_ranges]
    ax.boxplot(data, labels=['X Width', 'Y Width', 'Z Height'])
    ax.set_title('Dimension Distribution by Direction')
    plt.show()

if __name__ == "__main__":
    folder_path = "intermediate_data/line_only_3D_seq_processed_clipped"
    x_ranges, y_ranges, z_ranges = analyze_step_files(folder_path)
    print_range_counts(x_ranges, y_ranges, z_ranges)
    plot_distributions(x_ranges, y_ranges, z_ranges) 