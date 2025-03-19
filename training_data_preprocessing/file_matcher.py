# Author: Bohan Yang (Jan 3, 2025)

import os
import shutil
from enum import Enum
from typing import List, Tuple, Optional, Set

"""
Match files between two folders based on specified mode
"""
class MatchMode(Enum):
    """Matching modes for file synchronization"""
    MATCH_2_TO_1 = 1  # Match folder2 to folder1, output new folder2
    MATCH_1_TO_2 = 2  # Match folder1 to folder2, output new folder1
    MATCH_BOTH = 3    # Match both folders (union), output both

def remove_suffixes(filename: str, suffixes: List[str]) -> str:
    """Remove specified suffixes from filename"""
    base_name = os.path.splitext(filename)[0]
    for suffix in suffixes:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
    return base_name

def get_matching_files(folder1: str, folder2: str, 
                      extensions1: List[str], extensions2: List[str],
                      suffixes1: List[str], suffixes2: List[str]) -> Tuple[Set[str], Set[str]]:
    """
    Get sets of base filenames (without extensions and suffixes) from both folders
    
    Args:
        folder1: Path to first folder
        folder2: Path to second folder
        extensions1: List of extensions to consider for folder1 (e.g., ['.json', '.txt'])
        extensions2: List of extensions to consider for folder2 (e.g., ['.step', '.stp'])
        suffixes1: List of suffixes to remove for folder1 (e.g., ['_extracted', '_processed'])
        suffixes2: List of suffixes to remove for folder2
    
    Returns:
        Tuple of two sets containing base filenames from each folder
    """
    # Get filenames from folder1
    files1 = set()
    for filename in os.listdir(folder1):
        if any(filename.lower().endswith(ext.lower()) for ext in extensions1):
            base_name = remove_suffixes(filename, suffixes1)
            files1.add(base_name)
    
    # Get filenames from folder2
    files2 = set()
    for filename in os.listdir(folder2):
        if any(filename.lower().endswith(ext.lower()) for ext in extensions2):
            base_name = remove_suffixes(filename, suffixes2)
            files2.add(base_name)
    
    return files1, files2

def match_files(folder1: str, folder2: str, 
                output_dir1: Optional[str], output_dir2: Optional[str],
                extensions1: List[str], extensions2: List[str],
                mode: MatchMode,
                suffixes1: List[str] = [], suffixes2: List[str] = []) -> Tuple[int, int]:
    """
    Match files between two folders based on specified mode
    
    Args:
        folder1: Path to first folder
        folder2: Path to second folder
        output_dir1: Output directory for folder1 files (if needed)
        output_dir2: Output directory for folder2 files (if needed)
        extensions1: List of extensions to consider for folder1
        extensions2: List of extensions to consider for folder2
        mode: MatchMode specifying how to match files
        suffixes1: List of suffixes to remove when matching for folder1
        suffixes2: List of suffixes to remove when matching for folder2
    
    Returns:
        Tuple of (number of files in output1, number of files in output2)
    """
    # Get sets of base filenames from both folders
    files1, files2 = get_matching_files(folder1, folder2, extensions1, extensions2, suffixes1, suffixes2)
    print(f"Files in folder1: {files1}")
    print(f"Files in folder2: {files2}")
    
    # Determine which files to keep based on mode
    if mode == MatchMode.MATCH_2_TO_1:
        keep_files = files1.intersection(files2)
        if output_dir2:
            os.makedirs(output_dir2, exist_ok=True)
    elif mode == MatchMode.MATCH_1_TO_2:
        keep_files = files1.intersection(files2)
        if output_dir1:
            os.makedirs(output_dir1, exist_ok=True)
    else:  # MATCH_BOTH
        keep_files = files1.intersection(files2)
        if output_dir1:
            os.makedirs(output_dir1, exist_ok=True)
        if output_dir2:
            os.makedirs(output_dir2, exist_ok=True)
    
    # Copy matching files to output directories
    count1 = 0
    count2 = 0
    
    # Process folder1 if needed
    if output_dir1 and (mode in [MatchMode.MATCH_1_TO_2, MatchMode.MATCH_BOTH]):
        for filename in os.listdir(folder1):
            base_name = remove_suffixes(filename, suffixes1)
            if base_name in keep_files:
                src = os.path.join(folder1, filename)
                dst = os.path.join(output_dir1, filename)
                shutil.copy2(src, dst)
                count1 += 1
    
    # Process folder2 if needed
    if output_dir2 and (mode in [MatchMode.MATCH_2_TO_1, MatchMode.MATCH_BOTH]):
        for filename in os.listdir(folder2):
            base_name = remove_suffixes(filename, suffixes2)
            if base_name in keep_files:
                src = os.path.join(folder2, filename)
                dst = os.path.join(output_dir2, filename)
                shutil.copy2(src, dst)
                count2 += 1
    
    return count1, count2

if __name__ == "__main__":
    # Example usage:
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