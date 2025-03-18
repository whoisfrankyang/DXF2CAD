import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import numpy as np
import os
from utils import quantize_values
from torch.nn import functional as F
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import psutil


class CADDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 max_input_length: int = 512, # used for padding
                 max_output_length: int = 512, # used for padding
                 input_vec_dim: int = 13, # not used in this dataset, used for input_dim in model
                 target_vec_dim: int = 1033, # not used in this dataset, used for output_dim in model
                 file_list: List[str] = None,  # Add file_list parameter
                 is_validation: bool = False,  # Add this parameter
                 epoch: int = 0,  # Add epoch parameter
                 save_dir: str = 'checkpoints'  # Add save_dir parameter
                 ):
        """
            data_path: Base path to the data files
            max_input_length: Maximum length of the input sequence
            max_output_length: Maximum length of the output sequence
            input_vec_dim: Dimension of each input vector
            target_vec_dim: Dimension of each target vector 
            file_list: List of files to process
            is_validation: Whether this is a validation set
            epoch: Current epoch
            save_dir: Directory to save seed information
        """
        self.data_path = data_path
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.input_vec_dim = input_vec_dim
        self.target_vec_dim = target_vec_dim
        self.file_list = file_list  # Store file list
        self.is_validation = is_validation  # Store whether this is validation set
        self.epoch = epoch  # Store current epoch
        self.save_dir = save_dir  # Store save_dir
        
        # Load and preprocess data
        self.input_seq, self.output_seq, self.filenames = self.load_data()
        
    @staticmethod
    def _sort_curve_sequence(curve_seq: np.ndarray) -> np.ndarray:
        """Sort the curve sequence before transformation
        Each curve has format: [view, visibility, x1, y1, x2, y2]

        1. Group curves by their view number
        2. For each curve, ensure x1, y1 < x2, y2, if not, swap
        3. Sort curves based on their coordinates
        """
        
        # 1. Group curves by their view number
        view_groups = {}
        for curve in curve_seq:
            view_key = curve[0]  
            if view_key not in view_groups:
                view_groups[view_key] = []
            view_groups[view_key].append(curve)
        
        # print("Before sort: ", curve_seq)
        sorted_curves = []
        for view_key in sorted(view_groups.keys()):
            curves = view_groups[view_key]
            
            # 2. For each curve, ensure x1, y1 < x2, y2, if not, swap
            for curve in curves:
                x1, y1 = curve[2], curve[3]
                x2, y2 = curve[4], curve[5]
                # print("Before swap: ", curve)
                if x1 > x2 or (x1 == x2 and y1 > y2):
                    temp = curve[2:4].copy()
                    curve[2:4] = curve[4:6]
                    curve[4:6] = temp
            
            # 3. Sort curves based on their coordinates
            def curve_sort_key(curve):
                return (curve[2], curve[4], curve[3], curve[5]) 
            
            curves.sort(key=curve_sort_key)
            sorted_curves.extend(curves)
        
        return np.array(sorted_curves)
    
    @staticmethod
    def _transform_curve_sequence(curve_seq: np.ndarray) -> np.ndarray:
        """
        Transform curve sequence into point sequence.
        Input: curve_seq with format [view, vis, x1, y1, x2, y2]
        Output: Four tokens per curve: [coord_value, view_onehot, vis_onehot, coord_type] where:
            - coord_value is the quantized coordinate value
            - view_onehot is a 6-dim one-hot vector for view
            - vis_onehot is a 2-dim one-hot vector for visibility
            - coord_type is one-hot vector [1,0,0,0] for x1, [0,1,0,0] for y1, etc.
        """
        transformed_seq = []
        
        for curve in curve_seq:
            
            view, vis, x1, y1, x2, y2 = curve
            # views = [left, front, right, top, bottom, back]


            # Convert view to 6-dim one-hot
            view_onehot = np.zeros(6)
            view_onehot[int(curve[0])] = 1
            # print("view_onehot: ", view_onehot)
            
            # Convert visibility to 2-dim one-hot
            vis_onehot = np.zeros(2)
            vis_onehot[int(curve[1])] = 1
            
            coords = curve[2:6].copy()  # [x1, y1, x2, y2]
            
            # Create four tokens for each curve (x1, y1, x2, y2)
            for i in range(4):
                if view == 0 or view == 2:
                    if i == 0 or i == 2:
                        dimension = [0, 1, 0]
                    else:
                        dimension = [0, 0, 1]
                elif view == 1 or view == 5:
                    if i == 0 or i == 2:
                        dimension = [1, 0, 0]
                    else:
                        dimension = [0, 0, 1]
                elif view == 3 or view == 4:
                    if i == 0 or i == 2:
                        dimension = [0, 1, 0]
                    else:
                        dimension = [1, 0, 0]

                quantized_coord = quantize_values(coords[i])


                # Create one-hot vector for coordinate type
                coord_type = np.zeros(4)
                coord_type[i] = 1
                
                # Create token: [quantized_coord, view_onehot, vis_onehot, coord_type]
                point_token = np.concatenate([
                    [quantized_coord],  # quantized coordinate value (1-dim)
                    view_onehot,                   # view one-hot [left, front, right, top, bottom, back] (6-dim)
                    vis_onehot,                    # visibility one-hot [visible, invisible] (2-dim)
                    coord_type,                     # coordinate type one-hot [x1, y1, x2, y2] (4-dim)
                    dimension                      # dimension one-hot [x, y, z] (3-dim)
                ])
                transformed_seq.append(point_token)
        
        return np.array(transformed_seq)

    def _transform_3d_sequence(self, seq: np.ndarray, filename: str = "unknown") -> np.ndarray:
        """
        Transform 3D sequence by quantizing continuous values and separate into token and value sequences.
        - Values <= 1023: Quantize them
        - Special tokens (1025-1035): Keep as is
        - Other values: Invalid

        Returns:
            transformed_seq (np.ndarray): Sequence of tokens
        """
        transformed_seq = []
        
        # Quantize values and separate into token and value sequences
        for value in seq:
            if value <= 1023:
                quantized_value = quantize_values(value)
                transformed_seq.append(quantized_value)
            elif self._is_special_token(value):
                transformed_seq.append(int(value))
            else:
                print(f"Invalid value {value} found in file: {filename}")
                raise ValueError(f"Invalid value in sequence: {value}")
        
        # Separate seq into token seq and val seq
        return np.array(transformed_seq)


    def _is_special_token(self, value):
        """Any token that's an operation or type indicator"""
        return value in [1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035]

    def _augment_data(self, raw_input_data: np.ndarray, raw_output_data: np.ndarray, filename: str = "unknown"):
        """Data Augmentation: 
        - Scale input and output data based on epoch
        - Generate 100 different scaled versions
        """
        augmented_pairs = []
        
        # Generate a seed based on epoch and filename for reproducibility
        seed = hash(f"{self.epoch}_{filename}") % (2**32)
        
        # Save seed information
        seed_info = {
            'epoch': self.epoch,
            'filename': filename,
            'seed': seed,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Create directory for seed information
        seed_dir = Path(self.save_dir) / 'seeds'
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save seed information to file
        seed_file = seed_dir / f'epoch_{self.epoch}_seeds.json'
        if not seed_file.exists():
            seeds_list = []
        else:
            with open(seed_file, 'r') as f:
                seeds_list = json.load(f)
        
        seeds_list.append(seed_info)
        with open(seed_file, 'w') as f:
            json.dump(seeds_list, f, indent=2)
        
        rng = np.random.RandomState(seed)
        
        # Generate scale factors
        scale_factors = rng.uniform(low=0.3, high=0.95, size=100)
        
        # Process original file first (no scaling)
        if self._validate_raw_pair(raw_input_data, raw_output_data):
            input_data = raw_input_data.copy()
            output_data = raw_output_data.copy()
            input_data = self._sort_curve_sequence(input_data)
            input_data = self._transform_curve_sequence(input_data)
            output_seq = self._transform_3d_sequence(output_data, filename)
            augmented_pairs.append((input_data, output_seq))
        
        # Apply scaling based on factors
        for scale in scale_factors:
            # Scale input and output data
            scaled_input = raw_input_data.copy()
            scaled_input[:, 2:6] *= scale
            
            scaled_output = raw_output_data.copy()
            i = 0
            while i < len(scaled_output):
                value = scaled_output[i]
                if value == 1026:  # LINE
                    # Scale next 6 coordinates
                    for j in range(6):
                        if i + 1 + j < len(scaled_output):
                            if not self._is_special_token(scaled_output[i + 1 + j]):
                                scaled_output[i + 1 + j] *= scale
                    i += 7
                elif value == 1027:  # EXTRUSION
                    # Skip feature operation and extent type
                    i += 2
                    # Scale next 4 values
                    for j in range(4):
                        if i + 1 + j < len(scaled_output):
                            if not self._is_special_token(scaled_output[i + 1 + j]):
                                scaled_output[i + 1 + j] *= scale
                    i += 4
                else:
                    i += 1
            
            if self._validate_raw_pair(scaled_input, scaled_output):
                input_data = self._sort_curve_sequence(scaled_input)
                input_data = self._transform_curve_sequence(input_data)
                output_seq = self._transform_3d_sequence(scaled_output, f"{filename}_scale_{scale:.2f}")
                augmented_pairs.append((input_data, output_seq))

        return augmented_pairs

    def _validate_raw_pair(self, input_data: np.ndarray, output_data: np.ndarray) -> bool:
        """
        Validate raw data pair before transformation:
        1. No line segments with length < 1 in input
        2. All continuous values must be <= 1023 and >= 0BEFORE quantization
        """
        # Check input data line lengths and coordinates >= 0
        for line in input_data:
            x1, y1 = line[2], line[3]
            x2, y2 = line[4], line[5]
            
            # Check for negative coordinates
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 > 1023 or y1 > 1023 or x2 > 1023 or y2 > 1023:
                return False
            
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 1:
                return False
        
        # Check output data values
        i = 0
        while i < len(output_data):
            value = output_data[i]
            if value == 1026:  # LINE
                # Check next 6 coordinates
                for j in range(6):
                    if i + 1 + j < len(output_data):
                        coord = output_data[i + 1 + j]
                        if not self._is_special_token(coord):
                            if coord < 0 or coord > 1023:
                                return False
                i += 7
            elif value == 1027:  # EXTRUSION
                # Skip feature operation and extent type
                i += 2
                # Check next 4 values
                for j in range(4):
                    if i + 1 + j < len(output_data):
                        val = output_data[i + 1 + j]
                        if not self._is_special_token(val):
                            if val > 1023 or val < 0: 
                                return False
                i += 4
            else:
                if not self._is_special_token(value) and value > 1023:
                    return False
                i += 1
        
        return True

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """
        Load input/output sequences from npy files.
         1. perform augmentation on raw data and validation on raw data.
         2. processing raw data: 
            - sort curve sequence
            - transform curve sequence
            - transform 3d sequence
        
        Returns:
            input_seqs: List[np.ndarray]
            output_seqs: List[np.ndarray]
            filenames: List[str]
        """
        input_seqs = []
        output_seqs = []
        filenames = [] 
        
        blueprint_path = os.path.join(self.data_path, 'input')
        cad_path = os.path.join(self.data_path, 'output')
        
        # Create directory for processed sequences
        processed_dir = os.path.join(self.data_path, 'dataset_processed')
        processed_2d_dir = os.path.join(processed_dir, '2d_sequences')
        processed_3d_dir = os.path.join(processed_dir, '3d_sequences')
        os.makedirs(processed_2d_dir, exist_ok=True)
        os.makedirs(processed_3d_dir, exist_ok=True)
        
        print(f"Looking for files in {blueprint_path}")
        blueprint_files = [f for f in os.listdir(blueprint_path)]
        print(f"Found {len(blueprint_files)} blueprint files")
        
        print(f"Processing {len(self.file_list)} files")
        for bp_file in tqdm(self.file_list, desc="Processing files"):
            base_name = bp_file
            cad_file = base_name
            
            if os.path.exists(os.path.join(cad_path, cad_file)):
                raw_input_data = np.load(os.path.join(blueprint_path, bp_file))
                raw_output_data = np.load(os.path.join(cad_path, cad_file))
                
                # For validation, only process original
                if self.is_validation: 
                    input_data = self._sort_curve_sequence(raw_input_data)
                    input_data = self._transform_curve_sequence(input_data)
                    output_data = self._transform_3d_sequence(raw_output_data, bp_file)
                    
                    # Save processed validation sequence
                    np.save(os.path.join(processed_2d_dir, f"{bp_file}_input.npy"), input_data)
                    np.save(os.path.join(processed_3d_dir, f"{bp_file}_output.npy"), output_data)
                    
                    if len(input_data) <= self.max_input_length and len(output_data) <= self.max_output_length:
                        input_seqs.append(input_data)
                        output_seqs.append(output_data)
                        filenames.append(bp_file)

                # For training, do augmentation
                else: 
                    augmented_pairs = self._augment_data(raw_input_data, raw_output_data, bp_file)
                    # augmented_pairs = [(raw_input_data, raw_output_data)]
                    # Process and save valid pairs
                    for idx, (input_data, output_data) in enumerate(augmented_pairs):
                        # # Save processed training sequences with augmentation
                        # np.save(os.path.join(processed_2d_dir, f"{bp_file}_aug{idx}_input.npy"), input_data)
                        # np.save(os.path.join(processed_3d_dir, f"{bp_file}_aug{idx}_output.npy"), output_data)
                        
                        # Ensure that sequences after processing are less than max_length
                        if len(input_data) <= self.max_input_length and len(output_data) <= self.max_output_length:
                            input_seqs.append(input_data)
                            output_seqs.append(output_data)
            else:
                print(f"Missing CAD file for {bp_file}")
        
        print(f"Total valid sequences: {len(input_seqs)}")
        print(f"Saved processed sequences to {processed_dir}")
        
        if len(input_seqs) == 0:
            raise ValueError(f"No valid data was loaded from {self.data_path}!")
            
        return input_seqs, output_seqs, filenames
    
    
    def __len__(self) -> int:
        return len(self.input_seq)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Acces and reutrn one sample from the dataset at a time during training or inference.
        """
        input_seq = self.input_seq[idx]
        output_seq = self.output_seq[idx]


        return torch.FloatTensor(input_seq), torch.LongTensor(output_seq)
def collate_fn(batch):
    """
    Handles dynamic padding across the batch. All samples in the batch are padded 
    to match the longest sequence in the batch.
    Used in DataLoader to create batches.
    """
    input_seqs, output_seqs = zip(*batch)
    
    # Determine maximum lengths within the batch
    max_len_src = max(seq.size(0) for seq in input_seqs)
    max_len_tgt = max(seq.size(0) for seq in output_seqs)

    # Pad sequences dynamically
    input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=1024)
    output_seqs = pad_sequence(output_seqs, batch_first=True, padding_value=1024)

    # Create padding masks
    # For input sequences, we need to mask at sequence level, not feature level
    # Convert mask to shape [batch_size, seq_length]
    src_masks = torch.any(input_seqs != 1024, dim=-1).float()  
    tgt_masks = (output_seqs != 1024).float()

    return input_seqs, output_seqs, src_masks, tgt_masks



def create_dataloaders(config, epoch=0):
    print(f"Creating dataset for epoch {epoch}...")
    
    # Check if we're resuming training by looking for existing split file
    split_file = os.path.join('data/dataset_processed', 'train_val_split.json')
    
    if os.path.exists(split_file):
        # Load existing train/val split
        print(f"Loading existing train/val split from {split_file}")
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        train_files = split_info['train_files']
        val_files = split_info['val_files']
        print(f"Loaded {len(train_files)} train and {len(val_files)} val files from existing split")
    
    else:
        # Create new train/val split
        blueprint_path = os.path.join('data', 'input')
        blueprint_files = [f for f in os.listdir(blueprint_path)]
        
        # Split files into train/val
        split_idx = int(len(blueprint_files) * config.get('train_ratio', 0.98))
        train_files = blueprint_files[:split_idx]
        val_files = blueprint_files[split_idx:]
        
        print(f"Created new split with {len(blueprint_files)} files into {len(train_files)} train and {len(val_files)} val")
        
        # Save train/val split information
        split_info = {
            'train_files': train_files,
            'val_files': val_files,
            'train_ratio': config.get('train_ratio', 0.98),
            'total_files': len(blueprint_files),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Save to JSON
        os.makedirs('data/dataset_processed', exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Saved train/val split information to {split_file}")
    
    # Create datasets with their respective files
    train_dataset = CADDataset(
        data_path='data',
        max_input_length=config['max_input_length'],
        max_output_length=config['max_output_length'],
        input_vec_dim=config['input_dim'],
        target_vec_dim=1,
        file_list=train_files,
        is_validation=False,
        epoch=epoch,
        save_dir=config['save_dir']
    )
    
    val_dataset = CADDataset(
        data_path='data',
        max_input_length=config['max_input_length'],
        max_output_length=config['max_output_length'],
        input_vec_dim=config['input_dim'],
        target_vec_dim=1,
        file_list=val_files,
        is_validation=True,
        epoch=epoch,
        save_dir=config['save_dir']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader, val_loader