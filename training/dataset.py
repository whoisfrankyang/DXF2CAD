import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import numpy as np
import os
from utils import quantize_values

class CADDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 max_length: int = 512,
                 input_vec_dim: int = 19,
                 target_vec_dim: int = 1,
                 add_special_tokens: bool = True):
        """
            data_path: Base path to the data files
            max_length: Maximum length of the sequence
            input_vec_dim: Dimension of each input vector (2D blueprint - 28)
            target_vec_dim: Dimension of each target vector (3D sequence - 1 or 10)
            add_special_tokens: Whether to add <sos> and <eos> tokens to target sequence
        """
        self.data_path = data_path
        self.max_length = max_length - 2 if add_special_tokens else max_length  # Reserve space for special tokens
        self.input_vec_dim = input_vec_dim
        self.target_vec_dim = target_vec_dim
        self.add_special_tokens = add_special_tokens
        
        # Special token vectors
        # self.sos_token = np.zeros(target_vec_dim)  # Start of sequence token
        # self.eos_token = np.ones(target_vec_dim)   # End of sequence token
        
        # Load and preprocess data
        self.input_seq, self.output_seq = self.load_data()
        
    def sort_curve_sequence(self, curve_seq: np.ndarray) -> np.ndarray:
        """Sort the curve sequence before transformation"""
        
        # First group by views (indices 0:7)
        view_groups = {}
        for curve in curve_seq:
            view_key = tuple(curve[0:7])
            if view_key not in view_groups:
                view_groups[view_key] = []
            view_groups[view_key].append(curve)
        
        sorted_curves = []
        for view_key in sorted(view_groups.keys()):
            curves = view_groups[view_key]
            
            # Reorder endpoints within each curve (for lines and arcs)
            for curve in curves:
                curve_type = curve[9:12]
                if curve_type[0] == 1 or curve_type[2] == 1:  # Line or Arc
                    sx, sy = curve[12], curve[13]
                    ex, ey = curve[14], curve[15]
                    if sx > ex or (sx == ex and sy > ey):
                        # Swap endpoints
                        curve[12:14], curve[14:16] = curve[14:16], curve[12:14]
            
            # Sort curves based on their coordinates
            def curve_sort_key(curve):
                if curve[9] == 1 or curve[11] == 1:  # Line or Arc
                    return (curve[12], curve[14], curve[13], curve[15])  # sx, ex, sy, ey
                else:  # Circle
                    return (curve[16], curve[16], curve[17], curve[17])  # cx, cx, cy, cy
            
            curves.sort(key=curve_sort_key)
            sorted_curves.extend(curves)
        
        return np.array(sorted_curves)
    
    def transform_curve_sequence(self, curve_seq: np.ndarray) -> np.ndarray:
        """
        Transform curve sequence into point sequence.
        Input vector dim: 29
            - indices 9-11: one-hot encoding for [Line, Circle, Arc]
        Output: Multiple tokens per curve, with curve type preserved
        """
        transformed_seq = []
        
        for curve in curve_seq:
            # Check curve type (indices 9-11)
            view = curve[0:7]
            vis = curve[7:9]
            curve_type = curve[9:12]
            params = curve[12:]
            
            if curve_type[0] == 1:  # Line
                # Create 4 point tokens (start_x, start_y, end_x, end_y)
                for i in range(4):
                    point_token = np.concatenate([view, vis, curve_type, [quantize_values(params[12+i])], [quantize_values(params[26])]])
                    transformed_seq.append(point_token)
                    
            elif curve_type[1] == 1:  # Circle
                # Create 2 point tokens (center_x, center_y)
                for i in range(2):
                    point_token = np.concatenate([view, vis, curve_type, [quantize_values(params[16+i])], [quantize_values(params[26])]])
                    transformed_seq.append(point_token)
                    
            elif curve_type[2] == 1:  # Arc
                # Create 4 point tokens (start_x, start_y, end_x, end_y)
                for i in range(4):
                    point_token = np.concatenate([view, vis, curve_type, [quantize_values(params[12+i])], [quantize_values(params[26])]])
                    transformed_seq.append(point_token)
                    
            else:
                # If not a recognized curve type, keep as is
                print("UNEXPECTED CURVE TYPE")
        
        return np.array(transformed_seq)
    
    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load input (2D seq) and output (3D seq) from npy files."""
        input_seqs = []
        output_seqs = []
        
        blueprint_path = os.path.join(self.data_path, 'blueprint_seq')
        cad_path = os.path.join(self.data_path, 'cad_seq')
        
        blueprint_files = [f for f in os.listdir(blueprint_path) if f.endswith('_combined.npy')]
        
        for bp_file in blueprint_files:
            base_name = bp_file.split('_combined')[0]
            cad_file = f"{base_name}.npy"
            
            if os.path.exists(os.path.join(cad_path, cad_file)):
                input_data = np.load(os.path.join(blueprint_path, bp_file))

                input_data = self.sort_curve_sequence(input_data)  # Sort first
                input_data = self.transform_curve_sequence(input_data)  # Then transform into tokens
                
                # Load output sequence
                output_data = np.load(os.path.join(cad_path, cad_file))
                if output_data.ndim == 1:
                    print("WHAT what is this?")
                
                input_seqs.append(input_data)
                output_seqs.append(output_data)
        
        return input_seqs, output_seqs
    
    def pad_sequence(self, seq: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
        """Pad or truncate sequence to max_length"""
        curr_len = len(seq)
        if curr_len < self.max_length:
            if seq.ndim == 2:  # For blueprint 2D seq (vec based)
                padding = np.zeros((self.max_length - curr_len, seq.shape[1])) + pad_value
                padded_seq = np.vstack([seq, padding])
            else:  # For CAD 3D seq (token based)
                padding = np.zeros(self.max_length - curr_len) + pad_value
                padded_seq = np.concatenate([seq, padding])
        else:
            raise ValueError("Sequence length exceeds max_length")
        
        return padded_seq
    
    def create_padding_mask(self, seq_length: int) -> torch.Tensor:
        """Create padding mask for attention (1 for real tokens, 0 for padding)"""
        mask = torch.zeros(self.max_length)
        mask[:seq_length] = 1
        return mask
    
    def __len__(self) -> int:
        return len(self.input_seq)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_seq = self.input_seq[idx]
        output_seq = self.output_seq[idx]
        
        # Store original lengths before padding
        input_length = len(input_seq)
        output_length = len(output_seq)
        
        # Pad sequences
        input_seq = self.pad_sequence(input_seq)
        output_seq = self.pad_sequence(output_seq)
        
        # Create attention masks
        input_mask = self.create_padding_mask(input_length)
        output_mask = self.create_padding_mask(output_length + 2 if self.add_special_tokens else output_length)
        
        return (
            torch.tensor(input_seq, dtype=torch.float32),      # Input sequence
            torch.tensor(output_seq, dtype=torch.float32),     # Target sequence
            input_mask,                                        # Input padding mask
            output_mask                                        # Output padding mask
        )

def create_dataloaders(
    train_dataset: CADDataset,
    val_dataset: CADDataset,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader