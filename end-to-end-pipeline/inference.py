import torch
import numpy as np
import os
import json
from model import CADTransformer
from dataset import CADDataset
import argparse
from torch.utils.data import DataLoader


def print_numbers(array, name="Array"):
    print(f"Numbers in {name}:")
    for row in array:
        for value in row:
            print(value, end=" ")
    print("\n")

def quantize_values(arr, n_bits=10):
    """
    Quantizes an array of values in the range [1, 1024] to n_bits discrete levels.
    
    Args:
      arr (np.array or list): Array of continuous values to quantize.
      n_bits (int): Number of bits for quantization (e.g., 10 bits for 1024 levels).
      
    Returns:
      np.array: Discrete tokens in the range [1, 2^n_bits].
    """
    min_range = 0
    max_range = 1023
    n_levels = 2**n_bits  
    
    quantized_arr = (arr - min_range) * (n_levels - 1) / (max_range - min_range)
    
    return np.round(quantized_arr).astype(int) 

# List your input files here
INPUT_FILES = [
    # "data/input/91792_65f165a7_0000.npy",
    "data/input/148098_33ec30c9_0003.npy",
    # "data/input/84608_204c01e6_0002.npy",
    # "data/input/136716_076bbbb3_0008.npy",
    # "data/input/122328_4d8636de_0000.npy",
    # "data/input/142680_cd829f9e_0004.npy",
    # "data/input/100243_9fb796fe_0005.npy",
    # "data/input/110871_4b62f82f_0007.npy",
    # "data/input/23011_f267137c_0001.npy",
    # "data/input/85638_2ab1040c_0003.npy",
    # "data/input/137837_9c9f163d_0007.npy",
    # "data/input/21893_0500d043_0038.npy",
    # "data/input/22463_c48bb23e_0002.npy",
    # "data/input/41744_5c450f9d_0001.npy",
    # "data/input/55707_c78416ed_0019.npy",
    # "data/input/23144_88ca00a5_0005.npy",
    # "data/input/30379_f1d5e193_0004.npy",
    # "data/input/115535_a171c715_0000.npy",
    # "data/input/23144_88ca00a5_0007.npy",
    # "data/input/73081_906f09d5_0000.npy",
    # "data/input/88680_0fb9b042_0000.npy",
    # "data/input/146135_0ca63e9c_0003.npy"
]

# Output directory for results
OUTPUT_DIR = "inference_results"

class CADPredictor:
    def __init__(self, checkpoint_dir):
        # Load config
        with open(os.path.join(checkpoint_dir, 'config.json'), 'r') as f:
            self.config = json.load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = CADTransformer(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_encoder_layers=self.config['num_encoder_layers'],
            num_decoder_layers=self.config['num_decoder_layers']
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), 
                              map_location=self.device,
                              weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Special tokens
        self.START_TOKEN = 1025
        self.END_TOKEN = 1028
        
    def predict_batch(self, input_files):
        """
        Generate predictions for multiple input files
        """
        results = {}
        
        # Filter out non-existent files
        valid_files = [f for f in input_files if os.path.exists(f)]
        if len(valid_files) != len(input_files):
            print(f"\nWarning: {len(input_files) - len(valid_files)} files not found:")
            for f in input_files:
                if f not in valid_files:
                    print(f"  - Missing: {f}")
            print()
        
        for input_file in valid_files:
            print(f"\nProcessing: {input_file}")
            try:
                result = self.predict(input_file)
                results[input_file] = result
                print(f"Generated sequence length: {result['sequence_length']}")
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                results[input_file] = {"error": str(e)}
                
        return results
        
    def predict(self, input_file):
        """Generate prediction for a single input file"""
        try:
            if not os.path.exists(input_file):
                raise ValueError(f"File does not exist: {input_file}")
            
            # Load and verify input data
            input_data = np.load(input_file)
            if input_data is None:
                raise ValueError(f"Failed to load data from {input_file}")
            if len(input_data) == 0:
                raise ValueError(f"Empty data in {input_file}")
                
            print(f"Loaded input shape: {input_data.shape}")  # Debug print
            
            # Process input data
            sorted_data = CADDataset._sort_curve_sequence(input_data)
            # print_numbers(sorted_data, "sorted_data")
            processed_data = CADDataset._transform_curve_sequence(sorted_data)
            # print_numbers(processed_data, "processed_data")
            # Convert to tensor and add batch dimension
            src = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
            src_mask = torch.ones(1, src.size(1)).to(self.device)
            
            print(f"Processed input shape: {src.shape}")  # Debug print
            
            with torch.no_grad():
                batch_size = src.size(0)
                max_length = self.config['max_output_length']
                
                # Initialize with START token
                tgt_input = torch.full((batch_size, 1), self.START_TOKEN, 
                                     device=self.device)
                outputs = []
                
                # Generate sequence
                for _ in range(max_length - 1):
                    logits = self.model(src, tgt_input, src_mask)
                    
                    # Use argmax for deterministic output
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    
                    tgt_input = torch.cat([tgt_input, next_token], dim=-1)
                    outputs.append(next_token)
                    
                    # Stop if END token generated
                    if next_token.item() == self.END_TOKEN:
                        break
                
                generated_sequence = torch.cat(outputs, dim=1)[0].cpu().tolist()
                
                return {
                    'generated_sequence': generated_sequence,
                    'input_shape': input_data.shape,
                    'sequence_length': len(generated_sequence)
                }
        except Exception as e:
            raise ValueError(f"Error processing input file: {e}")

def save_results(results, output_dir):
    """Save results to text and numpy files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results to text file
    txt_file = os.path.join(output_dir, 'inference_results.txt')
    
    with open(txt_file, 'w') as f:
        f.write("Inference Results\n")
        f.write("=" * 50 + "\n\n")
        
        for input_file, result in results.items():
            f.write(f"Input File: {input_file}\n")
            if "error" in result:
                f.write(f"Error: {result['error']}\n")
            else:
                f.write(f"Input Shape: {result['input_shape']}\n")
                f.write(f"Sequence Length: {result['sequence_length']}\n")
                f.write(f"Generated Sequence: {result['generated_sequence']}\n")
            f.write("\n" + "-" * 50 + "\n\n")
    
    print(f"\nResults saved to: {txt_file}")
    
    # Save sequences to individual npy files
    for input_file, result in results.items():
        if "error" not in result:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            npy_file = os.path.join(output_dir, f'{base_name}_output.npy')
            np.save(npy_file, result['generated_sequence'])

def main():
    # Initialize predictor
    predictor = CADPredictor('checkpoints/011325')
    
    # Generate predictions
    results = predictor.predict_batch(INPUT_FILES)
    
    # Save results
    save_results(results, OUTPUT_DIR)

if __name__ == "__main__":
    main()