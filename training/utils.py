import numpy as np



# def quantize_values(arr, n_bits = 9):
#   """
#   Takes in an array that's normalized between 0 and 1 and quantizes it to n_bits.
#   """
#   min_range = 0
#   max_range = 1
#   n_levels = 2**n_bits - 1
#   quantize_arr = (arr - min_range) * n_levels / (max_range - min_range) 
#   return np.round(quantize_arr).astype('long')

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


def dequantize_values(arr, n_bits=9):
    """
    Takes in a quantized array of integers and converts it back to floats between 0 and 1.
    """
    min_range = 0
    max_range = 1
    n_levels = 2**n_bits - 1
    dequantize_arr = arr * (max_range - min_range) / n_levels + min_range
    return dequantize_arr.astype('float32')

