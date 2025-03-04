
import numpy as np

def quantize_values(arr, n_bits = 9):
  """
  Takes in an array that's normalized between 0 and 1 and quantizes it to n_bits.
  """
  min_range = 0
  max_range = 1
  n_levels = 2**n_bits - 1
  quantize_arr = (arr - min_range) * n_levels / (max_range - min_range) 
  return np.round(quantize_arr).astype('long')

def dequantize_values(arr, n_bits=9):
    """
    Takes in a quantized array of integers and converts it back to floats between 0 and 1.
    """
    min_range = 0
    max_range = 1
    n_levels = 2**n_bits - 1
    dequantize_arr = arr * (max_range - min_range) / n_levels + min_range
    return dequantize_arr.astype('float32')

