An end-to-end pipeline for converting 2D blueprints in DXF format to 3D CAD models in STEP format
using parametric sequence modeling. 

## Dataset 
Autodesk Reconstruction Dataset: 
 - 3D CAD models (.STEP)
 - Raw 3D CAD sequences (.JSON)
## Training Data Preprocessing and Encoding (DXF2Vec)
Design Automation: a semi-automation script leveraging the Autodesk CAD2Design feature to generate 2D blueprints from STEP files. 

1. seq_extractor.py: filter raw 3D CAD json files with only Line and Extrusion.
2. seq_processor.py: process the LE-only 3D CAD json files to remove negative extrusion distances, shift coordinates, determine extrusion direction, and clip small line length or large coordinate values.
3. seq2npy.py: convert the processed 3D CAD json files into vector sequence for training.
4. file_matcher.py: match the processed seq files with the step files. 
5. step_project.py: using functions defined in orthographic_projection.py, project the 3D CAD step files into 6 orthographic views. 

Final training data:
X: generated orthographic views 
Y: vector-encoded construction sequence

Helpers:
seq_stats.py: perform some basic statistics and analysis on the processed 3D CAD json files.
step_stats.py: perform some basic statistics and analysis on the 3D CAD step files. 
file_matcher.py: match files between two folders based on specified mode.

## Model Training (Vec2Seq)
model.py: define model architecture.
dataset.py: define logics for data augmentation, data validity check, and data padding
train.py: define training logic and hyperparameters. 
utils.py: quantization functions

## Model Inference 
model checkpoint( 01082503_checkpoint): contains best_model.pt, config.json, and training_history.csv
## CAD Generation (Seq2CAD)
