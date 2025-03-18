# DXF2CAD 
An AI-powered end-to-end pipeline for converting 2D blueprints in DXF format to 3D CAD models in STEP format using parametric sequence modeling. 


An example of DXF to CAD where I input a DXF file and the pipeline outputs a STEP file:
![Example of DXF-to-CAD](README_DXF2CAD_demo.png)



The inference flowmap:
![Inference flowmap](README_inference_flowmap.png)



The training and data preparation flowmap:
![Training flowmap](README_training_flowmap.png)





Due to data limitation, especially in the lack of blueprint and 3D CAD pairs, we transforms STEP files into orthographic views and then into vector-encoded features for the purpose of training. 
Then during inference stage, we extract entities from DXF files and store the entities into vector-encoded features in a similar fashion. 


## Contributions: 
  - A novel end-to-end pipeline for converting 2D blueprints in DXF format to 3D CAD in STEP format, including DXF entity extraction, a foundation model that transforms orthographic views into CAD parametric sequences, and a CAD rendering script. 
  - A dataset with (orthographic views and vector-encoded parametric sequence) for model training. 
  - A script for generating orthographic views from 3D CAD.
  - A foundation model for transforming orthographic views into CAD parametric sequences. 

## Training Data Preprocessing Generation:
1. `data_pipeline.py`: runs the entire data generation pipeline. Input should be the STEP files and JSON sequence files from the Fusion 360 Reconstruction dataset. But of coruse, this pipeline can be accomodated to other datasets where there exists only 3D STEP files but not the corresponding orthographic viewers.

Below is a detailed flowmap for our data generation pipeline:
![Data Generation Pipeline flowmap](README_data_generation_pipeline.png)

Final training data:
X: curve features of 2D orthographic views
Y: vector-encoded construction sequence of 3D CAD

Helpers:
- `seq_stats.py`: perform some statistic analysis on the processed 3D CAD json files.
- `step_stats.py`: perform some statistic analysis on the 3D CAD step files. 
- `file_matcher.py`: match files between two folders based on specified mode.

## Model Training (Vec2Seq)
- `model.py`: define model architecture.
- `dataset.py`: define logics for data augmentation, data validity check, and data padding
- `train.py`: define training logic and hyperparameters. 
- `utils.py`: quantization functions

## Model Inference (DXF2CAD Deployment Pipeline)
- `feature_encode.py`: extract features from curve entities in DXF files and store the features in vector-encoded entities. Using the Union-Find clustering algorithm in `clustering.py` to produce cluster labels on different orthographic view in a blueprint. 
- `inference.py`:
- `seq2CAD.py`:

## Raw Dataset 
Autodesk Reconstruction Dataset: 
 - 3D CAD models (.STEP)
 - Raw 3D CAD sequences (.JSON)

## Model Checkpoint:
checkpoint available at: https://drive.google.com/drive/folders/1tG59lJtayAfvI42L1VXMiYu0qd7chHQY?usp=share_link


Interested for futher collaboration? Please contact me at: 
- Email: by93@cornell.edu
- LinkedIn: https://www.linkedin.com/in/bohan-yang-b42959220/

