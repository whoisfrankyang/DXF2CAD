from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge, 
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.TopoDS import TopoDS_Wire, TopoDS_Face, TopoDS_Shape
from typing import List
import numpy as np
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
import os

def create_line(start_coords, end_coords):
    """
    Create a line (edge) from start point to end point
    
    Args:
        start_coords: tuple/list of (x,y,z) coordinates for start point
        end_coords: tuple/list of (x,y,z) coordinates for end point
    
    Returns:
        TopoDS_Edge: The created edge
    """
    # Create points
    start_point = gp_Pnt(*start_coords)
    end_point = gp_Pnt(*end_coords)
    
    # Create edge
    edge = BRepBuilderAPI_MakeEdge(start_point, end_point).Edge()
    return edge

def create_wire_from_edges(edges):
    """
    Create a wire from a list of edges
    
    Args:
        edges: List of TopoDS_Edge objects
    
    Returns:
        TopoDS_Wire: The created wire
    """
    # Create wire builder
    wire_maker = BRepBuilderAPI_MakeWire()
    
    # Add edges to wire
    for edge in edges:
        wire_maker.Add(edge)
    
    if not wire_maker.IsDone():
        raise Exception("Failed to create wire from edges")
    
    return wire_maker.Wire()

def extrude_wire(wire: TopoDS_Wire, distance: float, direction: str = 'z') -> TopoDS_Shape:
    """
    Extrude a wire in the specified direction
    
    Args:
        wire: TopoDS_Wire to extrude
        distance: Extrusion distance
        direction: 'x', 'y', or 'z' for extrusion direction
    
    Returns:
        TopoDS_Shape: The extruded shape
    """
    # Create face from wire
    face_maker = BRepBuilderAPI_MakeFace(wire)
    if not face_maker.IsDone():
        raise Exception("Failed to create face from wire")
    face = face_maker.Face()
    
    # Set extrusion direction
    dir_map = {
        'x': gp_Dir(1, 0, 0),
        'y': gp_Dir(0, 1, 0),
        'z': gp_Dir(0, 0, 1)
    }
    extrusion_dir = dir_map.get(direction.lower(), gp_Dir(0, 0, 1))
    
    # Create extrusion vector
    vec = gp_Vec(extrusion_dir).Multiplied(distance)
    
    # Perform extrusion
    prism_maker = BRepPrimAPI_MakePrism(face, vec)
    if not prism_maker.IsDone():
        raise Exception("Failed to create extrusion")
    
    return prism_maker.Shape()

# Add these token definitions
START_TOKEN = 1025
LINE_TOKEN = 1026
EXTRUSION_TOKEN = 1027
END_TOKEN = 1028

# Operation types
EXTRUSION_TYPE_TOKENS = {
    1029: 'NewBodyFeatureOperation',
    1030: 'JoinFeatureOperation',
    1031: 'CutFeatureOperation',
    1032: 'IntersectFeatureOperation'
}

# Extent types
EXTRUSION_EXTENT_TYPE_TOKENS = {
    1033: 'OneSideFeatureExtentType',    # Positive direction
    1034: 'SymmetricFeatureExtentType',  # Both directions equally
    1035: 'TwoSidesFeatureExtentType'    # Two different distances
}

def construct_from_sequence(sequence: List[int]) -> TopoDS_Shape:
    """
    Construct a CAD model from a token sequence
    
    Sequence format:
    - Line: [1026, x1, y1, z1, x2, y2, z2]
    - Extrusion: [1027, op_type, extent_type, dist1, angle1, dist2, angle2]
    
    Args:
        sequence: List of integers representing the construction sequence
    
    Returns:
        TopoDS_Shape: The constructed shape
    """
    current_edges = []
    current_shape = None
    i = 0
    
    while i < len(sequence):
        token = sequence[i]
        
        if token == LINE_TOKEN:
            # Extract line coordinates
            if i + 6 >= len(sequence):
                print("Error: Incomplete LINE token sequence")
                break
                
            start_coords = sequence[i+1:i+4]
            end_coords = sequence[i+4:i+7]
            
            try:
                edge = create_line(start_coords, end_coords)
                current_edges.append(edge)
            except Exception as e:
                print(f"Error creating line: {e}")
            
            i += 7
            
        elif token == EXTRUSION_TOKEN:
            # Extract extrusion parameters
            if i + 6 >= len(sequence):
                print("Error: Incomplete EXTRUSION token sequence")
                break
                
            op_type = sequence[i+1]
            extent_type = sequence[i+2]
            dist1 = sequence[i+3]
            angle1 = sequence[i+4]  # Currently unused
            dist2 = sequence[i+5]
            angle2 = sequence[i+6]  # Currently unused
            
            # Create wire from collected edges
            if not current_edges:
                print("Warning: No edges to extrude")
                i += 7
                continue
                
            try:
                wire = create_wire_from_edges(current_edges)
                
                # Determine extrusion parameters based on extent type
                if extent_type == 1033:  # One side positive
                    shape = extrude_wire(wire, dist1, 'z')
                elif extent_type == 1034:  # Symmetric
                    # For symmetric, use dist1 in both directions
                    shape = extrude_wire(wire, dist1/2, 'z')
                    # This is likely incorrect - you can't extrude the same wire twice
                    # shape = extrude_wire(wire, -dist1/2, 'z')
                elif extent_type == 1035:  # Two sides
                    if dist1 > 0:
                        shape = extrude_wire(wire, dist1, 'z')
                    if dist2 > 0:
                        # This is likely incorrect - you can't extrude the same wire twice
                        # shape = extrude_wire(wire, -dist2, 'z')
                        pass
                
                current_shape = shape
            except Exception as e:
                print(f"Error during extrusion: {e}")
                
            current_edges = []  # Reset edges for next potential operation
            i += 7
            
        elif token == END_TOKEN:
            break
            
        else:
            i += 1
    
    if current_shape is None:
        print("Warning: No shape was created")
        
    return current_shape

def test_sequence(sequence):
    """Test the construction with a sample sequence"""

    shape = construct_from_sequence(sequence)
    return shape

def export_to_step(shape: TopoDS_Shape, filename: str) -> bool:
    """
    Export a shape to STEP file
    
    Args:
        shape: TopoDS_Shape to export
        filename: Output STEP file path
    
    Returns:
        bool: True if export successful
    """
    # Make sure the shape is valid
    if shape is None:
        print("Error: Cannot export None shape")
        return False
        
    # Set units to MM
    Interface_Static.SetCVal("write.step.unit", "MM")
    
    # Create STEP writer
    step_writer = STEPControl_Writer()
    
    # Transfer shape
    status = step_writer.Transfer(shape, STEPControl_AsIs)
    if status != IFSelect_RetDone:
        print("Error: Failed to transfer shape to STEP format")
        return False
    
    # Write file
    status = step_writer.Write(filename)
    return status == IFSelect_RetDone

def read_sequence(text_path: str) -> list:
    """
    Read a text file containing tokens, where each line may contain one or more numbers 
    separated by whitespace. Returns the tokens as a list of integers.
    
    Args:
        text_path: The path to the text file containing the sequence.

    Returns:
        A list of integers representing the token sequence.
    """
    sequence = []
    with open(text_path, 'r') as file:
        for line in file:
            # Ignore empty lines
            tokens = line.strip().split()
            if tokens:
                numbers = [int(token) for token in tokens]
                sequence.extend(numbers)
    return sequence

# Example usage:
if __name__ == "__main__":
    text_path = "inference_sample/test_input.txt"
    sequence = read_sequence(text_path)
    print("Sequence read from file:", sequence)

    shape = test_sequence(sequence)
    
    # Export to STEP
    output_path = "test_output.step"  # Changed to current directory
    if export_to_step(shape, output_path):
        print(f"Successfully exported STEP file to: {output_path}")
    else:
        print("Failed to export STEP file")
