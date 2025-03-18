# Author: Bohan Yang (Jan 3, 2025)

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# PythonOCC / OCC imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopoDS import TopoDS_Iterator
from OCC.Core.Interface import Interface_Static

def round_coords(value, digits=2):
    """Round a number or tuple of numbers to specified digits"""
    if isinstance(value, tuple):
        return tuple(round(x, digits) for x in value)
    return value

def get_step_unit(step_reader):
    """Get the unit system from the STEP file"""
    try:
        # Get the unit name from STEP file
        unit_name = Interface_Static.CVal("xstep.cascade.unit")
        
        # Define conversion factors to keep in millimeters
        unit_factors = {
            "MM": 1.0,      # keep as millimeters
            "CM": 10.0,     # centimeters to millimeters
            "M": 1000.0,    # meters to millimeters
            "IN": 25.4,     # inches to millimeters
            "FT": 304.8     # feet to millimeters
        }
        
        print(f"STEP file unit system: {unit_name}")
        return unit_factors.get(unit_name.upper(), 1.0)
    
    except Exception as e:
        print(f"Warning: Could not determine STEP file units ({e})")
        print("Defaulting to millimeters (MM)")
        return 1.0  # Keep as millimeters

def extract_shapes(compound, unit_scale=1.0, view_name=None):
    """
    Given a shape (compound) from HLR output, extract line segments and circles.
    Applies appropriate flipping based on view direction before making coordinates positive.
    
    View coordinate systems:
    - Left/Right view: horizontal=y, vertical=z (flip y for right view)
    - Top/Bottom view: horizontal=y, vertical=x (flip y for top view)
    - Front/Back view: horizontal=x, vertical=z (flip x for back view)
    """
    lines = []
    circles = []

    if compound.IsNull():
        return lines, circles

    # First collect all points
    points = []
    explorer = TopExp_Explorer(compound, TopAbs_EDGE)
    while explorer.More():
        edge = explorer.Current()
        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            pnt = BRep_Tool.Pnt(vertex)
            points.append((pnt.X(), pnt.Y(), pnt.Z()))
            vertex_explorer.Next()
        explorer.Next()

    # Apply flipping based on view
    if view_name:
        flipped_points = []
        for x, y, z in points:
            if view_name == 'Right View':
            
                # Flip y-axis for right view
                flipped_points.append((-x, -y, -z))
            elif view_name == 'Bottom View':
                # Flip x-axis for bottom view
                flipped_points.append((-x, y, z))
            elif view_name == 'Back View':
                # Flip y-axis for back view
                flipped_points.append((-x, y, z))
            else:
                # No flipping for left, top, front views
                flipped_points.append((x, y, z))
        points = flipped_points

    # Find min values after flipping

    # comment on Jan 5, 2025:
    # the x, y contains true values while z always 0. 
    x_coords, y_coords, z_coords = zip(*points)
    x_min = min(x_coords)
    y_min = min(y_coords)
    z_min = min(z_coords)

    # Calculate offsets to make all coordinates positive
    x_offset = abs(x_min) if x_min < 0 else 0
    y_offset = abs(y_min) if y_min < 0 else 0
    z_offset = abs(z_min) if z_min < 0 else 0

    # Extract shapes with flipped coordinates and offsets
    explorer = TopExp_Explorer(compound, TopAbs_EDGE)
    while explorer.More():
        edge = explorer.Current()
        curve = BRepAdaptor_Curve(edge)
        curve_type = curve.GetType()
        
        if curve_type == GeomAbs_Circle:
            circ = curve.Circle()
            center = circ.Location()
            radius = circ.Radius()
            
            # Apply flipping and offset
            x, y, z = center.X(), center.Y(), center.Z()
            if view_name == 'Right View':
                x = -x
                
            elif view_name == 'Bottom View':
                x = -x
            elif view_name == 'Back View':
                x = -x
                
            circles.append((
                round_coords((x + x_offset) * unit_scale),
                round_coords((y + y_offset) * unit_scale),
                round_coords(radius * unit_scale)
            ))
        else:
            vertices = []
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            while vertex_explorer.More():
                vertex = vertex_explorer.Current()
                pnt = BRep_Tool.Pnt(vertex)
                
                # Apply flipping and offset
                x, y, z = pnt.X(), pnt.Y(), pnt.Z()
                if view_name == 'Right View':
                    x = -x
                elif view_name == 'Bottom View':
                    x = -x

                elif view_name == 'Back View':
                    x = -x
                    
                vertices.append((
                    round_coords((x + x_offset) * unit_scale),
                    round_coords((y + y_offset) * unit_scale)
                ))
                vertex_explorer.Next()
            
            if len(vertices) == 2:
                lines.append(vertices)
        
        explorer.Next()
    
    return lines, circles

def get_shape_center(shape):
    """
    Calculate the center point of a shape using its bounding box.
    Returns gp_Pnt of the center.
    """
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    center_z = (zmin + zmax) / 2
    
    return gp_Pnt(center_x, center_y, center_z)

def perform_hlr_projection(shape, view_dir, hide=True):
    """
    Perform a Hidden Line Removal projection for a given shape and view direction.
    
    Args:
        shape: TopoDS_Shape to project.
        view_dir: gp_Dir for the projection direction.
        hide: whether to call hlr.Hide() (hides hidden lines). 
              If True, hidden lines will be in the hidden compound.
    Returns:
        (visible_edges, hidden_edges): the two compounds
    """
    # Create the HLR algorithm
    hlr = HLRBRep_Algo()
    hlr.Add(shape)

    # Compute center of shape for the projection eye point
    shape_center = get_shape_center(shape)
    # print(f"Shape center: {shape_center.X()}, {shape_center.Y()}, {shape_center.Z()}")

    # Setup the projector
    # gp_Ax2(Origin, Direction) sets the direction of the Z-axis of the 
    # coordinate system. That becomes the projection direction in HLR.
    projector = HLRAlgo_Projector(gp_Ax2(shape_center, view_dir))
    
    hlr.Projector(projector)
    hlr.Update()

    # If hide=True, hidden edges go into the hidden compound
    if hide:
        hlr.Hide()

    # Convert to shape
    hlr_shape = HLRBRep_HLRToShape(hlr)
    visible_edges = hlr_shape.VCompound()
    hidden_edges = hlr_shape.HCompound()

    return visible_edges, hidden_edges

def are_points_equal(p1, p2, tolerance=1e-6):
    """Check if two points are equal within tolerance"""
    # Round points before comparison
    p1 = round_coords(p1)
    p2 = round_coords(p2)
    return p1 == p2

def are_lines_equal(line1, line2, tolerance=1e-6):
    """Check if two lines have the same endpoints (in either direction) within tolerance"""
    p1, p2 = line1
    q1, q2 = line2
    return (are_points_equal(p1, q1, tolerance) and are_points_equal(p2, q2, tolerance)) or \
           (are_points_equal(p1, q2, tolerance) and are_points_equal(p2, q1, tolerance))

def are_circles_equal(circle1, circle2):
    """Check if two circles have exactly the same center and radius"""
    return circle1 == circle2

def get_line_range(line):
    """Get the range of a line (works for vertical or horizontal lines)"""
    p1, p2 = line
    if p1[0] == p2[0]:  # Vertical line
        return p1[0], min(p1[1], p2[1]), max(p1[1], p2[1])
    else:  # Horizontal line
        return p1[1], min(p1[0], p2[0]), max(p1[0], p2[0])

def merge_ranges(ranges):
    """Merge overlapping ranges"""
    if not ranges:
        return []
    
    # Sort ranges by start point
    ranges = sorted(ranges)
    merged = [ranges[0]]
    
    for current in ranges[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:
            # Ranges overlap, merge them
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            # No overlap, add new range
            merged.append(current)
    
    return merged

def is_point_on_line_segment(point, line_start, line_end, tolerance=1e-6):
    """Check if a point lies on a line segment"""
    # Convert to numpy arrays for easier calculation
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)
    
    # Calculate distances
    ab = b - a  # Line vector
    ap = p - a  # Point to line start vector
    
    # Calculate projection
    t = np.dot(ap, ab) / np.dot(ab, ab)
    
    # Check if point is within line segment bounds
    if 0 <= t <= 1:
        # Calculate perpendicular distance
        projection = a + t * ab
        distance = np.linalg.norm(p - projection)
        return distance < tolerance
    return False

def is_line_covered_by_lines(hidden_line, visible_lines, tolerance=1e-6):
    """Check if a hidden line is covered by any combination of visible lines"""
    h1, h2 = hidden_line
    
    # First check for exact matches (including reversed)
    if any(are_lines_equal(hidden_line, vline) for vline in visible_lines):
        return True
        
    # Get points along the hidden line
    num_points = 30  # Number of points to check
    points_to_check = []
    for i in range(num_points + 1):
        t = i / num_points
        x = h1[0] + t * (h2[0] - h1[0])
        y = h1[1] + t * (h2[1] - h1[1])
        points_to_check.append((x, y))
    
    # Check if each point is covered by any visible line
    for point in points_to_check:
        point_covered = False
        for vline in visible_lines:
            if is_point_on_line_segment(point, vline[0], vline[1]):
                point_covered = True
                break
        if not point_covered:
            return False
    
    return True

def remove_overlapping_curves(visible_lines, hidden_lines, visible_circles, hidden_circles):
    """Remove hidden curves that are covered by any combination of visible lines"""
    # First remove hidden lines that are covered by visible lines
    filtered_hidden_lines = [
        hidden_line for hidden_line in hidden_lines
        if not is_line_covered_by_lines(hidden_line, visible_lines)
    ]

    # Remove duplicate hidden lines
    unique_hidden_lines = []
    for line in filtered_hidden_lines:
        if not any(are_lines_equal(line, unique_line) 
                  for unique_line in unique_hidden_lines):
            unique_hidden_lines.append(line)

    # Process circles (unchanged)
    filtered_hidden_circles = [
        hidden_circle for hidden_circle in hidden_circles
        if not any(are_circles_equal(hidden_circle, visible_circle)
                  for visible_circle in visible_circles)
    ]
    unique_hidden_circles = list(set(filtered_hidden_circles))

    return unique_hidden_lines, unique_hidden_circles

def plot_projection(visible_lines, hidden_lines, visible_circles, hidden_circles, view_name):
    """Plot a single projection view with both visible and hidden lines/circles"""
    plt.figure(figsize=(10, 10))
    
    # Define axis labels based on view
    if view_name in ['Left View', 'Right View']:
        horizontal_axis = 'Y'
        vertical_axis = 'Z'
    elif view_name in ['Top View', 'Bottom View']:
        horizontal_axis = 'Y'
        vertical_axis = 'X'
    else:  # front, back
        horizontal_axis = 'X'
        vertical_axis = 'Z'
    
    # Plot visible lines in solid black
    for line in visible_lines:
        x, y = zip(*line)
        plt.plot(x, y, 'k-', linewidth=1, label='Visible Lines')
        plt.plot(x[0], y[0], 'ko', markersize=5)
        plt.plot(x[1], y[1], 'ko', markersize=5)

    # Plot hidden lines in dashed red
    for line in hidden_lines:
        x, y = zip(*line)
        plt.plot(x, y, 'r--', linewidth=1, label='Hidden Lines')
        plt.plot(x[0], y[0], 'ro', markersize=5)
        plt.plot(x[1], y[1], 'ro', markersize=5)

    # Plot visible circles in solid black
    for cx, cy, r in visible_circles:
        circle = plt.Circle((cx, cy), r, color='black', fill=False)
        plt.gca().add_patch(circle)
        plt.plot(cx, cy, 'ko', markersize=5)

    # Plot hidden circles in dashed red
    for cx, cy, r in hidden_circles:
        circle = plt.Circle((cx, cy), r, color='red', fill=False, linestyle='--')
        plt.gca().add_patch(circle)
        plt.plot(cx, cy, 'ro', markersize=5)

    plt.title(f'2D Orthographic Projection - {view_name.capitalize()} View')
    plt.xlabel(f'{horizontal_axis} Coordinate')
    plt.ylabel(f'{vertical_axis} Coordinate')
    plt.grid(True)
    plt.axis('equal')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.save

def count_solids(shape):
    """Count number of solid bodies in a shape"""
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    
    return count

def load_step_file(step_path):
    """Load a STEP file and return the shape"""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_path)
    
    if status != IFSelect_RetDone:
        raise Exception("Could not read the STEP file.")
    
    # Get unit scale factor from STEP file
    unit_scale = get_step_unit(step_reader)
    
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    if shape.IsNull():
        raise Exception("Null shape. No geometry loaded from STEP.")
        
    return shape

def check_body_count(shape):
    """Check if shape has exactly one solid body"""
    num_solids = count_solids(shape)
    return num_solids == 1

def get_projected_edges_with_visibility(shape, view_name):
    """Get projected edges with visibility information for a given view"""
    # Map view names to directions
    view_directions = {
        'left':   gp_Dir( 1,  0,  0),
        'front':  gp_Dir( 0,  0,  1),
        'right':  gp_Dir(-1,  0,  0),
        'top':    gp_Dir( 0,  1,  0),
        'bottom': gp_Dir( 0, -1,  0),
        'back':   gp_Dir( 0,  0, -1)
    }
    
    direction = view_directions[view_name]
    visible_edges, hidden_edges = perform_hlr_projection(shape, direction, hide=True)
    
    # Extract lines and circles
    visible_lines, visible_circles = extract_shapes(visible_edges)
    hidden_lines, hidden_circles = extract_shapes(hidden_edges)
    
    # Remove overlapping curves
    hidden_lines, hidden_circles = remove_overlapping_curves(
        visible_lines, hidden_lines, visible_circles, hidden_circles)
    
    return visible_lines, hidden_lines

def get_projected_edges(shape, view_name, save_path=None):
    """Get projected edges and optionally save as image"""
    visible_lines, hidden_lines = get_projected_edges_with_visibility(shape, view_name)
    
    if save_path:
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Plot visible lines in solid black
        for line in visible_lines:
            x, y = zip(*line)
            plt.plot(x, y, 'k-', linewidth=1)
        
        # Plot hidden lines in dashed red
        for line in hidden_lines:
            x, y = zip(*line)
            plt.plot(x, y, 'r--', linewidth=1)
        
        # Define correct axis labels based on view
        if view_name in ['left', 'right']:
            horizontal_axis = 'Y'
            vertical_axis = 'Z'
        elif view_name in ['top', 'bottom']:
            horizontal_axis = 'Y'
            vertical_axis = 'X'
        else:  # front, back
            horizontal_axis = 'X'
            vertical_axis = 'Z'
        
        plt.title(f'2D Orthographic Projection - {view_name}')
        plt.xlabel(f'{horizontal_axis} Coordinate')
        plt.ylabel(f'{vertical_axis} Coordinate')
        plt.grid(True)
        plt.axis('equal')
        
        # Save and close
        plt.savefig(save_path)
        plt.close()
    
    return visible_lines, hidden_lines

def main():
    step_file_path = "2D_data_preprocess/data/line_only_step/20241_6bced5ac_0000.step"

    if not os.path.exists(step_file_path):
        print(f"Error: STEP file not found at '{step_file_path}'")
        sys.exit(1)

    # Load STEP file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file_path)

    if status != IFSelect_RetDone:
        print("Error: Could not read the STEP file.")
        sys.exit(1)
    
    # Get unit scale factor from STEP file
    unit_scale = get_step_unit(step_reader)
    print(f"Unit scale factor: {unit_scale}")
    
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    if shape.IsNull():
        print("Error: Null shape. No geometry loaded from STEP.")
        sys.exit(1)

    # Check number of solids
    num_solids = count_solids(shape)
    if num_solids > 1:
        print(f"Warning: STEP file contains {num_solids} solid bodies.")
        print("Skipping projection as multiple bodies detected.")
        return
    elif num_solids == 0:
        print("Warning: No solid bodies found in STEP file.")
        return

    # Prepare the list of views
    views = [
        ("Left View",  gp_Dir( 1,  0,  0)),
        ("Right View", gp_Dir(-1,  0,  0)),
        ("Top View",   gp_Dir( 0,  1,  0)),
        ("Bottom View",gp_Dir( 0, -1,  0)),
        ("Front View", gp_Dir( 0,  0,  1)),
        ("Back View",  gp_Dir( 0,  0, -1)),
    ]

    # For each view, perform HLR and extract geometry
    for view_name, direction in views:
        print("\n============================================")
        print(f"Processing {view_name}")

        # Perform HLR
        visible_edges, hidden_edges = perform_hlr_projection(shape, direction, hide=True)

        # Extract lines/circles with unit conversion
        visible_lines, visible_circles = extract_shapes(visible_edges, unit_scale, view_name)
        hidden_lines, hidden_circles = extract_shapes(hidden_edges, unit_scale, view_name)

        hidden_lines, hidden_circles = remove_overlapping_curves(
            visible_lines, hidden_lines, visible_circles, hidden_circles)

        # Print results
        print(f"\nVisible Lines ({len(visible_lines)}):")
        for line in visible_lines:
            print(f"    {line}")

        print(f"\nHidden Lines ({len(hidden_lines)}):")
        for line in hidden_lines:
            print(f"    {line}")

        print(f"\nVisible Circles ({len(visible_circles)}):")
        for cx, cy, r in visible_circles:
            print(f"    Center=({cx:.3f}, {cy:.3f}), Radius={r:.3f}")

        print(f"\nHidden Circles ({len(hidden_circles)}):")
        for cx, cy, r in hidden_circles:
            print(f"    Center=({cx:.3f}, {cy:.3f}), Radius={r:.3f}")
            
        # Plot the projection
        plot_projection(visible_lines, hidden_lines, visible_circles, hidden_circles, view_name)

if __name__ == "__main__":
    main()
