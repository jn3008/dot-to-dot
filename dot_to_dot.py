# dot_to_dot.py
import numpy as np
from matplotlib import pyplot as plt
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from svgpathtools import parse_path, svg2paths, Line, QuadraticBezier, CubicBezier
import argparse
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from scipy.spatial.distance import pdist, squareform
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random 
from scipy.spatial.distance import pdist, squareform

def find_closest_pair(list1, list2):
    """
    Finds the closest pair of points between two lists of complex numbers.
    Returns the indices of the closest points in list1 and list2.
    """
    min_distance = float('inf')
    index1, index2 = -1, -1
    for i, point1 in enumerate(list1):
        for j, point2 in enumerate(list2):
            distance = abs(point1 - point2)
            if distance < min_distance:
                min_distance = distance
                index1, index2 = i, j
    return index1, index2

def rotate_and_split(lst, split_index):
    """
    Rotates the list so that the element at split_index becomes the first element.
    """
    return lst[split_index:] + lst[:split_index]

def stitch_dots(all_dots):
    """
    Stitches together multiple sublists of dots by finding the closest points between consecutive sublists,
    rotating, and splitting them to create a seamless path, while preserving original connections.
    """
    if not all_dots:
        return []
    
    # Start with the first sublist
    stitched_path = all_dots[0]
    
    # Iterate over each consecutive pair of sublists
    for i in range(len(all_dots) - 1):
        current_list = stitched_path
        next_list = all_dots[i + 1]
        
        # Find the closest pair of points between the current list and the next list
        idx1, idx2 = find_closest_pair(current_list, next_list)
        
        # Rotate both lists so that the closest points are at the beginning
        current_list = rotate_and_split(current_list, idx1)
        next_list = rotate_and_split(next_list, idx2)
        
        # Stitch the lists together in the desired overlapping manner, preserving original connections
        stitched_path = current_list + [current_list[0]] + next_list + [next_list[0]]
    
    return stitched_path

# Generate SVG Path from a glyph
def get_svg_path_from_glyph(glyph, font_path):
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    char_map = font.getBestCmap()
    
    if ord(glyph) not in char_map:
        raise ValueError(f"Glyph '{glyph}' not found in the specified font.")
    
    glyph_name = char_map[ord(glyph)]
    glyph_obj = glyph_set[glyph_name]
    
    pen = SVGPathPen(glyph_set)
    glyph_obj.draw(pen)
    path_data = pen.getCommands()
    return parse_path(path_data)

def get_evenly_spaced_points_per_curve(curve, num_points):
    """
    Generate evenly spaced points along a single curve.
    
    Parameters:
    - curve: A segment of a path (e.g., Line, CubicBezier, QuadraticBezier).
    - num_points: Number of points to generate along the curve.
    
    Returns:
    - A list of complex numbers representing the points.
    """
    total_length = curve.length()
    distances = np.linspace(0, total_length, num=num_points)
    points = [curve.point(curve.ilength(distance)) for distance in distances]
    return points

def get_evenly_spaced_points_from_subpath(subpath, num_dots_subpath):
    """
    Generate evenly spaced points across all subcurves of a set of subpaths.
    
    Parameters:
    - subpaths: List of Path objects representing continuous subpaths.
    - num_points_per_curve: Number of points to generate on each subcurve.
    
    Returns:
    - A list of points (complex numbers) from all subcurves.
    """

    curves = [curve for curve in subpath]
    # Calculate the length of each path and the total length
    curve_lengths = [curve.length() for curve in curves]
    total_length = subpath.length()

    num_dots_per_curve = [int(num_dots_subpath * (length / total_length)) for length in curve_lengths]
    # Adjust the number of dots to ensure the total matches args.dots
    # i = 0
    # while sum(num_dots_per_curve) < num_dots_subpath:
    #     num_dots_per_curve[num_dots_per_curve.index(i)] += 1
    #     i+=1

    # print("dots_for_subpath", max(1, num_dots_subpath))

    # print(num_dots_per_curve)

    # print(curve_lengths)


    all_points = []
    # for subpath in subpaths:
    for i, curve in enumerate(subpath):
        # points = get_evenly_spaced_points_per_curve(curve, num_dots_per_curve[i])
        points = get_evenly_spaced_points_per_curve(curve, max(1, num_dots_per_curve[i]))
        all_points.extend(points)

    return all_points

# Function to convert svgpathtools path to matplotlib path
def convert_to_mpl_path(svg_path):
    vertices = []
    codes = []
    for segment in svg_path:
        if isinstance(segment, (Line, QuadraticBezier, CubicBezier)):
            vertices.append((segment.start.real, -segment.start.imag))
            codes.append(MplPath.MOVETO)
            for point in segment:
                vertices.append((point.real, -point.imag))
                codes.append(MplPath.LINETO)
        vertices.append((segment.end.real, -segment.end.imag))
        codes.append(MplPath.LINETO)
    return MplPath(vertices, codes)


# Utility function to get paths from SVG or text input
def get_paths(text=None, font=None):
    paths = []
    if text and font:
        # paths.append(get_svg_path_from_glyph(args.text, args.font))
        glyph_path = get_svg_path_from_glyph(text, font)
        paths = [parse_path(subpath.d()) for subpath in glyph_path.continuous_subpaths()]  # Separate disconnected pieces
    else:
        print("Please provide text with a font file.")
        exit()

    return paths
    
def remove_close_points_with_angle_updated(ordered_points, distance_threshold, angle_threshold, sample_attempts=2):
    """
    Removes points from the ordered list based on distance and angle thresholds by sampling.
    
    Parameters:
        ordered_points (list): A list of complex numbers representing points in order.
        distance_threshold (float): Minimum allowed distance between consecutive points.
        angle_threshold (float): Minimum allowed angle (in degrees) at a point.
        sample_attempts (int): Number of attempts to sample and test a point's validity.
    
    Returns:
        list: A filtered list of points.
    """
    def calculate_angle(p1, p2, p3):
        """Calculate the angle (in degrees) at p2 formed by points p1, p2, and p3."""
        v1 = np.array([p1.real - p2.real, p1.imag - p2.imag])
        v2 = np.array([p3.real - p2.real, p3.imag - p2.imag])
        # Normalize vectors
        if np.linalg.norm(v1) > 0:
            v1 = v1 / np.linalg.norm(v1)
        if np.linalg.norm(v2) > 0:
            v2 = v2 / np.linalg.norm(v2)
        # Calculate the angle in radians and convert to degrees
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return np.degrees(angle)

    filtered_points = ordered_points.copy()  # Start with the original points
    n = len(filtered_points)
    
    # Iterate through points and test for removal
    for _ in range(sample_attempts):
        for i in range(n):
            if n <= 3:  # Stop if only 3 points remain (minimum for a polygon)
                print("break")
                break
            
            # Pick a random point to test
            i += 1
            i %= n
            
            prev_index = (i - 1) % n
            next_index = (i + 1) % n
            
            prev_point = filtered_points[prev_index]
            curr_point = filtered_points[i]
            next_point = filtered_points[next_index]

            prev_dist = abs(curr_point - prev_point)
            next_dist = abs(curr_point - next_point)
            
            # Calculate distance and angle
            distance = min(prev_dist, next_dist)
            
            angle = calculate_angle(prev_point, curr_point, next_point)
            angle_threshold_variable = angle_threshold - (5 if distance < 35 else 0)
            
            # Check if point should be removed
            if (
                distance < distance_threshold
                and angle > angle_threshold_variable
            ):
                # Remove current point and restart validation
                filtered_points.pop(i)
                n -= 1

    return filtered_points

# Function to get dots from SVG or text input
def get_dots(text=None, font=None, total_dots=100, distance_threshold=20, angle_threshold=160):
    print(text)

    all_dots = []
    
    # Example usage with parsed arguments
    paths = get_paths(text, font)

    # Calculate the length of each path and the total length
    path_lengths = [path.length() for path in paths]
    total_length = sum(path_lengths)

    num_dots_per_path = [int(total_dots * (length / total_length)) for length in path_lengths]
    # Adjust the number of dots to ensure the total matches args.dots
    while sum(num_dots_per_path) < total_dots:
        num_dots_per_path[num_dots_per_path.index(max(num_dots_per_path))] += 1

    # Plot the dots and lines
    for path, num_dots in zip(paths, num_dots_per_path):
        dots = get_evenly_spaced_points_from_subpath(path, num_dots)

        if not abs(dots[-1] - dots[0]) < 1e-5:
            dots.append(dots[0])
            
        
        all_dots.append(dots)

    def scale_dots_to_max_y(all_dots, target_extent=100):
        all_y_coords = [dot.imag for sublist in all_dots for dot in sublist]
        y_max, y_min = max(all_y_coords), min(all_y_coords)
        scaling_factor = target_extent / (y_max - y_min) if y_max != y_min else 1
        return [[dot * scaling_factor for dot in sublist] for sublist in all_dots]
    
    all_dots = scale_dots_to_max_y(all_dots, 1000)
    
    for i, dots in enumerate(all_dots):
        
        if num_dots > 3:
        # if False:
            size_diff = len(all_dots[i])
            all_dots[i] = remove_close_points_with_angle_updated(
                dots, distance_threshold, 
                angle_threshold, 
                sample_attempts=10)
            size_diff -= len(all_dots[i])
            if size_diff > 0:
                print("removed", size_diff)
    
        j = 0
        while j < len(all_dots[i]):
            if abs(all_dots[i][j] - all_dots[i][(j + 1) % len(all_dots[i])]) < 1:
                all_dots[i].pop(j)
                # No need to increment j because we just removed an element at this position
            else:
                j += 1

    if len(all_dots) > 1:
        stitched_dots = stitch_dots(all_dots)
    else:
        stitched_dots = all_dots[0]
        if not abs(stitched_dots[-1]-stitched_dots[0]) < 1e-5:
            stitched_dots.append(stitched_dots[0])
    
    return stitched_dots

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert an SVG file or text into a dot-to-dot representation.")
    parser.add_argument("--svg_file", help="Path to the SVG file", default=None)
    parser.add_argument("--text", help="Text to convert to SVG", default=None)
    parser.add_argument("--font", help="Path to the font file (required if using text)", default=None)
    parser.add_argument("--distance_threshold", type=int, default=20, help="Distance threshold for reducing points")
    parser.add_argument("--angle_threshold", type=int, default=160, help="Angle factor for reducing points")

    args = parser.parse_args()

    fig, ax = plt.subplots()

    paths = get_paths(text=args.text, font=args.font, total_dots=args.dots)

    # Plot the underlying SVG
    for path in paths:
        mpl_path = convert_to_mpl_path(path)
        patch = patches.PathPatch(mpl_path, facecolor='none', edgecolor='gray', lw=1, alpha=0.9)
        ax.add_patch(patch)

    all_dots = get_dots(svg_file=args.svg_file, text=args.text, font=args.font, 
                        total_dots=args.dots, distance_threshold=args.distance_threshold,
                        angle_threshold = args.angle_threshold)

    for dots in all_dots:

        x_coords = [dot.real for dot in dots]
        y_coords = [-dot.imag for dot in dots]
        
        # Plotting the dots
        plt.scatter(x_coords, y_coords, color='blue')
        
        # Plotting lines between consecutive dots
        plt.plot(x_coords, y_coords, color='lightgreen', alpha=0.5)
        
        # Numbering the dots
        for i, (x, y) in enumerate(zip(x_coords, y_coords), start=1):
            plt.text(x, y, str(i), fontsize=8, ha='right', va='bottom', color='red')
        
    plt.gca().invert_yaxis()
    plt.title("Dot-to-Dot Representation with Lines, Numbered Dots, and Underlying SVG")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.axis('equal')
    plt.show()
