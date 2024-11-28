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

def solve_tsp_ortools(points):
    """Solves TSP using OR-Tools."""
    # Create the distance matrix
    distance_matrix = squareform(pdist(points))

    # OR-Tools setup
    tsp_size = len(points)
    manager = pywrapcp.RoutingIndexManager(tsp_size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Solve TSP
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        # Extract the route
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route
    else:
        raise ValueError("No solution found!")


def solve_tsp_optimized(points):
    """Solve TSP using OR-Tools with advanced optimization."""
    distance_matrix = squareform(pdist(points))
    tsp_size = len(points)
    manager = pywrapcp.RoutingIndexManager(tsp_size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Define search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 1  # Increase for better solutions
    search_parameters.log_search = True  # Enable logging to see progress

    # Solve the TSP
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        # Extract the route
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route
    else:
        raise ValueError("No solution found!")
    
def remove_close_points(ordered_points, threshold_factor, preserve_corners=True):
    """Removes points that are too close, preserving corner points."""
    # Calculate the total extent (height) of the points
    ys = [p.imag for p in ordered_points]
    # extent = max(ys) - min(ys)
    # threshold = threshold_factor * extent
    threshold = threshold_factor

    # Identify corner points (extreme x and y values)
    if preserve_corners:
        xs = [p.real for p in ordered_points]
        corners = set([
            ordered_points[np.argmin(xs)],
            ordered_points[np.argmax(xs)],
            ordered_points[np.argmin(ys)],
            ordered_points[np.argmax(ys)],
        ])
    else:
        corners = set()

    # Remove points that are too close
    filtered_points = [ordered_points[0]]  # Always keep the first point
    for i in range(1, len(ordered_points)):
        prev_point = filtered_points[-1]
        curr_point = ordered_points[i]

        # Calculate the distance to the previous point
        distance = abs(curr_point - prev_point)

        # Keep the point if it's far enough or if it's a corner
        if distance > threshold or curr_point in corners:
            filtered_points.append(curr_point)

    # return np.array(filtered_points)
    return filtered_points

import numpy as np

def remove_close_points_with_angle(ordered_points, distance_threshold, angle_threshold, preserve_corners=True):
    """
    Removes points that are too close or have shallow angles, preserving corner points if specified.

    Parameters:
        ordered_points (list of complex): Ordered points representing a closed curve.
        distance_threshold (float): Minimum distance between points.
        angle_threshold (float): Minimum angle (in degrees) to keep a point.
        preserve_corners (bool): Whether to preserve extreme corner points.

    Returns:
        list of complex: Filtered list of points.
    """
    def calculate_angle(p1, p2, p3):
        """Calculate the angle (in degrees) at p2 formed by points p1, p2, and p3."""
        v1 = np.array([p1.real - p2.real, p1.imag - p2.imag])
        v2 = np.array([p3.real - p2.real, p3.imag - p2.imag])
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        # Calculate the angle in radians and convert to degrees
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return np.degrees(angle)

    # Identify corner points (extreme x and y values)
    if preserve_corners:
        xs = [p.real for p in ordered_points]
        ys = [p.imag for p in ordered_points]
        corners = set([
            ordered_points[np.argmin(xs)],
            ordered_points[np.argmax(xs)],
            ordered_points[np.argmin(ys)],
            ordered_points[np.argmax(ys)],
        ])
    else:
        corners = set()

    # Remove points based on distance and angle thresholds
    filtered_points = []
    n = len(ordered_points)
    
    for i in range(n):
        prev_point = ordered_points[i - 1]  # Previous point (wraps around)
        curr_point = ordered_points[i]     # Current point
        next_point = ordered_points[(i + 1) % n]  # Next point (wraps around)

        # Calculate distance from the previous point
        distance = abs(curr_point - prev_point)

        # Calculate the angle at the current point
        angle = calculate_angle(prev_point, curr_point, next_point)

        # Keep the point if it's far enough, has a sharp enough angle, or is a corner
        if (
            distance > distance_threshold or
            angle < angle_threshold or
            curr_point in corners
        ):
            filtered_points.append(curr_point)

    return filtered_points


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

# Function to get evenly spaced dots along a path
def get_evenly_spaced_dots(path, num_dots):
    total_length = path.length()
    distances = np.linspace(0, total_length, num=num_dots)
    points = [path.point(distance / total_length) for distance in distances]
    return points

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
def get_paths(svg_file=None, text=None, font=None, total_dots=100):
    paths = []
    if text and font:
        # paths.append(get_svg_path_from_glyph(args.text, args.font))
        glyph_path = get_svg_path_from_glyph(text, font)
        paths = [parse_path(subpath.d()) for subpath in glyph_path.continuous_subpaths()]  # Separate disconnected pieces
    elif svg_file:
        # Load and process SVG file
        paths, attributes = svg2paths(svg_file)
    else:
        print("Please provide either an SVG file or text with a font file.")
        exit()

    return paths

# Function to get dots from SVG or text input
def get_dots(svg_file=None, text=None, font=None, total_dots=100, distance_threshold=20, angle_threshold=160):

    all_dots = []
    
    # Example usage with parsed arguments
    paths = get_paths(svg_file, text, font, total_dots)

    # Calculate the length of each path and the total length
    path_lengths = [path.length() for path in paths]
    total_length = sum(path_lengths)

    num_dots_per_path = [int(total_dots * (length / total_length)) for length in path_lengths]
    # Adjust the number of dots to ensure the total matches args.dots
    while sum(num_dots_per_path) < total_dots:
        num_dots_per_path[num_dots_per_path.index(max(num_dots_per_path))] += 1

    # fig, ax = plt.subplots()

    # # Plot the underlying SVG
    # for path in paths:
    #     mpl_path = convert_to_mpl_path(path)
    #     patch = patches.PathPatch(mpl_path, facecolor='none', edgecolor='gray', lw=1, alpha=0.9)
    #     ax.add_patch(patch)

    # Plot the dots and lines
    # for path in paths:
    for path, num_dots in zip(paths, num_dots_per_path):
        dots = get_evenly_spaced_dots(path, num_dots)


        # Solve TSP with OR-Tools
        # if True:
        if False:
            # Convert to 2D array for TSP solver (scipy needs explicit coordinates)
            points_2d = np.array([[p.real, p.imag] for p in dots])

            # tsp_order = solve_tsp_ortools(points_2d)
            tsp_order = solve_tsp_optimized(points_2d)
            ordered_points_2d = points_2d[tsp_order]
            dots = []
            dots = np.array([complex(p[0], p[1]) for p in ordered_points_2d])


        # Remove points if path is too dense
        if True and num_dots > 5:
        # if False:
            size_diff = len(dots)
            # final_points = remove_close_points(dots, threshold_factor)
            final_points = remove_close_points_with_angle(dots, distance_threshold, angle_threshold)
            dots = final_points
            size_diff -= len(dots)
            if size_diff > 0:
                print(size_diff)
        
        all_dots.append(dots)
    
    return all_dots


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

    paths = get_paths(svg_file=args.svg_file, text=args.text, font=args.font, total_dots=args.dots)

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
