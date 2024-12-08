# dot_to_dot.py
import math
import numpy as np
from matplotlib import pyplot as plt
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from svgpathtools import parse_path, Line, QuadraticBezier, CubicBezier, Path
import argparse
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon
import uharfbuzz as hb
import freetype

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
  # Rotates the list so that the element at split_index becomes the first element.
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
  if isinstance(glyph, str) and len(glyph) > 1:
    return get_svg_path_from_string(glyph, font_path)

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


def get_svg_path_from_string(string, font_path):
  # Load the font data
  with open(font_path, "rb") as font_file:
    font_data = font_file.read()

  # Create HarfBuzz Face and Font objects
  hb_face = hb.Face(font_data)
  hb_font = hb.Font(hb_face)

  # Create a HarfBuzz buffer and add the text
  buf = hb.Buffer()
  buf.add_str(string)
  buf.guess_segment_properties()

  # Shape the text
  hb.shape(hb_font, buf)

  # Get glyph information and positions
  infos = buf.glyph_infos
  positions = buf.glyph_positions

  # Load the font with fontTools for glyph extraction
  font = TTFont(font_path)
  glyph_set = font.getGlyphSet()

  # Combine glyph paths into a single SVG path
  combined_path = Path()
  x, y = 0, 0  # Starting pen position

  for info, pos in zip(infos, positions):
    glyph_name = font.getGlyphName(info.codepoint)
    glyph = glyph_set[glyph_name]

    # Create an SVG path for the glyph
    pen = SVGPathPen(glyph_set)
    glyph.draw(pen)
    glyph_path = parse_path(pen.getCommands())

    # Translate glyph path to its correct position
    # glyph_path = glyph_path.translated((x + pos.x_offset, y + pos.y_offset))
    glyph_path = glyph_path.translated(complex(x + pos.x_offset, y + pos.y_offset))

    # Update the pen position
    x += pos.x_advance
    y += pos.y_advance

    # Add the glyph path to the combined path
    combined_path += glyph_path

  return combined_path
def get_evenly_spaced_dots_from_path(path, path_num_dots):
  path_num_dots = max(1, path_num_dots)

  # Calculate the length of each path and the total length
  curve_lengths = [curve.length() for curve in path]
  total_length = path.length()

  num_dots_per_curve = [math.ceil(path_num_dots * (length / total_length)) for length in curve_lengths]

  dots = [segment.point(t) for (segment, num_dots) in zip(path, num_dots_per_curve)
             for t in [i * 1.0 / (num_dots) for i in range(num_dots)]]
  return dots

# Function to convert svgpathtools path to matplotlib path
def convert_to_mpl_path(svg_path):
  vertices = []
  codes = []
  for segment in svg_path:
    if isinstance(segment, (Line, QuadraticBezier, CubicBezier)):
      vertices.append((segment.start.real, segment.start.imag))
      codes.append(MplPath.MOVETO)
      for point in segment:
        vertices.append((point.real, point.imag))
        codes.append(MplPath.LINETO)
    vertices.append((segment.end.real, segment.end.imag))
    codes.append(MplPath.LINETO)
  return MplPath(vertices, codes)


# Utility function to get paths from SVG or text input
def get_paths(text=None, font=None):
  paths = []
  if text and font:
    glyph_path = get_svg_path_from_glyph(text, font)
    paths = [parse_path(subpath.d()) for subpath in glyph_path.continuous_subpaths()]  # Separate disconnected pieces
  else:
    print("Please provide text with a font file.")
    exit()

  return paths

# Convert a path to a polygon for overlap detection
def path_to_polygon(path, path_num_dots):
  dots = get_evenly_spaced_dots_from_path(path, path_num_dots)
  return Polygon([(p.real, p.imag) for p in dots])

# Merge overlapping shapes
def merge_overlapping_shapes(paths_with_counts):
  polygons = [path_to_polygon(path, dot_count) for (path, dot_count) in paths_with_counts]
  merged_shapes = []

  for poly in polygons:
    added = False
    for i, merged in enumerate(merged_shapes):
      if merged.intersects(poly) and not merged.contains(poly) and not poly.contains(merged):
        merged_shapes[i] = merged.union(poly)
        added = True
        break
    if not added:
      merged_shapes.append(poly)

  return merged_shapes



# Alternative to get_paths: parses and merges overlapping paths while preserving subpaths
def merge_paths(paths_with_counts):
  
  # Merge overlapping shapes while preserving small components
  merged_shapes = merge_overlapping_shapes(paths_with_counts)

  all_dots = []
  for polygon in merged_shapes:
    all_dots.append([complex(v[0], v[1]) for v in list(polygon.exterior.coords)])

  return all_dots

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
    
def remove_close_dots_with_angle_updated(ordered_dots, min_distance_threshold_max, max_angle_threshold_min, sample_attempts=2, reduce_straights=False):
  if (sample_attempts < 1):
    return

  filtered_dots = ordered_dots.copy()  # Start with the original dots
  n = len(filtered_dots)  

  strictest_adjustment_ratio_angle = 0.1; 
  strictest_adjustment_ratio_distance = 0.7; 

  # Iterate through dots and test for removal, starting with stricter thresholds
  for j in range(sample_attempts):
    if (sample_attempts == 1):
      distance_threshold = min_distance_threshold_max
      angle_threshold = max_angle_threshold_min 
    else:
      adjustment_ratio_distance = (strictest_adjustment_ratio_distance + 
        (1 - strictest_adjustment_ratio_distance) * j / (sample_attempts - 1))
      distance_threshold = min_distance_threshold_max * adjustment_ratio_distance

      adjustment_ratio_angle = (strictest_adjustment_ratio_angle + 
        (1 - strictest_adjustment_ratio_angle) * j / (sample_attempts - 1))
      angle_threshold = 180 - adjustment_ratio_angle * (180 - max_angle_threshold_min)
    for i in range(n):
      if n <= 3:  # Stop if only 3 dots remain (minimum for a polygon)
        print("break")
        break
      
      # Pick a random dots to test
      i += 1
      i %= n
      
      prev_index = (i - 1) % n
      next_index = (i + 1) % n
      
      prev_dot = filtered_dots[prev_index]
      curr_dot = filtered_dots[i]
      next_dot = filtered_dots[next_index]

      prev_dist = abs(curr_dot - prev_dot)
      next_dist = abs(curr_dot - next_dot)
      
      # Calculate distance and angle
      angle = calculate_angle(prev_dot, curr_dot, next_dot)

      if j == 0 and sample_attempts > 1: # on the first pass take average of angles
      # if j < sample_attempts-1: # on the first pass take average of angles
        prev_angle = calculate_angle(filtered_dots[(i - 2) % n], prev_dot, curr_dot)
        next_angle = calculate_angle(curr_dot, next_dot, filtered_dots[(i + 2) % n])
        angle += prev_angle + next_angle
        angle /= 3
      
      # Points that are further away and are on a very straight line can be removed
      max_distance = max(prev_dist, next_dist)
      if reduce_straights:
        min_distance = min(prev_dist, next_dist)
        # distance = max_distance if angle < 176 and j > 0 else (min_distance*0.7+max_distance*0.3)
        distance = max_distance if angle < 177 and j < sample_attempts - 1 else (min_distance*0.7+max_distance*0.3)
      else:
        distance = max_distance

      angle_threshold_variable = angle_threshold - (10 if distance < min_distance_threshold_max*0.3 else 0)

      # Check if point should be removed
      if (
        distance < distance_threshold
        and angle > angle_threshold_variable
      ):
        # Remove current point and restart validation
        filtered_dots.pop(i)
        n -= 1

  return filtered_dots

def remove_consecutive_duplicates(point_list):
  j = 0
  while j < len(point_list):
    if abs(point_list[j] - point_list[(j + 1) % len(point_list)]) < 1:
      point_list.pop(j)
      # No need to increment j because we just removed an element at this position
    else:
      j += 1


# Function to get dots from SVG or text input
def get_dots(text=None, font=None, total_dots=100, distance_threshold=20, angle_threshold=160, merge=True, reduce_straights=False):
  print(text)

  all_dots = []
  
  paths = get_paths(text, font)

  # Calculate the length of each path and the total length
  path_lengths = [path.length() for path in paths]
  total_length = sum(path_lengths)

  num_dots_per_path = [math.ceil(total_dots * (length / total_length)) for length in path_lengths]
  # Adjust the number of dots to ensure the total matches args.dots
  while sum(num_dots_per_path) < total_dots:
      num_dots_per_path[num_dots_per_path.index(max(num_dots_per_path))] += 1

  
  # Example usage with parsed arguments
  if merge:
    all_dots = merge_paths(zip(paths, num_dots_per_path))
  else:

    # Plot the dots and lines
    for path, num_dots in zip(paths, num_dots_per_path):
      dots = get_evenly_spaced_dots_from_path(path, num_dots)

      all_dots.append(dots)

  for i, dots in enumerate(all_dots):
    if not abs(dots[-1] - dots[0]) < 1e-5:
      all_dots[i].append(dots[0])
    print(len(all_dots[i]))
  
  for i, dots in enumerate(all_dots):

    # Remove any consecutive duplicate dots
    remove_consecutive_duplicates(all_dots[i])

    if len(all_dots[i]) > 3:
    # if False:
      size_diff = len(all_dots[i])
      all_dots[i] = remove_close_dots_with_angle_updated(
        dots, distance_threshold,
        angle_threshold,
        sample_attempts=5,
        reduce_straights=reduce_straights)
      size_diff -= len(all_dots[i])
      if size_diff > 0:
        print("removed", size_diff)

    
  stitched_dots = stitch_dots(all_dots)
  
  print("total_dots", len(stitched_dots))
  
  return stitched_dots
