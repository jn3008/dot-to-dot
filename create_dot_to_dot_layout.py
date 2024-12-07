# create_dot_to_dot_layout.py
import argparse
import matplotlib.pyplot as plt
from dot_to_dot import get_dots, stitch_dots, calculate_angle
from matplotlib.font_manager import FontProperties
import os

label_font = FontProperties(fname='fonts/RocknRollOne-Regular.ttf')

char_fontsize = 40
dot_size = 16
number_label_fontsize = 8
number_label_offset_distance = dot_size * 1.75  # Offset distance for overlap calculations
overlap_thresh = 105  # For fixing overlapping labels


def check_overlap(x, y, existing_labels, existing_points):
  """
  Checks if the given point overlaps with any existing labels or points.
  """
  for label in existing_labels:
    if abs(x - label['position'].real) < number_label_offset_distance and abs(y - label['position'].imag) < number_label_offset_distance:
      return True
  for px, py in existing_points:
    if abs(x - px) < number_label_offset_distance and abs(y - py) < number_label_offset_distance:
      return True
  return False


if __name__ == "__main__":
  # Command-line argument parsing
  parser = argparse.ArgumentParser(
    description="Convert a text into a dot-to-dot representation."
  )
  parser.add_argument("--text", help="Text to convert to SVG (required)", required=True)
  parser.add_argument("--font", help="Path to the font file (required)", required=True)
  parser.add_argument("--dots", type=int, default=100, help="Number of dots to generate per character before dot reduction")
  parser.add_argument("--output", help="Path to save the output PDF", default=None)
  parser.add_argument("--distance_threshold", type=int, default=20, help="Distance threshold for reducing points")
  parser.add_argument("--angle_threshold", type=int, default=160, help="Angle threshold for reducing points")
  parser.add_argument("--no_merge", action="store_true", help="Merge overlapping shapes")
  parser.add_argument("--chars", action="store_true", help="Show hints")
  parser.add_argument("--lines", action="store_true", help="Draw lines between dots")
  parser.add_argument("--visual_label_offset", type=float, default=0.4, help="Multiplier to adjust the visual distance of labels from the dots")
  parser.add_argument("--reduce_straights", action="store_true", help="Reduce further consecutive points on straight lines")

  args = parser.parse_args()
  print(args.no_merge)

  # Set default output name if not provided
  if args.output is None:
    fontname = os.path.splitext(os.path.basename(args.font))[0] if args.font else "no_font"
    args.output = f"output/text{args.text if args.text else 'no_text'}"
    args.output += f"-{fontname}-dots{args.dots}"
    args.output += f"-dist{args.distance_threshold}-angle{args.angle_threshold}"
    args.output += f"{'-no_merge' if args.no_merge else ''}"
    args.output += f"{'-lines' if args.lines else ''}"
    args.output += f"{'-chars' if args.chars else ''}"
    args.output += f".pdf"

  # Set up the figure for printing layout
  word_dots = []
  word_bounds = []

  all_x_coords = []
  all_y_coords = []
  label_data = []
  char_positions = []  # To store the positions of each character for plotting underneath

  global_scale = 1

  for i, char in enumerate(args.text):
    glyph_dots = get_dots(
      text=char,
      font=args.font,
      total_dots=args.dots,
      merge=not args.no_merge,
      distance_threshold=args.distance_threshold,
      angle_threshold=args.angle_threshold,
      reduce_straights=args.reduce_straights,
    )

    dots_x = [dot.real for dot in glyph_dots]
    dots_y = [dot.imag for dot in glyph_dots]
    word_bounds.append({"min_x":min(dots_x), "max_x":max(dots_x), 
                        "min_y":min(dots_y), "max_y":max(dots_y)})

    # if i > 0:
    #   # Extract y-coordinates of previous glyph
    #   flat_dots_imag = [dot.imag for dot in word_dots[i - 1]]
    #   imag_distance = max(flat_dots_imag) - min(flat_dots_imag)  # Calculate vertical distance

    #   for j, dot in enumerate(glyph_dots):
    #     glyph_dots[j] = complex(dot.real, dot.imag - (imag_distance * 1.1) * i)

    word_dots.append(glyph_dots)

    # Calculate the approximate center position of the current glyph for character plotting
    flat_dots_real = [dot.real for dot in glyph_dots]
    flat_dots_imag = [dot.imag for dot in glyph_dots]
    char_x = max(flat_dots_real)
    char_y = min(flat_dots_imag) - 50  # Increase positioning the character below the dots to make it clearly visible
    char_positions.append((char_x, char_y, char))

    # Create a single figure for plotting
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_aspect("equal")  # Set aspect ratio to be equal to prevent squishing

  max_height = max([glyph_bounds["max_y"] - glyph_bounds["min_y"] for glyph_bounds in word_bounds])

  for i, glyph_dots in enumerate(word_dots):
    if i == 0:
      continue
    for j, dot in enumerate(glyph_dots):
      glyph_dots[j] -= complex(0, max_height * i)

  all_dots =[]

  if len(word_dots) > 1:
    all_dots = stitch_dots(word_dots)
  else:
    all_dots = word_dots[0]

  # Extract x and y coordinates from the complex points
  x_coords = [dot.real for dot in all_dots]
  y_coords = [dot.imag for dot in all_dots]  # Flip y-axis for correct orientation

  # Debug: draw lines for the dot-to-dot
  if args.lines:
    ax.plot(x_coords, y_coords, color="#a0c0a0", zorder=2)

  # Add coordinates to the master list for plotting later
  all_x_coords.extend(x_coords)
  all_y_coords.extend(y_coords)

  # Prepare label data with subshape indicator
  for i in range(len(all_dots)):
    curr_point = all_dots[i]
    # Default positions for the label
    offset_options = [
      complex(curr_point.real + number_label_offset_distance, curr_point.imag),  # Right
      complex(curr_point.real - number_label_offset_distance, curr_point.imag),  # Left
      complex(curr_point.real, curr_point.imag + number_label_offset_distance),  # Above
      complex(curr_point.real, curr_point.imag - number_label_offset_distance),  # Below
    ]

    # Choose the best offset that does not overlap with any existing points or labels
    offset_point = None
    for option in offset_options:
      if not check_overlap(option.real, option.imag, label_data, zip(all_x_coords, all_y_coords)):
        offset_point = option
        break

    # If still no suitable position is found, use an arbitrary distant offset
    if offset_point is None:
      offset_point = complex(curr_point.real + number_label_offset_distance * 3, curr_point.imag)

    # Determine alignment based on chosen offset
    ha = "center"
    va = "center"
    if offset_point == offset_options[0]:
      ha = "left"
    elif offset_point == offset_options[1]:
      ha = "right"
    elif offset_point == offset_options[2]:
      va = "bottom"
    elif offset_point == offset_options[3]:
      va = "top"

    # Store the new label position and data with the original label text
    # Adjust the final position of the label for visual purposes only
    visual_offset_point = complex(
      curr_point.real + (offset_point.real - curr_point.real) * args.visual_label_offset,
      curr_point.imag + (offset_point.imag - curr_point.imag) * args.visual_label_offset
    )
    # visual_offset_point = offset_point
    label_data.append({"position": offset_point, "visual_position": visual_offset_point, 
                       "text": str(i + 1), "ha": ha, "va": va})

    # circle = plt.Circle((curr_point.real, curr_point.imag), args.distance_threshold, 
    #                     edgecolor='#c0a0a0', facecolor='none', linewidth=1, zorder=-1)
    # ax.add_patch(circle)

  # Plot all points at once
  ax.scatter(all_x_coords, all_y_coords, color="#808080", s=dot_size, zorder=2)  # Reduced the size of the points

  # Plot all labels at once
  for label in label_data:
    # Adjust label position for visual purposes only
    visual_position = complex(
      label["visual_position"].real, label["visual_position"].imag
    )
    ax.text(
      visual_position.real, visual_position.imag,
      label["text"], 
      fontsize=number_label_fontsize, ha=label["ha"], va=label["va"],
      color="#a0a0a0", fontproperties=label_font, zorder=1,
    )  # Reduced the font size

  if args.chars:
    # Plot all characters underneath the dots
    for (char_x, char_y, char) in char_positions:
      ax.text(
        char_x, char_y, char, fontsize=char_fontsize, ha="center", va="center",
        color="#808080", fontproperties=FontProperties(fname=args.font), zorder=1,
      )

  # Remove axes, ticks, and labels
  ax.set_axis_off()

  # Set limits to fit around the data, adjust to ensure characters are also visible
  margin = 40
  ax.set_xlim(min(all_x_coords) - margin, max(all_x_coords) + margin)
  ax.set_ylim(min(all_y_coords) - margin, max(all_y_coords) + margin)

  # Save the plot to a PDF file
  plt.tight_layout()
  plt.savefig(args.output, format="pdf", dpi=300, bbox_inches="tight")
  plt.close(fig)
