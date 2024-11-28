# create_dot_to_dot_layout.py
import argparse
import matplotlib.pyplot as plt
from dot_to_dot import get_dots
from matplotlib.font_manager import FontProperties
import os

label_font = FontProperties(fname='fonts/ZCOOLKuaiLe-Regular.ttf')

show_characters = True
draw_lines = False
char_fontsize = 7
dot_size = 0.5
number_label_fontsize = 2.5
number_label_offset_distance = 10  # Fixed offset distance
overlap_thresh = 15 # For fixing overlapping labels

if __name__ == "__main__":
  # Command-line argument parsing
  parser = argparse.ArgumentParser(description="Convert an SVG file or text into a dot-to-dot representation.")
  parser.add_argument("--svg_file", help="Path to the SVG file", default=None)
  parser.add_argument("--text", help="Text to convert to SVG", default=None)
  parser.add_argument("--font", help="Path to the font file (required if using text)", default=None)
  parser.add_argument("--dots", type=int, default=100, help="Number of dots to generate per path")
  parser.add_argument("--output", help="Path to save the output PDF", default=None)
  parser.add_argument("--dpi", type=int, default=300, help="DPI for the output PDF to control scaling")
  parser.add_argument("--distance_threshold", type=int, default=20, help="Distance threshold for reducing points")
  parser.add_argument("--angle_threshold", type=int, default=160, help="Angle factor for reducing points")

  args = parser.parse_args()

  # Set default output name if not provided
  if args.output is None:
    # Extract the font name without the path and the ".ttf" extension, if font is provided
    fontname = os.path.splitext(os.path.basename(args.font))[0] if args.font else "no_font"

    # Use provided arguments to construct the output filename
    args.output = f"output/text{args.text if args.text else 'no_text'}_{fontname}_dots{args.dots}_dist{args.distance_threshold}_angle{args.angle_threshold}.pdf"


  # Set up the figure for printing layout
  dots_word = []
  all_x_coords = []
  all_y_coords = []
  label_data = []
  char_positions = []  # To store the positions of each character for plotting underneath

  for i, char in enumerate(args.text):
    dots_glyph = get_dots(svg_file=args.svg_file, text=char, 
                          font=args.font, total_dots=args.dots, 
                          distance_threshold=args.distance_threshold,
                          angle_threshold = args.angle_threshold)

    for dots_subpath in dots_glyph:
      if not draw_lines: # If we don't want to preview the drawn dot-to-dot
        dots_subpath.pop() # Remove endpoint since it's same as startpoint
      for dot in dots_subpath:
        dot = complex(dot.real, -dot.imag)

    if i > 0:
      # Extract y-coordinates of previous glyph
      flat_dots_imag = [dot.imag for dots_subpath in dots_word[i - 1] for dot in dots_subpath]
      imag_distance = max(flat_dots_imag) - min(flat_dots_imag)  # Calculate vertical distance

      for dots_subpath in dots_glyph:
        for j, dot in enumerate(dots_subpath):
          # Apply vertical offset for each consecutive glyph
          dots_subpath[j] = complex(dot.real, dot.imag - (imag_distance * 1.1) * i)
    
    dots_word.append(dots_glyph)

    # Calculate the approximate center position of the current glyph for character plotting
    flat_dots_real = [dot.real for dots_subpath in dots_glyph for dot in dots_subpath]
    flat_dots_imag = [dot.imag for dots_subpath in dots_glyph for dot in dots_subpath]
    char_x = max(flat_dots_real) # np.mean(flat_dots_real)
    char_y = min(flat_dots_imag) - 50  # Increase positioning the character below the dots to make it clearly visible
    char_positions.append((char_x, char_y, char))


  # Create a single figure for plotting
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.set_aspect('equal')  # Set aspect ratio to be equal to prevent squishing

  # Prepare data for plotting
  for dots_glyph in dots_word:
    subshape_counter = 1  # Counter for subshape differentiation
    for dots in dots_glyph:
      if not dots:  # Skip empty subpaths
        continue

      # Extract x and y coordinates from the complex points
      x_coords = [dot.real for dot in dots]
      y_coords = [dot.imag for dot in dots]  # Flip y-axis for correct orientation
          
      # Debug: draw lines for the dot-to-dot
      if draw_lines:
        ax.plot(x_coords, y_coords, color='#a0c0a0', zorder=2)  # Reduced the size of the points

      # Add coordinates to the master list for plotting later
      all_x_coords.extend(x_coords)
      all_y_coords.extend(y_coords)

      # Prepare label data with subshape indicator
      for i in range(len(dots)):
        curr_point = dots[i]
        # Default positions for the label
        offset_point_right = curr_point + complex(number_label_offset_distance, 0)
        offset_point_left = curr_point - complex(number_label_offset_distance, 0)
        offset_point_top = curr_point + complex(0, number_label_offset_distance)
        offset_point_bottom = curr_point - complex(0, number_label_offset_distance)

        # Check for overlap with existing labels
        overlaps = {
            'right': any(abs(offset_point_right - label['position']) < overlap_thresh*1.25 for label in label_data),
            'left': any(abs(offset_point_left - label['position']) < overlap_thresh*1.25 for label in label_data),
            'top': any(abs(offset_point_top - label['position']) < overlap_thresh for label in label_data),
            'bottom': any(abs(offset_point_bottom - label['position']) < overlap_thresh for label in label_data)
        }

        # Determine the best position for the label based on available space
        if not overlaps['right']:
          offset_point = offset_point_right
          ha = 'left'
          va = 'center'
        elif not overlaps['top']:
          offset_point = offset_point_top
          ha = 'center'
          va = 'bottom'
        elif not overlaps['bottom']:
          offset_point = offset_point_bottom
          ha = 'center'
          va = 'top'
        elif not overlaps['left']:
          offset_point = offset_point_left
          ha = 'right'
          va = 'center'
        else:
          # Default to right if all positions are overlapping
          offset_point = offset_point_right
          ha = 'left'
          va = 'center'

        # Store the new label position and data with the original label text
        label_data.append({'position': offset_point, 'text': str(i + 1), 'ha': ha, 'va': va})

      # Plot the subshape index inside each dot
      for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.text(x, y, f"{subshape_counter}", fontsize=2, ha='center', va='center', 
                color='#ffffff', fontproperties=label_font, zorder=3)

      subshape_counter += 1  # Increment the subshape counter for the next subshape

  # Plot all points at once
  # if not draw_lines:
  ax.scatter(all_x_coords, all_y_coords, color='#808080', s=dot_size, zorder=2)  # Reduced the size of the points

  # Plot all labels at once
  for label in label_data:
    ax.text(label['position'].real, label['position'].imag, label['text'], 
            fontsize=number_label_fontsize, ha=label['ha'], va=label['va'], color='#a0a0a0', 
            fontproperties=label_font, zorder=1)  # Reduced the font size

  if show_characters:
    # Plot all characters underneath the dots
    for (char_x, char_y, char) in char_positions:
      ax.text(char_x, char_y, char, fontsize=char_fontsize, ha='center', va='center',
              color='#808080', fontproperties=FontProperties(fname=args.font), zorder=1)

  # Remove axes, ticks, and labels
  ax.set_axis_off()

  # Set limits to fit around the data, adjust to ensure characters are also visible
  ax.set_xlim(min(all_x_coords) - 10, max(all_x_coords) + 10)
  ax.set_ylim(min(all_y_coords) - 70, max(all_y_coords) + 10)

  # Save the plot to a PDF file
  plt.tight_layout()
  plt.savefig(args.output, format='pdf', dpi=args.dpi, bbox_inches='tight')
  plt.close(fig)
  # plt.show()
