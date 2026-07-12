"""
ConvJson.py
===========

Converts LabelMe-style JSON annotation files into binary segmentation
masks (PNG images).

Each JSON file is expected to follow the LabelMe annotation schema, i.e.
it contains:
    - "imageHeight" / "imageWidth": the dimensions of the original image.
    - "shapes": a list of annotated shapes, each with a "shape_type"
      (e.g. "line", "linestrip", "polygon") and a list of "points"
      describing the shape's coordinates.

For every JSON file found in `jsons_dir`, this script:
    1. Reads the annotation.
    2. Creates a black (all-zero) mask the same size as the original image.
    3. Draws every line/linestrip/polygon annotation onto the mask in white.
    4. Saves the resulting mask as a PNG with the same base name as the
       JSON file, inside `masks_output_dir`.

This is typically used to turn manual annotations of SAR (Synthetic
Aperture Radar) internal-wave signatures into ground-truth masks for
training a segmentation model.
"""

import os
import json
import numpy as np
import cv2

# --------------------------------------------------------------------------
# Directory configuration
# --------------------------------------------------------------------------
# images_dir:        folder containing the original SAR images (PNG).
#                     Note: currently unused inside ConvertJson, but kept as
#                     a parameter in case future versions need to read the
#                     image itself (e.g. to verify dimensions).
# jsons_dir:          folder containing the LabelMe JSON annotation files.
# masks_output_dir:   folder where the generated binary masks will be saved.
images_dir = "PATH"
jsons_dir = "PATH"
masks_output_dir = "PATH"

# Make sure the output directory exists before we try to write masks into it.
# exist_ok=True prevents an error if the folder is already there.
os.makedirs(masks_output_dir, exist_ok=True)


def ConvertJson(images_dir, jsons_dir, masks_output_dir):
    """
    Convert every LabelMe JSON annotation in `jsons_dir` into a binary
    segmentation mask and save it to `masks_output_dir`.

    Parameters
    ----------
    images_dir : str
        Path to the folder containing the original images. Not directly
        read in this function, but passed in for context / future use.
    jsons_dir : str
        Path to the folder containing the ".json" annotation files
        (LabelMe format) to be converted.
    masks_output_dir : str
        Path to the folder where the generated ".png" mask files will
        be written. One mask is created per JSON file, sharing the same
        base filename.

    Returns
    -------
    None
        The function writes mask images to disk as a side effect and
        prints "Done!" when finished; it does not return a value.

    Notes
    -----
    - Only shapes of type "line", "linestrip", or "polygon" are drawn.
      Other shape types (e.g. "point", "circle", "rectangle") are ignored.
    - Shapes are drawn as open polylines (isClosed=False) in white (255)
      with a thickness of 5 pixels, regardless of whether the original
      shape was a closed polygon. This effectively produces a mask that
      highlights the *outline/path* of each annotation rather than a
      filled region.
    """
    # Gather all annotation files in the folder that end with ".json".
    json_files = [f for f in os.listdir(jsons_dir) if f.endswith('.json')]

    # Process each annotation file individually.
    for json_file in json_files:
        json_path = os.path.join(jsons_dir, json_file)

        # Load the JSON annotation as a Python dictionary.
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get the original image dimensions so the mask matches its size.
        height = data['imageHeight']
        width = data['imageWidth']

        # Create an empty (all-black) single-channel mask the same size
        # as the original image. uint8 keeps memory usage low and matches
        # the expected format for saving as a standard grayscale PNG.
        mask = np.zeros((height, width), dtype=np.uint8)

        # Iterate over every annotated shape in this JSON file.
        for shape in data['shapes']:
            # Only draw shapes that represent lines/curves/polygons;
            # ignore other annotation types (points, rectangles, etc.).
            if shape['shape_type'] in ['line', 'linestrip', 'polygon']:
                # Convert the list of [x, y] points into a NumPy array of
                # integer coordinates, as required by cv2.polylines.
                points = np.array(shape['points'], dtype=np.int32)

                # Draw the annotation onto the mask as a white (255) open
                # polyline with a 5-pixel thickness. isClosed=False means
                # the first and last points are NOT connected, so even
                # polygon annotations are rendered as open paths here.
                cv2.polylines(mask, [points], isClosed=False, color=255, thickness=5)

        # Build the output filename: same base name as the JSON file,
        # but with a ".png" extension instead of ".json".
        mask_name = os.path.splitext(json_file)[0] + ".png"

        # Write the mask image to disk in the output directory.
        cv2.imwrite(os.path.join(masks_output_dir, mask_name), mask)

    # Simple completion message once all JSON files have been processed.
    print("Done!")


# Script entry point: run the conversion using the directories configured
# at the top of the file.
ConvertJson(images_dir, jsons_dir, masks_output_dir)

#teste