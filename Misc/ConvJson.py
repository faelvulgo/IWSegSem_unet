import os
import json
import numpy as np
import cv2

images_dir = "PATH"
jsons_dir = "PATH"
masks_output_dir = "PATH"

# Verify if the directory exists
os.makedirs(masks_output_dir, exist_ok=True)


def ConvertJson(images_dir, jsons_dir, masks_output_dir):
    """
    Converts LabelMe JSON annotation files into binary segmentation
    masks (PNG images).

    This script:
    1. Reads the annotation.
    2. Creates a black (all-zero) mask the same size as the original image.
    3. Draws every line/linestrip/polygon annotation onto the mask in white.
    4. Saves the resulting mask as a PNG with the same base name as the
       JSON file, inside `masks_output_dir`.
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

        # Create an empty (all-black) single-channel mask the same size as the original image.
        mask = np.zeros((height, width), dtype=np.uint8)

        # Iterate over every annotated shape in this JSON file.
        for shape in data['shapes']:
            # Only draw shapes that represent lines/curves/polygons;
            if shape['shape_type'] in ['line', 'linestrip', 'polygon']:
                # Convert the list of [x, y] points into a NumPy array of integer coordinates, as required by cv2.polylines.
                points = np.array(shape['points'], dtype=np.int32)

                # Draw the annotation onto the mask as a white (255) open polyline with a 5-pixel thickness.
                cv2.polylines(mask, [points], isClosed=False, color=255, thickness=5)

        # Build the output filename: same base name as the JSON file, but a PNG.
        mask_name = os.path.splitext(json_file)[0] + ".png"

        # Write the mask image to disk in the output directory.
        cv2.imwrite(os.path.join(masks_output_dir, mask_name), mask)

    print("Done!")

ConvertJson(images_dir, jsons_dir, masks_output_dir)
