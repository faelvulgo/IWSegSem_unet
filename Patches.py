"""
Patches.py
==========

Splits large SAR (Synthetic Aperture Radar) images and their binary
segmentation masks into smaller, fixed-size, non-overlapping patches
suitable for training a segmentation model.

For each large image/mask pair:
    - The mask is scanned with a sliding window of size
      `patch_size x patch_size` (no overlap, stride = patch_size).
    - Any patch that contains at least one non-zero (wave) pixel in the
      mask is always saved ("wave" patch).
    - Patches that are pure background (mask is entirely zero, i.e. pure
      ocean with no internal-wave signature) are only saved with a small
      probability (`chance_salvar_oceano_puro`), to avoid flooding the
      dataset with uninformative background patches while still keeping
      a representative sample of "no wave" examples.

This script is meant to be run directly (not imported) and produces
patch images/masks on disk plus a short summary printed to the console.
"""

import os
import cv2
import numpy as np
import random

# Fix the random seed so that which "pure ocean" patches get kept is
# reproducible across runs.
random.seed(42)

# --------------------------------------------------------------------------
# Input directories: large (full-size) SAR images and their masks.
# Mask filenames are expected to match the image's base name with a
# ".png" extension (see `mask_path` construction below).
# --------------------------------------------------------------------------
imagens_grandes_dir = "PATH"
mascaras_grandes_dir = "PATH"

# --------------------------------------------------------------------------
# Output directories: where generated patches (images and masks) will be
# written. Each patch image and its corresponding patch mask share the
# same filename, just in different folders.
# --------------------------------------------------------------------------
imagens_patches_dir = "PATH"
mascaras_patches_dir = "PATH"

# Make sure the output directories exist before writing any files.
os.makedirs(imagens_patches_dir, exist_ok=True)
os.makedirs(mascaras_patches_dir, exist_ok=True)

# Size (in pixels) of each square patch, e.g. 256 -> 256x256 patches.
patch_size = 256

# Probability of keeping a "pure ocean" patch (mask entirely zero).
# Since background patches vastly outnumber patches containing waves,
# only ~10% of them are kept to keep the dataset reasonably balanced.
chance_salvar_oceano_puro = 0.10

# List every image file in the input folder with a supported extension.
image_files = [f for f in os.listdir(imagens_grandes_dir) if f.endswith(('.tif', '.tiff', '.jpg', '.png'))]

# Running counters used for the final summary report.
ondas_count = 0    # number of patches saved that contain wave signatures
oceano_count = 0   # number of pure-background ("ocean") patches saved

# Process each large image (and its corresponding mask) one at a time.
for file_name in image_files:
    img_path = os.path.join(imagens_grandes_dir, file_name)
    name_base = os.path.splitext(file_name)[0]
    # The mask is assumed to be a PNG with the same base name as the image.
    mask_path = os.path.join(mascaras_grandes_dir, name_base + ".png")

    # Skip this image if there is no matching mask file, warning the user.
    if not os.path.exists(mask_path):
        print(f"Aviso: Máscara não encontrada para {file_name}. Pulando...")
        continue

    # Load the full-size image (color) and its mask (grayscale).
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    # Use the mask's shape to determine how many patches fit in the image.
    height, width = mask.shape

    # Slide a non-overlapping window of size patch_size across the image,
    # stepping by patch_size in both dimensions (range(...) with a step of
    # patch_size ensures no overlap between neighboring patches). Any
    # remainder smaller than patch_size at the edges is simply discarded.
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):

            # Crop out the current patch from both the mask and the image
            # using the same (y, x) window.
            patch_mask = mask[y:y+patch_size, x:x+patch_size]
            patch_img = img[y:y+patch_size, x:x+patch_size]
            patch_name = f"{name_base}_patch_y{y}_x{x}.png"

            # If the mask patch has any non-zero pixel, it contains at
            # least part of an internal-wave annotation -> always save it.
            if np.sum(patch_mask) > 0:
                cv2.imwrite(os.path.join(imagens_patches_dir, patch_name), patch_img)
                cv2.imwrite(os.path.join(mascaras_patches_dir, patch_name), patch_mask)
                ondas_count += 1

            # Otherwise, this is a "pure ocean" patch (mask is all zeros).
            # We only keep a random subset of these to avoid an overwhelming
            # majority of background-only examples in the final dataset.
            else:
                # Draw a number between 0 and 1. If it's less than 0.10
                # (10%), save the patch; otherwise discard it.
                if random.random() < chance_salvar_oceano_puro:
                    cv2.imwrite(os.path.join(imagens_patches_dir, patch_name), patch_img)
                    cv2.imwrite(os.path.join(mascaras_patches_dir, patch_name), patch_mask)
                    oceano_count += 1

# Final summary of how many patches were generated, split by category.
print("---Processing Summary---")
print(f"Patches with internal waves: {ondas_count}")
print(f"Patches of pure ocean: {oceano_count}")
print(f"Total of patches for training: {ondas_count + oceano_count}")
