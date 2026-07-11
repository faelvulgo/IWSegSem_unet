"""
SeparateData.py
================

Splits an existing folder of image/mask patches (previously generated,
e.g. by Patches.py) into training, validation, and test subsets by
*moving* files into new folders.

The split is:
    - 80% of patches remain in the original training folders
      (src_waves / src_masks).
    - 10% are moved into validation folders (ValWaves / ValMasks).
    - 10% are moved into test folders (TestWaves / TestMasks).

The file list is shuffled once (with a fixed random seed for
reproducibility) before slicing, so the validation and test sets are a
random sample of the full patch collection rather than, e.g., always the
first N files.

Note: this script assumes every image patch in `src_waves` has a
correspondingly named mask file in `src_masks`, and moves both files
together to keep image/mask pairs aligned across the train/val/test
splits.
"""

import os
import random
import shutil

# --------------------------------------------------------------------------
# 1. Paths to the current patch folders (source of the split).
# --------------------------------------------------------------------------
data_dir = "PATH"
src_waves = os.path.join(data_dir, 'Waves')
src_masks = os.path.join(data_dir, 'Masks')

# --------------------------------------------------------------------------
# 2. Paths to the destination folders (Validation and Test).
#    Files that get moved out of src_waves/src_masks land here.
# --------------------------------------------------------------------------
val_waves_dir = os.path.join(data_dir, 'ValWaves')
val_masks_dir = os.path.join(data_dir, 'ValMasks')
test_waves_dir = os.path.join(data_dir, 'TestWaves')
test_masks_dir = os.path.join(data_dir, 'TestMasks')

# Ensure the destination folders exist before moving any files into them.
os.makedirs(val_waves_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)
os.makedirs(test_waves_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

# --------------------------------------------------------------------------
# 3. List and shuffle the files (fixed seed guarantees the same random
#    split is produced every time this script is run).
# --------------------------------------------------------------------------
random.seed(42)
all_files = sorted(os.listdir(src_waves))  # Grab the names of all patch files
random.shuffle(all_files)                  # Shuffle in place using the fixed seed

# --------------------------------------------------------------------------
# 4. Compute how many files go to validation and test (10% each).
# --------------------------------------------------------------------------
total_files = len(all_files)
num_val = int(total_files * 0.10)   # ~10% of files for validation
num_test = int(total_files * 0.10)  # ~10% of files for test

# Split the shuffled filename list into validation and test subsets.
val_files = all_files[:num_val]
test_files = all_files[num_val : num_val + num_test]
# The remaining ~80% of filenames are left untouched in src_waves/src_masks,
# implicitly forming the training set.

# --------------------------------------------------------------------------
# 5. Move the validation files (both image and corresponding mask).
# --------------------------------------------------------------------------
for file_name in val_files:
    # Move the image patch.
    shutil.move(os.path.join(src_waves, file_name), os.path.join(val_waves_dir, file_name))
    # Move the matching mask patch (same filename, different folder).
    shutil.move(os.path.join(src_masks, file_name), os.path.join(val_masks_dir, file_name))

# --------------------------------------------------------------------------
# 6. Move the test files (both image and corresponding mask).
# --------------------------------------------------------------------------
for file_name in test_files:
    # Move the image patch.
    shutil.move(os.path.join(src_waves, file_name), os.path.join(test_waves_dir, file_name))
    # Move the matching mask patch (same filename, different folder).
    shutil.move(os.path.join(src_masks, file_name), os.path.join(test_masks_dir, file_name))

# --------------------------------------------------------------------------
# Final report: how many files ended up in each split.
# --------------------------------------------------------------------------
print("---Separation done!---")
print(f"Patches for training: {len(os.listdir(src_waves))}")
print(f"Patches for validation: {len(os.listdir(val_waves_dir))}")
print(f"Patches for test: {len(os.listdir(test_waves_dir))}")
