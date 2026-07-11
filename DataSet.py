"""
DataSet.py
==========

Defines a PyTorch `Dataset` for loading SAR (Synthetic Aperture Radar)
image patches and their corresponding binary segmentation masks
(internal-wave signatures), along with helper functions for data
augmentation and visualization.

This module can be:
    - Imported elsewhere to reuse the `Dataset` class, `visualize`,
      `get_training_augmentation`, and `get_validation_augmentation`.
    - Run directly as a script (see the `if __name__ == "__main__":`
      block at the bottom), which builds training and validation
      datasets/dataloaders and displays one sample image/mask pair.

Expected folder layout (relative to `data`):
    Waves/       -> training images
    Masks/       -> training masks (same filenames as Waves/)
    ValWaves/    -> validation images
    ValMasks/    -> validation masks
    TestWaves/   -> test images
    TestMasks/   -> test masks
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------------------------------------------------------------
# Directory paths for the dataset
# --------------------------------------------------------------------------
# `data` is the root folder containing all image/mask subfolders below.
data = "PATH"

# Training image/mask folders.
waves = os.path.join(data, 'Waves')
masks = os.path.join(data, 'Masks')

# Validation image/mask folders.
valid_waves = os.path.join(data, 'ValWaves')
valid_masks = os.path.join(data, 'ValMasks')

# Test image/mask folders (defined for completeness; not used in the
# __main__ block below, but available for a future test-time evaluation).
test_waves = os.path.join(data, 'TestWaves')
test_masks = os.path.join(data, 'TestMasks')

# Names of the segmentation classes. Only one class is used here:
# "internalwave" — the phenomenon being segmented in the SAR images.
classes = ['internalwave']


class Dataset(BaseDataset):
    """
    PyTorch Dataset for loading SAR image / binary mask pairs
    ("InternalWaveDataset").

    Reads an image and its corresponding mask from disk, converts the
    mask into a normalized single-channel float array, and optionally
    applies augmentation and/or preprocessing transforms (e.g. from the
    `albumentations` library) before returning the pair.

    Parameters
    ----------
    waves : str
        Path to the folder containing the input images (SAR patches).
    masks : str
        Path to the folder containing the corresponding binary masks.
        Filenames in this folder are expected to match those in `waves`.
    classes : list of str, optional
        List of class names relevant to this dataset. Stored on the
        instance but not currently used for multi-class logic (the mask
        is treated as a single binary channel).
    augmentation : albumentations.Compose, optional
        A composed set of augmentation transforms (e.g. flips, rotations)
        applied to the image/mask pair, typically used for training data.
    preprocessing : albumentations.Compose, optional
        A composed set of preprocessing transforms (e.g. normalization,
        tensor conversion) applied after augmentation, typically used for
        both training and validation/test data.

    Attributes
    ----------
    ids_x : list of str
        Sorted list of image filenames found in `waves`.
    ids_y : list of str
        Sorted list of mask filenames found in `masks`.
    images_paths : list of str
        Full paths to each image, built from `ids_x`.
    masks_paths : list of str
        Full paths to each mask, built from `ids_y`.
    """

    def __init__(
            self,
            waves,
            masks,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        # List and sort filenames so that images and masks line up by
        # index (assumes matching filenames/order between the two folders).
        self.ids_x = sorted(os.listdir(waves))
        self.ids_y = sorted(os.listdir(masks))

        # Sanity check: warn (but don't fail) if the image and mask
        # folders contain a different number of files, since this usually
        # indicates a data preparation problem.
        if len(self.ids_x) != len(self.ids_y):
            print(f"Aviso: Número de imagens ({len(self.ids_x)}) e máscaras ({len(self.ids_y)}) é diferente!")

        # Precompute full file paths for fast indexing in __getitem__.
        self.images_paths = [os.path.join(waves, image_id) for image_id in self.ids_x]
        self.masks_paths = [os.path.join(masks, image_id) for image_id in self.ids_y]

        # Store configuration for use in __getitem__.
        self.classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        """
        Load and return the image/mask pair at index `i`.

        Parameters
        ----------
        i : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            (image, mask) after any configured augmentation and
            preprocessing has been applied. Types depend on the
            transforms used (e.g. NumPy arrays if no transform sets a
            tensor conversion, or PyTorch tensors if `ToTensorV2` is
            included in `augmentation`/`preprocessing`).
        """
        # Read the SAR image. OpenCV loads images in BGR order by default,
        # so we convert to RGB to match the convention expected by most
        # downstream tools (matplotlib, torchvision, albumentations, etc.).
        image = cv2.imread(self.images_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the mask in grayscale mode (flag 0 = cv2.IMREAD_GRAYSCALE).
        mask = cv2.imread(self.masks_paths[i], 0)

        # Binarize the mask: any pixel greater than 0 becomes 1.0, and
        # cast to float32 for compatibility with loss functions / models
        # that expect floating-point targets.
        mask = (mask > 0).astype('float32')

        # Add an explicit channel dimension: (H, W) -> (H, W, 1). This
        # matches the (H, W, C) format expected by albumentations and
        # most segmentation model output heads.
        mask = np.expand_dims(mask, axis=-1)

        # Apply data augmentation (e.g. flips/rotations) if configured.
        # Albumentations transforms operate on named keyword arguments
        # and return a dict with the same keys.
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply any additional preprocessing (e.g. normalization, tensor
        # conversion) after augmentation.
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The number of images (equivalently, masks under normal
            conditions) available in this dataset.
        """
        return len(self.ids_x)


def visualize(image, mask, label=None):
    """
    Display a side-by-side plot of an original SAR image and its
    corresponding segmentation mask.

    Parameters
    ----------
    image : numpy.ndarray or torch.Tensor
        The image to display. If a PyTorch tensor in (C, H, W) format is
        given, it is automatically permuted to (H, W, C) for plotting.
    mask : numpy.ndarray or torch.Tensor
        The mask to display alongside the image. If it has a `.squeeze`
        method (e.g. a PyTorch tensor), any singleton dimensions (such
        as the channel dimension) are removed before plotting.
    label : str, optional
        A text label describing the mask/class, shown in the mask's
        subplot title.

    Returns
    -------
    None
        This function only displays a matplotlib figure; it does not
        return a value.
    """
    plt.figure(figsize=(10, 5))

    # Left subplot: the original SAR image.
    plt.subplot(1, 2, 1)
    # If the image is a PyTorch tensor (C, H, W), convert to a NumPy
    # array and reorder axes to (H, W, C) for correct display.
    if hasattr(image, 'permute'):
        image = image.permute(1, 2, 0).numpy()

    plt.imshow(image)
    plt.title("Imagem SAR Original")
    plt.axis('off')

    # Right subplot: the corresponding segmentation mask.
    plt.subplot(1, 2, 2)

    # If the mask is a tensor with an extra channel dimension, squeeze it
    # out and convert to a NumPy array so it can be shown as a 2D image.
    if hasattr(mask, 'squeeze'):
        mask = mask.squeeze().numpy()

    plt.imshow(mask, cmap='gray')
    plt.title(f"Máscara: {label}")
    plt.axis('off')

    plt.show()


def get_training_augmentation():
    """
    Build the augmentation pipeline used for training samples.

    Applies random geometric and photometric transforms to increase
    dataset diversity and reduce overfitting, then converts the result
    to a PyTorch tensor.

    Returns
    -------
    albumentations.Compose
        A composed pipeline of training-time augmentations, ready to be
        passed as the `augmentation` argument of `Dataset`.
    """
    train_transform = [
        A.HorizontalFlip(p=0.5),   # Horizontal flip with 50% probability
        A.VerticalFlip(p=0.5),     # Vertical flip with 50% probability
        A.RandomRotate90(p=0.5),   # Rotate 90 degrees with 50% probability
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),  # Random brightness and contrast with 50% probability
        ToTensorV2(),              # Transform to PyTorch tensor
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """
    Build the augmentation pipeline used for validation/test samples.

    Validation data should not be randomly augmented (to keep evaluation
    consistent and reproducible), so this pipeline only performs the
    tensor conversion required by the model.

    Returns
    -------
    albumentations.Compose
        A composed pipeline containing only a tensor-conversion step,
        ready to be passed as the `augmentation` argument of `Dataset`.
    """
    # In validation, we typically don't apply augmentations, but we can
    # still convert to tensor so the output format matches training data.
    return A.Compose([ToTensorV2()])


# ==============================================================================
# Execution block to create datasets and dataloaders
# ==============================================================================
if __name__ == "__main__":

    # Instantiate the Dataset for training (with augmentations applied
    # on every access, to increase variety across epochs).
    train_dataset = Dataset(
        waves=waves,  # Pasta 'Waves' com seus patches de treino
        masks=masks,  # Pasta 'Masks' com seus patches de treino
        classes=classes,
        augmentation=get_training_augmentation()
    )

    # Instantiate the Dataset for validation (deterministic, no random
    # augmentations, just tensor conversion).
    valid_dataset = Dataset(
        waves=valid_waves,
        masks=valid_masks,
        classes=classes,
        augmentation=get_validation_augmentation()
    )

    # Wrap each Dataset in a DataLoader to handle batching and shuffling.
    # batch_size=8 is a common choice, but you can adjust it based on your
    # GPU memory. Training data is shuffled each epoch; validation data
    # is kept in a fixed order for consistent evaluation.
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Print basic dataset/loader statistics for a quick sanity check.
    print(f"Dataset de treino carregado: {len(train_dataset)} patches.")
    print(f"Dataset de validação carregado: {len(valid_dataset)} patches.")
    print(f"Número de lotes (batches) por época no treino: {len(train_loader)}")

    # Quick visual check: grab a single sample from the training dataset
    # (index 2) and display the image alongside its mask.
    image, mask = train_dataset[2]
    visualize(image=image, mask=mask, label=classes[0])
