import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Directory containing the patches for training, validation and tests.
data = "PATH"

# Training image/mask folders.
waves = os.path.join(data, 'Waves')
masks = os.path.join(data, 'Masks')

# Validation image/mask folders.
valid_waves = os.path.join(data, 'ValWaves')
valid_masks = os.path.join(data, 'ValMasks')

# Test image/mask folders.
test_waves = os.path.join(data, 'TestWaves')
test_masks = os.path.join(data, 'TestMasks')

# Names of the segmentation classes.
classes = ['internalwave']


class Dataset(BaseDataset):
    """
    PyTorch Dataset for loading SAR image / binary mask pairs

    Reads an image and its corresponding mask from disk, converts the
    mask into a normalized single-channel float array, and
    applies augmentation and preprocessing transforms.

    It receives the path of the data, path of the masks, list of classes.

    Performs aumengtations and preprocessing.
    """

    def __init__(
            self,
            waves,
            masks,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        # List and sort filenames so that images and masks line up by index.
        self.ids_x = sorted(os.listdir(waves))
        self.ids_y = sorted(os.listdir(masks))

        # Verify if the image directory and masks directory have the same number of files.
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
        """
        # Read the SAR image.
        image = cv2.imread(self.images_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the mask in grayscale mode.
        mask = cv2.imread(self.masks_paths[i], 0)

        # Binarize the mask: any pixel greater than 0 becomes 1.0
        mask = (mask > 0).astype('float32')

        # Add an explicit channel dimension: (H, W) -> (H, W, 1).
        mask = np.expand_dims(mask, axis=-1)

        # Apply data augmentation (e.g. flips/rotations) if configured.
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply any additional preprocessing (e.g. normalization, tensor conversion) after augmentation.
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.ids_x)


def visualize(image, mask, label=None):
    """
    Display a side-by-side plot of an original SAR image and its
    corresponding segmentation mask.
    """
    plt.figure(figsize=(10, 5))

    # Left subplot: the original SAR image.
    plt.subplot(1, 2, 1)
    # If the image is a PyTorch tensor (C, H, W), convert to a NumPy array and reorder axes to (H, W, C) for correct display.
    if hasattr(image, 'permute'):
        image = image.permute(1, 2, 0).numpy()

    plt.imshow(image)
    plt.title("Imagem SAR Original")
    plt.axis('off')

    # Right subplot: the corresponding segmentation mask.
    plt.subplot(1, 2, 2)

    # If the mask is a tensor with an extra channel dimension, squeeze it out and convert to a NumPy array so it can be shown as a 2D image.
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
    """
    # Convert to tensor so the output format matches training data.
    return A.Compose([ToTensorV2()])


if __name__ == "__main__":

    # Instantiate the Dataset for training.
    train_dataset = Dataset(
        waves=waves,
        masks=masks,
        classes=classes,
        augmentation=get_training_augmentation()
    )

    # Instantiate the Dataset for validation.
    valid_dataset = Dataset(
        waves=valid_waves,
        masks=valid_masks,
        classes=classes,
        augmentation=get_validation_augmentation()
    )

    # Wrap each Dataset in a DataLoader to handle batching and shuffling.
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Print basic dataset/loader statistics for a quick check.
    print(f"Dataset de treino carregado: {len(train_dataset)} patches.")
    print(f"Dataset de validação carregado: {len(valid_dataset)} patches.")
    print(f"Número de lotes (batches) por época no treino: {len(train_loader)}")

    # Quick visual check: grab a single sample from the training dataset.
    image, mask = train_dataset[2]
    visualize(image=image, mask=mask, label=classes[0])
