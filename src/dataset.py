from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from disk and return it as an RGB numpy array."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_albu_transform(
    size: tuple[int, int] = (224, 224),
    augmentations: list | None = None,
) -> Callable[[np.ndarray], torch.Tensor]:
    """Build an albumentations pipeline: resize/crop to `size`, then
    optional augmentations, then normalize and convert to a tensor.
    """
    transforms = [
        A.SmallestMaxSize(max_size=max(size)),
        A.CenterCrop(height=size[0], width=size[1]),
    ]

    if augmentations is not None:
        transforms.extend(augmentations)

    transforms.extend(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    albu_transform = A.Compose(transforms)

    def transform(image: np.ndarray) -> torch.Tensor:
        return albu_transform(image=image)["image"]

    return transform


class GestureDataset(Dataset):
    """Loads gesture images and their labels from a dataframe of file IDs."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: Path,
        filename: str = "img_IDS",
        targetname: str = "Label_idx",
        transform: Callable | None = None,
    ) -> None:
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.filename = filename
        self.targetname = targetname
        self.transform = transform if transform is not None else make_albu_transform()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = (
            self.images_dir / f"{self.dataframe[self.filename].iloc[index]}.jpg"
        )

        image = read_image(image_path)
        image = self.transform(image)

        target = torch.tensor(
            self.dataframe[self.targetname].iloc[index], dtype=torch.long
        )

        return image, target


def make_loaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    images_dir: Path,
    size: tuple[int, int] = (224, 224),
    train_augmentations: list | None = None,
    batch_size_train: int = 64,
    batch_size_valid: int = 256,
    num_workers: int = 6,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Each worker reads and transforms an image on the CPU, and batches are
    moved to the GPU during training/validation. Only the train loader gets
    `train_augmentations`
    """
    train_transform = make_albu_transform(size=size, augmentations=train_augmentations)
    valid_transform = make_albu_transform(size=size)

    train_dataset = GestureDataset(train_df, images_dir, transform=train_transform)
    valid_dataset = GestureDataset(valid_df, images_dir, transform=valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader


class SquarePad:
    """Pads an image with black borders so its height and width match."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]

        max_size = max(h, w)
        pad_h = max_size - h
        pad_w = max_size - w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        return v2.Pad((left, top, right, bottom))(image)


def build_cache(
    dataframe: pd.DataFrame,
    images_dir: Path,
    size: tuple[int, int] = (224, 224),
    batch_size: int = 256,
    num_workers: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and resize every image once (no augmentation) and return them
    as two tensors (images, labels).
    """
    ds = GestureDataset(
        dataframe,
        images_dir=images_dir,
        transform=make_albu_transform(size=size),
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)

    return torch.cat(xs), torch.cat(ys)
