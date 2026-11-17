"""
SDIQA Dataset
=============

PyTorch ``Dataset`` for the ID-AIGC-6K database.

Each annotation file (``.xlsx`` or ``.csv``) is expected to contain the
following columns:

    filename      - Image filename (integer ID, e.g. ``3866`` -> ``3866.png``)
    prompt        - Text prompt describing the design intent
    consistency   - Text-image consistency score in [1, 5]
    material      - Material detail quality score   in [1, 5]
    illumination  - Light realism score             in [1, 5]
    creativity    - Creativity score                in [1, 5]
    aesthetics    - Aesthetics score                in [1, 5]
    overall       - Overall (average) score         in [1, 5]

All scores are normalized to ``[0, 1]`` by dividing by 5.0 in ``__getitem__``.
"""

import os
import logging
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Score columns ordered by dimension index (used by the QR heads).
SCORE_COLUMNS = ["consistency", "material", "illumination", "creativity", "aesthetics"]
OVERALL_COLUMN = "overall"

# The five dimension names exposed by the model output, in the same order as
# the columns of the model's output tensor.
DIMENSION_NAMES = ["consistency", "material", "illumination", "creativity", "aesthetics"]


def build_transforms(image_size: Tuple[int, int] = (448, 448)) -> transforms.Compose:
    """Build the standard image preprocessing pipeline.

    The model expects images resized to a fixed resolution and converted to
    float tensors with pixel values in ``[0, 1]``.

    Args:
        image_size: ``(H, W)`` target size for resizing.

    Returns:
        A composed ``torchvision`` transform.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


class IDIQADataset(Dataset):
    """ID-AIGC-6K dataset returning ``(image, prompt, scores)`` triplets.

    The ``scores`` tensor has shape ``(6,)`` and stores the five per-dimension
    scores followed by the overall score. The training loop should compute its
    loss only on the first five elements; the last element can be used to
    report the overall PLCC / SRCC.

    Args:
        annotation_file: Path to the annotation file (``.xlsx`` or ``.csv``).
        image_root: Directory containing all image files (``<filename>.png``).
        transform: Optional torchvision transform applied to the loaded image.
        score_norm_factor: Divisor used to normalize raw scores. Default 5.0.
        return_filename: If ``True``, ``__getitem__`` additionally returns the
            image filename as a string. Useful for inference / per-sample
            result export.
    """

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        transform: Optional[Callable] = None,
        score_norm_factor: float = 5.0,
        return_filename: bool = False,
    ):
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Image directory not found: {image_root}")

        if annotation_file.lower().endswith(".xlsx"):
            self.df = pd.read_excel(annotation_file)
        else:
            self.df = pd.read_csv(annotation_file)

        self.image_root = image_root
        self.transform = transform
        self.score_norm_factor = score_norm_factor
        self.return_filename = return_filename

        self.filenames = [f"{int(fid)}.png" for fid in self.df["filename"].tolist()]
        self.prompts = self.df["prompt"].astype(str).tolist()
        self.scores = self.df[SCORE_COLUMNS].values.astype("float32")

        if OVERALL_COLUMN in self.df.columns:
            self.avg_scores = self.df[OVERALL_COLUMN].values.astype("float32")
        else:
            self.avg_scores = self.scores.mean(axis=1).astype("float32")

        logger.info(
            "Loaded %d samples from %s", len(self.df), annotation_file,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.filenames[idx]
        img_path = os.path.join(self.image_root, filename)
        image = Image.open(img_path).convert("RGB")

        prompt = self.prompts[idx]

        dim_scores = self.scores[idx] / self.score_norm_factor
        avg_score = self.avg_scores[idx] / self.score_norm_factor
        scores = torch.tensor(
            list(dim_scores) + [float(avg_score)], dtype=torch.float32
        )

        if self.transform is not None:
            image = self.transform(image)

        if self.return_filename:
            return image, prompt, scores, filename
        return image, prompt, scores
