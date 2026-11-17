"""
SDIQA Inference Script
======================

Predict the five quality dimensions and the overall quality of
AI-generated industrial-design images given their text prompts.

Usage
-----

    # Single image
    python infer.py --image demo.png --prompt "photo of a minimalist ceramic teapot"

    # A folder of images sharing one prompt
    python infer.py --image_dir my_images/ --prompt "photo of ..." --save_csv results/pred.csv

All reported scores are on the original 1-5 MOS scale and clipped to [1, 5].
"""

import argparse
import os
from typing import List

import pandas as pd
import torch
from PIL import Image

from dataset import DIMENSION_NAMES, build_transforms
from model import SDIQAModel, load_checkpoint

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SDIQA on one image or a folder of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Path to a single image.")
    src.add_argument("--image_dir", type=str, help="Folder of images.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt used to generate the image(s).")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(ROOT, "checkpoints", "final.pth"))
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Optional path to save the predictions.")
    return parser.parse_args()


def collect_images(args: argparse.Namespace) -> List[str]:
    if args.image:
        return [args.image]
    return sorted(
        os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )


def main() -> None:
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )

    paths = collect_images(args)
    if not paths:
        raise FileNotFoundError("No images found.")

    model = SDIQAModel(visual_pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    transform = build_transforms((args.image_size, args.image_size))
    rows = []
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]
            images = torch.stack([
                transform(Image.open(p).convert("RGB")) for p in batch_paths
            ]).to(device)
            # The QR heads are unbounded, so clip to the valid MOS range
            scores = (model(images, [args.prompt] * len(batch_paths)) * 5.0).clamp(1.0, 5.0)
            for p, s in zip(batch_paths, scores.cpu().tolist()):
                rows.append([os.path.basename(p)] + s + [sum(s) / len(s)])

    df = pd.DataFrame(rows, columns=["filename"] + DIMENSION_NAMES + ["overall"])
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 120):
        print(df.to_string(index=False))

    if args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        df.to_csv(args.save_csv, index=False, float_format="%.4f")
        print(f"\nPredictions saved to {args.save_csv}")


if __name__ == "__main__":
    main()
