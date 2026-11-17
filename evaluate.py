"""
SDIQA Evaluation Script
=======================

Load a trained ``SDIQAModel`` checkpoint and report PLCC / SRCC on the
overall quality and on each of the five dimensions.

Usage
-----

    python evaluate.py                                   # checkpoints/final.pth on data/test.xlsx
    python evaluate.py --checkpoint checkpoints/best_plcc.pth --save_csv results/test_pred.csv
"""

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import DIMENSION_NAMES, IDIQADataset, build_transforms
from metrics import (
    compute_plcc, compute_plcc_per_dim,
    compute_srcc, compute_srcc_per_dim,
)
from model import SDIQAModel, load_checkpoint

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SDIQA on an annotation file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(ROOT, "checkpoints", "final.pth"))
    parser.add_argument("--test_csv", type=str,
                        default=os.path.join(ROOT, "data", "test.xlsx"))
    parser.add_argument("--image_dir", type=str,
                        default=os.path.join(ROOT, "data", "image"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Optional path to save per-image predictions "
                             "(1-5 scale, clipped to [1, 5]).")
    return parser.parse_args()


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # ---- Data ----
    transform = build_transforms((args.image_size, args.image_size))
    test_set = IDIQADataset(args.test_csv, args.image_dir, transform=transform,
                            return_filename=True)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Model ----
    # ``visual_pretrained=False`` skips the ImageNet weight download since the
    # checkpoint already supplies the trained backbone weights.
    model = SDIQAModel(visual_pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    num_dims = len(DIMENSION_NAMES)
    all_preds, all_targets, all_gt_avg, all_names = [], [], [], []

    with torch.no_grad():
        for images, prompts, scores, names in test_loader:
            images = images.to(device)
            outputs = model(images, prompts).cpu()
            all_preds.append(outputs)
            all_targets.append(scores[:, :num_dims])
            all_gt_avg.append(scores[:, num_dims])
            all_names.extend(names)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    gt_avg = torch.cat(all_gt_avg, dim=0)

    plcc_per_dim, avg_plcc = compute_plcc_per_dim(preds, targets)
    srcc_per_dim, avg_srcc = compute_srcc_per_dim(preds, targets)
    pred_avg = preds.mean(dim=1).numpy()
    overall_plcc = compute_plcc(pred_avg, gt_avg.numpy())
    overall_srcc = compute_srcc(pred_avg, gt_avg.numpy())

    # ---- Report ----
    print("\n" + "=" * 50)
    print("  SDIQA Evaluation")
    print("=" * 50)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Samples    : {len(test_set)}")
    print("-" * 50)
    print(f"  {'Dimension':14s} | {'PLCC':>8s} | {'SRCC':>8s}")
    print("-" * 50)
    print(f"  {'Overall':14s} | {overall_plcc:8.4f} | {overall_srcc:8.4f}")
    for i, name in enumerate(DIMENSION_NAMES):
        print(f"  {name:14s} | {plcc_per_dim[i]:8.4f} | {srcc_per_dim[i]:8.4f}")
    print("-" * 50)
    print(f"  {'Dim Average':14s} | {avg_plcc:8.4f} | {avg_srcc:8.4f}")
    print("=" * 50)

    # ---- Optional per-image export (scores mapped back and clipped to [1, 5]) ----
    # PLCC / SRCC above are computed on the raw outputs; clipping only affects
    # the exported values since the QR heads are unbounded linear layers.
    if args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        df = pd.DataFrame((preds.numpy() * 5.0).clip(1.0, 5.0), columns=DIMENSION_NAMES)
        df.insert(0, "filename", all_names)
        df["overall"] = df[DIMENSION_NAMES].mean(axis=1)
        df.to_csv(args.save_csv, index=False, float_format="%.4f")
        print(f"  Predictions saved to {args.save_csv}")


if __name__ == "__main__":
    main()
