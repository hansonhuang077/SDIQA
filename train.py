"""
SDIQA Training Script
=====================

End-to-end training of ``SDIQAModel`` on ID-AIGC-6K:

* Visual backbone (Swin V2-B) is **trainable**.
* Text encoder    (OpenCLIP)  is **frozen** and only used as a feature
  extractor (no gradient flows through it).

Default hyper-parameters follow the paper: AdamW (lr 2e-5, weight decay
0.01), cosine annealing, batch size 16, at most 50 epochs with early stopping
when the validation loss does not decrease for 10 epochs, Smooth L1 loss with
beta = 1.0.

Usage
-----

    python train.py                       # uses ./data/{train,val,test}.xlsx
    python train.py --image_dir /path/to/image --batch_size 8

Outputs
-------

* ``checkpoints/best_plcc.pth``  - Best validation PLCC checkpoint.
* ``checkpoints/best_loss.pth``  - Best validation loss checkpoint.
* ``checkpoints/final.pth``      - Last-epoch checkpoint.
* ``results/training_log.csv``   - Per-epoch metrics log.
"""

import argparse
import csv
import logging
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DIMENSION_NAMES, IDIQADataset, build_transforms
from metrics import (
    compute_plcc, compute_plcc_per_dim,
    compute_srcc, compute_srcc_per_dim,
)
from model import SDIQAModel, load_checkpoint

ROOT = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("sdiqa.train")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SDIQA on ID-AIGC-6K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data (paths are relative to the repository root by default)
    parser.add_argument("--train_csv", type=str,
                        default=os.path.join(ROOT, "data", "train.xlsx"))
    parser.add_argument("--val_csv", type=str,
                        default=os.path.join(ROOT, "data", "val.xlsx"))
    parser.add_argument("--test_csv", type=str,
                        default=os.path.join(ROOT, "data", "test.xlsx"),
                        help="Test annotation file; evaluated after training. "
                             "Pass an empty string to skip.")
    parser.add_argument("--image_dir", type=str,
                        default=os.path.join(ROOT, "data", "image"))
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--num_workers", type=int, default=4)

    # Optimization
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--smooth_l1_beta", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-7)

    # Gate regularization (keeps the PGF gates from collapsing to ~0)
    parser.add_argument("--gate_reg_weight", type=float, default=0.01,
                        help="Weight of the gate-activity regularization term. "
                             "Set to 0 to disable.")
    parser.add_argument("--gate_reg_target", type=float, default=0.2,
                        help="Lower bound on the average gate value per branch.")

    # Early stopping
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_stop_metric", type=str, default="loss",
                        choices=["loss", "plcc"],
                        help="Stop when this validation metric has not improved "
                             "for `patience` epochs (PLCC: higher is better, "
                             "loss: lower is better).")

    # Misc
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(ROOT, "results"))

    return parser.parse_args()


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    path: str,
    model: SDIQAModel,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_plcc: float,
) -> None:
    """Save weights (without the frozen text encoder) and optimizer state."""
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.trainable_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_plcc": val_plcc,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: SDIQAModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_dims: int = 5,
) -> Dict:
    """Evaluate the model on a data loader and return a metrics dict."""
    model.eval()
    total_loss = 0.0
    loss_per_dim = [0.0] * num_dims
    all_preds, all_targets, all_gt_avg = [], [], []

    for images, prompts, scores in loader:
        images = images.to(device)
        scores = scores.to(device)
        outputs = model(images, prompts)

        for i in range(num_dims):
            dim_loss = criterion(outputs[:, i], scores[:, i])
            total_loss += dim_loss.item()
            loss_per_dim[i] += dim_loss.item()

        all_preds.append(outputs)
        all_targets.append(scores[:, :num_dims])
        all_gt_avg.append(scores[:, num_dims])

    n = max(len(loader), 1)
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    gt_avg = torch.cat(all_gt_avg, dim=0)

    plcc_per_dim, avg_plcc = compute_plcc_per_dim(preds, targets)
    srcc_per_dim, avg_srcc = compute_srcc_per_dim(preds, targets)

    # Overall quality = arithmetic mean of the five predicted dimensions
    pred_avg = preds.mean(dim=1).cpu().numpy()
    gt_avg_np = gt_avg.cpu().numpy()
    overall_plcc = compute_plcc(pred_avg, gt_avg_np)
    overall_srcc = compute_srcc(pred_avg, gt_avg_np)

    return {
        "avg_loss": total_loss / n,
        "loss_per_dim": [v / n for v in loss_per_dim],
        "plcc_per_dim": plcc_per_dim,
        "avg_plcc": avg_plcc,
        "srcc_per_dim": srcc_per_dim,
        "avg_srcc": avg_srcc,
        "overall_plcc": overall_plcc,
        "overall_srcc": overall_srcc,
    }


def print_test_report(label: str, metrics: Dict, dim_names: List[str]) -> None:
    print(f"\n--- [{label}] Test Set ---")
    print(
        f"  Overall PLCC: {metrics['overall_plcc']:.4f} | "
        f"Overall SRCC: {metrics['overall_srcc']:.4f} | "
        f"Avg PLCC: {metrics['avg_plcc']:.4f} | "
        f"Avg SRCC: {metrics['avg_srcc']:.4f}"
    )
    for i, name in enumerate(dim_names):
        print(
            f"  {name:14s} | PLCC: {metrics['plcc_per_dim'][i]:.4f} | "
            f"SRCC: {metrics['srcc_per_dim'][i]:.4f}"
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    # ---- Data ----
    transform = build_transforms((args.image_size, args.image_size))
    train_set = IDIQADataset(args.train_csv, args.image_dir, transform=transform)
    val_set   = IDIQADataset(args.val_csv,   args.image_dir, transform=transform)
    test_set  = (IDIQADataset(args.test_csv, args.image_dir, transform=transform)
                 if args.test_csv else None)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = (DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
                   if test_set is not None else None)
    logger.info(
        "Datasets: train=%d, val=%d, test=%s",
        len(train_set), len(val_set),
        len(test_set) if test_set is not None else "n/a",
    )

    # ---- Model ----
    model = SDIQAModel().to(device)

    summary = model.get_parameter_summary()
    print("\n" + "=" * 60)
    print("SDIQA model parameter summary")
    print("=" * 60)
    print(f"  Total params      : {summary['total']:>12,} ({summary['total_M']} M)")
    print(f"  Trainable params  : {summary['trainable']:>12,} ({summary['trainable_M']} M)")
    print(f"  Frozen params     : {summary['frozen']:>12,}")
    print("=" * 60 + "\n")

    # ---- Optimizer / scheduler / loss ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    optimizer = optim.AdamW(trainable_params, lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.scheduler_eta_min)

    # ---- Output paths ----
    ensure_dir(args.ckpt_dir)
    ensure_dir(args.results_dir)
    best_loss_path = os.path.join(args.ckpt_dir, "best_loss.pth")
    best_plcc_path = os.path.join(args.ckpt_dir, "best_plcc.pth")
    final_path     = os.path.join(args.ckpt_dir, "final.pth")
    log_csv_path   = os.path.join(args.results_dir, "training_log.csv")

    dim_names = DIMENSION_NAMES
    num_dims = len(dim_names)

    # ---- CSV logger header ----
    csv_header = ["epoch", "lr", "train_loss"]
    csv_header += [f"train_loss_{n}" for n in dim_names]
    csv_header += ["val_loss"] + [f"val_loss_{n}" for n in dim_names]
    csv_header += [f"val_plcc_{n}" for n in dim_names] + ["val_avg_plcc"]
    csv_header += [f"val_srcc_{n}" for n in dim_names] + ["val_avg_srcc"]
    csv_header += ["val_overall_plcc", "val_overall_srcc"]
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_header)

    # ---- Training loop ----
    best_val_loss = float("inf")
    best_val_plcc = 0.0
    early_stop_counter = 0
    use_early_stopping = not args.no_early_stopping

    logger.info(
        "Training for %d epochs | batch_size=%d | lr=%.2e",
        args.epochs, args.batch_size, args.lr,
    )
    logger.info(
        "Early stopping: %s (metric=%s, patience=%d)",
        use_early_stopping, args.early_stop_metric, args.patience,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_loss_per_dim = [0.0] * num_dims
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for images, prompts, scores in progress:
            images = images.to(device)
            scores = scores.to(device)
            optimizer.zero_grad()

            outputs = model(images, prompts)
            # Equal-weighted sum of the five per-dimension Smooth L1 losses
            loss = torch.tensor(0.0, device=device)
            for i in range(num_dims):
                dim_loss = criterion(outputs[:, i], scores[:, i])
                loss = loss + dim_loss
                running_loss_per_dim[i] += dim_loss.item()

            if args.gate_reg_weight > 0:
                gates = model.last_gate_values
                reg = (
                    torch.relu(args.gate_reg_target - gates["light"].mean())
                    + torch.relu(args.gate_reg_target - gates["material"].mean())
                    + torch.relu(args.gate_reg_target - gates["aesthetics"].mean())
                )
                loss = loss + args.gate_reg_weight * reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params,
                                           max_norm=args.max_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item() / num_dims:.4f}"})

        scheduler.step()
        n_train = max(len(train_loader), 1)
        avg_train_loss = running_loss / n_train
        avg_train_loss_per_dim = [v / n_train for v in running_loss_per_dim]
        cur_lr = optimizer.param_groups[0]["lr"]

        # ---- Validation ----
        val_metrics = evaluate(model, val_loader, criterion, device, num_dims)

        print(
            f"\nEpoch {epoch}/{args.epochs} | LR: {cur_lr:.2e}\n"
            f"  Train Loss: {avg_train_loss / num_dims:.4f} | "
            f"Val Loss: {val_metrics['avg_loss'] / num_dims:.4f} | "
            f"Val PLCC: {val_metrics['avg_plcc']:.4f} | "
            f"Val SRCC: {val_metrics['avg_srcc']:.4f} | "
            f"Overall PLCC: {val_metrics['overall_plcc']:.4f}"
        )
        per_dim_str = ", ".join(
            f"{dim_names[i]}: {val_metrics['plcc_per_dim'][i]:.4f}"
            for i in range(num_dims)
        )
        print(f"  Val PLCC per dim: {per_dim_str}")

        # ---- CSV logging ----
        row = [epoch, f"{cur_lr:.2e}", f"{avg_train_loss / num_dims:.6f}"]
        row += [f"{v:.6f}" for v in avg_train_loss_per_dim]
        row += [f"{val_metrics['avg_loss'] / num_dims:.6f}"]
        row += [f"{v:.6f}" for v in val_metrics["loss_per_dim"]]
        row += [f"{v:.6f}" for v in val_metrics["plcc_per_dim"]]
        row += [f"{val_metrics['avg_plcc']:.6f}"]
        row += [f"{v:.6f}" for v in val_metrics["srcc_per_dim"]]
        row += [
            f"{val_metrics['avg_srcc']:.6f}",
            f"{val_metrics['overall_plcc']:.6f}",
            f"{val_metrics['overall_srcc']:.6f}",
        ]
        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        # ---- Checkpoint saving ----
        loss_improved = val_metrics["avg_loss"] < best_val_loss
        plcc_improved = val_metrics["avg_plcc"] > best_val_plcc

        if loss_improved:
            best_val_loss = val_metrics["avg_loss"]
            save_checkpoint(best_loss_path, model, optimizer, epoch,
                            val_metrics["avg_loss"], val_metrics["avg_plcc"])
            print(f"  >> Saved best loss model (val_loss={best_val_loss / num_dims:.4f})")

        if plcc_improved:
            best_val_plcc = val_metrics["avg_plcc"]
            save_checkpoint(best_plcc_path, model, optimizer, epoch,
                            val_metrics["avg_loss"], val_metrics["avg_plcc"])
            print(f"  >> Saved best PLCC model (val_plcc={best_val_plcc:.4f})")

        improved = plcc_improved if args.early_stop_metric == "plcc" else loss_improved
        early_stop_counter = 0 if improved else early_stop_counter + 1

        stop_now = use_early_stopping and early_stop_counter >= args.patience
        if epoch == args.epochs or stop_now:
            save_checkpoint(final_path, model, optimizer, epoch,
                            val_metrics["avg_loss"], val_metrics["avg_plcc"])
            print(f"  >> Saved final model (epoch {epoch})")
        if stop_now:
            print(f"\n  >> Early stopping triggered at epoch {epoch}")
            break

    print("\n" + "=" * 60)
    print(f"Training complete. Best val PLCC: {best_val_plcc:.4f}")
    print("=" * 60)

    # ---- Final test evaluation ----
    if test_loader is not None:
        for ckpt_path, label in (
            (best_loss_path, "Best Loss"),
            (best_plcc_path, "Best PLCC"),
            (final_path,     "Final"),
        ):
            if not os.path.isfile(ckpt_path):
                continue
            load_checkpoint(model, ckpt_path, map_location=device)
            test_metrics = evaluate(model, test_loader, criterion, device, num_dims)
            print_test_report(label, test_metrics, dim_names)


if __name__ == "__main__":
    main()
