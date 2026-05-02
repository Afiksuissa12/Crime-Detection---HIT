"""
Evaluate a trained violence classifier on a dataset split.

Metrics reported:
    accuracy, precision, recall, F1 (weighted), confusion matrix,
    full per-class classification report.

Usage:
    cd backend/training
    python scripts/evaluate_i3d_violence.py \\
        --checkpoint models/best_model.pt \\
        --data_root  datasets/rwf2000     \\
        --split      test

Imports VideoClipDataset and build_model from the training script to
guarantee evaluation uses the identical preprocessing and model architecture.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train_i3d_violence import CLASS_NAMES, VideoClipDataset, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """
    Run the model on every batch in loader and compute classification metrics.

    Returns a dict with keys:
        accuracy, precision, recall, f1, confusion_matrix (list[list[int]]),
        report (str — sklearn classification report)
    """
    model.eval()
    all_preds:  list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for clips, labels in loader:
            preds = model(clips.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":           recall_score(y_true, y_pred,    average="weighted", zero_division=0),
        "f1":               f1_score(y_true, y_pred,        average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report":           classification_report(y_true, y_pred, target_names=CLASS_NAMES),
    }


def print_results(metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    col_w = 14

    print("\n=== Evaluation Results ===")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")

    print("\nConfusion Matrix  (rows = true label, cols = predicted label)")
    header = f"  {'':>{col_w}}" + "".join(f"{c:>{col_w}}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {CLASS_NAMES[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row)
        print(row_str)

    print("\nPer-class Classification Report")
    print(metrics["report"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate I3D/R3D-18 violence classifier")
    p.add_argument("--checkpoint",  required=True,            help="Path to .pt checkpoint file")
    p.add_argument("--data_root",   default="datasets/rwf2000")
    p.add_argument("--split",       default="test",           help="Dataset split: train | val | test")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(
        "Loaded checkpoint — epoch %d  val_acc=%.4f  path=%s",
        ckpt["epoch"], ckpt["val_acc"], args.checkpoint,
    )

    split_root  = str(Path(args.data_root) / args.split)
    dataset     = VideoClipDataset(split_root, augment=False)
    loader      = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    metrics = evaluate(model, loader, device)
    print_results(metrics)


if __name__ == "__main__":
    main()
