"""
Evaluate the trained violence baseline on a dataset split.

Imports VideoClipDataset and ViolenceBaselineModel from the training script
to guarantee identical preprocessing and model architecture.

Metrics:
    accuracy, precision, recall, F1 (weighted + per-class), confusion matrix

Usage:
    cd backend/training
    python scripts/evaluate_violence_baseline.py \\
        --checkpoint models/violence_baseline_model.pth \\
        --data_root  datasets/rwf2000                   \\
        --split      test
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

# Allow importing from the same scripts/ directory
sys.path.insert(0, str(Path(__file__).parent))
from train_violence_baseline import CLASS_NAMES, VideoClipDataset, ViolenceBaselineModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Collect predictions and compute all classification metrics."""
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
    cm    = metrics["confusion_matrix"]
    col_w = 16

    print("\n=== Evaluation Results ===")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}  (weighted)")
    print(f"  Recall    : {metrics['recall']:.4f}  (weighted)")
    print(f"  F1 Score  : {metrics['f1']:.4f}  (weighted)")

    print("\n  Confusion Matrix  (rows = true label, cols = predicted label)")
    header = f"  {'':>{col_w}}" + "".join(f"  {c:>{col_w}}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {CLASS_NAMES[i]:>{col_w}}" + "".join(f"  {v:>{col_w}}" for v in row)
        print(row_str)

    print("\n  Per-class Classification Report")
    print(metrics["report"])

    # Production-readiness indicator
    f1 = metrics["f1"]
    if f1 >= 0.85:
        print(f"  [OK]      F1 {f1:.4f} meets the production target (>= 0.85)")
    else:
        print(f"  [BELOW]   F1 {f1:.4f} below production target (>= 0.85) -- continue training or tune")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate violence baseline classifier")
    p.add_argument("--checkpoint",  required=True,            help="Path to .pth checkpoint")
    p.add_argument("--data_root",   default="datasets/rwf2000")
    p.add_argument("--split",       default="test",           help="train | val | test")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    logger.info(
        "Checkpoint — epoch=%d  val_acc=%.4f  val_f1=%.4f",
        ckpt["epoch"], ckpt["val_acc"], ckpt["val_f1"],
    )

    # Verify the checkpoint was saved with compatible settings
    ckpt_frames = ckpt.get("num_frames", "unknown")
    ckpt_size   = ckpt.get("frame_size", "unknown")
    logger.info("Checkpoint config — num_frames=%s  frame_size=%s", ckpt_frames, ckpt_size)

    model = ViolenceBaselineModel().to(device)
    model.load_state_dict(ckpt["model_state"])

    split_root = str(Path(args.data_root) / args.split)
    dataset    = VideoClipDataset(split_root, augment=False)
    loader     = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    metrics = evaluate(model, loader, device)
    print_results(metrics)


if __name__ == "__main__":
    main()
