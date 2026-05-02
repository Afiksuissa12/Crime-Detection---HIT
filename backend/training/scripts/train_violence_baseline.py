"""
Practical violence detection baseline — training script.

Architecture:
    ResNet-18 (ImageNet pretrained) → temporal mean-pool → binary head.

    Each video is decoded to NUM_FRAMES equally-spaced frames, each processed
    independently by the shared 2D backbone.  Frame-level feature vectors
    (512-d) are averaged across the temporal dimension and fed into a linear
    binary classifier.

    This "frame-averaging" approach is a well-known strong baseline for video
    classification.  It is deliberately simpler than the R3D-18 (I3D) script
    so that it can be trained quickly on a CPU or a single low-end GPU.

Why ResNet-18 + mean pool instead of full I3D:
    - No custom CUDA ops required
    - Converges in 10–20 epochs on RWF-2000
    - ImageNet weights provide strong visual features out of the box
    - Easy to interpret and debug at each stage

After training, copy the checkpoint to the backend's Docker bind-mount so the
real-time service can load it:

    cp models/violence_baseline_model.pth ../../data/models/

Dataset layout:
    datasets/rwf2000/
      train/violence/        <- .mp4 / .avi clips
      train/no_violence/
      val/violence/
      val/no_violence/

Usage:
    cd backend/training
    pip install torch torchvision opencv-python scikit-learn
    python scripts/train_violence_baseline.py \\
        --data_root  datasets/rwf2000 \\
        --epochs     20               \\
        --batch_size 8                \\
        --lr         1e-4             \\
        --output_dir models/
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_FRAMES  = 16       # frames sampled per clip
FRAME_SIZE  = 224      # spatial resolution — H = W
# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["no_violence", "violence"]   # index 0 = negative, 1 = positive
EXTENSIONS  = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# NOTE: This class definition MUST stay in sync with the copy in
#       backend/app/services/real_violence_detection_service.py.
#       If you change the architecture here, update the service too.
# ---------------------------------------------------------------------------

class ViolenceBaselineModel(nn.Module):
    """
    Binary violence classifier.

    1. A shared ResNet-18 backbone (ImageNet weights) extracts a 512-d feature
       vector from each frame independently.
    2. Frame vectors are averaged across the temporal dimension (mean pooling).
    3. A single linear layer maps the 512-d vector to 2 class logits.

    Input tensor shape:  (B, T, C, H, W)
      B = batch size, T = NUM_FRAMES, C = 3, H = W = FRAME_SIZE
    Output tensor shape: (B, 2)
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        backbone       = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove the original classification head; keep the feature extractor
        self.backbone  = nn.Sequential(*list(backbone.children())[:-1])  # → (B*T, 512, 1, 1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        features = self.backbone(x.view(B * T, C, H, W))   # (B*T, 512, 1, 1)
        features = features.view(B, T, 512).mean(dim=1)     # (B, 512) — temporal mean pool
        return self.classifier(features)                     # (B, 2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoClipDataset(Dataset):
    """
    Loads NUM_FRAMES equally-spaced frames from each video file.

    augment=True  (training):  random temporal offset + random horizontal flip
    augment=False (val/test):  deterministic centre-offset, no flip
    """

    def __init__(
        self,
        root: str,
        num_frames: int  = NUM_FRAMES,
        frame_size: int  = FRAME_SIZE,
        augment: bool    = False,
    ) -> None:
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment    = augment
        self.samples: list[tuple[Path, int]] = []

        root_path = Path(root)
        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = root_path / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected class directory: {class_dir}")
            for video_file in sorted(class_dir.iterdir()):
                if video_file.suffix.lower() in EXTENSIONS:
                    self.samples.append((video_file, label_idx))

        if not self.samples:
            raise RuntimeError(f"No video files found under {root}")
        logger.info("Dataset: %d clips  root=%s", len(self.samples), root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        return self._to_tensor(frames), label

    # ------------------------------------------------------------------

    def _load_frames(self, path: Path) -> np.ndarray:
        cap   = cv2.VideoCapture(str(path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.num_frames)

        if self.augment:
            start = np.random.randint(0, max(1, total - self.num_frames + 1))
        else:
            start = (total - self.num_frames) // 2

        indices = np.linspace(start, start + self.num_frames - 1, self.num_frames, dtype=int)
        frames  = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            frame = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                (self.frame_size, self.frame_size),
            )
            frames.append(frame)
        cap.release()
        return np.stack(frames)   # (T, H, W, 3)  uint8

    def _to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        # (T, H, W, 3) uint8 → (T, 3, H, W) float32 normalised
        t    = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        if self.augment and torch.rand(1).item() > 0.5:
            t = t.flip(dims=[-1])   # random horizontal flip
        return (t - mean) / std


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(clips), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
    return total_loss / len(loader.dataset)


def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    """Returns (val_loss, val_accuracy, val_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits      = model(clips)
            total_loss += criterion(logits, labels).item() * clips.size(0)
            preds       = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n       = len(loader.dataset)
    acc     = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    f1      = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / n, acc, f1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train violence baseline classifier")
    p.add_argument("--data_root",   default="datasets/rwf2000")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--output_dir",  default="models")
    p.add_argument("--no_pretrain", action="store_true", help="Train from scratch")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_ds = VideoClipDataset(f"{args.data_root}/train", augment=True)
    val_ds   = VideoClipDataset(f"{args.data_root}/val",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    model = ViolenceBaselineModel().to(device)
    if args.no_pretrain:
        logger.warning("Training from scratch — no ImageNet weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t0                       = time.time()
        train_loss               = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %02d/%02d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_f1=%.4f  (%.1fs)",
            epoch, args.epochs, train_loss, val_loss, val_acc, val_f1, time.time() - t0,
        )

        if val_f1 > best_f1:
            best_f1   = val_f1
            ckpt_path = output_dir / "violence_baseline_model.pth"
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_acc":     val_acc,
                    "val_f1":      val_f1,
                    "num_frames":  NUM_FRAMES,
                    "frame_size":  FRAME_SIZE,
                    "class_names": CLASS_NAMES,
                },
                ckpt_path,
            )
            logger.info("  [BEST] New best checkpoint (F1=%.4f) -> %s", best_f1, ckpt_path)

    logger.info("Training complete.  Best val F1: %.4f", best_f1)
    logger.info(
        "Deploy: cp %s ../../data/models/violence_baseline_model.pth",
        output_dir / "violence_baseline_model.pth",
    )


if __name__ == "__main__":
    main()
