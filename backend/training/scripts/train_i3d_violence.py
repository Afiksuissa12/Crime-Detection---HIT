"""
I3D-based binary violence classifier — training script.

Model:
    R3D-18 (3D ResNet-18) from torchvision.models.video, fine-tuned for binary
    classification (violence / no_violence).  R3D-18 inflates 2D ResNet-18
    convolutions to 3D, capturing spatio-temporal features in the same way as
    the original Inflated 3D ConvNet (I3D) proposed by Carreira & Zisserman
    (CVPR 2017).  It ships with Kinetics-400 pre-trained weights, providing a
    strong temporal prior for violence action recognition.

Dataset layout expected (see datasets/rwf2000/ for the empty scaffold):
    datasets/rwf2000/
      train/
        violence/      <- *.mp4 / *.avi clips
        no_violence/
      val/
        violence/
        no_violence/

Usage:
    cd backend/training
    python scripts/train_i3d_violence.py \\
        --data_root datasets/rwf2000 \\
        --epochs    30               \\
        --batch_size 8               \\
        --lr        1e-4             \\
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
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18, R3D_18_Weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters / constants
# ---------------------------------------------------------------------------

CLIP_LEN    = 16          # temporal depth fed to the model (frames)
FRAME_SIZE  = 112         # spatial resolution — H = W = 112
MEAN        = [0.43216, 0.394666, 0.37645]   # Kinetics-400 RGB channel mean
STD         = [0.22803,  0.22145, 0.216989]  # Kinetics-400 RGB channel std
CLASS_NAMES = ["no_violence", "violence"]     # index 0 = negative, 1 = positive


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoClipDataset(Dataset):
    """
    Loads fixed-length RGB clips from a class-folder directory tree.

    Each video is decoded with OpenCV.  CLIP_LEN frames are sampled from the
    video using either:
      - a random temporal offset  (augment=True,  for training)
      - the centre of the video   (augment=False, for val / test)

    Returns a (C, T, H, W) float32 tensor normalised with Kinetics-400 stats.
    """

    EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        root: str,
        clip_len: int   = CLIP_LEN,
        frame_size: int = FRAME_SIZE,
        augment: bool   = False,
    ) -> None:
        self.clip_len   = clip_len
        self.frame_size = frame_size
        self.augment    = augment
        self.samples: list[tuple[Path, int]] = []

        root_path = Path(root)
        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = root_path / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected class directory: {class_dir}")
            for video_file in sorted(class_dir.iterdir()):
                if video_file.suffix.lower() in self.EXTENSIONS:
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
    # Private helpers
    # ------------------------------------------------------------------

    def _load_frames(self, path: Path) -> np.ndarray:
        cap   = cv2.VideoCapture(str(path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.clip_len)

        if self.augment:
            start = np.random.randint(0, max(1, total - self.clip_len + 1))
        else:
            start = (total - self.clip_len) // 2  # centre crop

        indices = np.linspace(start, start + self.clip_len - 1, self.clip_len, dtype=int)
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
        return np.stack(frames)  # (T, H, W, 3)

    def _to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        # (T, H, W, C) uint8  →  (C, T, H, W) float32 normalised
        t    = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        std  = torch.tensor(STD,  dtype=torch.float32).view(3, 1, 1, 1)
        return (t - mean) / std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Build an R3D-18 model with a binary classification head.

    Kinetics-400 pre-trained weights give the model a strong temporal prior,
    allowing it to converge faster and generalise better on small datasets such
    as RWF-2000 (~1,600 training clips).

    To swap in a true Inception-I3D backbone, replace the r3d_18() call and
    adjust the fc layer accordingly — the rest of the script is model-agnostic.
    """
    weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
    model   = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training loop
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
) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits      = model(clips)
            total_loss += criterion(logits, labels).item() * clips.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train I3D/R3D-18 violence classifier")
    p.add_argument("--data_root",   default="datasets/rwf2000", help="Dataset root directory")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--output_dir",  default="models",           help="Checkpoint output directory")
    p.add_argument("--no_pretrain", action="store_true",        help="Train from scratch")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_ds = VideoClipDataset(f"{args.data_root}/train", augment=True)
    val_ds   = VideoClipDataset(f"{args.data_root}/val",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model     = build_model(pretrained=not args.no_pretrain).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0                = time.time()
        train_loss        = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %02d/%02d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  (%.1fs)",
            epoch, args.epochs, train_loss, val_loss, val_acc, time.time() - t0,
        )

        if val_acc > best_acc:
            best_acc  = val_acc
            ckpt_path = output_dir / "best_model.pt"
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc},
                ckpt_path,
            )
            logger.info("  ↑  New best checkpoint → %s", ckpt_path)

    logger.info("Training complete.  Best val_acc: %.4f", best_acc)


if __name__ == "__main__":
    main()
