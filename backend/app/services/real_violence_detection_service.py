"""
Real-time violence detection service.

Loads the trained ViolenceBaselineModel checkpoint and runs inference on a
temporal window of frames.  Falls back gracefully if the checkpoint does not
exist yet (i.e., training has not been run).

Model file location (inside Docker bind-mount):
    /app/data/models/violence_baseline_model.pth

To deploy a trained checkpoint:
    cp backend/training/models/violence_baseline_model.pth backend/data/models/

--------------------------------------------------------------------------------
IMPORTANT — Architecture sync:
    The ViolenceBaselineModel class defined below MUST stay identical to the
    one in backend/training/scripts/train_violence_baseline.py.
    Both files share the same architecture so the checkpoint can be loaded here
    without importing from the training package (which is not inside Docker).
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)

# Path to the trained checkpoint, relative to the container working directory.
# backend/data/ is bind-mounted at /app/data/ in Docker.
MODEL_PATH = Path("data/models/violence_baseline_model.pth")

NUM_FRAMES  = 16
FRAME_SIZE  = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ["no_violence", "violence"]


# ---------------------------------------------------------------------------
# Model architecture  (must match train_violence_baseline.py)
# ---------------------------------------------------------------------------

class ViolenceBaselineModel(nn.Module):
    """
    ResNet-18 backbone → temporal mean-pool → binary classifier.
    Input: (B, T, C, H, W)   Output: (B, 2) logits
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        backbone        = resnet18(weights=None)   # weights loaded from checkpoint
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        features = self.backbone(x.view(B * T, C, H, W))
        features = features.view(B, T, 512).mean(dim=1)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_frames(frames: list, num_frames: int, frame_size: int) -> torch.Tensor:
    """
    Sample num_frames from the input list, resize, convert to normalised tensor.

    Args:
        frames:     list of BGR numpy arrays (from OpenCV / VideoIngestionService)
        num_frames: number of frames to sample
        frame_size: target spatial resolution (H = W)

    Returns:
        torch.Tensor of shape (1, T, 3, H, W) ready for the model
    """
    total   = len(frames)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    processed = []
    for i in indices:
        frame = frames[min(i, total - 1)]
        frame = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            (frame_size, frame_size),
        )
        processed.append(frame)

    # (T, H, W, 3) uint8 → (T, 3, H, W) float32 normalised
    arr  = np.stack(processed)
    t    = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    t    = (t - mean) / std
    return t.unsqueeze(0)  # (1, T, 3, H, W)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class RealViolenceDetectionService:
    """
    Wraps the trained ViolenceBaselineModel for real-time inference.

    If the checkpoint file does not exist, the service initialises without a
    model and returns a clear error on every inference call — this allows the
    rest of the pipeline to run without the model being present.

    Interface compatible with I3DViolenceService.analyze_video_window() so it
    can be swapped in without changing the ingestion pipeline.
    """

    def __init__(self) -> None:
        self._model:      ViolenceBaselineModel | None = None
        self._device:     torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_frames: int = NUM_FRAMES
        self._frame_size: int = FRAME_SIZE
        self._load_model()

    def _load_model(self) -> None:
        if not MODEL_PATH.exists():
            logger.warning(
                "Violence model not found at %s — "
                "run training and copy the checkpoint to data/models/. "
                "Service will return a placeholder result until the model is available.",
                MODEL_PATH,
            )
            return

        try:
            ckpt = torch.load(MODEL_PATH, map_location=self._device)
            self._num_frames = ckpt.get("num_frames", NUM_FRAMES)
            self._frame_size = ckpt.get("frame_size", FRAME_SIZE)

            model = ViolenceBaselineModel()
            model.load_state_dict(ckpt["model_state"])
            model.to(self._device).eval()
            self._model = model

            logger.info(
                "ViolenceBaselineModel loaded — epoch=%d  val_acc=%.4f  val_f1=%.4f  device=%s",
                ckpt.get("epoch", "?"),
                ckpt.get("val_acc", 0.0),
                ckpt.get("val_f1", 0.0),
                self._device,
            )
        except Exception as exc:
            logger.error("Failed to load violence model: %s", exc)
            self._model = None

    def analyze_video_window(self, frames: list) -> dict:
        """
        Run binary violence classification on a temporal window of frames.

        Args:
            frames: list of BGR numpy arrays (one per sampled frame).

        Returns:
            {
                "label":      "violence" | "no_violence",
                "confidence": float,          # probability of the predicted class
                "model_name": "ViolenceBaseline"
            }
            or, if the model is not loaded:
            {
                "label":      "no_violence",
                "confidence": 0.0,
                "model_name": "ViolenceBaseline",
                "error":      "Violence model not trained yet"
            }
        """
        if self._model is None:
            return {
                "label":      "no_violence",
                "confidence": 0.0,
                "model_name": "ViolenceBaseline",
                "error":      "Violence model not trained yet",
            }

        if not frames:
            return {
                "label":      "no_violence",
                "confidence": 0.0,
                "model_name": "ViolenceBaseline",
                "error":      "Empty frame list",
            }

        tensor = _preprocess_frames(frames, self._num_frames, self._frame_size)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)                     # (1, 2)
            probs  = torch.softmax(logits, dim=1)[0]         # (2,)

        pred_idx   = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        label      = CLASS_NAMES[pred_idx]

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "model_name": "ViolenceBaseline",
        }


real_violence_detection_service = RealViolenceDetectionService()
