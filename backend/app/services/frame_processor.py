import cv2
from datetime import datetime
from pathlib import Path

SNAPSHOTS_DIR = Path("data/snapshots")
SNAPSHOT_INTERVAL = 100


class FrameProcessor:
    def __init__(self) -> None:
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    def process(self, frame, frames_processed: int) -> bool:
        """Save a snapshot every SNAPSHOT_INTERVAL processed frames. Returns True if saved."""
        if frames_processed % SNAPSHOT_INTERVAL != 0:
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}_frame_{frames_processed}.jpg"
        cv2.imwrite(str(SNAPSHOTS_DIR / filename), frame)
        return True
