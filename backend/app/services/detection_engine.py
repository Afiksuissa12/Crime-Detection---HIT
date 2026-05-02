import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0
MODEL_NAME = "yolov8n.pt"


class DetectionEngine:
    def __init__(self) -> None:
        logger.info("Loading YOLO model: %s", MODEL_NAME)
        self._model = YOLO(MODEL_NAME)
        logger.info("YOLO model ready.")

    def detect_persons(self, frame) -> list[dict]:
        """
        Run YOLO inference on a single frame.

        Returns a list of person detections, each with pixel-space bounding box
        coordinates, dimensions, and confidence score.  The bounding box origin
        (x1, y1) is the top-left corner; (x2, y2) is the bottom-right corner.
        Width and height are pre-computed for downstream aspect-ratio checks
        (e.g. fallen-person detection).
        """
        results = self._model(frame, classes=[PERSON_CLASS_ID], verbose=False)
        detections: list[dict] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                detections.append({
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width": round(w, 2),
                    "height": round(h, 2),
                    "confidence": round(float(box.conf[0]), 4),
                })
        return detections
