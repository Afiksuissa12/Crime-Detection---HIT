import logging

logger = logging.getLogger(__name__)


class I3DViolenceService:
    """
    Primary violence detection model.

    I3D (Inflated 3D ConvNet) analyses a short temporal window of frames
    and classifies the clip as violence or no_violence.

    Real inference: not yet implemented.
    Expected input: a list of consecutive BGR frames (numpy arrays) sampled
                    at a fixed rate (e.g. 16 or 32 frames per window).
    """

    def __init__(self) -> None:
        logger.info("I3DViolenceService initialised (placeholder mode — no model loaded).")

    def analyze_video_window(self, frames: list) -> dict:
        """
        Classify a temporal window of frames for violent activity.

        Args:
            frames: list of BGR numpy arrays, one per sampled frame.

        Returns:
            {
                "label":      "violence" | "no_violence",
                "confidence": float,          # 0.0 – 1.0
                "model_name": "I3D"
            }
        """
        if not frames:
            logger.warning("analyze_video_window called with empty frame list.")
            return {"label": "no_violence", "confidence": 0.0, "model_name": "I3D"}

        # TODO: load I3D weights, preprocess frames, run inference
        logger.debug("I3D placeholder — %d frames received, returning default.", len(frames))
        return {"label": "no_violence", "confidence": 0.0, "model_name": "I3D"}


i3d_violence_service = I3DViolenceService()
