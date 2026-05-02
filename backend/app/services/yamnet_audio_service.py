import logging

logger = logging.getLogger(__name__)

AGGRESSIVE_SOUND_CLASSES = [
    "Screaming",
    "Shouting",
    "Crying",
    "Gunshot",
    "Explosion",
    "Glass breaking",
    "Alarm",
    "Siren",
]


class YAMNetAudioService:
    """
    Supporting audio-based aggression detection model.

    YAMNet classifies a short audio window against 521 AudioSet sound classes.
    This service filters results to the subset of classes associated with
    aggressive or crime-related events.

    Real inference: not yet implemented.
    Expected input: path to a WAV/MP3 audio segment (a few seconds).
    """

    def __init__(self) -> None:
        logger.info("YAMNetAudioService initialised (placeholder mode — no model loaded).")

    def analyze_audio_window(self, audio_path: str) -> dict:
        """
        Classify a short audio segment for aggressive sounds.

        Args:
            audio_path: filesystem path to the audio file to analyse.

        Returns:
            {
                "label":           "aggressive_audio" | "normal_audio",
                "confidence":      float,          # 0.0 – 1.0
                "detected_sounds": list[str],      # YAMNet class names that fired
                "model_name":      "YAMNet"
            }
        """
        if not audio_path:
            logger.warning("analyze_audio_window called with empty audio_path.")
            return {
                "label": "normal_audio",
                "confidence": 0.0,
                "detected_sounds": [],
                "model_name": "YAMNet",
            }

        # TODO: load YAMNet (TF-Hub), resample to 16 kHz mono, run inference,
        #       filter scores against AGGRESSIVE_SOUND_CLASSES threshold
        logger.debug("YAMNet placeholder — audio_path=%s, returning default.", audio_path)
        return {
            "label": "normal_audio",
            "confidence": 0.0,
            "detected_sounds": [],
            "model_name": "YAMNet",
        }


yamnet_audio_service = YAMNetAudioService()
