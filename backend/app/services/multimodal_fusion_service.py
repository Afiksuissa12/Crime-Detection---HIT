import logging

logger = logging.getLogger(__name__)

# Confidence threshold above which a single modality alone can trigger a positive
SINGLE_MODALITY_THRESHOLD = 0.80
# Combined weighted threshold when both modalities contribute
FUSION_THRESHOLD = 0.60

# Weight given to each modality in the combined score
VIDEO_WEIGHT = 0.65
AUDIO_WEIGHT = 0.35


class MultimodalFusionService:
    """
    Combines I3D video results and YAMNet audio results into a single
    violence-suspected / normal decision.

    Fusion strategy (to be refined once real inference is in place):
    - Either modality alone at high confidence → violence_suspected
    - Weighted average of both confidences above threshold → violence_suspected
    - Otherwise → normal
    """

    def __init__(self) -> None:
        logger.info("MultimodalFusionService initialised.")

    def combine(self, video_result: dict, audio_result: dict) -> dict:
        """
        Fuse video and audio classification results.

        Args:
            video_result: output of I3DViolenceService.analyze_video_window()
            audio_result: output of YAMNetAudioService.analyze_audio_window()

        Returns:
            {
                "final_label":      "violence_suspected" | "normal",
                "final_confidence": float,   # 0.0 – 1.0
                "reason":           str      # human-readable explanation
            }
        """
        video_positive = video_result.get("label") == "violence"
        audio_positive = audio_result.get("label") == "aggressive_audio"
        video_conf = float(video_result.get("confidence", 0.0))
        audio_conf = float(audio_result.get("confidence", 0.0))

        # High-confidence single modality override
        if video_positive and video_conf >= SINGLE_MODALITY_THRESHOLD:
            return {
                "final_label": "violence_suspected",
                "final_confidence": round(video_conf, 4),
                "reason": f"I3D high-confidence violence detection ({video_conf:.2%})",
            }
        if audio_positive and audio_conf >= SINGLE_MODALITY_THRESHOLD:
            return {
                "final_label": "violence_suspected",
                "final_confidence": round(audio_conf, 4),
                "reason": f"YAMNet high-confidence aggressive audio ({audio_conf:.2%})",
            }

        # Weighted fusion
        weighted = (video_conf * VIDEO_WEIGHT) + (audio_conf * AUDIO_WEIGHT)
        if (video_positive or audio_positive) and weighted >= FUSION_THRESHOLD:
            signals = []
            if video_positive:
                signals.append(f"I3D={video_conf:.2%}")
            if audio_positive:
                signals.append(f"YAMNet={audio_conf:.2%}")
            return {
                "final_label": "violence_suspected",
                "final_confidence": round(weighted, 4),
                "reason": f"Fusion threshold exceeded — {', '.join(signals)}",
            }

        return {
            "final_label": "normal",
            "final_confidence": round(1.0 - weighted, 4),
            "reason": "No modality exceeded violence threshold",
        }


multimodal_fusion_service = MultimodalFusionService()
