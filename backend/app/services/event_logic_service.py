"""
Event Logic Service — the sole authority for deciding whether a situation is suspicious.

All rules are expressed as plain conditionals so they can be read, cited, and
adjusted without touching any model code.  Each rule returns either a fully
populated event dict or None (meaning "this rule did not fire").  Rules are
evaluated in priority order: violence → person_fallen → crowd_gathering → normal.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — change these to tune sensitivity without touching logic
# ---------------------------------------------------------------------------

# Violence: I3D alone
VIOLENCE_I3D_HIGH = 0.65

# Violence: I3D + YAMNet corroboration
VIOLENCE_I3D_LOW = 0.55
VIOLENCE_YAMNET_SUPPORT = 0.50

# Fusion weights for corroborated-violence confidence
VIOLENCE_VIDEO_WEIGHT = 0.65
VIOLENCE_AUDIO_WEIGHT = 0.35

# Person fallen: bounding box is "landscape" (wider than tall)
FALLEN_ASPECT_RATIO = 1.4       # width / height must exceed this
FALLEN_MIN_CONFIDENCE = 0.60    # YOLO detection confidence floor

# Crowd gathering
CROWD_MIN_PERSONS = 5


# ---------------------------------------------------------------------------
# Helper: build a standard result dict
# ---------------------------------------------------------------------------

def _event(event_type: str, confidence: float, severity: str, reason: str) -> dict:
    return {
        "event_type": event_type,
        "confidence": round(confidence, 4),
        "severity": severity,
        "reason": reason,
        "requires_human_review": True,
    }


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def _check_violence(video_result: dict, audio_result: dict) -> dict | None:
    """
    Rule 1 — High-confidence I3D alone.
    Rule 2 — Medium-confidence I3D corroborated by YAMNet aggressive audio.
    """
    i3d_violence = video_result.get("label") == "violence"
    i3d_conf = float(video_result.get("confidence", 0.0))

    yamnet_aggressive = audio_result.get("label") == "aggressive_audio"
    yamnet_conf = float(audio_result.get("confidence", 0.0))

    if i3d_violence and i3d_conf >= VIOLENCE_I3D_HIGH:
        return _event(
            "violence_suspected",
            i3d_conf,
            "high",
            f"I3D detected violence with confidence {i3d_conf:.2%}",
        )

    if (
        i3d_violence
        and i3d_conf >= VIOLENCE_I3D_LOW
        and yamnet_aggressive
        and yamnet_conf >= VIOLENCE_YAMNET_SUPPORT
    ):
        combined = (i3d_conf * VIOLENCE_VIDEO_WEIGHT) + (yamnet_conf * VIOLENCE_AUDIO_WEIGHT)
        return _event(
            "violence_suspected",
            combined,
            "high",
            (
                f"I3D ({i3d_conf:.2%}) corroborated by YAMNet aggressive audio "
                f"({yamnet_conf:.2%}); weighted confidence {combined:.2%}"
            ),
        )

    return None


def _check_fallen(person_detections: list[dict]) -> dict | None:
    """
    Rule: a person whose bounding box is significantly wider than it is tall
    is likely horizontal — indicative of a fall.

    Aspect ratio threshold: width / height > FALLEN_ASPECT_RATIO (default 1.4).
    Only boxes whose YOLO confidence meets FALLEN_MIN_CONFIDENCE are considered.
    """
    for det in person_detections:
        w = float(det.get("width", 0))
        h = float(det.get("height", 0))
        conf = float(det.get("confidence", 0.0))

        if h <= 0:
            continue

        aspect_ratio = w / h
        if aspect_ratio > FALLEN_ASPECT_RATIO and conf >= FALLEN_MIN_CONFIDENCE:
            return _event(
                "person_fallen",
                conf,
                "high",
                (
                    f"Person bounding box aspect ratio {aspect_ratio:.2f} "
                    f"exceeds fallen threshold {FALLEN_ASPECT_RATIO} "
                    f"(YOLO confidence {conf:.2%})"
                ),
            )

    return None


def _check_crowd(person_detections: list[dict]) -> dict | None:
    """
    Rule: when five or more persons are visible simultaneously the scene is
    classified as a potential crowd gathering.

    Confidence scales linearly with count: 5 persons → 0.50, 10+ persons → 1.00.
    """
    count = len(person_detections)
    if count >= CROWD_MIN_PERSONS:
        confidence = min(1.0, count / (CROWD_MIN_PERSONS * 2))
        return _event(
            "crowd_gathering",
            confidence,
            "medium",
            f"{count} persons detected simultaneously (threshold: {CROWD_MIN_PERSONS})",
        )

    return None


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------

class EventLogicService:
    """
    Converts raw model outputs into a single, actionable event decision.

    This is the only class that should determine whether a situation is
    suspicious.  All downstream components (alert service, WebSocket push)
    consume its output without re-interpreting model scores.
    """

    def evaluate(
        self,
        video_result: dict,
        audio_result: dict,
        person_detections: list[dict],
        frame_meta: dict,
    ) -> dict:
        """
        Apply rules in priority order and return the first match.

        Args:
            video_result:      Output of I3DViolenceService.analyze_video_window()
            audio_result:      Output of YAMNetAudioService.analyze_audio_window()
            person_detections: Output of DetectionEngine.detect_persons()
            frame_meta:        Contextual info, e.g. {"frame_number": int}

        Returns:
            A dict with keys: event_type, confidence, severity, reason,
            requires_human_review.
        """
        frame_num = frame_meta.get("frame_number", "?")

        result = (
            _check_violence(video_result, audio_result)
            or _check_fallen(person_detections)
            or _check_crowd(person_detections)
        )

        if result is None:
            result = _event("normal", 1.0, "low", "No suspicious pattern detected")

        logger.debug(
            "Frame %s → event_type=%s confidence=%.4f reason=%s",
            frame_num,
            result["event_type"],
            result["confidence"],
            result["reason"],
        )

        return result


event_logic_service = EventLogicService()
