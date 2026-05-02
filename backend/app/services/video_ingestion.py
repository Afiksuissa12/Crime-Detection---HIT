import threading
import cv2
from typing import Optional
from app.services.frame_processor import FrameProcessor
from app.services.detection_engine import DetectionEngine
from app.services.real_violence_detection_service import real_violence_detection_service
from app.services.i3d_violence_service import i3d_violence_service
from app.services.yamnet_audio_service import yamnet_audio_service
from app.services.multimodal_fusion_service import multimodal_fusion_service
from app.services.event_logic_service import event_logic_service

AI_WINDOW_SIZE = 64


class VideoIngestionService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_processor = FrameProcessor()
        self._detection_engine = DetectionEngine()
        self._state: dict = {
            "is_running": False,
            "source": None,
            "frames_read": 0,
            "frames_processed": 0,
            "snapshots_saved": 0,
            "detections_run": 0,
            "persons_detected_total": 0,
            "last_person_count": 0,
            "ai_windows_analyzed": 0,
            "latest_video_label": None,
            "latest_video_confidence": None,
            "latest_audio_label": None,
            "latest_audio_confidence": None,
            "latest_final_label": None,
            "latest_final_confidence": None,
            "latest_ai_reason": None,
            "latest_event_type": None,
            "latest_event_confidence": None,
            "latest_event_severity": None,
            "latest_event_reason": None,
            "requires_human_review": False,
            "last_error": None,
        }

    def _run_ai_window(self, frames: list, person_detections: list, test_event: str | None) -> None:
        if test_event == "violence":
            video_result = {"label": "violence", "confidence": 0.85, "model_name": "I3D"}
            audio_result = {"label": "aggressive_audio", "confidence": 0.70, "detected_sounds": [], "model_name": "YAMNet"}
        else:
            video_result = real_violence_detection_service.analyze_video_window(frames)
            if "error" in video_result:
                video_result = i3d_violence_service.analyze_video_window(frames)
            audio_result = yamnet_audio_service.analyze_audio_window("")

        fusion_result = multimodal_fusion_service.combine(video_result, audio_result)
        event_result = event_logic_service.evaluate(
            video_result,
            audio_result,
            person_detections,
            {"frame_number": "window"},
        )
        with self._lock:
            self._state["ai_windows_analyzed"] += 1
            self._state["latest_video_label"] = video_result["label"]
            self._state["latest_video_confidence"] = video_result["confidence"]
            self._state["latest_audio_label"] = audio_result["label"]
            self._state["latest_audio_confidence"] = audio_result["confidence"]
            self._state["latest_final_label"] = fusion_result["final_label"]
            self._state["latest_final_confidence"] = fusion_result["final_confidence"]
            self._state["latest_ai_reason"] = fusion_result["reason"]
            self._state["latest_event_type"] = event_result["event_type"]
            self._state["latest_event_confidence"] = event_result["confidence"]
            self._state["latest_event_severity"] = event_result["severity"]
            self._state["latest_event_reason"] = event_result["reason"]
            self._state["requires_human_review"] = event_result["requires_human_review"]

    def _read_loop(self, source: str, stop_event: threading.Event, test_event: str | None) -> None:
        cap_source: int | str = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_source)

        if not cap.isOpened():
            with self._lock:
                self._state["last_error"] = f"Cannot open source: {source}"
                self._state["is_running"] = False
            return

        frame_buffer: list = []
        last_detections: list = []
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    with self._lock:
                        self._state["last_error"] = "Stream ended or failed to read frame"
                        self._state["is_running"] = False
                    break

                with self._lock:
                    self._state["frames_read"] += 1
                    should_process = self._state["frames_read"] % 10 == 0
                    if should_process:
                        self._state["frames_processed"] += 1
                        frames_processed = self._state["frames_processed"]

                if should_process:
                    saved = self._frame_processor.process(frame, frames_processed)
                    detections = self._detection_engine.detect_persons(frame)
                    person_count = len(detections)
                    last_detections = detections
                    frame_buffer.append(frame)
                    with self._lock:
                        if saved:
                            self._state["snapshots_saved"] += 1
                        self._state["detections_run"] += 1
                        self._state["persons_detected_total"] += person_count
                        self._state["last_person_count"] = person_count

                    if len(frame_buffer) >= AI_WINDOW_SIZE:
                        self._run_ai_window(frame_buffer, last_detections, test_event)
                        frame_buffer.clear()
        finally:
            cap.release()
            with self._lock:
                self._state["is_running"] = False

    def start(self, source: int | str, test_event: str | None = None) -> tuple[bool, str]:
        source_str = str(source)

        with self._lock:
            if self._state["is_running"]:
                return False, "Already running"
            self._state.update({
                "is_running": True,
                "source": source_str,
                "frames_read": 0,
                "frames_processed": 0,
                "snapshots_saved": 0,
                "detections_run": 0,
                "persons_detected_total": 0,
                "last_person_count": 0,
                "ai_windows_analyzed": 0,
                "latest_video_label": None,
                "latest_video_confidence": None,
                "latest_audio_label": None,
                "latest_audio_confidence": None,
                "latest_final_label": None,
                "latest_final_confidence": None,
                "latest_ai_reason": None,
                "latest_event_type": None,
                "latest_event_confidence": None,
                "latest_event_severity": None,
                "latest_event_reason": None,
                "requires_human_review": False,
                "last_error": None,
            })

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop,
            args=(source_str, self._stop_event, test_event),
            daemon=True,
        )
        self._thread.start()
        return True, "Video started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self._state["is_running"]:
                return False, "Not running"

        self._stop_event.set()
        return True, "Stopped"

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._state)


video_service = VideoIngestionService()
