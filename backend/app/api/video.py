from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.services.video_ingestion import video_service

UPLOAD_DIR = Path("data/videos/uploads")
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

router = APIRouter(tags=["video"])


class StartRequest(BaseModel):
    source: Union[int, str] = "0"
    test_event: Optional[str] = None


class VideoStatus(BaseModel):
    is_running: bool = False
    source: Optional[str] = None
    frames_read: int = 0
    frames_processed: int = 0
    snapshots_saved: int = 0
    detections_run: int = 0
    persons_detected_total: int = 0
    last_person_count: int = 0
    ai_windows_analyzed: int = 0
    latest_video_label: Optional[str] = None
    latest_video_confidence: Optional[float] = None
    latest_audio_label: Optional[str] = None
    latest_audio_confidence: Optional[float] = None
    latest_final_label: Optional[str] = None
    latest_final_confidence: Optional[float] = None
    latest_ai_reason: Optional[str] = None
    latest_event_type: Optional[str] = None
    latest_event_confidence: Optional[float] = None
    latest_event_severity: Optional[str] = None
    latest_event_reason: Optional[str] = None
    requires_human_review: bool = False
    last_error: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class StartResponse(BaseModel):
    message: str
    status: VideoStatus


@router.post("/video/start", response_model=StartResponse)
def start_video(payload: StartRequest):
    _, message = video_service.start(payload.source, test_event=payload.test_event)
    return StartResponse(message=message, status=VideoStatus(**video_service.get_status()))


@router.post("/video/stop")
def stop_video():
    ok, message = video_service.stop()
    return {"success": ok, "message": message}


@router.get("/video/status", response_model=VideoStatus)
def video_status():
    return video_service.get_status()


@router.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)
    source = f"data/videos/uploads/{file.filename}"
    return {"message": "File uploaded", "filename": file.filename, "source": source}
