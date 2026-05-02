# Project PLAN – Real-Time Crime Detection System

## 1. Project Overview

System that connects to video sources (camera / file / RTSP)
and detects suspicious events in real time.

Goal:
Generate alerts for human review (NOT automatic enforcement).

---

## 2. Current Progress (Completed)

### Backend Infrastructure
- FastAPI backend running in Docker
- SQLite database (persistent via named Docker volume)
- SQLAlchemy ORM

### Entities
- Camera model
- Alert model

### API Endpoints
- GET  /health
- GET  /api/cameras
- POST /api/cameras
- GET  /api/alerts
- POST /api/alerts
- POST /api/video/start
- POST /api/video/stop
- GET  /api/video/status

### Video Ingestion
- OpenCV integration (opencv-python-headless)
- Video source support: webcam (source "0"), local file path, RTSP URL
- Background thread reads frames continuously
- Every 10th frame is processed
- State tracking: is_running, source, frames_read, frames_processed, last_error
- VideoStatus Pydantic model on GET /api/video/status (proper JSON schema in Swagger)
- StartRequest accepts source as int or str (fixes webcam source 0)
- StartResponse Pydantic model on POST /api/video/start returns message + full status object
- Refactored to VideoIngestionService class with single global instance (video_service)
- VideoStatus fields have proper defaults (null/0) — Swagger no longer shows fake "string" placeholders

### System Checks
- /health endpoint
- Swagger UI at /docs
- DB write + read working
- Docker build + run verified

---

## 3. Next Steps

### Local video file testing
- backend/data/ bind-mounted into container at /app/data (replaces named sqlite_data volume)
- SQLite DB persists at backend/data/db.sqlite3 on host
- Drop test.mp4 in backend/data/videos/ and call POST /api/video/start with {"source": "data/videos/test.mp4"}
- backend/data/.gitignore excludes *.sqlite3 and video files from git

### Phase 1 – Frame Saving ✓ Done
- FrameProcessor saves one snapshot every 100 processed frames
- Snapshots saved to data/snapshots/ (bind-mounted, visible on host)
- Filename: snapshot_YYYYMMDD_HHMMSS_frame_<N>.jpg
- VideoStatus includes snapshots_saved counter

### Phase 2 – Detection ✓ Done
- Installed ultralytics (yolov8n.pt, COCO class 0 = person)
- DetectionEngine class in backend/app/services/detection_engine.py
- Model loaded once at service startup (singleton via VideoIngestionService)
- Model pre-downloaded into Docker image (no runtime network dependency)
- detect_persons(frame) → int runs on every processed frame
- VideoStatus extended: detections_run, persons_detected_total, last_person_count
- libgl1 added to Dockerfile apt-get for OpenCV compatibility

### Phase 2b – Multimodal AI Architecture ✓ Done (interfaces)
- Redesigned AI pipeline to use I3D + YAMNet as the primary detectors
- YOLO kept as optional person-count context only (not the violence signal)
- i3d_violence_service.py: analyze_video_window(frames) → label/confidence/model_name
- yamnet_audio_service.py: analyze_audio_window(audio_path) → label/confidence/detected_sounds/model_name
- multimodal_fusion_service.py: combine(video_result, audio_result) → final_label/final_confidence/reason
- Fusion weights: I3D 65%, YAMNet 35%; single-modality override at ≥80% confidence
- All three services are placeholder only — no real model inference yet
- README updated with full AI architecture table and model descriptions

### Phase 3 – AI Pipeline Wiring ✓ Done
- Frame buffer (64 processed frames) accumulated in video ingestion loop (local to read thread, no extra lock)
- Every 64 processed frames triggers: I3D → YAMNet → Fusion
- YAMNet called with empty audio_path (placeholder; real audio extraction not yet implemented)
- VideoStatus extended with 8 new AI fields:
  ai_windows_analyzed, latest_video_label, latest_video_confidence,
  latest_audio_label, latest_audio_confidence,
  latest_final_label, latest_final_confidence, latest_ai_reason
- Window triggers every ~640 read frames (≈21 s at 30 fps)

### Phase 4 – Event Logic ✓ Done
- Created backend/app/services/event_logic_service.py — sole authority for suspicious event decisions
- Rule-based MVP, four event types in priority order: violence_suspected → person_fallen → crowd_gathering → normal
- Violence rule 1: I3D label=violence AND confidence ≥ 0.65
- Violence rule 2: I3D confidence ≥ 0.55 AND YAMNet aggressive_audio confidence ≥ 0.50 (weighted fusion 65/35)
- Person fallen rule: YOLO bounding box width/height ratio > 1.4 AND confidence ≥ 0.60
- Crowd rule: ≥ 5 persons detected simultaneously; confidence scales linearly (5→0.50, 10→1.00)
- Output: event_type, confidence, severity (low/medium/high), reason, requires_human_review=True
- DetectionEngine.detect_persons() upgraded: now returns list[dict] with full bounding box data
  (x1, y1, x2, y2, width, height, confidence) instead of bare int count
- video_ingestion.py updated: person_count = len(detections) from new return type

### Phase 5 – Event Logic Integration ✓ Done
- event_logic_service.evaluate() called inside _run_ai_window after fusion
- person_detections (last frame of window) passed into evaluate() for fallen/crowd rules
- test_event propagated: start() → thread args → _read_loop → _run_ai_window (no shared mutable state)
- test_event="violence" overrides I3D (confidence=0.85) and YAMNet (confidence=0.70) with forced values
- VideoStatus extended with 5 new event fields:
  latest_event_type, latest_event_confidence, latest_event_severity,
  latest_event_reason, requires_human_review
- StartRequest now accepts optional test_event field

### Phase 6 – Training Module ✓ Done (structure only, no training yet)
- backend/training/ module created, fully isolated from the Docker backend
- Dataset scaffold: datasets/rwf2000/{train,val,test}/{violence,no_violence}/ with .gitkeep files
- datasets/.gitignore excludes all video files from git
- scripts/train_i3d_violence.py: R3D-18 (I3D proxy) fine-tuning on Kinetics-400 weights
  - VideoClipDataset: OpenCV frame loader, random/centre temporal crop, Kinetics-400 normalisation
  - build_model(): torchvision r3d_18 with binary classification head
  - train_one_epoch() + validate() loops, cosine LR schedule, best checkpoint saving
- scripts/evaluate_i3d_violence.py: accuracy, precision, recall, F1, confusion matrix (sklearn)
- training/README.md: full step-by-step guide (download → validate → train → evaluate → deploy)
- Target: F1 ≥ 0.85 on test split before replacing I3DViolenceService placeholder

### Phase 6b – Real Violence Detection Service ✓ Done
- backend/app/services/real_violence_detection_service.py created
- Loads violence_baseline_model.pth from data/models/ (Docker bind-mount path)
- Graceful fallback: if checkpoint not found, returns {"error": "Violence model not trained yet"}
- ViolenceBaselineModel: ResNet-18 backbone + temporal mean-pool + binary head (identical arch to training script)
- _preprocess_frames(): samples NUM_FRAMES uniformly, resizes to 224×224, ImageNet normalisation
- I3DViolenceService placeholder preserved — RealViolenceDetectionService is the replacement once trained

### Phase 6c – Violence Baseline Training Scripts ✓ Done
- scripts/split_rwf2000.py: splits raw RWF-2000 (Fight/NonFight) into train/val/test (70/15/15, seed=42, copy not move)
- scripts/prepare_video_dataset.py: validates dataset structure, checks readability, prints per-split table
- scripts/train_violence_baseline.py: ResNet-18 baseline training, saves best F1 checkpoint
- scripts/evaluate_violence_baseline.py: accuracy/precision/recall/F1/confusion matrix + production target check
- backend/training/models/.gitkeep + .gitignore (excludes .pth from git)
- backend/data/models/.gitkeep (deploy path for Docker)
- backend/data/.gitignore updated: models/*.pth excluded

### Phase 6d – Real Model Inference ✓ Done
- Trained on RWF-2000 (ResNet-18 baseline): val F1=0.8557, test F1=0.8294 (epoch 13)
- Checkpoint deployed to backend/data/models/violence_baseline_model.pth (Docker bind-mount)
- video_ingestion.py: replaced i3d_violence_service with real_violence_detection_service in _run_ai_window
- Graceful fallback: if RealViolenceDetectionService returns "error" key, falls back to I3D placeholder
- YAMNet audio extraction not yet implemented (placeholder still used)

### Phase 7 – Auto Alerts
- Create alerts automatically when event_logic returns violence_suspected / person_fallen / crowd_gathering
- Save snapshot per alert
- Prevent duplicate alerts (cooldown window)

### Phase 8 – Real-Time
- WebSocket endpoint for live alert push

### Phase 9 – Frontend
- React dashboard
- Alerts view
- Camera status view

---

## 4. System Architecture

```
Video Source
    ↓
Video Ingestion (OpenCV, background thread)
    ↓
Frame Sampling (every Nth frame)
    ├── YOLO (optional)  →  person detections (bounding boxes + count)
    ├── I3D              →  video violence classification  ─┐
    └── YAMNet           →  audio aggression detection    ─┤
                                                            ↓
                                               Fusion Service
                                                            ↓
                                           Event Logic Service  ←── YOLO detections
                                                            ↓
                              violence_suspected / person_fallen / crowd_gathering / normal
                                                            ↓
                                          Alert Service (not yet)
                                                            ↓
                                          Database + Dashboard (not yet)
```

---

## 5. Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| Backend   | FastAPI, Python 3.11, SQLAlchemy  |
| Video     | OpenCV (headless)                 |
| AI        | YOLO yolov8n, I3D (placeholder), YAMNet (placeholder) |
| Database  | SQLite                            |
| Frontend  | React + Tailwind (planned)        |
| DevOps    | Docker, docker-compose            |

---

## 6. Key Design Decisions

- MVP first — simple, working system before optimization
- Rule-based detection before custom ML training
- No face recognition (ethical boundary)
- All alerts require human validation
- Modular architecture: services layer separate from API layer

---

## 7. Risks

- False positives in detection
- Performance under real-time video load
- Limited dataset for violence/aggression detection

---

## 8. Future Improvements

- Train custom model on domain-specific data
- Multi-camera support
- Cloud deployment
- Notification system (SMS / email)
- Analytics dashboard

---

## 9. Status Summary

| Area             | Status      |
|------------------|-------------|
| Backend          | Done        |
| Database         | Done        |
| Docker           | Done        |
| Health check     | Done        |
| Video ingestion  | Done        |
| Local file mount | Done        |
| Frame saving     | Done        |
| YOLO detection   | Done        |
| I3D service      | Interface   |
| YAMNet service   | Interface   |
| Fusion service   | Interface   |
| Event logic      | Done        |
| Event logic wired| Done        |
| Test mode        | Done        |
| Training scripts | Done        |
| Real detection svc | Done (model deployed, wired) |
| Auto alerts      | Not started |
| WebSocket        | Not started |
| Frontend         | Not started |
