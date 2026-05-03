# AI Crime Detection — HIT

Real-time violence detection system using video analysis and a deep learning pipeline.  
Built with FastAPI, OpenCV, YOLO, and a trained ResNet-18 violence classifier.

---

## What it does

- Accepts a video source (file, webcam, RTSP stream)
- Samples frames continuously in a background thread
- Runs YOLO person detection on every processed frame
- Every 64 frames triggers the AI violence detection pipeline:
  - **ViolenceBaselineModel** (ResNet-18 + temporal mean-pool, trained on RWF-2000)
  - Audio analysis placeholder (YAMNet interface ready)
  - Multimodal fusion (video 65% / audio 35%)
  - Event logic: `violence_suspected` / `person_fallen` / `crowd_gathering` / `normal`
- All results accessible via REST API and a built-in Test Console UI

---

## Stack

| Layer      | Technology                                          |
|------------|-----------------------------------------------------|
| Backend    | FastAPI, Python 3.11, SQLAlchemy, SQLite            |
| Video      | OpenCV headless                                     |
| Detection  | YOLO yolov8n (person detection)                     |
| AI Model   | ResNet-18 + temporal mean-pool (RWF-2000, F1 0.85)  |
| Container  | Docker, docker-compose                              |

---

## Quick Start

### Requirements
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- No Python installation needed on the host

### 1. Clone the repo

```bash
git clone https://github.com/Afiksuissa12/Crime-Detection---HIT.git
cd Crime-Detection---HIT
```

### 2. Start the backend

```bash
docker compose up --build
```

First build takes 2–3 minutes (downloads dependencies and YOLO model).  
Subsequent starts are fast (all layers cached).

### 3. Open the Test Console

```
http://localhost:8000/test-ui
```

---

## Test Console

The Test Console at `/test-ui` lets you run the full detection pipeline without writing any code.

### What you can do

| Button | Action |
|--------|--------|
| **Check Health** | Verify the backend is running |
| **Start Violence Test Video** | Run pipeline on `data/videos/test.mp4` |
| **Start No-Violence Test Video** | Run pipeline on `data/videos/no_violence.mp4` |
| **Stop Video** | Stop the currently running video |
| **Get Status** | Refresh AI results manually |
| **Get Alerts** | Load alerts from the database |
| **Upload Video** | Upload any `.mp4 / .avi / .mov / .mkv` file |
| **Run Uploaded Video Test** | Run the pipeline on your uploaded video |

### Upload and test your own video

1. Drag and drop a video file onto the upload area (or click to browse)
2. Click **Upload Video** — the file is saved to `data/videos/uploads/`
3. Click **Run Uploaded Video Test** — the pipeline starts automatically
4. Status auto-refreshes every 2 seconds until the video ends

### Result cards

- **Video Status** — `is_running`, `source`, `frames_read`, `frames_processed`, `ai_windows_analyzed`
- **Latest AI Result** — `video_label`, `video_confidence`, `final_label`, `final_confidence`
- **Latest Event Decision** — `event_type`, `severity`, `confidence`, `reason`, `requires_human_review`
- **Alerts** — all stored alerts with event type, confidence, and timestamp

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/test-ui` | Test Console UI |
| POST | `/api/video/start` | Start video processing |
| POST | `/api/video/stop` | Stop video processing |
| GET | `/api/video/status` | Get current AI status |
| POST | `/api/video/upload` | Upload a video file |
| GET | `/api/cameras` | List cameras |
| POST | `/api/cameras` | Add a camera |
| GET | `/api/alerts` | List alerts |
| POST | `/api/alerts` | Create an alert |

Full interactive docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Adding test videos

Place your videos in the bind-mounted folder — no Docker rebuild needed:

```
backend/data/videos/
├── test.mp4          ← violence test video
├── no_violence.mp4   ← non-violence test video
└── uploads/          ← uploaded via /test-ui
```

The folder is mounted into the container at `/app/data/videos/`.

---

## Project structure

```
.
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── data/                        # bind-mounted at /app/data
│   │   ├── videos/                  # input videos
│   │   ├── snapshots/               # saved frame snapshots
│   │   └── models/                  # trained model checkpoint (.pth)
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── video.py             # video endpoints + upload
│   │   │   ├── cameras.py
│   │   │   ├── alerts.py
│   │   │   └── ui.py                # Test Console HTML
│   │   ├── services/
│   │   │   ├── video_ingestion.py   # background capture thread
│   │   │   ├── real_violence_detection_service.py
│   │   │   ├── detection_engine.py  # YOLO
│   │   │   ├── event_logic_service.py
│   │   │   ├── multimodal_fusion_service.py
│   │   │   ├── i3d_violence_service.py   # placeholder
│   │   │   └── yamnet_audio_service.py   # placeholder
│   │   ├── models/
│   │   └── database/
│   └── training/                    # offline training module
│       ├── README.md
│       ├── datasets/rwf2000/
│       └── scripts/
│           ├── split_rwf2000.py
│           ├── prepare_video_dataset.py
│           ├── train_violence_baseline.py
│           └── evaluate_violence_baseline.py
```

---

## Violence model training

The trained checkpoint (`violence_baseline_model.pth`) is not included in the repo.  
To train your own model on RWF-2000, follow the guide in [`backend/training/README.md`](backend/training/README.md).

After training, copy the checkpoint to the deploy path:

```bash
cp backend/training/models/violence_baseline_model.pth backend/data/models/
```

No Docker rebuild needed — the folder is bind-mounted.

---

## Design decisions

- All alerts require **human review** — the system never takes automatic action
- No face recognition
- Rule-based event logic (MVP) — replaceable with learned models
- Modular services layer — each AI component is independently swappable
