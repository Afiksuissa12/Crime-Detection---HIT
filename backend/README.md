# Crime Detection API — Backend

FastAPI backend with SQLite, served on port **8000**.

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

---

## Run with Docker

### 1. Build and start

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### 2. Run in the background

```bash
docker compose up --build -d
```

### 3. Stop

```bash
docker compose down
```

The `backend/data/` folder is bind-mounted into the container at `/app/data`.
This means both the SQLite database and video files persist on your local disk.

---

## Verify the service is up

```bash
# Health check
open http://localhost:8000/health

# Interactive API docs (Swagger UI)
open http://localhost:8000/docs
```

On Windows, replace `open` with `start`:

```bash
start http://localhost:8000/health
start http://localhost:8000/docs
```

---

## Endpoints

| Method | Path               | Description              |
|--------|--------------------|--------------------------|
| GET    | `/health`          | Health check             |
| GET    | `/api/cameras`     | List all cameras         |
| POST   | `/api/cameras`     | Create a camera          |
| GET    | `/api/alerts`      | List all alerts          |
| POST   | `/api/alerts`      | Create an alert          |
| POST   | `/api/video/start` | Start video ingestion    |
| POST   | `/api/video/stop`  | Stop video ingestion     |
| GET    | `/api/video/status`| Get ingestion status     |

### POST /api/cameras — example body

```json
{
  "name": "Front Entrance",
  "location": "Building A",
  "stream_url": "rtsp://192.168.1.10/stream",
  "is_active": true
}
```

### POST /api/alerts — example body

```json
{
  "camera_id": 1,
  "event_type": "intrusion",
  "confidence": 0.92,
  "snapshot_path": "/snapshots/frame_001.jpg",
  "status": "new"
}
```

---

## Testing video ingestion with a local file

### 1. Place a test video in the mounted folder

Manually copy any `.mp4` file to:

```
backend/data/videos/test.mp4
```

The entire `backend/data/` directory is bind-mounted into the container at `/app/data`.
Any file you put here is immediately accessible inside Docker — no rebuild needed.

### 2. Start Docker

```bash
docker compose up --build
```

### 3. Start video ingestion

```bash
curl -X POST http://localhost:8000/api/video/start \
  -H "Content-Type: application/json" \
  -d '{"source": "data/videos/test.mp4"}'
```

### 4. Check status

```bash
curl http://localhost:8000/api/video/status
```

### 5. Stop

```bash
curl -X POST http://localhost:8000/api/video/stop
```

---

## Video ingestion

### Start — webcam

```bash
curl -X POST http://localhost:8000/api/video/start \
  -H "Content-Type: application/json" \
  -d '{"source": "0"}'
```

### Start — local file

```bash
curl -X POST http://localhost:8000/api/video/start \
  -H "Content-Type: application/json" \
  -d '{"source": "data/videos/test.mp4"}'
```

### Start — RTSP stream

```bash
curl -X POST http://localhost:8000/api/video/start \
  -H "Content-Type: application/json" \
  -d '{"source": "rtsp://192.168.1.10/stream"}'
```

### Stop

```bash
curl -X POST http://localhost:8000/api/video/stop
```

### Status

```bash
curl http://localhost:8000/api/video/status
```

Example response:

```json
{
  "is_running": true,
  "source": "0",
  "frames_read": 150,
  "frames_processed": 15,
  "last_error": null
}
```

---

## AI Architecture

The system uses a **multimodal detection pipeline** to identify violence and
suspicious crime-related events:

| Model | Role | Input | Status |
|-------|------|-------|--------|
| **I3D** | Primary violence detector | Video clip (temporal window of frames) | Placeholder |
| **YAMNet** | Supporting audio detector | Short audio segment (WAV/MP3) | Placeholder |
| **YOLO** | Optional person context | Single frame | Active (yolov8n) |

### I3D — Inflated 3D ConvNet
The primary model. Analyses a short temporal window of consecutive frames and
classifies the clip as `violence` or `no_violence`. I3D captures both spatial
appearance and temporal motion patterns, making it the best fit for
action-level violence detection.

### YAMNet — Audio Event Classifier
Supporting model. Classifies a short audio segment against 521 AudioSet sound
classes and flags events such as screaming, shouting, gunshots, breaking glass,
or alarms as `aggressive_audio`. Provides a complementary signal when visual
cues alone are ambiguous.

### YOLO (yolov8n)
Optional supporting context only. Detects the number of persons in a frame.
Person count can feed into event logic (e.g. crowd aggression) but is **not**
the primary violence signal.

### Fusion
`MultimodalFusionService` combines I3D and YAMNet results into a single
`violence_suspected` / `normal` decision using weighted confidence scores
(I3D: 65 %, YAMNet: 35 %). A high-confidence result from either modality alone
can also trigger a positive independently.

---

## Project structure

```
backend/
  app/
    main.py          # FastAPI app + table creation
    config.py        # Environment-based settings
    database/
      db.py          # Engine, session, Base
    models/
      camera.py      # Camera ORM model
      alert.py       # Alert ORM model
    api/
      cameras.py     # Camera routes + Pydantic schemas
      alerts.py      # Alert routes + Pydantic schemas
  requirements.txt
  Dockerfile
docker-compose.yml
```
