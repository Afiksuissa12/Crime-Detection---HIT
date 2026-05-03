FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download yolov8n.pt into the image so runtime has no network dependency
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY . .

RUN mkdir -p /app/data /app/static && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js', '/app/static/redoc.standalone.js')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
