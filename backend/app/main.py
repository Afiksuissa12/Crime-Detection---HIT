from fastapi import FastAPI
from app.config import APP_NAME
from app.database.db import Base, engine
from app.models import camera, alert  # noqa: F401 — register models before create_all
from app.api import cameras, alerts, video

Base.metadata.create_all(bind=engine)

app = FastAPI(title=APP_NAME, version="0.1.0")

app.include_router(cameras.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(video.router, prefix="/api")


@app.get("/")
def root():
    return {"status": "ok", "service": APP_NAME}


@app.get("/health")
def health():
    return {"status": "ok", "service": "crime-detection-backend"}
