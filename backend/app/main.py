from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.config import APP_NAME
from app.database.db import Base, engine
from app.models import camera, alert  # noqa: F401 — register models before create_all
from app.api import cameras, alerts, video, ui

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Crime Detection API",
    description="Real-time violence detection system using video and audio analysis",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url=None,  # served below with a locally bundled JS to avoid CDN issues
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/redoc", include_in_schema=False)
async def redoc() -> HTMLResponse:
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="AI Crime Detection API",
        redoc_js_url="/static/redoc.standalone.js",
    )

app.include_router(cameras.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(video.router, prefix="/api")
app.include_router(ui.router)


@app.get("/")
def root():
    return {"status": "ok", "service": APP_NAME}


@app.get("/health")
def health():
    return {"status": "ok", "service": "crime-detection-backend"}
