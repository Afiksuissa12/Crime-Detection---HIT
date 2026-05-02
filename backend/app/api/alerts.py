from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database.db import get_db
from app.models.alert import Alert
from app.models.camera import Camera

router = APIRouter(tags=["alerts"])


class AlertCreate(BaseModel):
    camera_id: int
    event_type: str
    confidence: float
    snapshot_path: Optional[str] = None
    status: str = "new"


class AlertResponse(BaseModel):
    id: int
    camera_id: int
    event_type: str
    confidence: float
    snapshot_path: Optional[str]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/alerts", response_model=list[AlertResponse])
def list_alerts(db: Session = Depends(get_db)):
    return db.query(Alert).all()


@router.post("/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
def create_alert(payload: AlertCreate, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == payload.camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {payload.camera_id} not found",
        )
    alert = Alert(**payload.model_dump())
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert
