from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database.db import get_db
from app.models.camera import Camera

router = APIRouter(tags=["cameras"])


class CameraCreate(BaseModel):
    name: str
    location: str
    stream_url: str
    is_active: bool = True


class CameraResponse(BaseModel):
    id: int
    name: str
    location: str
    stream_url: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/cameras", response_model=list[CameraResponse])
def list_cameras(db: Session = Depends(get_db)):
    return db.query(Camera).all()


@router.post("/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    camera = Camera(**payload.model_dump())
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera
