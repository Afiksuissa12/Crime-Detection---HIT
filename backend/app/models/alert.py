from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from app.database.db import Base


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    event_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    snapshot_path = Column(String, nullable=True)
    status = Column(String, default="new", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
