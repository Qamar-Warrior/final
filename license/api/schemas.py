from datetime import datetime
from pydantic import BaseModel


class DetectionResponse(BaseModel):
    id: int
    source: str
    plate_text: str
    raw_text: str
    is_valid: bool
    confidence: float
    bbox_x1: int | None
    bbox_y1: int | None
    bbox_x2: int | None
    bbox_y2: int | None
    detected_at: str
    image_path: str | None = None


class DetectionListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[DetectionResponse]


class StatsResponse(BaseModel):
    total_detections: int
    valid_plates: int
    invalid_plates: int
    top_sources: list[dict]


class HealthResponse(BaseModel):
    status: str
    db_path: str
    model_loaded: bool
