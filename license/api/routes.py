import io
import json
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from api.schemas import (
    DetectionListResponse,
    DetectionResponse,
    HealthResponse,
    StatsResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _pipeline(request: Request):
    return request.app.state.pipeline


def _db(request: Request):
    return request.app.state.db


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    return {
        "status": "ok",
        "db_path": request.app.state.db.db_path,
        "model_loaded": True,
    }


# ------------------------------------------------------------------
# Detection endpoints
# ------------------------------------------------------------------

@router.post("/detect/image", response_model=list[DetectionResponse])
async def detect_image(
    file: UploadFile = File(...),
    save: bool = Query(True, description="Save results to database"),
    pipeline=Depends(_pipeline),
    db=Depends(_db),
):
    """Upload an image; returns all detected license plates."""
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    results = pipeline.process_frame(frame, source=file.filename or "upload")

    response_items = []
    for result in results:
        record_id = db.save_detection(result) if save else -1
        response_items.append({
            "id": record_id,
            "source": result.source,
            "plate_text": result.plate_text,
            "raw_text": result.raw_text,
            "is_valid": result.is_valid,
            "confidence": result.confidence,
            "bbox_x1": result.bbox[0] if result.bbox else None,
            "bbox_y1": result.bbox[1] if result.bbox else None,
            "bbox_x2": result.bbox[2] if result.bbox else None,
            "bbox_y2": result.bbox[3] if result.bbox else None,
            "detected_at": result.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    return response_items


@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    save: bool = Query(True, description="Save results to database"),
    pipeline=Depends(_pipeline),
    db=Depends(_db),
):
    """
    Upload a video; streams NDJSON results as plates are detected.
    Each line is a JSON object matching DetectionResponse.
    """
    data = await file.read()

    async def generate():
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "video.mp4").suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()

            def on_result(result):
                record_id = db.save_detection(result) if save else -1
                payload = {
                    "id": record_id,
                    "source": result.source,
                    "plate_text": result.plate_text,
                    "raw_text": result.raw_text,
                    "is_valid": result.is_valid,
                    "confidence": result.confidence,
                    "bbox_x1": result.bbox[0] if result.bbox else None,
                    "bbox_y1": result.bbox[1] if result.bbox else None,
                    "bbox_x2": result.bbox[2] if result.bbox else None,
                    "bbox_y2": result.bbox[3] if result.bbox else None,
                    "detected_at": result.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                return json.dumps(payload) + "\n"

            results_buffer = []

            def collect(result):
                results_buffer.append(on_result(result))

            pipeline.process_video(tmp.name, on_result=collect)
            for line in results_buffer:
                yield line

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ------------------------------------------------------------------
# Query endpoints
# ------------------------------------------------------------------

@router.get("/detections", response_model=DetectionListResponse)
def list_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    valid_only: bool = Query(False),
    db=Depends(_db),
):
    offset = (page - 1) * page_size
    rows, total = db.get_recent(limit=page_size, offset=offset, valid_only=valid_only)
    return {"total": total, "page": page, "page_size": page_size, "items": rows}


@router.get("/detections/search", response_model=list[DetectionResponse])
def search_detections(
    q: str = Query(..., min_length=1, description="Plate text to search"),
    db=Depends(_db),
):
    return db.search_plate(q)


@router.get("/detections/{record_id}", response_model=DetectionResponse)
def get_detection(record_id: int, db=Depends(_db)):
    row = db.get_by_id(record_id)
    if not row:
        raise HTTPException(status_code=404, detail="Detection not found")
    return row


@router.get("/stats", response_model=StatsResponse)
def get_stats(db=Depends(_db)):
    return db.get_stats()
