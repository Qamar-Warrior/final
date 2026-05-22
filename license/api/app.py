import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.routes import router
from core.pipeline import Pipeline
from db.manager import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Uzbekistan License Plate Recognition API",
    description="Detects and stores Uzbekistan vehicle license plates from images and video.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    app.state.pipeline = Pipeline(
        model_path=config.MODEL_PATH,
        confidence=config.YOLO_CONFIDENCE,
        languages=config.OCR_LANGUAGES,
        gpu=config.USE_GPU,
        video_frame_skip=config.VIDEO_FRAME_SKIP,
        dedup_window_seconds=config.DEDUP_WINDOW_SECONDS,
    )
    app.state.db = DatabaseManager(config.DB_PATH, images_dir=config.PLATE_IMAGES_DIR)


@app.on_event("shutdown")
def shutdown():
    app.state.db.close()


app.include_router(router)
