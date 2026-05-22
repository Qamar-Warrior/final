#!/usr/bin/env python3
"""
Uzbekistan License Plate Recognition — CLI entrypoint.

Usage:
    python main.py image car.jpg [--show] [--no-save]
    python main.py video traffic.mp4 [--skip 3] [--no-save]
    python main.py serve [--host 0.0.0.0] [--port 8000]
    python main.py query [--search 01A123BC] [--limit 20]
"""

import argparse
import json
import logging
import sys

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _make_pipeline():
    from core.pipeline import Pipeline
    return Pipeline(
        model_path=config.MODEL_PATH,
        confidence=config.YOLO_CONFIDENCE,
        languages=config.OCR_LANGUAGES,
        gpu=config.USE_GPU,
        video_frame_skip=config.VIDEO_FRAME_SKIP,
        dedup_window_seconds=config.DEDUP_WINDOW_SECONDS,
    )


def _make_db():
    from db.manager import DatabaseManager
    return DatabaseManager(config.DB_PATH, images_dir=config.PLATE_IMAGES_DIR)



def _result_to_dict(result, record_id: int = -1) -> dict:
    return {
        "id": record_id,
        "plate_text": result.plate_text,
        "raw_text": result.raw_text,
        "is_valid": result.is_valid,
        "confidence": result.confidence,
        "bbox": result.bbox,
        "source": result.source,
        "detected_at": result.timestamp.isoformat() + "Z",
    }


# ------------------------------------------------------------------
# Subcommand handlers
# ------------------------------------------------------------------

def cmd_image(args):
    pipeline = _make_pipeline()
    db = _make_db() if not args.no_save else None

    results = pipeline.process_image(args.path)

    if not results:
        logger.info("No license plates detected.")

    for result in results:
        should_save = db and result.confidence >= config.MIN_SAVE_CONFIDENCE
        record_id = db.save_detection(result) if should_save else -1
        if should_save and result.is_valid:
            db.save_valid_plate(result)
        print(json.dumps(_result_to_dict(result, record_id)))

    if args.show:
        import cv2
        frame = cv2.imread(args.path)
        annotated = pipeline.detector.draw_boxes(
            frame,
            [{'bbox': r.bbox, 'confidence': r.confidence} for r in results],
            labels=[r.plate_text or r.raw_text for r in results],
        )
        cv2.imshow("License Plate Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if db:
        db.close()


def cmd_video(args):
    pipeline = _make_pipeline()
    db = _make_db() if not args.no_save else None

    if args.skip:
        pipeline.frame_skip = args.skip

    def on_result(result):
        should_save = db and result.confidence >= config.MIN_SAVE_CONFIDENCE
        record_id = db.save_detection(result) if should_save else -1
        if should_save and result.is_valid:
            db.save_valid_plate(result)
        print(json.dumps(_result_to_dict(result, record_id)), flush=True)

    total = pipeline.process_video(args.path, on_result=on_result)
    logger.info(f"Video processing complete. {total} unique plates detected.")

    if db:
        db.close()


def cmd_serve(args):
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def cmd_query(args):
    db = _make_db()

    if args.search:
        rows = db.search_plate(args.search)
        for row in rows:
            print(json.dumps(row))
    elif args.stats:
        print(json.dumps(db.get_stats(), indent=2))
    else:
        rows, total = db.get_recent(limit=args.limit, valid_only=args.valid_only)
        print(json.dumps({"total": total, "items": rows}, indent=2))

    db.close()


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Uzbekistan License Plate Recognition System"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # image subcommand
    p_image = sub.add_parser("image", help="Process a single image file")
    p_image.add_argument("path", help="Path to image file")
    p_image.add_argument("--show", action="store_true", help="Display annotated image")
    p_image.add_argument("--no-save", action="store_true", help="Skip saving to database")

    # video subcommand
    p_video = sub.add_parser("video", help="Process a video file")
    p_video.add_argument("path", help="Path to video file")
    p_video.add_argument("--skip", type=int, help="Process every Nth frame (default: 3)")
    p_video.add_argument("--no-save", action="store_true", help="Skip saving to database")

    # serve subcommand
    p_serve = sub.add_parser("serve", help="Start REST API server")
    p_serve.add_argument("--host", default=config.API_HOST)
    p_serve.add_argument("--port", type=int, default=config.API_PORT)

    # query subcommand
    p_query = sub.add_parser("query", help="Query the database")
    p_query.add_argument("--search", help="Search by plate text")
    p_query.add_argument("--stats", action="store_true", help="Show aggregate statistics")
    p_query.add_argument("--limit", type=int, default=20)
    p_query.add_argument("--valid-only", action="store_true")

    args = parser.parse_args()

    dispatch = {
        "image": cmd_image,
        "video": cmd_video,
        "serve": cmd_serve,
        "query": cmd_query,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
