import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pipeline import DetectionResult
from db.manager import DatabaseManager


def _make_result(plate="01A123BC", valid=True, conf=0.87, source="test.jpg"):
    return DetectionResult(
        source=source,
        raw_text=plate,
        plate_text=plate,
        is_valid=valid,
        confidence=conf,
        bbox=[10, 20, 110, 50],
        timestamp=datetime(2026, 4, 1, 12, 0, 0),
    )


def _make_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return DatabaseManager(path), path


# ------------------------------------------------------------------
# Basic CRUD
# ------------------------------------------------------------------

def test_save_and_retrieve():
    db, path = _make_db()
    try:
        result = _make_result()
        row_id = db.save_detection(result)
        assert row_id > 0

        row = db.get_by_id(row_id)
        assert row is not None
        assert row["plate_text"] == "01A123BC"
        assert row["is_valid"] == 1
        assert row["confidence"] == 0.87
        assert row["bbox_x1"] == 10
        assert row["bbox_y2"] == 50
    finally:
        db.close()
        os.unlink(path)


def test_get_recent_returns_latest_first():
    db, path = _make_db()
    try:
        db.save_detection(_make_result(plate="01A111AA", source="img1.jpg"))
        db.save_detection(_make_result(plate="02B222BB", source="img2.jpg"))

        rows, total = db.get_recent(limit=10)
        assert total == 2
        assert rows[0]["plate_text"] == "02B222BB"   # last inserted comes first
        assert rows[1]["plate_text"] == "01A111AA"
    finally:
        db.close()
        os.unlink(path)


def test_get_recent_valid_only_filter():
    db, path = _make_db()
    try:
        db.save_detection(_make_result(plate="01A111AA", valid=True))
        db.save_detection(_make_result(plate="INVALID__", valid=False))

        rows, total = db.get_recent(valid_only=True)
        assert total == 1
        assert rows[0]["plate_text"] == "01A111AA"
    finally:
        db.close()
        os.unlink(path)


def test_search_plate_exact():
    db, path = _make_db()
    try:
        db.save_detection(_make_result(plate="01A123BC"))
        db.save_detection(_make_result(plate="02B456DE"))

        results = db.search_plate("01A123BC")
        assert len(results) == 1
        assert results[0]["plate_text"] == "01A123BC"
    finally:
        db.close()
        os.unlink(path)


def test_search_plate_partial():
    db, path = _make_db()
    try:
        db.save_detection(_make_result(plate="01A123BC"))
        db.save_detection(_make_result(plate="01A456DE"))
        db.save_detection(_make_result(plate="02B789FG"))

        results = db.search_plate("01A")
        assert len(results) == 2
    finally:
        db.close()
        os.unlink(path)


def test_search_case_insensitive():
    db, path = _make_db()
    try:
        db.save_detection(_make_result(plate="01A123BC"))
        results = db.search_plate("01a123bc")
        assert len(results) == 1
    finally:
        db.close()
        os.unlink(path)


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

def test_get_stats():
    db, path = _make_db()
    try:
        db.save_detection(_make_result(plate="01A111AA", valid=True, source="cam1.mp4"))
        db.save_detection(_make_result(plate="02B222BB", valid=True, source="cam1.mp4"))
        db.save_detection(_make_result(plate="GARBAGE_", valid=False, source="cam2.mp4"))

        stats = db.get_stats()
        assert stats["total_detections"] == 3
        assert stats["valid_plates"] == 2
        assert stats["invalid_plates"] == 1
        assert len(stats["top_sources"]) == 2
    finally:
        db.close()
        os.unlink(path)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
