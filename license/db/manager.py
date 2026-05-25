import sqlite3
import logging
from pathlib import Path

import cv2

from core.pipeline import DetectionResult

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class DatabaseManager:
    """
    Thin SQLite wrapper. All queries use parameterized statements.
    WAL mode is enabled for concurrent read/write (API + video ingestion).
    """

    def __init__(self, db_path: str, images_dir: str | None = None):
        self.db_path = db_path
        self.images_dir = Path(images_dir) if images_dir else None
        if self.images_dir:
            self.images_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,  # needed for FastAPI async context
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info(f"Database ready at {db_path}")

    def _init_db(self):
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            schema = _SCHEMA_PATH.read_text()
            self._conn.executescript(schema)
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(detections)")}
            if "image_path" not in cols:
                self._conn.execute("ALTER TABLE detections ADD COLUMN image_path TEXT")
            if "plate_type" not in cols:
                self._conn.execute("ALTER TABLE detections ADD COLUMN plate_type TEXT NOT NULL DEFAULT 'unknown'")
            vp_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(valid_plates)")}
            if "plate_type" not in vp_cols:
                self._conn.execute("ALTER TABLE valid_plates ADD COLUMN plate_type TEXT NOT NULL DEFAULT 'unknown'")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_detection(self, result: DetectionResult) -> int:
        """Insert a detection only if a frame image is saved. Returns row id or -1."""
        image_path = self._save_frame(result)
        if image_path is None:
            return -1

        bbox = result.bbox if len(result.bbox) == 4 else [None, None, None, None]
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO detections
                    (source, plate_text, raw_text, is_valid, plate_type, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2, detected_at, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.source,
                    result.plate_text,
                    result.raw_text,
                    int(result.is_valid),
                    result.plate_type.value,
                    result.confidence,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    result.timestamp.strftime("%d %B %Y %H:%M:%S"),
                    image_path,
                ),
            )
            return cursor.lastrowid

    def _save_frame(self, result: DetectionResult) -> str | None:
        """Save the full frame to disk named after the license plate. Returns the file path or None."""
        if self.images_dir is None or result.frame is None or result.frame.size == 0:
            return None
        plate = result.plate_text.strip() if result.plate_text else "unknown"
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{plate}_{timestamp}.jpg"
        filepath = self.images_dir / filename
        ok = cv2.imwrite(str(filepath), result.frame)
        if not ok:
            logger.warning(f"Failed to write frame image to {filepath}")
            return None
        return str(filepath)

    def save_valid_plate(self, result: DetectionResult) -> int:
        """Insert a confirmed valid plate into the valid_plates table. Returns new row id."""
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO valid_plates (plate_text, plate_type, confidence, source, detected_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.plate_text,
                    result.plate_type.value,
                    result.confidence,
                    result.source,
                    result.timestamp.strftime("%d %B %Y %H:%M:%S"),
                ),
            )
            return cursor.lastrowid

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, record_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM detections WHERE id = ?", (record_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_recent(self, limit: int = 50, offset: int = 0,
                   valid_only: bool = False) -> tuple[list[dict], int]:
        """Returns (rows, total_count)."""
        where = "WHERE is_valid = 1" if valid_only else ""
        total = self._conn.execute(
            f"SELECT COUNT(*) FROM detections {where}"
        ).fetchone()[0]
        rows = self._conn.execute(
            f"SELECT * FROM detections {where} "
            f"ORDER BY detected_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows], total

    def search_plate(self, plate_text: str) -> list[dict]:
        """Exact or partial match on plate_text (case-insensitive)."""
        pattern = f"%{plate_text.upper()}%"
        rows = self._conn.execute(
            "SELECT * FROM detections WHERE UPPER(plate_text) LIKE ? "
            "ORDER BY detected_at DESC LIMIT 100",
            (pattern,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        valid = self._conn.execute(
            "SELECT COUNT(*) FROM detections WHERE is_valid = 1"
        ).fetchone()[0]
        sources = self._conn.execute(
            "SELECT source, COUNT(*) as cnt FROM detections "
            "GROUP BY source ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        by_type = self._conn.execute(
            "SELECT plate_type, COUNT(*) as cnt FROM detections "
            "WHERE is_valid = 1 GROUP BY plate_type"
        ).fetchall()
        return {
            "total_detections": total,
            "valid_plates": valid,
            "invalid_plates": total - valid,
            "by_plate_type": {row["plate_type"]: row["cnt"] for row in by_type},
            "top_sources": [dict(r) for r in sources],
        }

    def close(self):
        self._conn.close()
