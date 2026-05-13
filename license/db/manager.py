import sqlite3
import logging
from pathlib import Path

from core.pipeline import DetectionResult

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class DatabaseManager:
    """
    Thin SQLite wrapper. All queries use parameterized statements.
    WAL mode is enabled for concurrent read/write (API + video ingestion).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
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

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_detection(self, result: DetectionResult) -> int:
        """Insert a detection result. Returns the new row id."""
        bbox = result.bbox if len(result.bbox) == 4 else [None, None, None, None]
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO detections
                    (source, plate_text, raw_text, is_valid, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.source,
                    result.plate_text,
                    result.raw_text,
                    int(result.is_valid),
                    result.confidence,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    result.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            return cursor.lastrowid

    def save_valid_plate(self, result: DetectionResult) -> int:
        """Insert a confirmed valid plate into the valid_plates table. Returns new row id."""
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO valid_plates (plate_text, confidence, source, detected_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    result.plate_text,
                    result.confidence,
                    result.source,
                    result.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        return {
            "total_detections": total,
            "valid_plates": valid,
            "invalid_plates": total - valid,
            "top_sources": [dict(r) for r in sources],
        }

    def close(self):
        self._conn.close()
