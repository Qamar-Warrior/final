CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,
    plate_text  TEXT    NOT NULL,
    raw_text    TEXT    NOT NULL,
    is_valid    INTEGER NOT NULL DEFAULT 0,
    plate_type  TEXT    NOT NULL DEFAULT 'unknown',
    confidence  REAL    NOT NULL,
    bbox_x1     INTEGER,
    bbox_y1     INTEGER,
    bbox_x2     INTEGER,
    bbox_y2     INTEGER,
    detected_at TEXT    NOT NULL DEFAULT (strftime('%d %m %Y', 'now')),
    image_path  TEXT
);

CREATE INDEX IF NOT EXISTS idx_plate_text   ON detections(plate_text);
CREATE INDEX IF NOT EXISTS idx_detected_at  ON detections(detected_at);
CREATE INDEX IF NOT EXISTS idx_is_valid     ON detections(is_valid);

CREATE TABLE IF NOT EXISTS valid_plates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_text  TEXT    NOT NULL,
    plate_type  TEXT    NOT NULL DEFAULT 'unknown',
    confidence  REAL    NOT NULL,
    source      TEXT    NOT NULL,
    detected_at TEXT    NOT NULL DEFAULT (strftime('%d %m %Y', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_vp_plate_text   ON valid_plates(plate_text);
CREATE INDEX IF NOT EXISTS idx_vp_detected_at  ON valid_plates(detected_at);
