import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import cv2
import numpy as np

from core.detector import PlateDetector
from core.recognizer import TextRecognizer
from core.validator import validate_plate

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    source: str
    raw_text: str
    plate_text: str
    is_valid: bool
    confidence: float         # combined YOLO + OCR confidence
    bbox: list[int]           # [x1, y1, x2, y2]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Pipeline:
    """
    Orchestrates detection → recognition → validation.
    Initialize once; call process_image() or process_video() as needed.
    """

    def __init__(self, model_path: str, confidence: float,
                 languages: list[str], gpu: bool,
                 video_frame_skip: int = 3,
                 dedup_window_seconds: int = 2):
        self.detector = PlateDetector(model_path, confidence)
        self.recognizer = TextRecognizer(languages, gpu)
        self.frame_skip = video_frame_skip
        self.dedup_window = dedup_window_seconds
        # plate_text -> last_seen datetime (for video deduplication)
        self._dedup_cache: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self, image_path: str) -> list[DetectionResult]:
        """Process a single image file. Returns all detections found."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self._process_frame(frame, source=image_path)

    def process_frame(self, frame: np.ndarray, source: str = "frame") -> list[DetectionResult]:
        """Process a raw BGR numpy frame directly."""
        return self._process_frame(frame, source=source)

    def process_video(
        self,
        video_path: str,
        on_result: Callable[[DetectionResult], None],
    ) -> int:
        """
        Process a video file. Calls on_result() for each detection found.
        Returns total number of detections emitted.

        Uses frame skipping and deduplication to avoid redundant writes.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = 0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % self.frame_skip != 0:
                    continue

                source = f"video:{video_path}:frame_{frame_idx}"
                results = self._process_frame(frame, source=source)

                for result in results:
                    if self._is_duplicate(result):
                        continue
                    self._update_dedup(result)
                    on_result(result)
                    total += 1
        finally:
            cap.release()

        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, source: str) -> list[DetectionResult]:
        detections = self.detector.detect(frame)
        results = []

        for det in detections:
            raw_text, ocr_conf = self.recognizer.recognize(det['crop'])

            if not raw_text:
                continue

            is_valid, plate_text = validate_plate(raw_text)

            # Combined confidence: geometric mean of YOLO and OCR scores
            combined_conf = round((det['confidence'] * ocr_conf) ** 0.5, 4)

            results.append(DetectionResult(
                source=source,
                raw_text=raw_text,
                plate_text=plate_text,
                is_valid=is_valid,
                confidence=combined_conf,
                bbox=det['bbox'],
            ))

        return results

    def _is_duplicate(self, result: DetectionResult) -> bool:
        if result.plate_text not in self._dedup_cache:
            return False
        delta = (result.timestamp - self._dedup_cache[result.plate_text]).total_seconds()
        return delta < self.dedup_window

    def _update_dedup(self, result: DetectionResult) -> None:
        self._dedup_cache[result.plate_text] = result.timestamp
        # Prune old entries to prevent unbounded growth
        if len(self._dedup_cache) > 1000:
            cutoff = result.timestamp.timestamp() - self.dedup_window * 10
            self._dedup_cache = {
                k: v for k, v in self._dedup_cache.items()
                if v.timestamp() > cutoff
            }
