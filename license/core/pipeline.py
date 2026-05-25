import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import cv2
import numpy as np

from core.detector import PlateDetector
from core.recognizer import TextRecognizer
from core.validator import PlateType, validate_plate

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    source: str
    raw_text: str
    plate_text: str
    is_valid: bool
    plate_type: PlateType
    confidence: float         # combined YOLO + OCR confidence
    bbox: list[int]           # [x1, y1, x2, y2]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    frame: np.ndarray | None = field(default=None, repr=False)  # full BGR frame


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
        Process a video file. For each unique valid plate, emits only the
        single highest-confidence detection across the entire video.
        Returns total number of detections emitted.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # plate_text -> best DetectionResult seen so far
        best: dict[str, DetectionResult] = {}
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
                for result in self._process_frame(frame, source=source):
                    if not result.is_valid:
                        continue
                    prev = best.get(result.plate_text)
                    if prev is None or result.confidence > prev.confidence:
                        best[result.plate_text] = result
        finally:
            cap.release()

        for result in best.values():
            on_result(result)

        return len(best)

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

            is_valid, plate_text, plate_type = validate_plate(raw_text)

            # Combined confidence: geometric mean of YOLO and OCR scores
            combined_conf = round((det['confidence'] * ocr_conf) ** 0.5, 4)

            results.append(DetectionResult(
                source=source,
                raw_text=raw_text,
                plate_text=plate_text,
                is_valid=is_valid,
                plate_type=plate_type,
                confidence=combined_conf,
                bbox=det['bbox'],
                frame=frame,
            ))

        return results

