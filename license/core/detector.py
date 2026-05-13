import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlateDetector:
    """
    Wraps YOLOv8 for license plate region detection.
    Loads model once; reuses across all frames.
    """

    def __init__(self, model_path: str, confidence: float = 0.45):
        from ultralytics import YOLO

        self.confidence = confidence
        model_file = Path(model_path)

        if model_file.exists():
            logger.info(f"Loading plate detection model from {model_path}")
            self.model = YOLO(str(model_file))
        else:
            # Fall back to YOLOv8n (COCO) — less accurate for plates but functional
            logger.warning(
                f"Custom model not found at {model_path}. "
                "Falling back to YOLOv8n with aspect-ratio filtering. "
                "For best accuracy, provide a plate-specific model."
            )
            self.model = YOLO("yolov8n.pt")
            self._fallback_mode = True
            return

        self._fallback_mode = False

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a BGR frame (as returned by cv2.imread / VideoCapture).

        Returns a list of dicts:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'crop': np.ndarray  # cropped plate region (BGR)
            }
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # In fallback mode, filter by plate-like aspect ratio (width > 2x height)
                if self._fallback_mode:
                    w, h = x2 - x1, y2 - y1
                    if h == 0 or (w / h) < 2.0 or (w / h) > 8.0:
                        continue
                    # Only accept detections that look like they could be a plate
                    # (small-ish bounding box relative to frame)
                    frame_area = frame.shape[0] * frame.shape[1]
                    box_area = w * h
                    if box_area / frame_area > 0.15:
                        continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'crop': crop,
                })

        return detections

    def draw_boxes(self, frame: np.ndarray, detections: list[dict],
                   labels: list[str] | None = None) -> np.ndarray:
        """Draw bounding boxes on frame. Returns annotated copy."""
        annotated = frame.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            label = labels[i] if labels else f"{det['confidence']:.2f}"
            color = (0, 255, 0)  # green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
        return annotated
