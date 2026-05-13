"""
Synthetic pipeline test — does not require a real YOLO model.
Creates a synthetic plate image and tests the recognizer + validator path.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_synthetic_plate(text: str = "01A123BC", width: int = 300, height: int = 80) -> np.ndarray:
    """Generate a white plate image with black text — simulates a crop from detector."""
    import cv2
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    cv2.rectangle(img, (2, 2), (width - 2, height - 2), (0, 0, 0), 2)
    font_scale = height / 60
    thickness = max(1, int(height / 30))
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return img


def test_recognizer_preprocesses_without_crash():
    """Recognizer preprocessing should not raise on valid input."""
    from core.recognizer import TextRecognizer
    rec = TextRecognizer.__new__(TextRecognizer)
    crop = make_synthetic_plate("01A123BC")
    preprocessed = rec._preprocess(crop)
    assert preprocessed is not None
    assert preprocessed.shape[0] == 64  # target height


def test_validator_catches_valid_ocr_output():
    """Validator should accept clean OCR output for valid plates."""
    from core.validator import validate_plate
    ok, text = validate_plate("01A123BC")
    assert ok is True
    assert text == "01A123BC"


def test_synthetic_plate_image_creation():
    """Ensure synthetic plate image has correct shape."""
    plate = make_synthetic_plate("14K789MN", width=300, height=80)
    assert plate.shape == (80, 300, 3)
    assert plate.dtype == np.uint8


def test_detection_result_dataclass():
    """DetectionResult fields are accessible correctly."""
    from datetime import datetime
    from core.pipeline import DetectionResult
    r = DetectionResult(
        source="test.jpg",
        raw_text="01A123BC",
        plate_text="01A123BC",
        is_valid=True,
        confidence=0.91,
        bbox=[0, 0, 200, 60],
        timestamp=datetime(2026, 4, 1),
    )
    assert r.is_valid is True
    assert r.confidence == 0.91
    assert len(r.bbox) == 4


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
