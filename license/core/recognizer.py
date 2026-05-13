import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Target height for plate crop before OCR (empirically optimal for EasyOCR)
_TARGET_HEIGHT = 128


class TextRecognizer:
    """
    Wraps EasyOCR for license plate text extraction.
    The reader is expensive to initialize — create once and reuse.
    """

    def __init__(self, languages: list[str], gpu: bool = False):
        import easyocr
        logger.info(f"Initializing EasyOCR (languages={languages}, gpu={gpu})")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self._warm_up()

    def _warm_up(self):
        """Run OCR once on a blank image to JIT-compile the model."""
        blank = np.ones((64, 200), dtype=np.uint8) * 255
        try:
            self.reader.readtext(blank)
        except Exception:
            pass

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Extract text from a cropped plate image.

        Args:
            crop: BGR numpy array of the plate region.

        Returns:
            (raw_text, confidence) where raw_text is all detected segments joined,
            and confidence is the mean of per-segment confidences.
        """
        preprocessed = self._preprocess(crop)
        results = self.reader.readtext(
            preprocessed, detail=1, paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        )

        if not results:
            return "", 0.0

        texts = []
        confidences = []
        for (_bbox, text, conf) in results:
            texts.append(text.strip())
            confidences.append(conf)

        raw_text = "".join(texts)
        mean_conf = sum(confidences) / len(confidences)
        return raw_text, round(mean_conf, 4)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Prepare crop for OCR:
        1. Grayscale
        2. Resize to fixed height (maintains aspect ratio)
        3. OTSU binarization (removes lighting variation)
        4. Slight sharpening
        """
        # Grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Resize to target height
        h, w = gray.shape
        if h == 0:
            return gray
        scale = _TARGET_HEIGHT / h
        new_w = max(1, int(w * scale))
        resized = cv2.resize(gray, (new_w, _TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

        # OTSU binarization
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(binary, -1, kernel)

        return sharpened
