import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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

        Tries multiple preprocessing strategies and returns the result with
        the highest mean OCR confidence, making reads more robust against
        lighting and binarization artifacts.

        Returns:
            (raw_text, confidence)
        """
        best_text, best_conf = "", 0.0

        for preprocessed in self._preprocess_variants(crop):
            raw_results = self.reader.readtext(
                preprocessed, detail=1, paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            )
            if not raw_results:
                continue

            text, conf = self._extract_dominant(raw_results)
            if conf > best_conf:
                best_conf, best_text = conf, text

        return best_text, best_conf

    def _extract_dominant(self, results: list) -> tuple[str, float]:
        """
        From a list of EasyOCR segments, keep only the dominant text line
        (tallest characters) and join left-to-right.
        """
        if len(results) > 1:
            heights = [abs(r[0][2][1] - r[0][0][1]) for r in results]
            max_h = max(heights)
            results = [r for r, h in zip(results, heights) if h >= max_h * 0.5]

        results.sort(key=lambda r: r[0][0][0])

        texts = [text.strip() for (_, text, _) in results]
        confidences = [conf for (_, _, conf) in results]
        return "".join(texts), round(sum(confidences) / len(confidences), 4)

    def _preprocess_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        """
        Return several preprocessed versions of the crop.
        The caller picks the result with the highest OCR confidence.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
        h, w = gray.shape
        if h == 0:
            return [gray]

        def _resize(target_h: int) -> np.ndarray:
            new_w = max(1, int(w * target_h / h))
            return cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

        # Strategy 1: Raw grayscale at 300 px — highest resolution without
        # binarization artifacts. EasyOCR's own internal contrast adjustment
        # handles well-lit plates and resolves digits that binarization blurs.
        variants = [_resize(300)]

        # Strategy 2: OTSU + sharpen at 128 px — reliable fallback for evenly
        # lit plates and low-resolution crops where raw gray is noisy.
        r128 = _resize(128)
        _, otsu = cv2.threshold(r128, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.filter2D(otsu, -1, sharpen))

        # Strategy 3: CLAHE then OTSU at 128 px — helps when part of the plate
        # is in shadow or overexposed.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(r128)
        _, otsu_clahe = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu_clahe)

        # Strategy 4: Adaptive threshold at 128 px — handles strongly uneven
        # illumination where global OTSU picks a bad threshold.
        adaptive = cv2.adaptiveThreshold(
            r128, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        variants.append(adaptive)

        return variants
