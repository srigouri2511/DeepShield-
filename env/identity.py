import cv2
import numpy as np


class IdentityVerifier:
    """
    Compares two face images and returns a similarity score.
    Uses histogram correlation as a lightweight heuristic.
    For production, replace with a face embedding model (e.g. FaceNet, ArcFace).
    """

    MATCH_THRESHOLD = 0.8

    def verify(self, image_path_a: str, image_path_b: str) -> dict:
        img_a = self._load(image_path_a)
        img_b = self._load(image_path_b)
        score = self._compare(img_a, img_b)
        match = score >= self.MATCH_THRESHOLD
        return {
            "match": match,
            "similarity": round(score, 4),
            "confidence": round(score if match else 1.0 - score, 4),
            "label": "same_person" if match else "different_person",
        }

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.resize(img, (128, 128))

    def _compare(self, a: np.ndarray, b: np.ndarray) -> float:
        scores = []
        for i in range(3):
            hist_a = cv2.calcHist([a], [i], None, [256], [0, 256])
            hist_b = cv2.calcHist([b], [i], None, [256], [0, 256])
            scores.append(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
        histogram_similarity = float(np.mean(scores))
        pixel_similarity = 1.0 - float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)
        hybrid_similarity = 0.3 * histogram_similarity + 0.7 * pixel_similarity
        return float(np.clip(hybrid_similarity, 0.0, 1.0))
