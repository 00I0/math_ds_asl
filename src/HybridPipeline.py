import os
import pickle
import warnings
from typing import Tuple, Any, Dict, List
import numpy as np
import mediapipe as mp
from PIL import Image
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from BasePipeline import BasePipeline
from utils import get_image_paths_and_labels, compute_metrics


class MPFeatureExtractor(TransformerMixin):
    """
    Transformer that extracts 64D hand landmarks using MediaPipe.
    """

    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1,
                                              min_tracking_confidence=0.1, model_complexity=0)

    def fit(self, X: Any, y: Any = None) -> 'MPFeatureExtractor':
        return self

    def transform(self, filepaths: list[str]) -> np.ndarray:
        features: list[np.ndarray] = []
        for path in tqdm(filepaths, desc="Extracting hand landmarks"):
            img = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
            res = self.hands.process(img)
            if not res.multi_hand_landmarks:
                features.append(np.zeros(64, dtype=np.float32))
            else:
                lm = res.multi_hand_landmarks[0]
                pts = np.array(
                    [[p.x * img.shape[1], p.y * img.shape[0], p.z]
                     for p in lm.landmark],
                    dtype=np.float32
                )
                x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                bw, bh = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
                norm_pts: list[float] = []
                for x_px, y_px, z in pts:
                    norm_pts.extend([(x_px - x1) / bw, (y_px - y1) / bh, z])
                conf = float(res.multi_handedness[0].classification[0].score)
                features.append(np.array(norm_pts + [conf], dtype=np.float32))
        return np.vstack(features)


class HybridPipeline(BasePipeline):
    """
    Pipeline combining MediaPipe feature extraction with RandomForest.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = SklearnPipeline(steps=[
            ('mp', MPFeatureExtractor()),
            ('clf', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=2,
                                           n_jobs=os.cpu_count() - 1 or 1)),
        ])

    def fit(self, paths: List[str], labels: np.ndarray) -> None:
        self._pipeline.fit(paths, labels)
        self._is_fitted = True

    def predict(self, paths: List[str]) -> np.ndarray:
        if not self.is_fitted: warnings.warn("Not fitted yet")
        return self._pipeline.predict(paths)

    def save(self, path: str) -> None:
        # Extract the classifier step
        clf = self._pipeline.named_steps['clf']
        with open(path, 'wb') as f:
            pickle.dump(clf, f)

    def load(self, path: str) -> 'HybridPipeline':
        with open(path, 'rb') as f:
            clf = pickle.load(f)

        self._pipeline = SklearnPipeline(steps=[
            ('mp', MPFeatureExtractor()),
            ('clf', clf),
        ])
        self._is_fitted = True
        return self
