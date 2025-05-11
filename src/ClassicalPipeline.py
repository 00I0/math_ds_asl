import os
import pickle
import warnings
from typing import Tuple, Any, Dict, List
import numpy as np
from PIL import Image
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from BasePipeline import BasePipeline
from src.utils import get_image_paths_and_labels, compute_metrics


class ImageLoader(TransformerMixin):
    """
    Load image files, convert to grayscale, resize to a fixed size,
    normalize to [0,1], and flatten to 1D arrays.
    """

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def fit(self, x: Any, y: Any = None) -> 'ImageLoader':
        return self

    def transform(self, paths: List[str]) -> np.ndarray:
        arrays: List[np.ndarray] = []
        for path in paths:
            img = Image.open(path).convert('L').resize(self.size)
            arrays.append(np.asarray(img, dtype=np.float32).ravel())
        return np.vstack(arrays)


class ClassicalPipeline(BasePipeline):
    """
    Classical ML pipeline: image loading, PCA on [0,1]-normalized pixels,
    feature scaling, and RandomForest classification.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = SklearnPipeline(steps=[
            ('loader', ImageLoader(size=(64, 64))),
            ('pca', PCA(n_components=20, random_state=42)),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, max_features=0.3,
                                           min_samples_split=2, n_jobs=os.cpu_count() - 1 or 1)),
        ])

    def fit(self, paths: List[str], labels: np.ndarray) -> None:
        self._pipeline.fit(paths, labels)
        self._is_fitted = True

    def predict(self, paths: List[str]) -> np.ndarray:
        if not self.is_fitted: warnings.warn("Not fitted yet")
        return self._pipeline.predict(paths)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self._pipeline, f)

    def load(self, path: str) -> 'ClassicalPipeline':
        with open(path, 'rb') as f:
            self._pipeline = pickle.load(f)
        self._is_fitted = True
        return self
