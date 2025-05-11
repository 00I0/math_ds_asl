import pickle
import warnings
from typing import List, Dict

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline as SklearnPipeline

import numpy as np

from BasePipeline import BasePipeline
from ClassicalPipeline import ImageLoader
from HybridPipeline import MPFeatureExtractor
from src.utils import get_image_paths_and_labels, compute_metrics


class CombinedPipeline(BasePipeline):
    """
    Pipeline combining PCA-reduced pixel features with hand landmarks.
    """

    def __init__(self) -> None:
        super().__init__()
        pca_branch = SklearnPipeline(steps=[
            ('loader', ImageLoader(size=(64, 64))),
            ('pca', PCA(n_components=20, random_state=42)),
            ('scaler_pix', StandardScaler()),
        ])
        landmark_branch = MPFeatureExtractor()
        features = FeatureUnion(transformer_list=[
            ('pca', pca_branch),
            ('landmarks', landmark_branch),
        ])

        self._pipeline = SklearnPipeline(steps=[
            ('features', features),
            ('clf', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, max_features=0.3,
                                           min_samples_split=2, n_jobs=-1)),
        ])

    def fit(self, paths: List[str], labels: np.ndarray) -> None:
        self._pipeline.fit(paths, labels)
        self._is_fitted = True

    def predict(self, paths: List[str]) -> np.ndarray:
        if not self.is_fitted: warnings.warn("Not fitted yet")
        return self._pipeline.predict(paths)

    def save(self, path: str) -> None:
        feat_union: FeatureUnion = self._pipeline.named_steps['features']
        pca_pipeline = feat_union.transformer_list[0][1]
        clf = self._pipeline.named_steps['clf']
        to_dump = {'pca': pca_pipeline, 'clf': clf}
        with open(path, 'wb') as f:
            pickle.dump(to_dump, f)

    def load(self, path: str) -> 'CombinedPipeline':
        with open(path, 'rb') as f:
            data: Dict[str, object] = pickle.load(f)
        pca_pipeline = data['pca']
        clf = data['clf']

        # rebuild feature union with fresh extractor
        features = FeatureUnion(transformer_list=[
            ('pca', pca_pipeline),
            ('landmarks', MPFeatureExtractor()),
        ])
        self._pipeline = SklearnPipeline([
            ('features', features),
            ('clf', clf),
        ])
        self._is_fitted = True
        return self
