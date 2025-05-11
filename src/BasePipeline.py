from abc import abstractmethod, ABC
from typing import Any, Dict, List

import numpy as np

from utils import compute_metrics


class BasePipeline(ABC):
    """
    Abstract base class defining the pipeline interface.
    """

    def __init__(self) -> None:
        # flag to indicate whether a pipeline is trained or loaded
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, paths: List[str], labels: Any) -> None:
        """
        Fit the pipeline on provided filepaths and labels.
        Args:
            paths: List of image filepaths.
            labels: Corresponding labels (numpy array).
        """
        pass

    @abstractmethod
    def predict(self, paths: List[str]) -> np.ndarray:
        """
        Return predicted labels for given filepaths.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained pipeline or model to disk.

        Args:
            path: File path for saving.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> 'BasePipeline':
        """
        Load a pipeline or model from disk.

        Args:
            path: File path from which to load.

        Returns:
            The pipeline instance.
        """
        pass

    def plot_pipeline(self) -> Any:
        """
        Plot the underlying sklearn pipeline as a diagram (if available).

        Returns:
             The pipeline object for display in Jupyter.
        """
        try:
            from sklearn import set_config
            set_config(display='diagram')
            # noinspection PyUnresolvedReferences
            return self._pipeline
        except Exception:
            raise NotImplementedError("plot_pipeline is only supported for sklearn-based pipelines.")

    def evaluate(self, paths: List[str], labels: np.ndarray) -> Dict[str, Any]:
        """
        Predict on `paths` and compare against `labels`, returning metrics.
        """
        preds = self.predict(paths)
        return compute_metrics(labels, preds)
