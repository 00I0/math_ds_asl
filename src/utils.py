from typing import Tuple, List, Dict

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def get_image_paths_and_labels(root_dir: str) -> Tuple[List[str], np.ndarray]:
    """
    Collect image file paths and integer labels from a directory structured with one subdirectory per class.

    Args:
        root_dir: Path to root directory containing class subfolders.

    Returns:
        A tuple of (paths, labels) where:
        - paths: list of file paths
        - labels: numpy array of integer labels corresponding to subfolder index
    """
    paths = []
    labels = []
    classes = sorted(os.listdir(root_dir))
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for file_name in os.listdir(cls_dir):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, file_name))
                labels.append(idx)
    return paths, np.array(labels, dtype=int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """
    Compute classification metrics: accuracy, F1, and confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing 'accuracy', 'f1_macro', and 'cm'.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'cm': confusion_matrix(y_true, y_pred),
    }


def plot_cm_subset(cm: np.ndarray, classes: list[str], subset: list[str], annot: bool = True) -> None:
    """
    Plot only the rows/columns of `cm` corresponding to `subset` classes.
    Args:
      cm: full confusion matrix (N×N)
      classes: list of all N class names in order
      subset: list of class names to include in the plot
      annot: whether to annotate cells
    """
    # Find indices for the subset
    idx = [classes.index(c) for c in subset]
    cm_sub = cm[np.ix_(idx, idx)]

    # Figure size scales with number of classes
    size = max(6, len(subset) * 0.3)
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(cm_sub, annot=annot, fmt='d', xticklabels=subset, yticklabels=subset, cbar=False, square=True,
                linewidths=0.5, linecolor='gray')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


import os
import gdown
from typing import List, Optional


class DataDownloader:
    """
    A utility class to download files from Google Drive using gdown.

    Usage:
        downloader = DataDownloader(output_dir="data")
        downloader.download("https://drive.google.com/uc?id=FILE_ID", filename="myfile.zip")
    """

    def __init__(self, output_dir: str = "downloads"):
        """
        Args:
            output_dir: Directory to save downloaded files. Will be created if it doesn't exist.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download(self, url: str, filename: Optional[str] = None, quiet: bool = False) -> str:
        """
        Download a single file from Google Drive.

        Args:
            url: Full gdown-compatible URL, or Drive share URL/file ID.
            filename: Desired name for the saved file. If None, gdown infers from URL.
            quiet: Suppress progress output if True.

        Returns:
            The path to the downloaded file.
        """

        if not url.startswith("http"):
            url = f"https://drive.google.com/uc?id={url}"

        output_path = os.path.join(self.output_dir, filename if filename else "")

        downloaded = gdown.download(url, output_path, quiet=quiet)
        if downloaded is None:
            raise RuntimeError(f"Failed to download from {url}")

        return downloaded
