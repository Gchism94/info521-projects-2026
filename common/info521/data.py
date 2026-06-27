"""Data loading plumbing.

Returns plain NumPy arrays so students build design matrices themselves. No
preprocessing decisions (scaling, splitting, polynomial expansion) are made for
them -- those are graded skills.
"""

from __future__ import annotations

import os
import numpy as np

_HERE = os.path.dirname(__file__)
_DEFAULT_CSV = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "clinical_reference.csv"))


def load_clinical(path: str | None = None):
    """Load the clinical dataset.

    Parameters
    ----------
    path : str, optional
        CSV path. Defaults to the shipped synthetic reference dataset. To deploy
        with a curated real dataset, point this at it (see
        instructor/dataset-curation.md) -- the column contract below must hold.

    Returns
    -------
    dict with keys:
        'X'        : (n, d) ndarray of all feature columns (target excluded)
        'y'        : (n,)   ndarray, continuous target ('risk')
        'features' : list[str], length d, feature column names
        'primary'  : str, the recommended single predictor for Project 1 ('age')
        'target'   : str, the target name ('risk')

    Notes
    -----
    Contract: the CSV has a header row; the last column is the continuous target
    named 'risk'; a column named 'age' is present to serve as the Project 1
    primary predictor.
    """
    path = path or _DEFAULT_CSV
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    M = np.genfromtxt(path, delimiter=",", skip_header=1)
    if "risk" not in header:
        raise ValueError("dataset contract violated: expected a 'risk' target column")
    target_idx = header.index("risk")
    feature_idx = [i for i in range(len(header)) if i != target_idx]
    return {
        "X": M[:, feature_idx],
        "y": M[:, target_idx],
        "features": [header[i] for i in feature_idx],
        "primary": "age" if "age" in header else header[feature_idx[0]],
        "target": "risk",
    }


def primary_predictor(ds) -> np.ndarray:
    """Return the single primary-predictor column (Project 1 works in 1-D)."""
    return ds["X"][:, ds["features"].index(ds["primary"])]


def binarize(y: np.ndarray, threshold: float | None = None):
    """Binarize the continuous target for the Project 2 conjugacy-break bridge.

    Returns (labels, threshold). Default threshold is the median (balanced
    split). This is a *framing* helper, not a model.
    """
    threshold = float(np.median(y)) if threshold is None else float(threshold)
    return (y > threshold).astype(int), threshold
