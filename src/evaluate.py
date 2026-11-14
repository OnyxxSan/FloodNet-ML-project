"""Helpers for standardized model-evaluation reporting."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def summarize_classification(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] | None = None) -> dict:
    """Return a compact dictionary of classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels) if labels is not None else [0, 1]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
