"""Minimal direct adapter for Tigramite CMIknn; no significance testing."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Standardization:
    mean: list[float]
    scale: list[float]


def standardize_selected(columns: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], Standardization]:
    """Standardize named one-dimensional inputs using only their supplied rows."""
    names = list(columns)
    matrix = np.column_stack([np.asarray(columns[name], dtype=float).copy() for name in names])
    if not np.isfinite(matrix).all():
        raise ValueError("CMI inputs must be finite")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    if np.any(scale == 0.0):
        bad = [name for name, value in zip(names, scale) if value == 0.0]
        raise ValueError(f"CMI inputs contain constant columns: {bad}")
    standardized = (matrix - mean) / scale
    return ({name: standardized[:, i].copy() for i, name in enumerate(names)}, Standardization(mean.tolist(), scale.tolist()))


def make_cmiknn(seed: int):
    """Construct official CMIknn with the fixed seed accepted by CondIndTest."""
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.independence_tests.independence_tests_base import CondIndTest

    if "seed" not in inspect.signature(CondIndTest).parameters:
        raise RuntimeError("Installed CondIndTest exposes no supported fixed-seed parameter")
    kwargs = {"knn": 0.10, "transform": False, "workers": 1, "seed": seed}
    return CMIknn(**kwargs), kwargs


def direct_score(array: np.ndarray, xyz: np.ndarray, seed: int) -> tuple[float, dict]:
    """Call official get_dependence_measure directly with validated low-level convention."""
    values = np.asarray(array, dtype=float).copy()
    labels = np.asarray(xyz, dtype=int).copy()
    if values.ndim != 2 or labels.shape != (values.shape[0],):
        raise ValueError("array must be (variables, samples) and xyz one label per variable")
    if values.shape[1] < 2 or not np.isfinite(values).all() or not set(labels).issubset({0, 1, 2}):
        raise ValueError("invalid CMI array or xyz labels")
    if list(labels).count(0) != 1 or list(labels).count(1) != 1:
        raise ValueError("exactly one X (0) and target (1) are required")
    estimator, kwargs = make_cmiknn(seed)
    return float(estimator.get_dependence_measure(values, labels)), kwargs
