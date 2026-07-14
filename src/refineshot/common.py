"""Shared dependency-light helpers used across training, evaluation and analysis.

Single home for the small utilities that used to be copy-pasted between
`eval.py`, `postprocess_calibration.py`,
`train_phase2.py` and the journal-study scripts. Only numpy is imported at
module load; torch and scipy are imported lazily inside the functions that
need them.

Note on F1: `utils.py` computes F1 as ``(p * r * 2) / (p + r)`` — a different
floating-point operation order than `f1_pr` here — and keeps its own formula
on purpose so historical numbers stay bit-identical.
"""

from __future__ import annotations

import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def clean_key(key: str) -> str:
    """Strip a ``dataset:`` prefix, keeping any file extension.

    Use when matching prediction keys against ground-truth dicts whose keys
    may still carry a ``.mp4``-style suffix.
    """
    return str(key).split(":", 1)[-1]


def normalize_key(value: str) -> str:
    """Strip a ``dataset:`` prefix AND the file extension (stem only).

    Use for cross-source joins where one side is keyed by bare video stems.
    Deliberately different from :func:`clean_key`; do not merge them.
    """
    return Path(clean_key(value)).stem


def load_pickle_payload(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_logits(path: Path) -> dict[str, np.ndarray]:
    """Load a logits pickle, unwrapping the optional ``{"config", "logits"}`` envelope."""
    payload = load_pickle_payload(path)
    return payload["logits"] if isinstance(payload, dict) and "logits" in payload else payload


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def scores_from_cache(
    scores: dict[str, np.ndarray],
    temperature: float,
    sigma: float,
    input_kind: str,
) -> dict[str, np.ndarray]:
    """Temperature-scale and Gaussian-smooth cached scores into predictions.

    Distinct from ``runtime.logits_to_probabilities``: that one clamps the
    temperature to 1e-6 and has no ``"probabilities"`` input branch — keep
    them separate so deployed inference and analysis stay independently pinned.
    """
    from scipy.ndimage import gaussian_filter1d

    pred: dict[str, np.ndarray] = {}
    for key, arr in scores.items():
        value = np.asarray(arr, dtype=np.float32).reshape(-1)
        if input_kind == "logits":
            value = sigmoid_np(value / temperature)
        elif input_kind == "probabilities":
            if temperature != 1.0:
                eps = np.finfo(np.float32).eps
                clipped = np.clip(value, eps, 1.0 - eps)
                logits = np.log(clipped / (1.0 - clipped))
                value = sigmoid_np(logits / temperature)
        else:
            raise ValueError(f"Unsupported input kind: {input_kind}")
        if sigma > 0:
            value = gaussian_filter1d(value, sigma=sigma)
        pred[clean_key(key)] = value[:, np.newaxis].astype(np.float32)
    return pred


def f1_pr(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def classification_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    f1, precision, recall = f1_pr(tp, fp, fn)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def f1_from_counts(values: np.ndarray) -> np.ndarray:
    """Vectorized F1 over ``(..., 3)`` arrays of (tp, fp, fn) counts."""
    tp = values[..., 0].astype(np.float64)
    fp = values[..., 1].astype(np.float64)
    fn = values[..., 2].astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )


def build_train_phase2_command(
    options: Mapping[str, object],
    extra: Sequence[object] = (),
) -> list[str]:
    """Assemble argv for a ``python -m refineshot.train_phase2`` subprocess.

    ``options`` maps ``--flag`` names to values, emitted in insertion order;
    ``extra`` holds raw trailing tokens (boolean flags or flag/value runs).
    """
    cmd = [sys.executable, "-m", "refineshot.train_phase2"]
    for key, value in options.items():
        cmd.extend([key, str(value)])
    cmd.extend(str(token) for token in extra)
    return cmd


def set_global_seeds(seed: int) -> None:
    """Seed python, numpy and torch RNGs (torch imported lazily)."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
