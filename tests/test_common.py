import pickle

import numpy as np
import pytest

from refineshot.common import (
    classification_metrics,
    clean_key,
    f1_from_counts,
    f1_pr,
    load_logits,
    normalize_key,
    scores_from_cache,
    set_global_seeds,
    sigmoid_np,
)


def test_clean_key_keeps_extension_normalize_key_strips_it():
    assert clean_key("clipshots:video.mp4") == "video.mp4"
    assert normalize_key("clipshots:video.mp4") == "video"
    assert clean_key("video.mp4") == "video.mp4"
    assert normalize_key("video.mp4") == "video"
    # only the first colon is a prefix separator
    assert clean_key("a:b:c.mp4") == "b:c.mp4"


def test_load_logits_unwraps_envelope_and_accepts_raw_dict(tmp_path):
    logits = {"video": np.zeros(4, dtype=np.float32)}

    wrapped = tmp_path / "wrapped.pkl"
    wrapped.write_bytes(pickle.dumps({"config": {"x": 1}, "logits": logits}))
    raw = tmp_path / "raw.pkl"
    raw.write_bytes(pickle.dumps(logits))

    assert set(load_logits(wrapped)) == {"video"}
    assert set(load_logits(raw)) == {"video"}


def test_f1_helpers_agree():
    for tp, fp, fn in [(0, 0, 0), (5, 0, 0), (0, 3, 4), (7, 2, 5)]:
        f1, precision, recall = f1_pr(tp, fp, fn)
        metrics = classification_metrics(tp, fp, fn)
        assert metrics["f1"] == f1
        assert metrics["precision"] == precision
        assert metrics["recall"] == recall
        assert f1_from_counts(np.array([tp, fp, fn]))[()] == f1


def test_scores_from_cache_logits_and_probabilities_kinds():
    logits = {"clipshots:video.mp4": np.array([-2.0, 0.0, 2.0], dtype=np.float32)}
    pred = scores_from_cache(logits, temperature=1.0, sigma=0.0, input_kind="logits")
    assert set(pred) == {"video.mp4"}
    assert pred["video.mp4"].shape == (3, 1)
    np.testing.assert_allclose(pred["video.mp4"].reshape(-1), sigmoid_np(np.array([-2.0, 0.0, 2.0])), rtol=1e-6)

    probs = {"video": np.array([0.1, 0.5, 0.9], dtype=np.float32)}
    identity = scores_from_cache(probs, temperature=1.0, sigma=0.0, input_kind="probabilities")
    np.testing.assert_allclose(identity["video"].reshape(-1), [0.1, 0.5, 0.9], rtol=1e-6)

    with pytest.raises(ValueError, match="Unsupported input kind"):
        scores_from_cache(probs, temperature=1.0, sigma=0.0, input_kind="bogus")


def test_build_train_phase2_command_preserves_order_and_stringifies():
    import sys

    from refineshot.common import build_train_phase2_command

    cmd = build_train_phase2_command(
        {"--epochs": 3, "--sigma": 2.0},
        ["--skip-test-eval", "--stop-after-minutes", 30.5, "--no-resume"],
    )
    assert cmd == [
        sys.executable,
        "-m",
        "refineshot.train_phase2",
        "--epochs",
        "3",
        "--sigma",
        "2.0",
        "--skip-test-eval",
        "--stop-after-minutes",
        "30.5",
        "--no-resume",
    ]


def test_set_global_seeds_makes_numpy_deterministic():
    set_global_seeds(123)
    first = np.random.rand(3)
    set_global_seeds(123)
    np.testing.assert_array_equal(first, np.random.rand(3))
