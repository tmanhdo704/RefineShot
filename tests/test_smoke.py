"""Minimal smoke tests for RefineShot.

Runs as a plain script (no pytest needed) or under pytest:

    python tests/test_smoke.py        # prints PASS/SKIP/FAIL per test
    pytest tests/test_smoke.py        # if pytest is installed

Covered:
  - edge case: predictions_to_scenes([]) must not raise (utils + inference copies)
  - basic correctness of predictions_to_scenes
  - calibration reproduction when its external caches are available
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Prefer the installed package (pip install -e .); fall back to src/ on path so the
# tests also run on a fresh checkout that has not been installed yet.
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np


class _SkipTest(Exception):
    pass


def _skip(msg: str):
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(msg)
    raise _SkipTest(msg)


def test_predictions_to_scenes_empty():
    from refineshot import eval as rie
    from refineshot import utils

    assert utils.predictions_to_scenes(np.array([])).tolist() == [[0, 0]]
    assert rie.predictions_to_scenes(np.array([])).tolist() == [[0, 0]]


def test_predictions_to_scenes_normal():
    from refineshot import eval as rie
    from refineshot import utils

    pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
    expected = [[0, 2], [4, 6], [8, 9]]
    assert utils.predictions_to_scenes(pred).tolist() == expected
    assert rie.predictions_to_scenes(pred).tolist() == expected


def test_reproduce_calibration():
    from refineshot import postprocess_calibration as pc

    representative = [
        pc.ARTIFACTS / "eval_cache_shot_clipshots" / "shot_test_logits.pkl",
        pc.ABLATION / "A1_phase2_bce_onehot" / "eval_cache" / "shot_test_logits.pkl",
    ]
    if not all(p.exists() for p in representative):
        _skip("cached logits not present; see docs/DATA.md")

    thresholds = [float(t) for t in pc.DEFAULT_THRESHOLDS]
    failures = pc.run_reproduce(thresholds)
    assert failures == 0, f"{failures} reproduce mismatch(es)"


def test_restem_key_matching():
    # GT keys often carry a `.mp4` suffix while logits use the bare stem; restem must align them.
    from refineshot import postprocess_calibration as pc

    logits = {"abc": 1, "x.y.mp4": 2}
    gt = {"abc.mp4": 10, "x.y.mp4": 20}
    common = set(pc.restem(logits)) & set(pc.restem(gt))
    assert common == {"abc", "x.y"}, common


def test_single_frame_edge_case():
    # A 1-frame clip must not collapse to a scalar via .squeeze(); output must stay (1, 1).
    from refineshot import common
    from refineshot import eval as rie

    out = rie.logits_to_predictions({"v": np.array([[0.3]], dtype=np.float32)}, temperature=1.0, sigma=2.0)
    assert out["v"].shape == (1, 1), out["v"].shape
    out2 = common.scores_from_cache(
        {"v": np.array([[2.0]], dtype=np.float32)}, temperature=1.0, sigma=0.0, input_kind="logits"
    )
    assert out2["v"].shape == (1, 1), out2["v"].shape


def test_eval_graceful_missing_checkpoint():
    # A missing checkpoint must not crash the cached-logits eval path (it is not in the bundle).
    from pathlib import Path

    from refineshot import eval as rie

    assert rie.load_checkpoint_config(Path("definitely_missing_ckpt.pth")) == {}


def _main() -> int:
    tests = [
        test_predictions_to_scenes_empty,
        test_predictions_to_scenes_normal,
        test_restem_key_matching,
        test_single_frame_edge_case,
        test_eval_graceful_missing_checkpoint,
        test_reproduce_calibration,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except _SkipTest as exc:
            print(f"SKIP  {test.__name__}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any unexpected error as a failure
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{'OK' if failed == 0 else 'FAILURES: ' + str(failed)}")
    return failed


if __name__ == "__main__":
    raise SystemExit(_main())
