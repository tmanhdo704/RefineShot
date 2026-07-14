import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

pytest.importorskip("fastapi")
pytest.importorskip("motor")
pytest.importorskip("cloudinary")
pytest.importorskip("cv2")

from app.core.config import Settings
from app.ml import scene_detector
from app.services.storage_service import safe_video_extension

from refineshot.runtime import DEFAULT_SIGMA, DEFAULT_TEMPERATURE, DEFAULT_THRESHOLD


def test_settings_support_new_and_legacy_checkpoint_env(tmp_path: Path):
    new_path = tmp_path / "new.pth"
    with patch.dict(
        "os.environ",
        {"REFINESHOT_MODEL_PATH": str(new_path), "REFINESHOT_CKPT": ""},
        clear=False,
    ):
        assert Settings(_env_file=None).refineshot_model_path == new_path

    legacy_path = tmp_path / "legacy.pth"
    with patch.dict(
        "os.environ",
        {"REFINESHOT_MODEL_PATH": "", "REFINESHOT_CKPT": str(legacy_path)},
        clear=False,
    ):
        assert Settings(_env_file=None).refineshot_model_path == legacy_path


def test_effective_postprocess_precedence():
    assert scene_detector._effective_value(0.2, 0.3, 0.4) == 0.2
    assert scene_detector._effective_value(None, 0.3, 0.4) == 0.3
    assert scene_detector._effective_value(None, None, 0.4) == 0.4
    assert DEFAULT_TEMPERATURE > 0
    assert DEFAULT_SIGMA == 2.0
    assert DEFAULT_THRESHOLD == 0.19


def test_auto_backend_falls_back_when_checkpoint_is_missing(tmp_path: Path):
    options = scene_detector.VideoAnalysisSettings(
        sensitivity="medium",
        min_scene_duration_sec=0.5,
        backend="auto",
    )
    baseline_result = {"processing": {"model": "opencv-histogram-baseline"}}

    with (
        patch.object(scene_detector, "get_runtime", side_effect=FileNotFoundError("missing model")),
        patch.object(
            scene_detector.baseline_opencv,
            "analyze_video",
            return_value=baseline_result,
        ),
    ):
        result = scene_detector.analyze_video("job_test", tmp_path / "video.mp4", options)

    assert result["processing"]["backend"] == "baseline"
    assert result["processing"]["requested_backend"] == "auto"
    assert "missing model" in result["processing"]["fallback_reason"]


def test_phase2_backend_does_not_hide_missing_checkpoint(tmp_path: Path):
    options = scene_detector.VideoAnalysisSettings(
        sensitivity="medium",
        min_scene_duration_sec=0.5,
        backend="phase2",
    )
    with patch.object(
        scene_detector,
        "get_runtime",
        side_effect=FileNotFoundError("missing model"),
    ):
        with pytest.raises(RuntimeError, match="model is unavailable"):
            scene_detector.analyze_video("job_test", tmp_path / "video.mp4", options)


def test_safe_video_extension():
    assert safe_video_extension("clip.MP4") == ".mp4"
    with pytest.raises(ValueError):
        safe_video_extension("payload.exe")
