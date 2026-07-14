from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from refineshot import runtime


def test_deployment_postprocess_defaults():
    config = runtime.PostprocessConfig.from_mapping({})
    assert config.temperature == runtime.DEFAULT_TEMPERATURE
    assert config.sigma == 2.0
    assert config.threshold == 0.19


def test_resolve_device_auto_and_cuda_fallback():
    with patch("refineshot.runtime.torch.cuda.is_available", return_value=False):
        assert runtime.resolve_device("auto") == "cpu"
        assert runtime.resolve_device("cuda") == "cpu"
    with patch("refineshot.runtime.torch.cuda.is_available", return_value=True):
        assert runtime.resolve_device("auto") == "cuda"
        assert runtime.resolve_device("cuda:0") == "cuda:0"


def test_extract_checkpoint_state_variants_and_module_prefix():
    value = torch.ones(1)
    assert runtime.extract_checkpoint_state({"net": {"weight": value}})["weight"] is value
    state = runtime.extract_checkpoint_state({"state_dict": {"module.weight": value}})
    assert list(state) == ["weight"]
    assert state["weight"] is value


def test_frame_batching_and_prediction_length():
    frames = np.zeros((51, 27, 48, 3), dtype=np.uint8)
    batches = list(runtime.iter_frame_batches(frames))
    assert len(batches) == 2
    assert all(batch.shape == (100, 27, 48, 3) for batch in batches)

    class FakeModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.ones((inputs.shape[0], inputs.shape[2], 1), device=inputs.device)

    logits = runtime.predict_frame_logits(FakeModel(), frames, "cpu")
    assert logits.shape == (51, 1)
    assert logits.dtype == np.float32

    with patch("refineshot.runtime.torch.cuda.is_available", return_value=False):
        fallback_logits = runtime.predict_frame_logits(FakeModel(), frames[:1], "cuda")
    assert fallback_logits.shape == (1, 1)


def test_single_frame_prediction_and_scene_conversion():
    frames = np.zeros((1, 27, 48, 3), dtype=np.uint8)

    class FakeModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.zeros((inputs.shape[0], inputs.shape[2], 1), device=inputs.device)

    logits = runtime.predict_frame_logits(FakeModel(), frames, "cpu")
    probabilities = runtime.logits_to_probabilities(logits, temperature=1.0, sigma=0.0)
    assert logits.shape == (1, 1)
    assert probabilities.shape == (1,)
    assert runtime.probabilities_to_scenes(probabilities, threshold=0.5).tolist() == [[0, 0]]


def test_missing_checkpoint_errors_are_explicit(tmp_path: Path):
    missing = tmp_path / "missing.pth"
    try:
        runtime.load_model(missing, "cpu")
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("load_model accepted a missing checkpoint")
