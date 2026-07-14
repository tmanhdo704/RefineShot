from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from refineshot import train_phase2 as train


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer_4_13 = torch.nn.Linear(2, 2)
        self.Layer_5_12 = torch.nn.Linear(2, 2)
        self.fc1_0 = torch.nn.Linear(2, 2)
        self.cls_layer1 = torch.nn.Linear(2, 1)
        self.cls_layer2 = torch.nn.Linear(2, 1)
        self.other = torch.nn.Linear(2, 2)


class TinyVideoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer_4_13 = torch.nn.Linear(1, 1)
        self.Layer_5_12 = torch.nn.Linear(1, 1)
        self.fc1_0 = torch.nn.Linear(1, 1)
        self.cls_layer1 = torch.nn.Linear(1, 1)
        self.cls_layer2 = torch.nn.Linear(1, 1)

    def forward(self, inputs):
        x = inputs.float().mean(dim=(1, 3, 4)).unsqueeze(-1)
        x = self.Layer_4_13(x)
        x = self.Layer_5_12(x)
        x = self.fc1_0(x)
        return self.cls_layer1(x), self.cls_layer2(x)


def trainable_names(model: torch.nn.Module) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def test_module_names_for_finetune_scope():
    assert train.module_names_for_finetune_scope("head_only") == ()
    assert train.module_names_for_finetune_scope("fc1_0") == ("fc1_0", "cls_layer1", "cls_layer2")
    assert train.module_names_for_finetune_scope("layer5") == (
        "Layer_5_12",
        "fc1_0",
        "cls_layer1",
        "cls_layer2",
    )
    assert train.module_names_for_finetune_scope("layer4_layer5") == (
        "Layer_4_13",
        "Layer_5_12",
        "fc1_0",
        "cls_layer1",
        "cls_layer2",
    )


def test_configure_finetune_modules_freezes_unselected_parameters():
    model = TinyModel()

    summary = train.configure_finetune_modules(model, "layer5")
    names = trainable_names(model)

    assert summary["scope"] == "layer5"
    assert all(name.startswith(("Layer_5_12.", "fc1_0.", "cls_layer1.", "cls_layer2.")) for name in names)
    assert not any(name.startswith(("Layer_4_13.", "other.")) for name in names)


def test_split_finetune_parameters_separates_head_and_backbone():
    model = TinyModel()
    train.configure_finetune_modules(model, "layer4_layer5")

    head_params, backbone_params = train.split_finetune_parameters(model)

    assert head_params
    assert backbone_params
    assert sum(parameter.numel() for parameter in head_params) == 12
    assert sum(parameter.numel() for parameter in backbone_params) == 12


def test_validate_training_mode_rejects_ema_for_end_to_end():
    args = Namespace(finetune_scope="layer5", use_ema=True, backbone_lr=1e-6)

    with pytest.raises(ValueError, match="use-ema"):
        train.validate_training_mode_args(args)


def test_validate_training_mode_allows_fc1_without_backbone_lr():
    args = Namespace(finetune_scope="fc1_0", use_ema=False, backbone_lr=0.0)

    train.validate_training_mode_args(args)


def test_select_finetune_loss_indices_keeps_positive_and_caps_negatives():
    rng = np.random.default_rng(7)
    one_hot = np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)
    boundary = np.array([0, 1, 1, 1, 0, 0], dtype=np.float32)

    selected = train.select_finetune_loss_indices(
        one_hot,
        boundary,
        max_frames=4,
        neg_per_pos=2,
        min_negatives=1,
        rng=rng,
    )

    assert boundary[selected].sum() >= 1
    assert len(selected) <= 4


def test_select_finetune_loss_indices_uses_all_frames_when_uncapped():
    rng = np.random.default_rng(7)

    selected = train.select_finetune_loss_indices(
        np.zeros(6, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        max_frames=0,
        neg_per_pos=2,
        min_negatives=1,
        rng=rng,
    )

    assert selected.tolist() == [0, 1, 2, 3, 4, 5]


def test_fine_tune_model_runs_one_synthetic_video(monkeypatch, tmp_path: Path):
    frames = np.ones((4, 27, 48, 3), dtype=np.uint8)
    batch = np.ones((100, 27, 48, 3), dtype=np.uint8)
    args = Namespace(
        finetune_scope="layer5",
        loss="bce",
        gamma=1.0,
        alpha=0.5,
        lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=0.0,
        manyhot_weight=0.3,
        boundary_window=1,
        max_train_videos=0,
        max_finetune_videos=1,
        finetune_batches_per_video=1,
        finetune_max_frames_per_batch=4,
        finetune_min_neg_per_batch=1,
        neg_per_pos=2,
        seed=42,
        resume_state=str(tmp_path / "resume.pt"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        no_resume=True,
        ignore_resume_config=False,
        save_every_epochs=1,
        epochs=1,
        log_every_batches=100,
        max_cache_video_frames=1000,
        max_cache_video_seconds=1000.0,
        _deadline=None,
    )

    monkeypatch.setattr(train, "device", "cpu")
    monkeypatch.setattr(train, "get_frames", lambda _path: frames)
    monkeypatch.setattr(train, "get_batches", lambda _frames: iter([batch]))
    monkeypatch.setattr(train, "check_sample_cache_video_budget", lambda *_args, **_kwargs: None)

    model, losses, stats = train.fine_tune_model(
        TinyVideoModel(),
        {"v": {"video_path": "synthetic.mp4", "transitions": np.array([[1, 1]], dtype=np.int32)}},
        ["v"],
        args,
    )

    assert isinstance(model, TinyVideoModel)
    assert len(losses) == 1
    assert stats["videos_seen"] == 1
    assert stats["batches"] == 1
    assert stats["frames_used"] > 0
    assert (tmp_path / "resume.pt").is_file()
