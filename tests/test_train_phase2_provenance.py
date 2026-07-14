import os
from argparse import Namespace

from refineshot.train_phase2 import (
    build_sample_cache_config,
    hash_keys,
    select_training_keys,
)


def cache_args(**overrides):
    values = {
        "max_train_videos": 400,
        "max_samples_per_video": 160,
        "max_total_samples": 0,
        "neg_per_pos": 3,
        "min_neg_per_video": 32,
        "boundary_window": 0,
        "max_cache_video_frames": 180000,
        "max_cache_video_seconds": 7200.0,
        "data_seed": 42,
    }
    values.update(overrides)
    return Namespace(**values)


def test_training_key_selection_is_deterministic_and_budgeted():
    keys = [f"video-{index:04d}" for index in range(1000)]
    selected = select_training_keys(keys, seed=42, max_train_videos=400)

    assert len(selected) == 400
    assert selected == select_training_keys(keys, seed=42, max_train_videos=400)
    assert selected != select_training_keys(keys, seed=43, max_train_videos=400)
    assert len(set(selected)) == len(selected)


def test_cache_identity_changes_with_selected_videos_and_metadata(tmp_path):
    metadata = tmp_path / "meta.pickle"
    metadata.write_bytes(b"metadata-v1")
    keys = ["a", "b", "c"]
    base = build_sample_cache_config(
        str(metadata),
        keys,
        "checkpoint-hash",
        cache_args(),
    )

    assert base["selected_keys_hash"] == hash_keys(keys)
    assert base["selected_keys_count"] == 3
    assert base != build_sample_cache_config(
        str(metadata),
        keys[:2],
        "checkpoint-hash",
        cache_args(max_train_videos=2),
    )

    previous_mtime_ns = metadata.stat().st_mtime_ns
    metadata.write_bytes(b"metadata-v2")
    # Some filesystems can retain the same timestamp for immediate, same-size
    # rewrites. Advance it explicitly because the production hash cache uses
    # path, mtime, and size as its file identity.
    os.utime(metadata, ns=(metadata.stat().st_atime_ns, previous_mtime_ns + 1_000_000_000))
    changed_metadata = build_sample_cache_config(
        str(metadata),
        keys,
        "checkpoint-hash",
        cache_args(),
    )
    assert changed_metadata["meta_sha256"] != base["meta_sha256"]


def test_cache_config_is_pinned_exactly(tmp_path):
    """Freeze cache identity: any key/value drift invalidates existing caches.

    If this test fails, sampled training data or cache reuse semantics changed.
    Never adapt the pinned literal casually — see SAMPLE_CACHE_SCHEMA_VERSION.
    """
    metadata = tmp_path / "meta.pickle"
    metadata.write_bytes(b"metadata-v1")

    config = build_sample_cache_config(
        str(metadata),
        ["a", "b", "c"],
        "checkpoint-hash",
        cache_args(),
    )

    assert config == {
        "schema_version": 2,
        "meta_path": os.path.abspath(str(metadata)),
        "meta_sha256": "8c642b95858fcf252e843b44bd7feb0225b774516a83a885f6ef1e8ce1ceeb55",
        "selected_keys_hash": "880553fca8fcea94e325ee2cfb48e5a985cc797f39a14cc6d3cedecfeb2ae4d2",
        "selected_keys_count": 3,
        "base_ckpt_hash": "checkpoint-hash",
        "max_train_videos": 400,
        "max_samples_per_video": 160,
        "max_total_samples": 0,
        "neg_per_pos": 3,
        "min_neg_per_video": 32,
        "boundary_window": 0,
        "max_cache_video_frames": 180000,
        "max_cache_video_seconds": 7200.0,
        "data_seed": 42,
    }


def test_training_seed_does_not_change_shared_data_cache_identity(tmp_path):
    metadata = tmp_path / "meta.pickle"
    metadata.write_bytes(b"metadata")
    selected = select_training_keys(["a", "b", "c"], seed=42, max_train_videos=2)

    first = build_sample_cache_config(
        str(metadata),
        selected,
        "checkpoint-hash",
        cache_args(),
    )
    second = build_sample_cache_config(
        str(metadata),
        selected,
        "checkpoint-hash",
        cache_args(),
    )
    assert first == second
