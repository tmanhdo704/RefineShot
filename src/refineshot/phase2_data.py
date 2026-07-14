"""Data and sample-cache side of phase-2 training.

Everything that decides WHICH frames become training samples lives here:
metadata loading, deterministic video selection, ffprobe budgets, backbone
feature extraction, balanced index sampling, and the resumable sample cache.

CRITICAL: the sample cache identity (build_sample_cache_config) and the RNG
call order inside build_or_load_sample_cache define the sampled training data.
Any change here invalidates caches or silently changes the data — guarded by
tests/test_train_phase2_provenance.py.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch

from refineshot.utils import get_batches, get_frames, scenes2zero_one_representation

SAMPLE_CACHE_SCHEMA_VERSION = 2

device = "cuda" if torch.cuda.is_available() else "cpu"


class TimeBudgetExpired(Exception):
    pass


# Keyed by (abspath, mtime_ns, size): metadata/checkpoint files are hashed
# repeatedly per run (once per ablation experiment + once per manifest write);
# memoizing avoids re-reading multi-hundred-MB files with identical content.
_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: str) -> str:
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
    cached = _SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA256_CACHE[key] = digest
    return digest


def hash_keys(keys: list[str]) -> str:
    h = hashlib.sha256()
    for key in sorted(keys):
        h.update(key.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def select_training_keys(
    train_keys: list[str],
    seed: int,
    max_train_videos: int,
) -> list[str]:
    """Return the exact, deterministic video order used to build the sample cache."""
    selected = list(train_keys)
    random.Random(seed).shuffle(selected)
    if max_train_videos > 0:
        selected = selected[:max_train_videos]
    return selected


def build_sample_cache_config(
    meta_path: str,
    selected_keys: list[str],
    base_ckpt_hash: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a strict cache identity from every input that changes sampled data."""
    return {
        "schema_version": SAMPLE_CACHE_SCHEMA_VERSION,
        "meta_path": os.path.abspath(meta_path),
        "meta_sha256": sha256_file(meta_path),
        "selected_keys_hash": hash_keys(selected_keys),
        "selected_keys_count": len(selected_keys),
        "base_ckpt_hash": base_ckpt_hash,
        "max_train_videos": args.max_train_videos,
        "max_samples_per_video": args.max_samples_per_video,
        "max_total_samples": args.max_total_samples,
        "neg_per_pos": args.neg_per_pos,
        "min_neg_per_video": args.min_neg_per_video,
        "boundary_window": args.boundary_window,
        "max_cache_video_frames": args.max_cache_video_frames,
        "max_cache_video_seconds": args.max_cache_video_seconds,
        "data_seed": args.data_seed,
    }


def write_training_data_manifest(
    path: str,
    meta_path: str,
    base_ckpt_path: str,
    entries: dict[str, dict[str, Any]],
    stats: dict[str, Any],
    data_seed: int,
) -> dict[str, Any]:
    selected_keys = list(stats["selected_keys"])
    completed_keys = set(stats["completed_keys"])
    skipped_by_key = {str(key): str(reason) for key, reason in stats["skipped"]}
    rows = []
    dataset_counts: dict[str, int] = {}
    for key in selected_keys:
        entry = entries[key]
        dataset = str(entry.get("dataset", "unknown"))
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        status = (
            "completed"
            if key in completed_keys
            else "skipped"
            if key in skipped_by_key
            else "not_processed"
        )
        rows.append(
            {
                "key": key,
                "dataset": dataset,
                "source_split": entry.get("source_split"),
                "source_name": entry.get("source_name"),
                "status": status,
                "samples": int(stats["sample_counts"].get(key, 0)),
                "skip_reason": skipped_by_key.get(key),
            }
        )
    payload = {
        "schema_version": 1,
        "data_seed": data_seed,
        "metadata": {
            "path": os.path.abspath(meta_path),
            "sha256": sha256_file(meta_path),
        },
        "base_checkpoint": {
            "path": os.path.abspath(base_ckpt_path),
            "sha256": sha256_file(base_ckpt_path),
        },
        "selected_keys_hash": hash_keys(selected_keys),
        "selected_videos": len(selected_keys),
        "completed_videos": len(completed_keys),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "sampling": stats["cache_config"],
        "videos": rows,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def make_boundary_labels(one_hot: np.ndarray, window: int) -> np.ndarray:
    labels = one_hot.copy().astype(np.float32)
    for idx in np.flatnonzero(one_hot > 0):
        start = max(0, idx - window)
        end = min(len(labels), idx + window + 1)
        labels[start:end] = 1.0
    return labels


def transitions_to_scenes(transitions: np.ndarray, n_frames: int) -> np.ndarray:
    transitions = np.asarray(transitions, dtype=np.int32)
    if n_frames <= 0:
        return np.asarray([[0, 0]], dtype=np.int32)
    if transitions.size == 0:
        return np.asarray([[0, n_frames - 1]], dtype=np.int32)

    transitions = transitions.reshape(-1, 2)
    transitions = transitions[np.argsort(transitions[:, 0])]
    transitions = np.clip(transitions, 0, n_frames - 1)
    transitions = transitions[transitions[:, 0] <= transitions[:, 1]]
    if len(transitions) == 0:
        return np.asarray([[0, n_frames - 1]], dtype=np.int32)

    scenes = [[0, int(transitions[0, 0])]]
    for i in range(1, len(transitions)):
        scenes.append([int(transitions[i - 1, 1]), int(transitions[i, 0])])
    scenes.append([int(transitions[-1, 1]), n_frames - 1])

    arr = np.asarray(scenes, dtype=np.int32)
    arr = np.clip(arr, 0, n_frames - 1)
    arr = arr[arr[:, 0] <= arr[:, 1]]
    if len(arr) == 0:
        arr = np.asarray([[0, n_frames - 1]], dtype=np.int32)
    return arr


def load_metadata(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    required = {"entries", "train_keys", "val_keys", "shot_test_entries"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Metadata file is missing fields: {sorted(missing)}")
    return payload


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return None
        return float(num) / den_f
    return float(value)


def probe_video_info(video_path: str) -> dict[str, float | int | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,r_frame_rate,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout or "{}")
    except Exception as exc:
        return {"probe_error": str(exc), "duration": None, "fps": None, "frames": None, "size": None}

    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    duration_raw = stream.get("duration") or fmt.get("duration")
    duration = None if duration_raw in {None, "N/A"} else float(duration_raw)
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))

    frames_raw = stream.get("nb_frames")
    frames = int(frames_raw) if frames_raw not in {None, "N/A"} else None
    if frames is None and duration is not None and fps is not None:
        frames = int(duration * fps)

    size_raw = fmt.get("size")
    size = int(size_raw) if size_raw not in {None, "N/A"} else None
    return {"probe_error": None, "duration": duration, "fps": fps, "frames": frames, "size": size}


def check_sample_cache_video_budget(video_path: str, args: argparse.Namespace) -> None:
    info = probe_video_info(video_path)
    frames = info.get("frames")
    duration = info.get("duration")
    fps = info.get("fps")

    if frames is not None and args.max_cache_video_frames > 0 and frames > args.max_cache_video_frames:
        raise RuntimeError(
            "video too long for sample cache: "
            f"frames={frames} fps={fps} duration={duration} "
            f"limit_frames={args.max_cache_video_frames} path={video_path}"
        )

    if duration is not None and args.max_cache_video_seconds > 0 and duration > args.max_cache_video_seconds:
        raise RuntimeError(
            "video too long for sample cache: "
            f"duration={duration:.1f}s frames={frames} fps={fps} "
            f"limit_seconds={args.max_cache_video_seconds} path={video_path}"
        )

    if info.get("probe_error"):
        raise RuntimeError(f"ffprobe failed before sample-cache decode: {info['probe_error']} path={video_path}")


def extract_backbone_features(backbone, video_path: str, captured: dict[str, torch.Tensor]) -> torch.Tensor:
    frames = get_frames(video_path)
    if len(frames) == 0:
        raise RuntimeError(f"No decoded frames: {video_path}")

    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in get_batches(frames):
            t = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float().to(device)
            captured.clear()
            backbone(t)
            if "feat" not in captured:
                raise RuntimeError("fc1_0 hook did not capture features")
            chunks.append(captured["feat"][0, 25:75, :].cpu())
    return torch.cat(chunks, 0)[: len(frames)]


def select_sample_indices(
    one_hot: np.ndarray,
    boundary: np.ndarray,
    max_samples_per_video: int,
    neg_per_pos: int,
    min_neg_per_video: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pos_idx = np.unique(np.concatenate([np.flatnonzero(one_hot > 0), np.flatnonzero(boundary > 0)]))
    neg_idx = np.flatnonzero(boundary == 0)

    if max_samples_per_video <= 0:
        max_samples_per_video = len(one_hot)

    if len(pos_idx) > 0:
        pos_cap = max(1, max_samples_per_video // max(neg_per_pos + 1, 1))
        if len(pos_idx) > pos_cap:
            pos_idx = rng.choice(pos_idx, size=pos_cap, replace=False)
        n_neg = min(
            len(neg_idx), max_samples_per_video - len(pos_idx), max(min_neg_per_video, len(pos_idx) * neg_per_pos)
        )
    else:
        n_neg = min(len(neg_idx), max_samples_per_video, min_neg_per_video)

    if n_neg > 0:
        neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)
        selected = np.concatenate([pos_idx, neg_idx])
    else:
        selected = pos_idx

    if len(selected) == 0:
        return selected.astype(np.int64)
    rng.shuffle(selected)
    return selected.astype(np.int64)


def deadline_expired(args: argparse.Namespace) -> bool:
    deadline = getattr(args, "_deadline", None)
    return deadline is not None and time.monotonic() >= deadline


def sample_cache_partial_paths(cache_path: str) -> tuple[str, str]:
    return cache_path + ".parts", cache_path + ".partial.pkl"


def _load_complete_cache(
    cache_path: str,
    cache_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]] | None:
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    if cached.get("config") == cache_config:
        print(f"Loading sample cache: {cache_path}")
        return cached["features"], cached["one_hot"], cached["boundary"], cached["stats"]
    print("Sample cache identity changed; rebuilding.")
    return None


def _resume_partial_state(
    partial_manifest_path: str,
    cache_config: dict[str, Any],
) -> tuple[set[str], list[str], list[tuple[str, str]], dict[str, int], int] | None:
    with open(partial_manifest_path, "rb") as f:
        manifest = pickle.load(f)
    if manifest.get("config") != cache_config:
        print("Partial sample cache exists but config changed; starting over.")
        return None
    done_keys = set(manifest.get("done_keys", []))
    part_files = list(manifest.get("part_files", []))
    skipped = list(manifest.get("skipped", []))
    sample_counts = {
        str(key): int(value)
        for key, value in manifest.get("sample_counts", {}).items()
    }
    total_samples = int(manifest.get("samples", 0))
    print(f"Resuming partial sample cache: done_videos={len(done_keys)} samples={total_samples}")
    return done_keys, part_files, skipped, sample_counts, total_samples


def _assemble_parts(part_files: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loaded_parts = [torch.load(path, map_location="cpu", weights_only=False) for path in part_files]
    x = torch.cat([part["features"] for part in loaded_parts], 0)
    y1 = torch.cat([part["one_hot"] for part in loaded_parts], 0)
    y2 = torch.cat([part["boundary"] for part in loaded_parts], 0)
    return x, y1, y2


def build_or_load_sample_cache(
    cache_path: str,
    meta_path: str,
    entries: dict[str, dict[str, Any]],
    train_keys: list[str],
    backbone,
    base_ckpt_hash: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    work_keys = select_training_keys(train_keys, args.data_seed, args.max_train_videos)
    cache_config = build_sample_cache_config(
        meta_path,
        work_keys,
        base_ckpt_hash,
        args,
    )
    parts_dir, partial_manifest_path = sample_cache_partial_paths(cache_path)

    if os.path.exists(cache_path) and not args.rebuild_sample_cache:
        cached = _load_complete_cache(cache_path, cache_config)
        if cached is not None:
            return cached

    # One generator, created here and consumed strictly in work_keys order by
    # select_sample_indices / rng.choice below — the call order IS the data.
    rng = np.random.default_rng(args.data_seed)

    done_keys: set[str] = set()
    part_files: list[str] = []
    skipped: list[tuple[str, str]] = []
    sample_counts: dict[str, int] = {}
    total_samples = 0

    if os.path.exists(partial_manifest_path) and not args.rebuild_sample_cache:
        resumed = _resume_partial_state(partial_manifest_path, cache_config)
        if resumed is not None:
            done_keys, part_files, skipped, sample_counts, total_samples = resumed

    os.makedirs(parts_dir, exist_ok=True)

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inp, _out):
        captured["feat"] = inp[0].detach().cpu()

    handle = backbone.fc1_0.register_forward_hook(hook)
    chunk_features: list[torch.Tensor] = []
    chunk_one_hot: list[torch.Tensor] = []
    chunk_boundary: list[torch.Tensor] = []
    chunk_keys: list[str] = []

    def write_manifest() -> None:
        with open(partial_manifest_path, "wb") as f:
            pickle.dump(
                {
                    "config": cache_config,
                    "done_keys": sorted(done_keys),
                    "part_files": part_files,
                    "skipped": skipped,
                    "sample_counts": sample_counts,
                    "samples": total_samples,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def flush_chunk() -> None:
        nonlocal chunk_features, chunk_one_hot, chunk_boundary, chunk_keys
        if not chunk_features:
            write_manifest()
            return
        part_path = os.path.join(parts_dir, f"part_{len(part_files):05d}.pt")
        torch.save(
            {
                "features": torch.cat(chunk_features, 0),
                "one_hot": torch.cat(chunk_one_hot, 0),
                "boundary": torch.cat(chunk_boundary, 0),
                "keys": chunk_keys,
            },
            part_path,
        )
        part_files.append(part_path)
        chunk_features = []
        chunk_one_hot = []
        chunk_boundary = []
        chunk_keys = []
        write_manifest()
        print(f"  partial sample cache saved: {part_path}", flush=True)

    try:
        for i, key in enumerate(work_keys, 1):
            if key in done_keys:
                continue
            if args.max_total_samples > 0 and total_samples >= args.max_total_samples:
                break

            entry = entries[key]
            try:
                print(f"  sample cache processing [{i}/{len(work_keys)}] key={key}", flush=True)
                check_sample_cache_video_budget(entry["video_path"], args)
                feats = extract_backbone_features(backbone, entry["video_path"], captured)
                n_frames = int(feats.shape[0])
                scenes = transitions_to_scenes(entry["transitions"], n_frames)
                one_hot_np, _ = scenes2zero_one_representation(scenes, n_frames)
                boundary_np = make_boundary_labels(one_hot_np, args.boundary_window)
                idx = select_sample_indices(
                    one_hot_np,
                    boundary_np,
                    args.max_samples_per_video,
                    args.neg_per_pos,
                    args.min_neg_per_video,
                    rng,
                )
                if len(idx) == 0:
                    skipped.append((key, "no sampled indices"))
                    continue
                if args.max_total_samples > 0 and total_samples + len(idx) > args.max_total_samples:
                    keep = args.max_total_samples - total_samples
                    idx = rng.choice(idx, size=keep, replace=False)

                chunk_features.append(feats[idx].half())
                chunk_one_hot.append(torch.from_numpy(one_hot_np[idx].astype(np.float32)))
                chunk_boundary.append(torch.from_numpy(boundary_np[idx].astype(np.float32)))
                chunk_keys.append(key)
                total_samples += len(idx)
                sample_counts[key] = len(idx)
                done_keys.add(key)
                if i == 1 or i % 25 == 0:
                    pct = 100.0 * i / max(len(work_keys), 1)
                    print(
                        f"  sample cache {pct:6.2f}% [{i}/{len(work_keys)}] samples={total_samples} key={key}",
                        flush=True,
                    )
                if args.save_every_videos > 0 and len(chunk_keys) >= args.save_every_videos:
                    flush_chunk()
                if deadline_expired(args):
                    flush_chunk()
                    raise TimeBudgetExpired("Time budget reached while building sample cache.")
            except Exception as exc:
                skipped.append((key, str(exc)))
                print(f"  [skip] {key}: {exc}")
                done_keys.add(key)
                if deadline_expired(args):
                    flush_chunk()
                    raise TimeBudgetExpired("Time budget reached while building sample cache.") from exc
    finally:
        handle.remove()

    flush_chunk()

    if not part_files:
        raise RuntimeError("No training samples were extracted.")

    x, y1, y2 = _assemble_parts(part_files)
    stats = {
        "videos_seen": len(work_keys),
        "videos_completed": len(done_keys),
        "samples": int(x.shape[0]),
        "one_hot_positive_rate": float(y1.mean().item()),
        "boundary_positive_rate": float(y2.mean().item()),
        "skipped": skipped,
        "selected_keys": work_keys,
        "selected_keys_hash": hash_keys(work_keys),
        "completed_keys": sorted(done_keys),
        "sample_counts": dict(sorted(sample_counts.items())),
        "cache_config": cache_config,
        "partial_parts": part_files,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(
            {"config": cache_config, "features": x, "one_hot": y1, "boundary": y2, "stats": stats},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Sample cache saved -> {cache_path}")
    return x, y1, y2, stats
