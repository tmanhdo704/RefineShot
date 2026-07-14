import argparse
import json
import os
import pickle
import random
import re
from typing import Any

import numpy as np

DEFAULT_SHOT_ROOT = os.path.join(".", "data", "ShotDataset")
DEFAULT_CLIPSHOTS_ROOT = os.path.join(".", "data", "ClipShots")
DEFAULT_OUT = "./shot_clipshots_trainval.pickle"


def parse_shot_ground_truth(path: str, video_dir: str, key_prefix: str) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    missing_videos: list[str] = []
    cur_name: str | None = None
    cur_n_frames: int | None = None
    cur_transitions: list[list[int]] = []

    def flush() -> None:
        nonlocal cur_name, cur_n_frames, cur_transitions
        if cur_name is None:
            return
        video_path = os.path.join(video_dir, cur_name + ".mp4")
        if not os.path.exists(video_path):
            missing_videos.append(cur_name)
            return
        key = f"{key_prefix}:{cur_name}"
        entries[key] = {
            "dataset": "shot",
            "source_split": key_prefix,
            "source_name": cur_name,
            "video_path": video_path,
            "transitions": np.asarray(cur_transitions, dtype=np.int32),
            "n_frames": cur_n_frames,
        }

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                flush()
                cur_name = None
                cur_n_frames = None
                cur_transitions = []
                continue

            if ".mp4" in line:
                flush()
                cur_name = None
                cur_n_frames = None
                cur_transitions = []

                parts = line.split()
                m = re.match(r"^(.+)\.mp4$", parts[0])
                if m is None:
                    continue
                cur_name = m.group(1)
                if len(parts) >= 2:
                    try:
                        cur_n_frames = int(float(parts[1]))
                    except ValueError:
                        cur_n_frames = None
                continue

            if cur_name is None:
                continue
            try:
                start, end = map(int, line.split(","))
            except ValueError:
                continue
            cur_transitions.append([start, end])

    flush()
    if missing_videos:
        print(
            f"WARNING [{key_prefix}]: {len(missing_videos)} videos listed in GT were not found under "
            f"{video_dir} and were skipped. Check --shot-root. First few: {missing_videos[:5]}",
            flush=True,
        )
    return entries


def load_clipshots_split(root: str, split: str) -> dict[str, dict[str, Any]]:
    ann_path = os.path.join(root, "annotations", f"{split}.json")
    list_path = os.path.join(root, "video_lists", f"{split}.txt")
    video_dir = os.path.join(root, "videos", split)

    with open(ann_path, encoding="utf-8") as f:
        annotations = json.load(f)
    with open(list_path, encoding="utf-8") as f:
        listed = [line.strip() for line in f if line.strip()]

    entries: dict[str, dict[str, Any]] = {}
    missing_ann = 0
    missing_video = 0
    for filename in listed:
        ann = annotations.get(filename)
        if ann is None:
            missing_ann += 1
            continue
        video_path = os.path.join(video_dir, filename)
        if not os.path.exists(video_path):
            missing_video += 1
            continue
        stem = filename[:-4] if filename.endswith(".mp4") else filename
        key = f"clipshots_{split}:{stem}"
        entries[key] = {
            "dataset": "clipshots",
            "source_split": split,
            "source_name": stem,
            "video_path": video_path,
            "transitions": np.asarray(ann.get("transitions", []), dtype=np.int32),
            "n_frames": int(float(ann.get("frame_num", 0))) if ann.get("frame_num") is not None else None,
        }
    if missing_ann or missing_video:
        print(
            f"WARNING [clipshots:{split}]: {len(listed)} listed, kept {len(entries)}; "
            f"skipped {missing_ann} without annotation and {missing_video} without video under "
            f"{video_dir}. Check --clipshots-root.",
            flush=True,
        )
    return entries


def split_keys_by_dataset(
    entries: dict[str, dict[str, Any]],
    val_ratio: float,
    max_val_videos: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    train_keys: list[str] = []
    val_keys: list[str] = []

    groups: dict[str, list[str]] = {}
    for key, entry in entries.items():
        groups.setdefault(f"{entry['dataset']}:{entry['source_split']}", []).append(key)

    for group_keys in groups.values():
        group_keys = sorted(group_keys)
        rng.shuffle(group_keys)
        n_val = max(1, int(round(len(group_keys) * val_ratio))) if len(group_keys) > 1 else 0
        val_keys.extend(group_keys[:n_val])
        train_keys.extend(group_keys[n_val:])

    if max_val_videos > 0 and len(val_keys) > max_val_videos:
        val_keys = sorted(val_keys)
        rng.shuffle(val_keys)
        keep_val = set(val_keys[:max_val_videos])
        overflow = [key for key in val_keys if key not in keep_val]
        train_keys.extend(overflow)
        val_keys = list(keep_val)

    return sorted(train_keys), sorted(val_keys)


def count_transitions(entries: dict[str, dict[str, Any]], keys: list[str]) -> int:
    return int(sum(len(entries[key]["transitions"]) for key in keys))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot-root", default=DEFAULT_SHOT_ROOT)
    parser.add_argument("--clipshots-root", default=DEFAULT_CLIPSHOTS_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--max-val-videos", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-clipshots-only-gradual",
        action="store_true",
        help="Also include ClipShots only_gradual split in the training pool.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail if the validation split is empty (on top of the always-on empty-train check).",
    )
    args = parser.parse_args()

    shot_train_gt = os.path.join(args.shot_root, "train", "GT", "ground_truth.txt")
    shot_train_video_dir = os.path.join(args.shot_root, "train", "videos")
    shot_test_gt = os.path.join(args.shot_root, "test", "GT", "ground_truth.txt")
    shot_test_video_dir = os.path.join(args.shot_root, "test", "videos")

    entries: dict[str, dict[str, Any]] = {}
    shot_train = parse_shot_ground_truth(shot_train_gt, shot_train_video_dir, "shot_train")
    shot_test = parse_shot_ground_truth(shot_test_gt, shot_test_video_dir, "shot_test")
    clip_train = load_clipshots_split(args.clipshots_root, "train")
    clip_test = load_clipshots_split(args.clipshots_root, "test")

    entries.update(shot_train)
    entries.update(clip_train)
    if args.include_clipshots_only_gradual:
        entries.update(load_clipshots_split(args.clipshots_root, "only_gradual"))

    train_keys, val_keys = split_keys_by_dataset(
        entries,
        val_ratio=args.val_ratio,
        max_val_videos=args.max_val_videos,
        seed=args.seed,
    )

    # Guard against silently writing broken metadata when datasets are mislocated.
    if not entries or not train_keys:
        raise RuntimeError(
            f"Refusing to write metadata: found {len(entries)} entries / {len(train_keys)} train keys. "
            "Check --shot-root / --clipshots-root (see the WARNING lines above)."
        )
    if args.strict and not val_keys:
        raise RuntimeError(
            "Refusing to write metadata in --strict mode: validation split is empty. "
            "Lower --val-ratio expectations or check the dataset."
        )

    payload = {
        "entries": entries,
        "train_keys": train_keys,
        "val_keys": val_keys,
        "shot_test_entries": shot_test,
        "excluded_clipshots_test_count": len(clip_test),
        "config": {
            "shot_root": os.path.abspath(args.shot_root),
            "clipshots_root": os.path.abspath(args.clipshots_root),
            "val_ratio": args.val_ratio,
            "max_val_videos": args.max_val_videos,
            "seed": args.seed,
            "include_clipshots_only_gradual": args.include_clipshots_only_gradual,
        },
    }

    with open(args.out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved -> {args.out}")
    print("Train/eval split:")
    print(f"  Shot train videos        : {len(shot_train)}")
    print(f"  Shot test videos         : {len(shot_test)}  (held out for final test only)")
    print(f"  ClipShots train videos   : {len(clip_train)}")
    print(f"  ClipShots test videos    : {len(clip_test)}  (excluded)")
    if args.include_clipshots_only_gradual:
        gradual_count = sum(1 for e in entries.values() if e["source_split"] == "only_gradual")
        print(f"  ClipShots only_gradual   : {gradual_count}")
    print(f"  Combined train videos    : {len(train_keys)}")
    print(f"  Combined val videos      : {len(val_keys)}")
    print(f"  Train transitions        : {count_transitions(entries, train_keys)}")
    print(f"  Val transitions          : {count_transitions(entries, val_keys)}")


if __name__ == "__main__":
    main()
