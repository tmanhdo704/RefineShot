"""Reproduce the final-v8 SHOT evaluation from bundled cached logits."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from refineshot import v8_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "checkpoints" / v8_config.FINAL_CHECKPOINT_NAME,
    )
    parser.add_argument(
        "--logits-cache",
        type=Path,
        default=ROOT / "data" / "eval" / "shot_gt200_logits_cv2.pkl",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=ROOT / "data" / "eval" / "gt_scenes_dict_baseline_v2.pickle",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "runs" / "v8_shot_cached_evaluation.json",
    )
    args = parser.parse_args()

    missing = [path for path in (args.checkpoint, args.logits_cache, args.ground_truth) if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Missing evaluation input(s):\n{formatted}\nSee docs/DATA.md.")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "refineshot.eval",
        "--checkpoint",
        str(args.checkpoint),
        "--logits-cache",
        str(args.logits_cache),
        "--gt",
        str(args.ground_truth),
        "--results",
        str(args.results),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
