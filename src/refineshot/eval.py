import argparse
import json
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from refineshot import runtime
from refineshot.common import clean_key, load_logits

DEFAULT_THRESHOLDS = np.array(
    [
        0.02,
        0.06,
        0.1,
        0.15,
        0.2,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.255,
        0.26,
        0.265,
        0.27,
        0.275,
        0.28,
        0.2833,
        0.2867,
        0.29,
        0.292,
        0.294,
        0.296,
        0.298,
        0.3,
        0.302,
        0.304,
        0.306,
        0.308,
        0.31,
        0.3133,
        0.3167,
        0.32,
        0.325,
        0.33,
        0.335,
        0.34,
        0.345,
        0.35,
        0.36,
        0.37,
        0.38,
        0.39,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ],
    dtype=np.float32,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def get_frames(video_path: Path, width: int = 48, height: int = 27) -> np.ndarray:
    return runtime.decode_video_frames(video_path, width=width, height=height)


def get_batches(frames: np.ndarray) -> Iterator[np.ndarray]:
    yield from runtime.iter_frame_batches(frames)


def load_checkpoint_config(checkpoint_path: Path) -> dict:
    # The checkpoint is only needed for its post-process config (temperature/sigma/threshold)
    # and for video inference. When evaluating from cached logits it may be absent (it is not
    # shipped in the bundle), so degrade gracefully instead of crashing.
    if not checkpoint_path.exists():
        print(
            f"WARNING: checkpoint not found ({checkpoint_path}); post-process config unavailable. "
            "Pass --temperature/--sigma/--threshold explicitly to reproduce the deploy numbers "
            "(deploy: 0.24128 / 2.0 / 0.19). See models/README.md.",
            flush=True,
        )
        return {}
    return runtime.load_checkpoint_config(checkpoint_path)


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    return runtime.load_model(checkpoint_path, device)


def predict_video_logits(model: torch.nn.Module, video_path: Path, device: str) -> np.ndarray:
    return runtime.predict_video_logits(model, video_path, runtime.resolve_device(device))


def video_files(videos_dir: Path) -> list[Path]:
    return sorted(p for p in videos_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def ground_truth_keys(gt_path: Path) -> set[str]:
    with gt_path.open("rb") as f:
        gt = pickle.load(f)
    return {clean_key(key) for key in gt}


def filter_video_files_for_keys(files: list[Path], include_keys: set[str] | None) -> list[Path]:
    if include_keys is None:
        return files
    available = {clean_key(path.stem): path for path in files}
    missing = sorted(include_keys - set(available))
    if missing:
        sample = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Video source is missing {len(missing)} ground-truth videos. Missing sample: {sample}"
        )
    return [available[key] for key in sorted(include_keys)]


def save_inference_logits(out_logits_path: Path, videos_dir: Path, logits: dict) -> None:
    out_logits_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_logits_path.with_suffix(out_logits_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump({"config": {"source": str(videos_dir)}, "logits": logits}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(out_logits_path)


def load_resume_logits(out_logits_path: Path, resume: bool) -> dict:
    if not resume or not out_logits_path.is_file():
        return {}
    logits = dict(load_logits(out_logits_path))
    if logits:
        print(f"resuming logits from {out_logits_path}: {len(logits)} videos already cached", flush=True)
    return logits


def run_video_inference(
    checkpoint_path: Path,
    videos_dir: Path,
    out_logits_path: Path,
    device: str,
    include_keys: set[str] | None = None,
    resume: bool = True,
) -> dict:
    files = video_files(videos_dir)
    total_files = len(files)
    files = filter_video_files_for_keys(files, include_keys)
    if not files:
        raise FileNotFoundError(f"No video files found under {videos_dir}")
    if include_keys is not None:
        print(f"filtered video source: {len(files)}/{total_files} files match ground truth", flush=True)

    model = load_model(checkpoint_path, device)
    logits = load_resume_logits(out_logits_path, resume=resume)
    completed_keys = {clean_key(key) for key in logits}
    for index, video_path in enumerate(files, 1):
        pct = 100.0 * index / len(files)
        key = video_path.stem
        if clean_key(key) in completed_keys:
            print(f"inference {pct:6.2f}% [{index}/{len(files)}] {video_path.name} [cached]", flush=True)
            continue
        print(f"inference {pct:6.2f}% [{index}/{len(files)}] {video_path.name}", flush=True)
        logits[video_path.stem] = predict_video_logits(model, video_path, device)
        completed_keys.add(clean_key(key))
        save_inference_logits(out_logits_path, videos_dir, logits)

    save_inference_logits(out_logits_path, videos_dir, logits)
    print(f"logits saved -> {out_logits_path}")
    return logits




def logits_to_predictions(logits: dict, temperature: float, sigma: float) -> dict:
    pred = {}
    for key, arr in logits.items():
        probs = runtime.logits_to_probabilities(arr, temperature=temperature, sigma=sigma)
        pred[clean_key(key)] = probs[:, np.newaxis].astype(np.float32)
    return pred


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    return runtime.predictions_to_scenes(predictions)


def evaluate_scenes(
    gt_scenes: np.ndarray, pred_scenes: np.ndarray, tolerance: int = 2, return_mistakes: bool = False
) -> tuple:
    """Canonical two-pointer scene-boundary matcher (tp, fp, fn).

    Adapted from https://github.com/gyglim/shot-detection-evaluation. This is the single
    source of truth for the matching algorithm; ``refineshot.utils.evaluate_scenes`` wraps
    it to additionally return precision/recall/F1.

    tolerance:
        Number of frames a detection may miss the ground-truth boundary by and still count
        as correct. Examples (with the transition midpoints each scene pair maps to):

        tolerance = 0
          pred [[0, 5], [6, 9]] -> pred_trans [[5.5, 5.5]]
          gt   [[0, 5], [6, 9]] -> gt_trans   [[5.5, 5.5]] -> HIT
          gt   [[0, 4], [5, 9]] -> gt_trans   [[4.5, 4.5]] -> MISS
        tolerance = 2
          pred [[0, 5], [6, 9]] -> pred_trans [[4.5, 6.5]]
          gt   [[0, 3], [4, 9]] -> gt_trans   [[2.5, 4.5]] -> HIT
          gt   [[0, 2], [3, 9]] -> gt_trans   [[1.5, 3.5]] -> MISS

    return_mistakes:
        When True, also return the lists of unmatched predicted/ground-truth transitions
        ``(tp, fp, fn, fp_mistakes, fn_mistakes)``.
    """
    shift = tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    i = j = tp = fp = fn = 0
    fp_mistakes, fn_mistakes = [], []
    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            if return_mistakes:
                fn_mistakes.append(gt_trans[i])
            i += 1
        elif i == len(gt_trans):
            fp += 1
            if return_mistakes:
                fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            if return_mistakes:
                fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            if return_mistakes:
                fn_mistakes.append(gt_trans[i])
            i += 1
        else:
            tp += 1
            i += 1
            j += 1
    if return_mistakes:
        return tp, fp, fn, fp_mistakes, fn_mistakes
    return tp, fp, fn


def eval_at_threshold(pred: dict, gt: dict, threshold: float) -> dict:
    tp = fp = fn = 0
    for key, scores in pred.items():
        if key not in gt:
            continue
        binary = (scores.squeeze() > threshold).astype(np.uint8)
        pred_scenes = predictions_to_scenes(binary)
        tp_i, fp_i, fn_i = evaluate_scenes(np.asarray(gt[key]), pred_scenes)
        tp += int(tp_i)
        fp += int(fp_i)
        fn += int(fn_i)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate(pred: dict, gt_path: Path, threshold: float) -> dict:
    with gt_path.open("rb") as f:
        gt = pickle.load(f)

    common_keys = sorted(set(pred) & set(gt))
    common_pred = {key: pred[key] for key in common_keys}
    common_gt = {key: gt[key] for key in common_keys}
    sweep = [eval_at_threshold(common_pred, common_gt, thr) for thr in DEFAULT_THRESHOLDS]
    best = max(sweep, key=lambda item: item["f1"])
    deploy = eval_at_threshold(common_pred, common_gt, threshold)
    return {
        "videos_evaluated": len(common_pred),
        "gt_videos": len(gt),
        "missing_prediction_keys": sorted(set(gt) - set(pred)),
        "extra_prediction_keys": sorted(set(pred) - set(gt)),
        "best_sweep": best,
        "deploy": deploy,
        "top_thresholds": sorted(sweep, key=lambda item: item["f1"], reverse=True)[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RefineShot Phase-2 inference/evaluation.")
    parser.add_argument("--checkpoint", default="models/checkpoints/refineshot_v8_final.pth")
    parser.add_argument("--videos-dir", default="")
    parser.add_argument("--logits-cache", default="data/eval/shot_gt200_logits_cv2.pkl")
    parser.add_argument("--out-logits", default="runs/inference_logits.pkl")
    parser.add_argument("--gt", default="data/eval/gt_scenes_dict_baseline_v2.pickle")
    parser.add_argument("--results", default="runs/inference_results.json")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--filter-to-gt", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    # Only consult the checkpoint for post-process config when a value is missing from the CLI.
    # If the user supplies --temperature/--sigma/--threshold, no checkpoint is needed (and no warning).
    need_config = args.temperature is None or args.sigma is None or args.threshold is None
    config = load_checkpoint_config(checkpoint_path) if need_config else {}
    temperature = float(
        args.temperature if args.temperature is not None else config.get("temperature", runtime.DEFAULT_TEMPERATURE)
    )
    sigma = float(args.sigma if args.sigma is not None else config.get("sigma", runtime.DEFAULT_SIGMA))
    threshold = float(
        args.threshold if args.threshold is not None else config.get("threshold", runtime.DEFAULT_THRESHOLD)
    )

    if args.videos_dir:
        include_keys = ground_truth_keys(Path(args.gt)) if args.filter_to_gt else None
        logits = run_video_inference(
            checkpoint_path,
            Path(args.videos_dir),
            Path(args.out_logits),
            args.device,
            include_keys=include_keys,
            resume=not args.no_resume,
        )
        logits_source = str(Path(args.out_logits))
    else:
        logits_path = Path(args.logits_cache)
        if not logits_path.exists():
            raise FileNotFoundError(
                f"Missing logits cache: {logits_path}. Provide --videos-dir to run inference from videos."
            )
        logits = load_logits(logits_path)
        logits_source = str(logits_path)

    pred = logits_to_predictions(logits, temperature=temperature, sigma=sigma)
    result = {
        "checkpoint": str(checkpoint_path),
        "logits_source": logits_source,
        "postprocess": {
            "temperature": temperature,
            "sigma": sigma,
            "threshold": threshold,
        },
    }

    if not args.no_eval:
        result.update(evaluate(pred, Path(args.gt), threshold))

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if not args.no_eval:
        deploy = result["deploy"]
        best = result["best_sweep"]
        print(
            "deploy: "
            f"F1={deploy['f1']:.6f} P={deploy['precision']:.6f} R={deploy['recall']:.6f} "
            f"thr={deploy['threshold']:.4f} TP={deploy['tp']} FP={deploy['fp']} FN={deploy['fn']}"
        )
        print(
            "best sweep: "
            f"F1={best['f1']:.6f} P={best['precision']:.6f} R={best['recall']:.6f} "
            f"thr={best['threshold']:.4f} TP={best['tp']} FP={best['fp']} FN={best['fn']}"
        )
    print(f"results saved -> {results_path}")


if __name__ == "__main__":
    main()
