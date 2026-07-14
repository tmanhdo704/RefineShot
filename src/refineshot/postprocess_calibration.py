"""Per-dataset post-process calibration on cached scores (no retraining).

This script answers a thesis experiment question: can per-dataset calibration of the
post-process knobs (temperature, Gaussian sigma, decision threshold) close the
ClipShots gap to the original AutoShot baseline (A0) *without retraining* and
*without tuning on the test set*?

It runs CPU-only on the cached scores that ship with this bundle. The
deploy-checkpoint logits behind the README "Kết Quả Đã Chạy" table are NOT in
the bundle (they lived in the git-ignored `eval_cache_shot_clipshots/`), so this
script does not attempt to reproduce those numbers. Instead it neighbours the
numbers that ARE reproducible from the bundle:

  - A0 baseline probabilities in `data/experiments/published_sweeps/`
  - Phase2 control A1 and full candidate B5 logits in
    `data/experiments/ablation_full/`

Honesty guardrail: the headline metric is a K-fold cross-validated deploy F1.
The post-process knobs are chosen on calibration folds and measured on a
held-out fold, so the reported number is NOT tuned on the data it is measured on.
The optimistic "best-sweep-on-test" ceiling is reported alongside it to expose
the optimism gap.

Usage:
    python postprocess_calibration.py --reproduce      # validate the harness
    python postprocess_calibration.py                  # run K-fold calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from refineshot.common import f1_pr, load_logits, scores_from_cache
from refineshot.eval import DEFAULT_THRESHOLDS, eval_at_threshold, evaluate_scenes, predictions_to_scenes

REPO_DIR = Path(__file__).resolve().parents[2]  # src/refineshot/X.py -> repo root
EXPERIMENTS = REPO_DIR / "data" / "experiments"
ARTIFACTS = EXPERIMENTS / "published_sweeps"
ABLATION = EXPERIMENTS / "ablation_full"

GT_PATHS = {
    "shot": ARTIFACTS / "gt_scenes_dict_baseline_v2.pickle",
    "clipshots": ARTIFACTS / "clipshots_test_gt_scenes.pickle",
    "bbc": ARTIFACTS / "bbc_shots_gt_scenes.pickle",
}

# Each model: input_kind + cached score file per dataset.
MODELS: dict[str, dict[str, Any]] = {
    "A0_autoshot_baseline": {
        "input_kind": "probabilities",
        "scores": {
            "shot": ARTIFACTS / "eval_cache_shot_clipshots" / "shot_test_logits.pkl",
            "clipshots": ARTIFACTS / "eval_cache_clipshots" / "clipshot_test_logits.pkl",
            "bbc": ARTIFACTS / "eval_cache_bbc" / "bbc_test_logits.pkl",
        },
    },
    "A1_phase2_control": {
        "input_kind": "logits",
        "scores": {
            "shot": ABLATION / "A1_phase2_bce_onehot" / "eval_cache" / "shot_test_logits.pkl",
            "clipshots": ABLATION / "A1_phase2_bce_onehot" / "eval_cache" / "clipshots_test_logits.pkl",
            "bbc": ABLATION / "A1_phase2_bce_onehot" / "eval_cache" / "bbc_test_logits.pkl",
        },
    },
    "B5_phase2_full": {
        "input_kind": "logits",
        "scores": {
            "shot": ABLATION / "B5_full_candidate" / "eval_cache" / "shot_test_logits.pkl",
            "clipshots": ABLATION / "B5_full_candidate" / "eval_cache" / "clipshots_test_logits.pkl",
            "bbc": ABLATION / "B5_full_candidate" / "eval_cache" / "bbc_test_logits.pkl",
        },
    },
}

# Reference numbers that ARE reproducible from this bundle (sanity targets).
REPRODUCE_TARGETS = [
    # name, model, dataset, temperature, sigma, threshold, mode, expected_f1
    ("A0 baseline (best thr)", "A0_autoshot_baseline", "shot", 1.0, 0.0, None, "best", 0.8405),
    ("A0 baseline (best thr)", "A0_autoshot_baseline", "clipshots", 1.0, 0.0, None, "best", 0.7649),
    ("A0 baseline (best thr)", "A0_autoshot_baseline", "bbc", 1.0, 0.0, None, "best", 0.9554),
    ("B4 temp+gauss (A1 logits)", "A1_phase2_control", "shot", 0.661785970550883, 2.0, 0.1, "fixed", 0.8540),
    ("B4 temp+gauss (A1 logits)", "A1_phase2_control", "clipshots", 0.661785970550883, 2.0, 0.1, "fixed", 0.7441),
    ("B4 temp+gauss (A1 logits)", "A1_phase2_control", "bbc", 0.661785970550883, 2.0, 0.1, "fixed", 0.9570),
    ("B5 full candidate", "B5_phase2_full", "shot", 0.3729196833931546, 2.0, 0.1, "fixed", 0.8542),
    ("B5 full candidate", "B5_phase2_full", "clipshots", 0.3729196833931546, 2.0, 0.1, "fixed", 0.7409),
    ("B5 full candidate", "B5_phase2_full", "bbc", 0.3729196833931546, 2.0, 0.1, "fixed", 0.9551),
]

DEFAULT_TEMPERATURES = (0.4, 0.5, 0.661785970550883, 1.0)
DEFAULT_SIGMAS = (0.0, 1.0, 2.0)
# Focused threshold grid over the action region (the cached scores concentrate here).
CALIB_THRESHOLDS = (0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5)


def restem(mapping: dict[str, Any]) -> dict[str, Any]:
    """Re-key by file stem so logits using stems and GT using `.mp4` keys match.

    This is the key-normalisation fix documented in the ablation report (section 9:
    ClipShots logits use the stem while ground truth keys carry the `.mp4` suffix).

    Close to `{common.normalize_key(k): v}` but deliberately kept separate:
    normalize_key also strips a "dataset:" prefix, and the bundled calibration
    caches this module reads cannot be re-verified here to prove the two are
    interchangeable. Do not swap without rerunning `--reproduce` on the bundle.
    """
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        out[Path(str(key)).stem] = value
    return out


def load_model_dataset(model: str, dataset: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    spec = MODELS[model]
    scores_path = spec["scores"][dataset]
    gt_path = GT_PATHS[dataset]
    scores = restem(load_logits(scores_path))
    with gt_path.open("rb") as f:
        import pickle

        gt = restem(pickle.load(f))
    common = sorted(set(scores) & set(gt))
    scores = {k: scores[k] for k in common}
    gt = {k: np.asarray(gt[k]) for k in common}
    return scores, gt, spec["input_kind"]


def make_pred(
    scores: dict[str, np.ndarray], input_kind: str, temperature: float, sigma: float
) -> dict[str, np.ndarray]:
    return scores_from_cache(scores, temperature=temperature, sigma=sigma, input_kind=input_kind)


def per_video_stats(
    scores: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    input_kind: str,
    temperatures: tuple[float, ...],
    sigmas: tuple[float, ...],
    thresholds: tuple[float, ...],
) -> dict[tuple[float, float], dict[str, np.ndarray]]:
    """Compute per-video (tp, fp, fn) at every threshold, once per (T, sigma).

    Returns stats[(T, sigma)][stem] = int array of shape (n_thresholds, 3). Folds are then
    cheap numpy sums over video subsets, so each scene conversion runs exactly once.
    """
    stats: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for temp in temperatures:
        for sigma in sigmas:
            pred = make_pred(scores, input_kind, temp, sigma)
            per_video: dict[str, np.ndarray] = {}
            for stem, arr in pred.items():
                probs = arr.squeeze()
                gt_scenes = np.asarray(gt[stem])
                rows = np.empty((len(thresholds), 3), dtype=np.int64)
                for ti, thr in enumerate(thresholds):
                    binary = (probs > thr).astype(np.uint8)
                    pred_scenes = predictions_to_scenes(binary)
                    rows[ti] = evaluate_scenes(gt_scenes, pred_scenes)
                per_video[stem] = rows
            stats[(temp, sigma)] = per_video
    return stats


def best_params(
    stats: dict[tuple[float, float], dict[str, np.ndarray]],
    stems: list[str],
    thresholds: tuple[float, ...],
) -> tuple[float, float, float, float]:
    """Pick (f1, temperature, sigma, threshold) maximising F1 over the given video subset."""
    best = (-1.0, 1.0, 0.0, thresholds[0])
    for (temp, sigma), per_video in stats.items():
        agg = np.sum([per_video[s] for s in stems], axis=0)  # (n_thr, 3)
        for ti, thr in enumerate(thresholds):
            f1, _, _ = f1_pr(int(agg[ti, 0]), int(agg[ti, 1]), int(agg[ti, 2]))
            if f1 > best[0]:
                best = (f1, temp, sigma, thr)
    return best


def kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(part) for part in np.array_split(idx, k)]


def cross_validate(
    model: str,
    dataset: str,
    temperatures: tuple[float, ...],
    sigmas: tuple[float, ...],
    thresholds: tuple[float, ...],
    k: int,
    seed: int,
) -> dict[str, Any]:
    scores, gt, input_kind = load_model_dataset(model, dataset)
    stems = sorted(scores)
    n = len(stems)
    thr_index = {thr: i for i, thr in enumerate(thresholds)}
    stats = per_video_stats(scores, gt, input_kind, temperatures, sigmas, thresholds)

    effective_k = min(k, n) if n > 1 else 1
    folds = kfold_indices(n, effective_k, seed)

    tp = fp = fn = 0
    chosen: list[dict[str, float]] = []
    for fold in folds:
        test_stems = [stems[i] for i in fold]
        test_set = set(test_stems)
        calib_stems = [s for s in stems if s not in test_set]
        if not calib_stems:  # tiny dataset fallback: calibrate on test (optimistic)
            calib_stems = test_stems
        _, temp, sigma, thr = best_params(stats, calib_stems, thresholds)
        agg = np.sum([stats[(temp, sigma)][s] for s in test_stems], axis=0)
        ti = thr_index[thr]
        f_tp, f_fp, f_fn = int(agg[ti, 0]), int(agg[ti, 1]), int(agg[ti, 2])
        tp += f_tp
        fp += f_fp
        fn += f_fn
        fold_f1, _, _ = f1_pr(f_tp, f_fp, f_fn)
        chosen.append({"temperature": temp, "sigma": sigma, "threshold": thr, "fold_f1": fold_f1})

    f1, precision, recall = f1_pr(tp, fp, fn)
    ceil_f1, ceil_t, ceil_s, ceil_thr = best_params(stats, stems, thresholds)

    return {
        "model": model,
        "dataset": dataset,
        "n_videos": n,
        "k_folds": effective_k,
        "cv_deploy": {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn},
        "cv_chosen_params": chosen,
        "test_ceiling": {"f1": ceil_f1, "temperature": ceil_t, "sigma": ceil_s, "threshold": ceil_thr},
    }


def run_reproduce(thresholds: list[float]) -> int:
    print("Reproduce check against numbers that are recoverable from this bundle:\n")
    print(f"{'config':28s} {'dataset':10s} {'got':>8s} {'expected':>9s} {'diff':>8s}  status")
    failures = 0
    for name, model, dataset, temp, sigma, thr, mode, expected in REPRODUCE_TARGETS:
        scores, gt, input_kind = load_model_dataset(model, dataset)
        pred = make_pred(scores, input_kind, temp, sigma)
        if mode == "best":
            got = max(eval_at_threshold(pred, gt, t)["f1"] for t in thresholds)
        else:
            got = eval_at_threshold(pred, gt, thr)["f1"]
        diff = got - expected
        ok = abs(diff) < 2e-3
        failures += 0 if ok else 1
        print(f"{name:28s} {dataset:10s} {got:8.4f} {expected:9.4f} {diff:+8.4f}  {'OK' if ok else 'MISMATCH'}")
    print()
    if failures:
        print(f"FAILED: {failures} mismatch(es). Harness does not match the bundle pipeline.")
    else:
        print("PASS: harness reproduces the bundle's A0/B4/B5 numbers. NOTE: the deploy-checkpoint")
        print("      logits behind the README table are not in the bundle and are not checked here.")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reproduce", action="store_true", help="Validate the harness against bundle numbers and exit."
    )
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model ids to calibrate.")
    parser.add_argument("--datasets", default="shot,clipshots,bbc")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="runs/calibration/results.json")
    parser.add_argument("--out-summary", default="runs/calibration/summary.md")
    args = parser.parse_args()

    thresholds = [float(t) for t in DEFAULT_THRESHOLDS]

    if args.reproduce:
        raise SystemExit(run_reproduce(thresholds))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    results: list[dict[str, Any]] = []
    for model in models:
        for dataset in datasets:
            print(f"calibrating {model} / {dataset} ...", flush=True)
            results.append(
                cross_validate(
                    model, dataset, DEFAULT_TEMPERATURES, DEFAULT_SIGMAS, CALIB_THRESHOLDS, args.k_folds, args.seed
                )
            )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "k_folds": args.k_folds, "results": results}, f, indent=2)
    print(f"results -> {out_json}")

    write_summary(Path(args.out_summary), results, args)
    print(f"summary -> {args.out_summary}")


def write_summary(path: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_md = {(r["model"], r["dataset"]): r for r in results}
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    lines = [
        "# Post-process Calibration (no retraining)",
        "",
        f"- Method: {args.k_folds}-fold cross-validation, seed {args.seed}. Post-process knobs",
        "  (temperature, Gaussian sigma, decision threshold) are chosen on calibration folds and",
        "  measured on the held-out fold (micro-averaged tp/fp/fn). This avoids tuning on the test set.",
        "- `cv_f1` is the honest, cross-validated deploy F1. `ceiling_f1` tunes on the full test set",
        "  (optimistic, shown only to expose the optimism gap).",
        "- The deploy-checkpoint logits behind the README table are not in the bundle; these models are",
        "  A0 baseline + the Phase2 controlled runs (A1, B5) whose caches ship with the bundle.",
        "",
        "## Cross-validated deploy F1 (honest)",
        "",
    ]
    header = "| Model | " + " | ".join(datasets) + " |"
    sep = "|" + "---|" * (len(datasets) + 1)
    lines.append(header)
    lines.append(sep)
    for model in models:
        cells = []
        for ds in datasets:
            r = by_md.get((model, ds))
            cells.append(f"{r['cv_deploy']['f1']:.4f}" if r else "-")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")

    lines += ["", "## Optimism gap (ceiling tuned on test vs CV)", "", header, sep]
    for model in models:
        cells = []
        for ds in datasets:
            r = by_md.get((model, ds))
            if r:
                cells.append(f"{r['test_ceiling']['f1']:.4f} / {r['cv_deploy']['f1']:.4f}")
            else:
                cells.append("-")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")

    # Headline: ClipShots Phase2 vs A0
    a0 = by_md.get(("A0_autoshot_baseline", "clipshots"))
    b5 = by_md.get(("B5_phase2_full", "clipshots"))
    if a0 and b5:
        delta = b5["cv_deploy"]["f1"] - a0["cv_deploy"]["f1"]
        verdict = "beats" if delta > 0 else "does NOT beat"
        lines += [
            "",
            "## Headline: does calibrated Phase2 beat A0 on ClipShots? (cross-validated)",
            "",
            f"- A0 baseline CV F1: {a0['cv_deploy']['f1']:.4f}",
            f"- B5 Phase2 CV F1: {b5['cv_deploy']['f1']:.4f}",
            f"- Delta (B5 - A0): {delta:+.4f} -> calibrated Phase2 **{verdict}** A0 on ClipShots.",
        ]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
