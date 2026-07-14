# Inference and evaluation

Run commands from the repository root after `pip install -e .` and
`git lfs pull`.

## Predict videos

```powershell
python scripts/predict.py `
  --checkpoint models/checkpoints/refineshot_v8_final.pth `
  --videos-dir C:\path\to\videos `
  --out-logits runs/inference_logits.pkl `
  --results runs/inference_results.json `
  --no-eval
```

Remove `--no-eval` and supply `--gt` to calculate metrics. Video stems must
match ground-truth keys.

## Cached evaluation

Prepare the inputs described in [`DATA.md`](DATA.md), then run:

```powershell
python scripts/evaluate.py
```

Defaults:

- checkpoint: `models/checkpoints/refineshot_v8_final.pth`;
- logits: `data/eval/shot_gt200_logits_cv2.pkl`;
- ground truth: `data/eval/gt_scenes_dict_baseline_v2.pickle`;
- output: `runs/v8_shot_cached_evaluation.json`.
