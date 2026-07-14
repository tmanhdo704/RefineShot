# Model registry

The repository includes the four unique checkpoints used by training and the
web demo. They are versioned with Git LFS under `models/checkpoints/`.

| Checkpoint | Purpose |
|---|---|
| `refineshot_v8_final.pth` | Default final-v8 inference model |
| `refineshot_standard.pth` | Standard RefineShot comparison model |
| `refineshot_heatmap.pth` | HeatMap variant used by the web comparison page |
| `autoshot_base.pth` | Original AutoShot initialization for Phase-2 training |

Exact byte sizes and SHA-256 digests are recorded in `models/registry.json`.
The pinned training and deployment settings are stored in `models/config.json`.
After cloning, run `git lfs pull` before training or model-backed inference.

Generated checkpoints belong under `runs/` and remain ignored by Git.
