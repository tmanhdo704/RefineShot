# Architecture

RefineShot has three execution layers:

1. `src/refineshot/` contains the model, training, evaluation, calibration, and
   shared inference runtime.
2. `scripts/` provides simple commands for AI engineering workflows.
3. `app/` exposes the runtime through FastAPI and a React interface.

## Inference flow

1. FFmpeg decodes and resizes video frames.
2. The runtime creates overlapping frame windows.
3. The model produces per-frame boundary logits.
4. Temperature scaling and Gaussian smoothing calibrate the scores.
5. Thresholded boundaries are converted into scene ranges.
6. The web worker creates thumbnails, storyboards, and exports.

`src/refineshot/runtime.py` is shared by both CLI and web inference.
`app/backend/app/ml/refineshot_runtime.py` is only a web adapter.

## Repository layers

| Location | Responsibility |
|---|---|
| `src/` | AI implementation |
| `models/` | Checkpoints, pinned config, hashes, and metadata |
| `scripts/` | Train, evaluate, predict, and calibrate |
| `app/` | Frontend, API, database integration, and jobs |
| `tests/` | Automated verification |
| `docs/` | Data instructions and experimental evidence |

Datasets, generated caches, new checkpoints, and uploaded media stay outside
Git. The four portfolio checkpoints are stored with Git LFS.
