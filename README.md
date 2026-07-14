# RefineShot

RefineShot is an end-to-end AI system for shot boundary detection. It combines
a PyTorch training and inference pipeline with a FastAPI and React web app.

The portfolio includes source code, tests, deployment files, experiment notes,
and four unique model checkpoints managed with Git LFS. Datasets and generated
caches are excluded.

## Highlights

- PyTorch video model with 3D convolutions and Transformer blocks.
- Phase-2 training with focal loss, many-hot supervision, and resumable caches.
- Evaluation, calibration, and frame-to-scene post-processing.
- Shared inference runtime for CLI and web deployment.
- React, TypeScript, FastAPI, MongoDB, Docker, and export APIs.

## Structure

```text
RefineShot/
|-- app/          React frontend and FastAPI backend
|-- src/          Reusable AI package: model, training, evaluation, inference
|-- models/       Git LFS weights, model config, registry, and metadata
|-- scripts/      Train, evaluate, predict, and calibrate commands
|-- tests/        Model, training, runtime, and web tests
`-- docs/         Architecture, data layout, results, inference, and web guides
```

## Main results

The headline thesis result uses the AutoShot evaluation protocol: sweep the
decision threshold and report the best F1 for each dataset.

| Dataset | Best-threshold F1 |
|---|---:|
| SHOT | **0.8607** |
| BBC | 0.9656 |
| ClipShots | 0.7706 |

SHOT reaches `0.8607` at threshold `0.12`. Fixed-deployment results are lower
and are reported separately in [`docs/RESULTS.md`](docs/RESULTS.md).

## Setup

Python 3.12 is used by CI. FFmpeg must be available on `PATH`.

```powershell
git lfs install
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . ruff pytest
ruff check .
pytest -ra
```

## Train and use the model

```powershell
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/predict.py --help
python scripts/calibrate.py --help
```

Example video inference:

```powershell
python scripts/predict.py `
  --checkpoint models/checkpoints/refineshot_v8_final.pth `
  --videos-dir C:\path\to\videos `
  --no-eval `
  --results runs/predictions.json
```

See [`models/README.md`](models/README.md) for checkpoint details and
[`docs/DATA.md`](docs/DATA.md) for the external dataset layout.

## Run the web app

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open `http://localhost:5173`; the API and Swagger documentation are available
at `http://localhost:8000` and `/docs`.

## Attribution

The network architecture is based on the
[AutoShot project](https://github.com/wentaozhu/AutoShot) and its TransNetV2
port. The Phase-2 training pipeline, calibration, reproducibility layer,
runtime integration, and web product are the project-specific work presented
here.
