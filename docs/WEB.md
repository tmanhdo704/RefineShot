# Web app

- `app/frontend`: React, TypeScript, and Vite.
- `app/backend`: FastAPI, MongoDB, and local/Cloudinary storage.
- `src/refineshot/runtime.py`: shared inference runtime.
- `models/checkpoints`: model files managed with Git LFS.

## Docker

```powershell
git lfs pull
Copy-Item .env.example .env
docker compose up --build
```

For NVIDIA GPUs:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The frontend runs at `http://localhost:5173`, while the API runs at
`http://localhost:8000`.

Docker mounts `models/checkpoints/` read-only. The default checkpoint is
`refineshot_v8_final.pth`.

## Local development

```powershell
pip install -r requirements.txt
pip install -r app/backend/requirements.txt
pip install -e .

Set-Location app/backend
uvicorn app.main:app --reload
```

In another terminal:

```powershell
Set-Location app/frontend
npm ci
npm run dev
```

Main endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Runtime status |
| `GET` | `/api/models` | Available model presets |
| `POST` | `/api/jobs/from-upload` | Upload and analyze a video |
| `GET` | `/api/jobs/{job_id}` | Job result |
| `GET` | `/api/jobs/{job_id}/exports/{kind}` | JSON, CSV, or TXT export |
| `DELETE` | `/api/jobs/{job_id}` | Remove a job |
