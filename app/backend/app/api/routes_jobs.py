import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.routes_health import DEFAULT_PRESET, PRESETS
from app.core.config import get_settings
from app.db.mongo import get_database
from app.services.job_serializer import serialize_job
from app.services.storage_service import (
    delete_cloudinary_assets,
    publish_asset,
    remove_job_files,
    safe_video_extension,
    upload_dir,
)
from app.worker.runner import process_job

router = APIRouter(prefix="/jobs", tags=["jobs"])

MAX_HISTORY = 7


async def _enforce_history_limit(db) -> None:
    total = await db.jobs.count_documents({})
    if total > MAX_HISTORY:
        oldest = await db.jobs.find(
            {}, {"_id": 1, "cloudinary": 1}
        ).sort("created_at", 1).limit(total - MAX_HISTORY).to_list(length=total - MAX_HISTORY)
        for old in oldest:
            await asyncio.to_thread(delete_cloudinary_assets, old)
            await db.jobs.delete_one({"_id": old["_id"]})
            remove_job_files(old["_id"])


@router.get("/")
async def list_jobs() -> dict:
    db = get_database()
    cursor = db.jobs.find(
        {},
        {"_id": 1, "status": 1, "stage": 1, "progress": 1, "created_at": 1,
         "input": 1, "processing": 1, "summary": 1, "error": 1}
    ).sort("created_at", -1).limit(MAX_HISTORY)
    jobs = await cursor.to_list(length=MAX_HISTORY)
    return {"jobs": [serialize_job(j) for j in jobs]}


@router.post("/from-upload")
async def create_job_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    preset: str = Form(DEFAULT_PRESET),
) -> dict:
    settings = get_settings()
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{preset}'")
    preset_cfg = PRESETS[preset]
    model_path = settings.refineshot_models_dir / preset_cfg["filename"]
    if not model_path.is_file():
        raise HTTPException(status_code=400, detail=f"Model file for preset '{preset}' not found")

    try:
        extension = safe_video_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = f"job_{uuid4().hex[:16]}"
    target_dir = upload_dir(job_id)
    video_path = target_dir / f"input{extension}"

    size_bytes = 0
    async with aiofiles.open(video_path, "wb") as handle:
        while chunk := await file.read(1024 * 1024):
            size_bytes += len(chunk)
            if size_bytes > settings.max_upload_bytes:
                remove_job_files(job_id)
                raise HTTPException(status_code=413, detail=f"Upload limit is {settings.max_upload_mb}MB")
            await handle.write(chunk)

    if size_bytes == 0:
        remove_job_files(job_id)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    video_asset = await asyncio.to_thread(publish_asset, video_path, job_id, "input", "video")
    cloudinary_assets = [video_asset] if "public_id" in video_asset else []

    now = datetime.now(timezone.utc)
    document = {
        "_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0.05,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "input": {
            "original_name": Path(file.filename or "video").name,
            "size_bytes": size_bytes,
            "content_type": file.content_type or "application/octet-stream",
        },
        "processing": {
            "model": "pending",
            "preset": preset,
            "display_name": preset_cfg["display_name"],
            "model_path": str(model_path),
            "threshold": preset_cfg["threshold"],
        },
        "storage": {
            "video_url": video_asset["url"],
        },
        "cloudinary": {
            "enabled": settings.use_cloudinary,
            "folder": settings.cloudinary_folder,
            "assets": cloudinary_assets,
        },
        "summary": None,
        "boundaries": [],
        "scenes": [],
        "exports": {},
        "artifacts": {},
        "error": None,
        "internal": {
            "video_path": str(video_path),
        },
    }

    db = get_database()
    await db.jobs.insert_one(document)
    await _enforce_history_limit(db)
    background_tasks.add_task(process_job, job_id)
    return serialize_job(document)


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    db = get_database()
    job = await db.jobs.find_one({"_id": job_id})
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return serialize_job(job)


@router.get("/{job_id}/scenes")
async def get_job_scenes(job_id: str) -> dict:
    db = get_database()
    job = await db.jobs.find_one({"_id": job_id}, {"scenes": 1})
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "scenes": job.get("scenes", [])}


@router.get("/{job_id}/boundaries")
async def get_job_boundaries(job_id: str) -> dict:
    db = get_database()
    job = await db.jobs.find_one({"_id": job_id}, {"boundaries": 1})
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "boundaries": job.get("boundaries", [])}


@router.get("/{job_id}/exports/{kind}")
async def download_export(job_id: str, kind: str) -> FileResponse:
    names = {
        "json": ("result.json", "application/json"),
        "csv": ("scenes.csv", "text/csv"),
        "txt": ("summary.txt", "text/plain"),
    }
    if kind not in names:
        raise HTTPException(status_code=404, detail="Export not found")

    filename, media_type = names[kind]
    path = get_settings().storage_dir / "jobs" / job_id / "exports" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export not ready")
    return FileResponse(path, media_type=media_type, filename=filename)


@router.delete("/{job_id}")
async def delete_job(job_id: str) -> dict[str, str]:
    db = get_database()
    job = await db.jobs.find_one({"_id": job_id})
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    await asyncio.to_thread(delete_cloudinary_assets, job)
    result = await db.jobs.delete_one({"_id": job_id})
    remove_job_files(job_id)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted"}
