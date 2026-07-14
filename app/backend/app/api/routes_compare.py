import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.api.routes_health import DEFAULT_PRESET, PRESETS
from app.core.config import get_settings
from app.db.mongo import get_database
from app.services.job_serializer import serialize_job
from app.services.storage_service import (
    publish_asset,
    remove_job_files,
    safe_video_extension,
    upload_dir,
)
from app.worker.runner import process_job

router = APIRouter(prefix="/compare", tags=["compare"])


def _make_job_doc(job_id: str, preset: str, preset_cfg: dict, model_path: Path,
                  video_path: Path, video_asset: dict, size_bytes: int,
                  original_name: str, content_type: str, settings) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0.05,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "input": {
            "original_name": original_name,
            "size_bytes": size_bytes,
            "content_type": content_type,
        },
        "processing": {
            "model": "pending",
            "preset": preset,
            "display_name": preset_cfg["display_name"],
            "model_path": str(model_path),
            "threshold": preset_cfg["threshold"],
        },
        "storage": {"video_url": video_asset["url"]},
        "cloudinary": {
            "enabled": settings.use_cloudinary,
            "folder": settings.cloudinary_folder,
            "assets": [video_asset] if "public_id" in video_asset else [],
        },
        "summary": None,
        "boundaries": [],
        "scenes": [],
        "exports": {},
        "artifacts": {},
        "error": None,
        "internal": {"video_path": str(video_path)},
    }


@router.post("/from-upload")
async def create_compare_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    preset_a: str = Form(DEFAULT_PRESET),
    preset_b: str = Form("best_clipshot"),
) -> dict:
    settings = get_settings()

    if preset_a not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{preset_a}'")
    if preset_b not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{preset_b}'")

    cfg_a = PRESETS[preset_a]
    cfg_b = PRESETS[preset_b]
    path_a = settings.refineshot_models_dir / cfg_a["filename"]
    path_b = settings.refineshot_models_dir / cfg_b["filename"]

    if not path_a.is_file():
        raise HTTPException(status_code=400, detail=f"Model file for '{preset_a}' not found")
    if not path_b.is_file():
        raise HTTPException(status_code=400, detail=f"Model file for '{preset_b}' not found")

    try:
        extension = safe_video_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Save file once, symlink / copy for second job
    job_id_a = f"job_{uuid4().hex[:16]}"
    job_id_b = f"job_{uuid4().hex[:16]}"

    video_path_a = upload_dir(job_id_a) / f"input{extension}"
    video_path_b = upload_dir(job_id_b) / f"input{extension}"

    size_bytes = 0
    async with aiofiles.open(video_path_a, "wb") as handle:
        while chunk := await file.read(1024 * 1024):
            size_bytes += len(chunk)
            if size_bytes > settings.max_upload_bytes:
                remove_job_files(job_id_a)
                remove_job_files(job_id_b)
                raise HTTPException(status_code=413, detail=f"Upload limit is {settings.max_upload_mb}MB")
            await handle.write(chunk)

    if size_bytes == 0:
        remove_job_files(job_id_a)
        remove_job_files(job_id_b)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Hard-link second job to same file (saves disk space)
    try:
        video_path_b.hardlink_to(video_path_a)
    except OSError:
        import shutil
        shutil.copy2(video_path_a, video_path_b)

    original_name = Path(file.filename or "video").name
    content_type = file.content_type or "application/octet-stream"

    video_asset_a = await asyncio.to_thread(publish_asset, video_path_a, job_id_a, "input", "video")
    video_asset_b = await asyncio.to_thread(publish_asset, video_path_b, job_id_b, "input", "video")

    doc_a = _make_job_doc(job_id_a, preset_a, cfg_a, path_a, video_path_a,
                          video_asset_a, size_bytes, original_name, content_type, settings)
    doc_b = _make_job_doc(job_id_b, preset_b, cfg_b, path_b, video_path_b,
                          video_asset_b, size_bytes, original_name, content_type, settings)

    db = get_database()
    await db.jobs.insert_many([doc_a, doc_b])
    background_tasks.add_task(process_job, job_id_a)
    background_tasks.add_task(process_job, job_id_b)

    return {
        "job_a": serialize_job(doc_a),
        "job_b": serialize_job(doc_b),
    }
