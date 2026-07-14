import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.db.mongo import get_database
from app.ml.scene_detector import VideoAnalysisSettings, analyze_video
from app.services.export_service import write_exports

_job_lock = asyncio.Lock()


async def process_job(job_id: str) -> None:
    async with _job_lock:
        db = get_database()
        job = await db.jobs.find_one({"_id": job_id})
        if not job:
            return

        try:
            await _set_job_state(job_id, status="running", stage="probing", progress=0.15)

            video_path = Path(job["internal"]["video_path"])
            processing = job.get("processing", {})
            options = VideoAnalysisSettings(
                model_path=Path(processing["model_path"]),
                threshold=processing.get("threshold"),
            )

            await _set_job_state(job_id, status="running", stage="analyzing", progress=0.35)
            analysis = await asyncio.to_thread(analyze_video, job_id, video_path, options)

            await _set_job_state(job_id, status="running", stage="finalizing", progress=0.85)
            result = {
                "job_id": job_id,
                "input": {**job.get("input", {}), **analysis["input"]},
                "processing": analysis["processing"],
                "summary": analysis["summary"],
                "boundaries": analysis["boundaries"],
                "scenes": analysis["scenes"],
                "artifacts": analysis["artifacts"],
                "created_at": job["created_at"].isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            exports = await asyncio.to_thread(write_exports, job_id, result)

            # Xóa video local sau khi đã có Cloudinary URL (tiết kiệm disk)
            if get_settings().use_cloudinary and video_path.exists():
                await asyncio.to_thread(shutil.rmtree, video_path.parent, True)

            export_assets = exports.pop("assets", [])
            artifact_assets = analysis["artifacts"].pop("assets", [])
            asset_refs = job.get("cloudinary", {}).get("assets", []) + artifact_assets + export_assets

            await db.jobs.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "done",
                        "stage": "done",
                        "progress": 1.0,
                        "finished_at": datetime.now(timezone.utc),
                        "input": result["input"],
                        "processing": result["processing"],
                        "summary": result["summary"],
                        "boundaries": result["boundaries"],
                        "scenes": result["scenes"],
                        "artifacts": result["artifacts"],
                        "exports": exports,
                        "cloudinary": {
                            "enabled": get_settings().use_cloudinary,
                            "folder": get_settings().cloudinary_folder,
                            "assets": asset_refs,
                        },
                        "error": None,
                    }
                },
            )
        except Exception as exc:
            await db.jobs.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "error",
                        "stage": "error",
                        "progress": 1.0,
                        "finished_at": datetime.now(timezone.utc),
                        "error": str(exc),
                    }
                },
            )


async def _set_job_state(job_id: str, status: str, stage: str, progress: float) -> None:
    db = get_database()
    await db.jobs.update_one(
        {"_id": job_id},
        {
            "$set": {
                "status": status,
                "stage": stage,
                "progress": progress,
                "started_at": datetime.now(timezone.utc),
            }
        },
    )
