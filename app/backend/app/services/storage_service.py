import shutil
from pathlib import Path

import cloudinary
import cloudinary.api
import cloudinary.uploader

from app.core.config import get_settings

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def ensure_storage_dirs() -> None:
    settings = get_settings()
    for child in ("uploads", "jobs", "tmp"):
        (settings.storage_dir / child).mkdir(parents=True, exist_ok=True)
    configure_cloudinary()


def configure_cloudinary() -> None:
    settings = get_settings()
    if not settings.use_cloudinary:
        return
    cloudinary.config(
        cloud_name=settings.cloudinary_cloud_name,
        api_key=settings.cloudinary_api_key,
        api_secret=settings.cloudinary_api_secret,
        secure=True,
    )


def safe_video_extension(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported video format")
    return suffix


def upload_dir(job_id: str) -> Path:
    path = get_settings().storage_dir / "uploads" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_dir(job_id: str) -> Path:
    path = get_settings().storage_dir / "jobs" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def media_url(relative_path: Path) -> str:
    return "/media/" + relative_path.as_posix().lstrip("/")


def publish_asset(local_path: Path, job_id: str, asset_path: str, resource_type: str) -> dict[str, str]:
    settings = get_settings()
    if not settings.use_cloudinary:
        return {"url": media_url(local_path.relative_to(settings.storage_dir))}

    configure_cloudinary()
    public_id = _cloudinary_public_id(job_id, asset_path)
    result = cloudinary.uploader.upload(
        str(local_path),
        public_id=public_id,
        resource_type=resource_type,
        overwrite=True,
    )
    return {
        "url": result["secure_url"],
        "public_id": result["public_id"],
        "resource_type": resource_type,
    }


def delete_cloudinary_assets(job: dict | None) -> None:
    if not job:
        return
    settings = get_settings()
    if not settings.use_cloudinary:
        return

    configure_cloudinary()
    assets = job.get("cloudinary", {}).get("assets", [])
    for asset in assets:
        public_id = asset.get("public_id")
        resource_type = asset.get("resource_type", "image")
        if public_id:
            cloudinary.uploader.destroy(public_id, resource_type=resource_type, invalidate=True)

    prefix = f"{settings.cloudinary_folder}/jobs/{job.get('_id')}"
    for resource_type in ("image", "video", "raw"):
        try:
            cloudinary.api.delete_resources_by_prefix(prefix, resource_type=resource_type)
        except Exception:
            pass


def remove_job_files(job_id: str) -> None:
    settings = get_settings()
    for path in (settings.storage_dir / "uploads" / job_id, settings.storage_dir / "jobs" / job_id):
        if path.exists():
            shutil.rmtree(path)


def _cloudinary_public_id(job_id: str, asset_path: str) -> str:
    settings = get_settings()
    clean_folder = settings.cloudinary_folder.strip("/")
    clean_asset_path = asset_path.strip("/")
    return f"{clean_folder}/jobs/{job_id}/{clean_asset_path}"
