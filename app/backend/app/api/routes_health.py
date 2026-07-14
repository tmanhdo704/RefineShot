from fastapi import APIRouter

from app.core.config import get_settings
from app.db.mongo import get_database
from refineshot.runtime import resolve_device

router = APIRouter()

DEFAULT_PRESET = "refineshot"

PRESETS: dict[str, dict] = {
    "refineshot": {
        "display_name": "RefineShot",
        "filename": "refineshot_v8_final.pth",
        "threshold": None,
    },
    "refineshot_heatmap": {
        "display_name": "RefineShot HeatMap",
        "filename": "refineshot_heatmap.pth",
        "threshold": None,
    },
    "best_shot": {
        "display_name": "RefineShot-SHOT",
        "filename": "refineshot_v8_final.pth",
        "threshold": 0.12,
    },
    "best_clipshot": {
        "display_name": "RefineShot-ClipShot",
        "filename": "refineshot_v8_final.pth",
        "threshold": 0.19,
    },
    "best_bbc": {
        "display_name": "RefineShot-BBC",
        "filename": "refineshot_standard.pth",
        "threshold": None,
    },
    "autoshot": {
        "display_name": "AutoShot",
        "filename": "autoshot_base.pth",
        "threshold": None,
    },
}


@router.get("/models")
async def list_models() -> dict:
    settings = get_settings()
    models_dir = settings.refineshot_models_dir
    available = [
        {
            "preset": key,
            "display_name": p["display_name"],
            "is_default": key == DEFAULT_PRESET,
            "available": (models_dir / p["filename"]).is_file(),
        }
        for key, p in PRESETS.items()
    ]
    return {"models": available, "default": DEFAULT_PRESET}


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    database_status = "ok"
    try:
        await get_database().command("ping")
    except Exception:
        database_status = "unavailable"

    checkpoint_exists = settings.refineshot_model_path.is_file()
    preferred_backend = (
        "baseline"
        if settings.refineshot_use_baseline or not checkpoint_exists
        else "phase2"
    )
    return {
        "status": "ok" if database_status == "ok" else "degraded",
        "database": database_status,
        "model": {
            "checkpoint": str(settings.refineshot_model_path),
            "checkpoint_exists": checkpoint_exists,
            "requested_device": settings.refineshot_device,
            "effective_device": resolve_device(settings.refineshot_device),
            "preferred_backend": preferred_backend,
        },
    }
