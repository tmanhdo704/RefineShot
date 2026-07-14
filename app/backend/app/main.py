from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes_compare import router as compare_router
from app.api.routes_health import router as health_router
from app.api.routes_jobs import router as jobs_router
from app.core.config import get_settings
from app.db.mongo import close_mongo, connect_to_mongo
from app.services.storage_service import ensure_storage_dirs


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_storage_dirs()
    await connect_to_mongo()
    yield
    await close_mongo()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="RefineShot Web API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api")
    app.include_router(jobs_router, prefix="/api")
    app.include_router(compare_router, prefix="/api")
    app.mount("/media", StaticFiles(directory=settings.storage_dir), name="media")
    return app


app = create_app()
