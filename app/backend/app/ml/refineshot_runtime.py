from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from refineshot.runtime import RefineShotRuntime


@lru_cache(maxsize=4)
def get_runtime(model_path: Path | None = None) -> RefineShotRuntime:
    settings = get_settings()
    path = model_path or settings.refineshot_model_path
    return RefineShotRuntime(path, settings.refineshot_device)


def clear_runtime_cache() -> None:
    get_runtime.cache_clear()
