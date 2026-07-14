from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    mongodb_uri: str = "mongodb://mongo:27017"
    mongodb_db: str = "refineshot"
    storage_dir: Path = Path("/app/storage")
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""
    cloudinary_folder: str = "refineshot"

    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 300
    max_duration_sec: int = 600
    job_ttl_hours: int = 24

    default_sensitivity: str = "medium"
    default_min_scene_duration_sec: float = 0.5

    refineshot_model_path: Path = Field(
        default=Path("/app/models/refineshot_v8_final.pth"),
        validation_alias=AliasChoices("REFINESHOT_MODEL_PATH", "REFINESHOT_CKPT"),
    )
    refineshot_models_dir: Path = Field(
        default=Path("/app/models"),
        validation_alias=AliasChoices("REFINESHOT_MODELS_DIR"),
    )
    refineshot_device: str = "auto"
    refineshot_default_temperature: float | None = None
    refineshot_default_sigma: float | None = None
    refineshot_default_threshold: float | None = None
    refineshot_use_baseline: bool = False

    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def use_cloudinary(self) -> bool:
        values = [
            self.cloudinary_cloud_name,
            self.cloudinary_api_key,
            self.cloudinary_api_secret,
        ]
        return all(
            value and not value.startswith(("your_", "<"))
            for value in values
        )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
