import asyncio

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo() -> None:
    global _client, _db
    settings = get_settings()
    last_error: Exception | None = None

    for _ in range(12):
        try:
            _client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
            _db = _client[settings.mongodb_db]
            await _db.command("ping")
            await _db.jobs.create_index("created_at")
            return
        except Exception as exc:
            last_error = exc
            if _client is not None:
                _client.close()
            _client = None
            _db = None
            await asyncio.sleep(2)

    raise RuntimeError("Cannot connect to MongoDB") from last_error


async def close_mongo() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


def get_database() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB is not connected")
    return _db
