from copy import deepcopy
from datetime import datetime
from typing import Any


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def serialize_job(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None

    job = deepcopy(document)
    job["id"] = str(job.pop("_id"))
    job.pop("internal", None)
    return _serialize_value(job)
