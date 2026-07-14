import csv
import json
from pathlib import Path
from typing import Any

from app.services.storage_service import job_dir, publish_asset


def write_exports(job_id: str, result: dict[str, Any]) -> dict[str, str]:
    export_dir = job_dir(job_id) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    json_path = export_dir / "result.json"
    csv_path = export_dir / "scenes.csv"
    txt_path = export_dir / "summary.txt"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_scene_csv(csv_path, result.get("scenes", []), result.get("boundaries", []))
    _write_summary_txt(txt_path, result)

    json_asset = publish_asset(json_path, job_id, "exports/result.json", "raw")
    csv_asset = publish_asset(csv_path, job_id, "exports/scenes.csv", "raw")
    txt_asset = publish_asset(txt_path, job_id, "exports/summary.txt", "raw")
    return {
        "json_url": json_asset["url"],
        "csv_url": csv_asset["url"],
        "txt_url": txt_asset["url"],
        "assets": [asset for asset in (json_asset, csv_asset, txt_asset) if "public_id" in asset],
    }


def _write_scene_csv(path: Path, scenes: list[dict[str, Any]], boundaries: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scene_index",
                "start_frame",
                "end_frame",
                "start_time_sec",
                "end_time_sec",
                "duration_sec",
                "next_boundary_time_sec",
                "next_boundary_confidence",
            ]
        )
        for scene in scenes:
            boundary = boundaries[scene["index"]] if scene["index"] < len(boundaries) else {}
            writer.writerow(
                [
                    scene["index"],
                    scene["start_frame"],
                    scene["end_frame"],
                    scene["start_time_sec"],
                    scene["end_time_sec"],
                    scene["duration_sec"],
                    boundary.get("time_sec", ""),
                    boundary.get("confidence", ""),
                ]
            )


def _write_summary_txt(path: Path, result: dict[str, Any]) -> None:
    summary = result.get("summary", {})
    lines = [
        "RefineShot Web Result",
        "",
        f"Job: {result.get('job_id', '')}",
        f"Video: {result.get('input', {}).get('original_name', '')}",
        f"Scenes: {summary.get('scene_count', 0)}",
        f"Boundaries: {summary.get('boundary_count', 0)}",
        f"Average scene duration: {summary.get('average_scene_duration_sec', 0)}s",
        f"Shortest scene: {summary.get('shortest_scene_sec', 0)}s",
        f"Longest scene: {summary.get('longest_scene_sec', 0)}s",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
