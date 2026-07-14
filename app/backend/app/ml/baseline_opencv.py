from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.services.storage_service import job_dir, publish_asset

THRESHOLDS = {
    "low": 0.55,
    "medium": 0.42,
    "high": 0.30,
}


@dataclass
class BaselineSettings:
    sensitivity: str
    min_scene_duration_sec: float


def analyze_video(job_id: str, video_path: Path, options: BaselineSettings) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot decode this video. Please try another file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1:
        fps = 25.0

    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_hint = frame_count_hint / fps if frame_count_hint else 0

    settings = get_settings()
    if duration_hint and duration_hint > settings.max_duration_sec:
        raise RuntimeError(f"Video is longer than the {settings.max_duration_sec}s local limit.")

    sensitivity = options.sensitivity if options.sensitivity in THRESHOLDS else "medium"
    threshold = THRESHOLDS[sensitivity]
    min_gap_frames = max(1, int(options.min_scene_duration_sec * fps))
    sample_every = max(1, int(round(fps / 4.0)))

    boundaries: list[dict] = []
    previous_hist: np.ndarray | None = None
    last_boundary_frame = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % sample_every == 0:
            hist = _frame_histogram(frame)
            if previous_hist is not None:
                diff = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if math.isnan(diff):
                    diff = 0.0
                if diff >= threshold and frame_index - last_boundary_frame >= min_gap_frames:
                    boundaries.append(
                        {
                            "index": len(boundaries),
                            "frame": frame_index,
                            "time_sec": round(frame_index / fps, 3),
                            "confidence": round(min(1.0, diff / 0.75), 3),
                            "type": "cut",
                        }
                    )
                    last_boundary_frame = frame_index
            previous_hist = hist

        frame_index += 1

    cap.release()

    total_frames = frame_index or frame_count_hint
    duration_sec = total_frames / fps if total_frames else duration_hint
    scenes, scene_assets = build_scenes(job_id, video_path, boundaries, fps, total_frames)
    storyboard_url, storyboard_asset = build_storyboard(job_id, scenes)
    summary = build_summary(scenes, boundaries)

    return {
        "input": {
            "fps": round(fps, 3),
            "duration_sec": round(duration_sec, 3),
            "total_frames": total_frames,
            "width": width,
            "height": height,
        },
        "processing": {
            "model": "opencv-histogram-baseline",
            "sensitivity": sensitivity,
            "threshold": threshold,
            "min_scene_duration_sec": options.min_scene_duration_sec,
            "device": "cpu",
        },
        "summary": summary,
        "boundaries": boundaries,
        "scenes": scenes,
        "artifacts": {
            "storyboard_url": storyboard_url,
            "assets": scene_assets + ([storyboard_asset] if storyboard_asset else []),
        },
    }


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def build_scenes(
    job_id: str,
    video_path: Path,
    boundaries: list[dict],
    fps: float,
    total_frames: int,
) -> tuple[list[dict], list[dict]]:
    scene_ranges: list[tuple[int, int]] = []
    start_frame = 0
    last_frame = max(0, total_frames - 1)

    for boundary in boundaries:
        end_frame = max(start_frame, int(boundary["frame"]) - 1)
        scene_ranges.append((start_frame, end_frame))
        start_frame = min(last_frame, int(boundary["frame"]))

    scene_ranges.append((start_frame, last_frame))

    thumbs_dir = job_dir(job_id) / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    scenes: list[dict] = []
    assets: list[dict] = []
    for index, (start, end) in enumerate(scene_ranges):
        mid = int((start + end) / 2)
        thumb_path = thumbs_dir / f"scene_{index:04d}.jpg"
        save_thumbnail(video_path, mid, thumb_path)
        thumb_asset = publish_asset(thumb_path, job_id, f"thumbnails/scene_{index:04d}", "image")
        if "public_id" in thumb_asset:
            assets.append(thumb_asset)
        scenes.append(
            {
                "index": index,
                "start_frame": start,
                "end_frame": end,
                "start_time_sec": round(start / fps, 3),
                "end_time_sec": round(end / fps, 3),
                "duration_sec": round(max(0.0, (end - start + 1) / fps), 3),
                "thumbnail_url": thumb_asset["url"],
            }
        )
    return scenes, assets


def save_thumbnail(video_path: Path, frame_index: int, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()

    if not ok:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

    frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])


def build_storyboard(job_id: str, scenes: list[dict]) -> tuple[str | None, dict | None]:
    if not scenes:
        return None, None

    settings = get_settings()
    thumb_paths = sorted((settings.storage_dir / "jobs" / job_id / "thumbs").glob("scene_*.jpg"))[:36]
    images = [cv2.imread(str(path)) for path in thumb_paths]
    images = [image for image in images if image is not None]
    if not images:
        return None, None

    cell_w, cell_h = 320, 180
    cols = min(4, len(images))
    rows = int(math.ceil(len(images) / cols))
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)

    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        canvas[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = image

    storyboard_path = job_dir(job_id) / "storyboard.jpg"
    cv2.imwrite(str(storyboard_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    storyboard_asset = publish_asset(storyboard_path, job_id, "storyboard", "image")
    return storyboard_asset["url"], storyboard_asset if "public_id" in storyboard_asset else None


def build_summary(scenes: list[dict], boundaries: list[dict]) -> dict:
    durations = [scene["duration_sec"] for scene in scenes]
    if not durations:
        durations = [0.0]

    return {
        "boundary_count": len(boundaries),
        "scene_count": len(scenes),
        "average_scene_duration_sec": round(sum(durations) / len(durations), 3),
        "shortest_scene_sec": round(min(durations), 3),
        "longest_scene_sec": round(max(durations), 3),
    }
