from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.config import get_settings
from app.ml import baseline_opencv
from app.ml.postprocess import merge_short_scenes, scene_ranges_to_payload
from app.ml.refineshot_runtime import get_runtime
from refineshot.runtime import (
    CheckpointLoadError,
    FrameDecodeError,
    RefineShotRuntime,
    decode_video_frames,
    logits_to_probabilities,
    probabilities_to_scenes,
)

logger = logging.getLogger(__name__)


@dataclass
class VideoAnalysisSettings:
    model_path: Path
    threshold: float | None = None


def analyze_video(job_id: str, video_path: Path, options: VideoAnalysisSettings) -> dict:
    try:
        runtime = get_runtime(options.model_path)
    except (FileNotFoundError, CheckpointLoadError) as exc:
        raise RuntimeError(f"Model unavailable: {exc}") from exc

    return _analyze_with_refineshot(job_id, video_path, options, runtime)


def _analyze_with_refineshot(
    job_id: str,
    video_path: Path,
    options: VideoAnalysisSettings,
    runtime: RefineShotRuntime,
) -> dict:
    settings = get_settings()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot decode this video. Please try another file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1:
        fps = 25.0
    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_hint = frame_count_hint / fps if frame_count_hint else 0.0
    if duration_hint and duration_hint > settings.max_duration_sec:
        raise RuntimeError(f"Video is longer than the {settings.max_duration_sec}s local limit.")

    defaults = runtime.defaults
    temperature = float(settings.refineshot_default_temperature or defaults.temperature)
    sigma = float(settings.refineshot_default_sigma or defaults.sigma)
    threshold = float(
        options.threshold
        if options.threshold is not None
        else (settings.refineshot_default_threshold or defaults.threshold)
    )

    try:
        frames = decode_video_frames(video_path)
    except FrameDecodeError as exc:
        raise RuntimeError(f"Frame decode failed: {exc}") from exc

    total_frames = int(len(frames))
    if total_frames == 0:
        raise RuntimeError("Video produced zero frames after decoding")

    logits = runtime.predict_logits(frames)
    probs = logits_to_probabilities(logits, temperature=temperature, sigma=sigma)
    raw_ranges = probabilities_to_scenes(probs, threshold=threshold)
    scene_ranges = merge_short_scenes(raw_ranges, min_gap_frames=1)

    scenes_payload, boundaries_payload = scene_ranges_to_payload(scene_ranges, probs, fps=fps)

    scenes_with_thumbs, scene_assets = _attach_thumbnails(job_id, video_path, scenes_payload)
    storyboard_url, storyboard_asset = baseline_opencv.build_storyboard(job_id, scenes_with_thumbs)
    summary = baseline_opencv.build_summary(scenes_with_thumbs, boundaries_payload)

    duration_sec = total_frames / fps if total_frames else duration_hint

    return {
        "input": {
            "fps": round(fps, 3),
            "duration_sec": round(duration_sec, 3),
            "total_frames": total_frames,
            "width": width,
            "height": height,
        },
        "processing": {
            "model": runtime.checkpoint_path.name,
            "backend": "phase2",
            "device": runtime.device,
        },
        "summary": summary,
        "boundaries": boundaries_payload,
        "scenes": scenes_with_thumbs,
        "artifacts": {
            "storyboard_url": storyboard_url,
            "assets": scene_assets + ([storyboard_asset] if storyboard_asset else []),
        },
    }


def _attach_thumbnails(
    job_id: str,
    video_path: Path,
    scenes: list[dict],
) -> tuple[list[dict], list[dict]]:
    from app.services.storage_service import job_dir, publish_asset

    thumbs_dir = job_dir(job_id) / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict] = []
    enriched: list[dict] = []

    for scene in scenes:
        mid = int((scene["start_frame"] + scene["end_frame"]) / 2)
        thumb_path = thumbs_dir / f"scene_{scene['index']:04d}.jpg"
        baseline_opencv.save_thumbnail(video_path, mid, thumb_path)
        thumb_asset = publish_asset(thumb_path, job_id, f"thumbnails/scene_{scene['index']:04d}", "image")
        if "public_id" in thumb_asset:
            assets.append(thumb_asset)
        enriched.append({**scene, "thumbnail_url": thumb_asset["url"]})

    return enriched, assets
