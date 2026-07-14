from __future__ import annotations

import numpy as np


def merge_short_scenes(scene_ranges: np.ndarray, min_gap_frames: int) -> np.ndarray:
    if len(scene_ranges) <= 1 or min_gap_frames <= 1:
        return scene_ranges
    merged: list[list[int]] = [list(scene_ranges[0])]
    for start, end in scene_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if (prev_end - prev_start + 1) < min_gap_frames:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    return np.array(merged, dtype=np.int32)


def scene_ranges_to_payload(
    scene_ranges: np.ndarray,
    probs: np.ndarray,
    fps: float,
) -> tuple[list[dict], list[dict]]:
    safe_fps = float(fps) if fps and fps > 0 else 25.0
    total = int(probs.shape[0]) if probs.ndim else 0

    scenes: list[dict] = []
    for index, (start, end) in enumerate(scene_ranges):
        start_i = int(max(0, start))
        end_i = int(max(start_i, end))
        scenes.append(
            {
                "index": index,
                "start_frame": start_i,
                "end_frame": end_i,
                "start_time_sec": round(start_i / safe_fps, 3),
                "end_time_sec": round(end_i / safe_fps, 3),
                "duration_sec": round(max(0.0, (end_i - start_i + 1) / safe_fps), 3),
            }
        )

    boundaries: list[dict] = []
    for index in range(len(scene_ranges) - 1):
        trans_start = int(scene_ranges[index][1])
        trans_end = int(scene_ranges[index + 1][0])
        if total:
            lo = max(0, trans_start)
            hi = min(total, trans_end + 1)
            window = probs[lo:hi] if hi > lo else probs[lo : lo + 1]
            peak_offset = int(np.argmax(window)) if window.size else 0
            frame_idx = lo + peak_offset
            confidence = float(window[peak_offset]) if window.size else 0.0
        else:
            frame_idx = trans_end
            confidence = 0.0
        boundaries.append(
            {
                "index": index,
                "frame": frame_idx,
                "time_sec": round(frame_idx / safe_fps, 3),
                "confidence": round(min(1.0, max(0.0, confidence)), 3),
                "type": "cut",
            }
        )
    return scenes, boundaries
