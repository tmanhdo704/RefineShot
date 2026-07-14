from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from refineshot import v8_config
from refineshot.model import TransNetV2Supernet

logger = logging.getLogger(__name__)

MODEL_WIDTH = 48
MODEL_HEIGHT = 27
DEFAULT_TEMPERATURE = v8_config.DEPLOY_TEMPERATURE
DEFAULT_SIGMA = v8_config.DEPLOY_SIGMA
DEFAULT_THRESHOLD = v8_config.DEPLOY_THRESHOLD


class FrameDecodeError(RuntimeError):
    pass


class CheckpointLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class PostprocessConfig:
    temperature: float = DEFAULT_TEMPERATURE
    sigma: float = DEFAULT_SIGMA
    threshold: float = DEFAULT_THRESHOLD

    @classmethod
    def from_mapping(cls, value: object) -> PostprocessConfig:
        config = value if isinstance(value, dict) else {}
        return cls(
            temperature=float(config.get("temperature", DEFAULT_TEMPERATURE)),
            sigma=float(config.get("sigma", DEFAULT_SIGMA)),
            threshold=float(config.get("threshold", DEFAULT_THRESHOLD)),
        )


def resolve_device(requested: str) -> str:
    requested = (requested or "cpu").strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU")
        return "cpu"
    return requested


def extract_checkpoint_state(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("net", "model", "state_dict"):
            state = payload.get(key)
            if isinstance(state, dict):
                return _strip_module_prefix(state)
        if payload and all(hasattr(value, "shape") for value in payload.values()):
            return _strip_module_prefix(payload)
    raise CheckpointLoadError("Unrecognized checkpoint format")


def load_checkpoint_config(checkpoint_path: Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (RuntimeError, OSError, ValueError) as exc:
        raise CheckpointLoadError(f"Cannot read checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(payload, dict):
        return {}
    config = payload.get("phase2_config")
    return dict(config) if isinstance(config, dict) else {}


def load_model(checkpoint_path: Path, device: str) -> TransNetV2Supernet:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    effective_device = resolve_device(device)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = extract_checkpoint_state(payload)
        model = TransNetV2Supernet()
        model.load_state_dict(state, strict=True)
        return model.eval().to(effective_device)
    except (CheckpointLoadError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        raise CheckpointLoadError(f"Cannot load checkpoint {checkpoint_path}: {exc}") from exc


def decode_video_frames(
    video_path: Path,
    width: int = MODEL_WIDTH,
    height: int = MODEL_HEIGHT,
    timeout_sec: float = 600.0,
) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError as exc:
        raise FrameDecodeError("ffmpeg executable was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise FrameDecodeError(f"ffmpeg decode timed out after {timeout_sec}s") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise FrameDecodeError(f"ffmpeg failed: {stderr or exc}") from exc

    buffer = np.frombuffer(proc.stdout, np.uint8)
    frame_size = height * width * 3
    if buffer.size == 0 or buffer.size % frame_size != 0:
        raise FrameDecodeError("ffmpeg produced no or incomplete frame data")
    return buffer.reshape([-1, height, width, 3])


def iter_frame_batches(frames: np.ndarray) -> Iterator[np.ndarray]:
    if frames.size == 0 or len(frames) == 0:
        return
    pad_tail = 50 - len(frames) % 50
    if pad_tail == 50:
        pad_tail = 0
    padded = np.concatenate(
        [frames[:1]] * 25 + [frames] + [frames[-1:]] * (pad_tail + 25),
        axis=0,
    )
    for index in range(0, len(padded) - 50, 50):
        yield padded[index : index + 100]


@torch.no_grad()
def predict_frame_logits(model: torch.nn.Module, frames: np.ndarray, device: str) -> np.ndarray:
    if frames.size == 0 or len(frames) == 0:
        raise RuntimeError("No decoded frames")

    effective_device = resolve_device(device)
    chunks = []
    for batch in iter_frame_batches(frames):
        tensor = (
            torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...])
            .float()
            .to(effective_device)
        )
        output = model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        chunks.append(output[0].detach().cpu().numpy()[25:75])

    logits = np.concatenate(chunks, axis=0)[: len(frames)]
    if logits.ndim == 1:
        logits = logits[:, np.newaxis]
    return logits.astype(np.float32)


def predict_video_logits(model: torch.nn.Module, video_path: Path, device: str) -> np.ndarray:
    return predict_frame_logits(model, decode_video_frames(video_path), device)


def logits_to_probabilities(logits: np.ndarray, temperature: float, sigma: float) -> np.ndarray:
    safe_temperature = max(1e-6, float(temperature))
    values = np.asarray(logits, dtype=np.float32).reshape(-1) / safe_temperature
    probabilities = 1.0 / (1.0 + np.exp(-values))
    if sigma > 0:
        probabilities = gaussian_filter1d(probabilities, sigma=float(sigma))
    return probabilities.astype(np.float32)


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions).reshape(-1)
    if len(predictions) == 0:
        return np.array([[0, 0]], dtype=np.int32)

    scenes: list[list[int]] = []
    previous = 0
    start = 0
    current = -1
    for index, current in enumerate(predictions):
        if previous == 1 and current == 0:
            start = index
        if previous == 0 and current == 1 and index != 0:
            scenes.append([start, index])
        previous = current
    if current == 0:
        scenes.append([start, index])
    if not scenes:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.asarray(scenes, dtype=np.int32)


def probabilities_to_scenes(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    binary = (np.asarray(probabilities).reshape(-1) > float(threshold)).astype(np.uint8)
    return predictions_to_scenes(binary)


class RefineShotRuntime:
    def __init__(self, checkpoint_path: Path, device: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        self._lock = threading.Lock()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        config = load_checkpoint_config(self.checkpoint_path)
        self.defaults = PostprocessConfig.from_mapping(config)
        self.model = load_model(self.checkpoint_path, self.device)

    def predict_logits(self, frames: np.ndarray) -> np.ndarray:
        with self._lock:
            return predict_frame_logits(self.model, frames, self.device)

    def predict_video(self, video_path: Path) -> np.ndarray:
        return self.predict_logits(decode_video_frames(video_path))


def _strip_module_prefix(state: dict) -> dict:
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    return state
