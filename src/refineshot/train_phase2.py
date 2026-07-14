import argparse
import hashlib
import json
import os
import pickle
import random
import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Dataset

from refineshot import v8_config
from refineshot.common import f1_pr, set_global_seeds, sigmoid_np
from refineshot.model.linear import Linear_

# Data/cache side lives in phase2_data; names are re-exported here because
# ablation, journal_protocol, paper_analysis and the tests import them from
# this module.
from refineshot.phase2_data import (  # noqa: F401
    SAMPLE_CACHE_SCHEMA_VERSION,
    TimeBudgetExpired,
    build_or_load_sample_cache,
    build_sample_cache_config,
    check_sample_cache_video_budget,
    deadline_expired,
    device,
    extract_backbone_features,
    hash_keys,
    load_metadata,
    make_boundary_labels,
    probe_video_info,
    sample_cache_partial_paths,
    select_sample_indices,
    select_training_keys,
    sha256_file,
    transitions_to_scenes,
    write_training_data_manifest,
)
from refineshot.utils import (
    evaluate_scenes,
    get_batches,
    get_frames,
    mAP_f1_p_fix_r,
    predictions_to_scenes,
    scenes2zero_one_representation,
)

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

DEFAULT_META = "./data/shot_clipshots_trainval.pickle"
DEFAULT_BASE_CKPT = "./models/checkpoints/autoshot_base.pth"
DEFAULT_OUT_CKPT = "./runs/training/refineshot_v8_final.pth"
DEFAULT_SAMPLE_CACHE = "./data/shot_clipshots_phase2_sample_cache.pkl"
DEFAULT_RESULTS = "./runs/training/phase2_results.pkl"
DEFAULT_EVAL_CACHE_DIR = "./data/eval"
DEFAULT_RESUME_STATE = "./runs/training/refineshot_resume.pt"
DEFAULT_CHECKPOINT_DIR = "./runs/training/checkpoints"

FINETUNE_SCOPE_CHOICES = ("head_only", "fc1_0", "layer5", "layer4_layer5")
HEAD_MODULE_NAMES = ("fc1_0", "cls_layer1", "cls_layer2")
BACKBONE_FINETUNE_MODULES = {
    "head_only": (),
    "fc1_0": (),
    "layer5": ("Layer_5_12",),
    "layer4_layer5": ("Layer_4_13", "Layer_5_12"),
}


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        a_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (a_t * (1.0 - p_t) ** self.gamma * ce).mean()


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int = 4864, hidden_dim: int = 1024, dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = Linear_(in_features, hidden_dim, bias=True, act="ReLU")
        # dropout_rate is the probability of zeroing an element (PyTorch convention),
        # not a keep-rate. Default 0.5 keeps the original behaviour unchanged.
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cls_layer1 = Linear_(hidden_dim, 1, bias=True, act="Identity")
        self.cls_layer2 = Linear_(hidden_dim, 1, bias=True, act="Identity")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x = self.dropout(x)
        return self.cls_layer1(x), self.cls_layer2(x)


class SampleFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, one_hot: torch.Tensor, boundary: torch.Tensor):
        self.features = features
        self.one_hot = one_hot
        self.boundary = boundary

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx].float(),
            self.one_hot[idx].float().unsqueeze(0),
            self.boundary[idx].float().unsqueeze(0),
        )


class ManualAdam:
    """Small Adam implementation to avoid torch.optim importing torch._dynamo/onnx."""

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.exp_avg = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params]
        self.exp_avg_sq = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count
        for param, exp_avg, exp_avg_sq in zip(self.params, self.exp_avg, self.exp_avg_sq):
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay != 0.0:
                grad = grad.add(param, alpha=self.weight_decay)
            exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(self.eps)
            step_size = self.lr / bias_correction1
            param.addcdiv_(exp_avg, denom, value=-step_size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "step_count": self.step_count,
            "exp_avg": [t.detach().cpu() for t in self.exp_avg],
            "exp_avg_sq": [t.detach().cpu() for t in self.exp_avg_sq],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lr = float(state["lr"])
        self.weight_decay = float(state["weight_decay"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.step_count = int(state["step_count"])
        for dst, src in zip(self.exp_avg, state["exp_avg"]):
            dst.copy_(src.to(dst.device))
        for dst, src in zip(self.exp_avg_sq, state["exp_avg_sq"]):
            dst.copy_(src.to(dst.device))


def hash_head_state(head: ClassificationHead) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(head.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_supernet(ckpt_path: str):
    from refineshot.model.supernet import TransNetV2Supernet

    model = TransNetV2Supernet().eval().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = model.state_dict()
    pretrained = {k: v for k, v in ckpt["net"].items() if k in sd}
    sd.update(pretrained)
    model.load_state_dict(sd)
    print(f"Loaded {len(pretrained)}/{len(sd)} tensors from {ckpt_path}")
    return model


def train_head(
    head: ClassificationHead,
    dataset: SampleFeatureDataset,
    args: argparse.Namespace,
) -> tuple[ClassificationHead, list[float]]:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha)
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")
    optimizer = ManualAdam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses: list[float] = []
    start_epoch = 1
    train_config = {
        "in_features": int(dataset.features.shape[1]),
        "n_samples": int(len(dataset)),
        "batch_size": args.batch_size,
        "loss": args.loss,
        "lr": args.lr,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "weight_decay": args.weight_decay,
        "manyhot_weight": args.manyhot_weight,
        "training_seed": args.seed,
    }

    if os.path.exists(args.resume_state) and not args.no_resume:
        state = torch.load(args.resume_state, map_location=device, weights_only=False)
        if state.get("train_config") != train_config and not args.ignore_resume_config:
            raise RuntimeError(
                f"Resume config mismatch in {args.resume_state}. "
                "Use --no-resume or --ignore-resume-config if this is intentional."
            )
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        losses = list(state.get("losses", []))
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resuming training from epoch {start_epoch}/{args.epochs}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    def save_training_state(epoch: int) -> None:
        payload = {
            "epoch": epoch,
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "losses": losses,
            "train_config": train_config,
            "args": vars(args),
        }
        torch.save(payload, args.resume_state)
        if args.save_every_epochs > 0 and (epoch % args.save_every_epochs == 0 or epoch == args.epochs):
            torch.save(payload, os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt"))

    if start_epoch > args.epochs:
        print(f"Training already complete at epoch {start_epoch - 1}.")
        return head, losses

    for epoch in range(start_epoch, args.epochs + 1):
        head.train()
        total_loss = 0.0
        n_seen = 0
        num_batches = max(len(loader), 1)
        for batch_idx, (feats, one_hot, boundary) in enumerate(loader, 1):
            feats = feats.to(device)
            one_hot = one_hot.to(device)
            boundary = boundary.to(device)
            logits1, logits2 = head(feats)
            loss = criterion(logits1, one_hot) + args.manyhot_weight * criterion(logits2, boundary)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(feats)
            n_seen += len(feats)
            if batch_idx == 1 or batch_idx % args.log_every_batches == 0 or batch_idx == num_batches:
                overall_pct = 100.0 * ((epoch - 1) + batch_idx / num_batches) / max(args.epochs, 1)
                running_loss = total_loss / max(n_seen, 1)
                print(
                    f"train {overall_pct:6.2f}% epoch {epoch:03d}/{args.epochs} "
                    f"batch {batch_idx:05d}/{num_batches:05d} loss={running_loss:.6f}",
                    flush=True,
                )

        avg_loss = total_loss / max(n_seen, 1)
        losses.append(avg_loss)
        print(f"epoch {epoch:03d}/{args.epochs} done loss={avg_loss:.6f}", flush=True)
        save_training_state(epoch)
        if deadline_expired(args):
            raise TimeBudgetExpired("Time budget reached after saving epoch checkpoint.")
    return head, losses


def hash_state_dict(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def module_names_for_finetune_scope(scope: str) -> tuple[str, ...]:
    if scope not in FINETUNE_SCOPE_CHOICES:
        raise ValueError(f"Unsupported finetune scope: {scope}")
    if scope == "head_only":
        return ()
    return BACKBONE_FINETUNE_MODULES[scope] + HEAD_MODULE_NAMES


def configure_finetune_modules(model: nn.Module, scope: str) -> dict[str, Any]:
    trainable_modules = module_names_for_finetune_scope(scope)
    for parameter in model.parameters():
        parameter.requires_grad = False

    module_by_name = dict(model.named_modules())
    missing = [name for name in trainable_modules if name not in module_by_name]
    if missing:
        raise ValueError(f"Model does not contain requested finetune module(s): {missing}")

    for name in trainable_modules:
        for parameter in module_by_name[name].parameters():
            parameter.requires_grad = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "scope": scope,
        "modules": list(trainable_modules),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }


def split_finetune_parameters(model: nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    head_prefixes = tuple(f"{name}." for name in HEAD_MODULE_NAMES)
    head_params: list[torch.nn.Parameter] = []
    backbone_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(head_prefixes):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    return head_params, backbone_params


def freeze_batch_norm_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def validate_training_mode_args(args: argparse.Namespace) -> None:
    if args.finetune_scope not in FINETUNE_SCOPE_CHOICES:
        raise ValueError(f"Unsupported finetune scope: {args.finetune_scope}")
    if args.finetune_scope != "head_only" and getattr(args, "use_ema", False):
        raise ValueError("--use-ema is only supported by the cached head_only training path")
    if BACKBONE_FINETUNE_MODULES[args.finetune_scope] and args.backbone_lr <= 0:
        raise ValueError("--backbone-lr must be positive when Layer_4/Layer_5 fine-tuning is selected")


def select_finetune_loss_indices(
    one_hot: np.ndarray,
    boundary: np.ndarray,
    max_frames: int,
    neg_per_pos: int,
    min_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_frames <= 0 or max_frames >= len(one_hot):
        return np.arange(len(one_hot), dtype=np.int64)
    return select_sample_indices(
        one_hot,
        boundary,
        max_samples_per_video=max_frames,
        neg_per_pos=neg_per_pos,
        min_neg_per_video=min_negatives,
        rng=rng,
    )


def fine_tune_model(
    model: nn.Module,
    entries: dict[str, dict[str, Any]],
    train_keys: list[str],
    args: argparse.Namespace,
) -> tuple[nn.Module, list[float], dict[str, Any]]:
    trainable_summary = configure_finetune_modules(model, args.finetune_scope)
    head_params, backbone_params = split_finetune_parameters(model)
    if not head_params:
        raise RuntimeError("No trainable head parameters were selected")

    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha)
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")

    head_optimizer = ManualAdam(head_params, lr=args.lr, weight_decay=args.weight_decay)
    backbone_optimizer = (
        ManualAdam(backbone_params, lr=args.backbone_lr, weight_decay=args.weight_decay)
        if backbone_params
        else None
    )
    losses: list[float] = []
    start_epoch = 1
    train_config = {
        "mode": "end_to_end_finetune",
        "scope": args.finetune_scope,
        "modules": trainable_summary["modules"],
        "lr": args.lr,
        "backbone_lr": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "manyhot_weight": args.manyhot_weight,
        "boundary_window": args.boundary_window,
        "max_train_videos": args.max_train_videos,
        "max_finetune_videos": args.max_finetune_videos,
        "finetune_batches_per_video": args.finetune_batches_per_video,
        "finetune_max_frames_per_batch": args.finetune_max_frames_per_batch,
        "finetune_min_neg_per_batch": args.finetune_min_neg_per_batch,
        "seed": args.seed,
    }

    if os.path.exists(args.resume_state) and not args.no_resume:
        state = torch.load(args.resume_state, map_location=device, weights_only=False)
        if state.get("train_config") != train_config and not args.ignore_resume_config:
            raise RuntimeError(
                f"Resume config mismatch in {args.resume_state}. "
                "Use --no-resume or --ignore-resume-config if this is intentional."
            )
        model.load_state_dict(state["model"])
        head_optimizer.load_state_dict(state["head_optimizer"])
        if backbone_optimizer is not None and state.get("backbone_optimizer") is not None:
            backbone_optimizer.load_state_dict(state["backbone_optimizer"])
        losses = list(state.get("losses", []))
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resuming end-to-end fine-tuning from epoch {start_epoch}/{args.epochs}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    def save_training_state(epoch: int) -> None:
        payload = {
            "epoch": epoch,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "head_optimizer": head_optimizer.state_dict(),
            "backbone_optimizer": backbone_optimizer.state_dict() if backbone_optimizer is not None else None,
            "losses": losses,
            "train_config": train_config,
            "trainable_summary": trainable_summary,
            "args": vars(args),
        }
        torch.save(payload, args.resume_state)
        if args.save_every_epochs > 0 and (epoch % args.save_every_epochs == 0 or epoch == args.epochs):
            torch.save(payload, os.path.join(args.checkpoint_dir, f"finetune_epoch_{epoch:03d}.pt"))

    work_keys = list(train_keys)
    random.Random(args.seed).shuffle(work_keys)
    cap = args.max_finetune_videos if args.max_finetune_videos > 0 else args.max_train_videos
    if cap > 0:
        work_keys = work_keys[:cap]

    stats: dict[str, Any] = {
        "mode": "end_to_end_finetune",
        "scope": args.finetune_scope,
        "trainable_summary": trainable_summary,
        "videos_requested": len(work_keys),
        "videos_seen": 0,
        "videos_skipped": [],
        "batches": 0,
        "frames_used": 0,
        "one_hot_positive_frames": 0,
        "boundary_positive_frames": 0,
        "one_hot_positive_rate": 0.0,
        "boundary_positive_rate": 0.0,
    }

    if start_epoch > args.epochs:
        print(f"End-to-end fine-tuning already complete at epoch {start_epoch - 1}.")
        return model, losses, stats

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        freeze_batch_norm_stats(model)
        total_loss = 0.0
        n_frames_seen = 0
        rng = np.random.default_rng(args.seed + epoch)
        for video_index, key in enumerate(work_keys, 1):
            entry = entries[key]
            try:
                check_sample_cache_video_budget(entry["video_path"], args)
                frames = get_frames(entry["video_path"])
            except Exception as exc:
                stats["videos_skipped"].append((key, str(exc)))
                continue
            if len(frames) == 0:
                stats["videos_skipped"].append((key, "no decoded frames"))
                continue

            n_frames = int(len(frames))
            scenes = transitions_to_scenes(entry["transitions"], n_frames)
            one_hot_np, _ = scenes2zero_one_representation(scenes, n_frames)
            boundary_np = make_boundary_labels(one_hot_np, args.boundary_window)
            stats["videos_seen"] += 1

            for batch_index, batch in enumerate(get_batches(frames), 1):
                if args.finetune_batches_per_video > 0 and batch_index > args.finetune_batches_per_video:
                    break
                start = (batch_index - 1) * 50
                valid = min(50, n_frames - start)
                if valid <= 0:
                    break

                one_hot_slice = one_hot_np[start : start + valid].astype(np.float32)
                boundary_slice = boundary_np[start : start + valid].astype(np.float32)
                selected = select_finetune_loss_indices(
                    one_hot_slice,
                    boundary_slice,
                    args.finetune_max_frames_per_batch,
                    args.neg_per_pos,
                    args.finetune_min_neg_per_batch,
                    rng,
                )
                if len(selected) == 0:
                    continue

                inputs = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float().to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits1, logits2 = outputs
                else:
                    logits1, logits2 = outputs, None
                logits1 = logits1[0, 25 : 25 + valid, :][selected]
                if logits2 is None:
                    if args.manyhot_weight > 0:
                        raise RuntimeError("Model did not return many-hot logits but --manyhot-weight is positive")
                    logits2 = logits1
                else:
                    logits2 = logits2[0, 25 : 25 + valid, :][selected]

                one_hot_target = torch.from_numpy(one_hot_slice[selected, np.newaxis]).float().to(device)
                boundary_target = torch.from_numpy(boundary_slice[selected, np.newaxis]).float().to(device)
                loss = criterion(logits1, one_hot_target) + args.manyhot_weight * criterion(logits2, boundary_target)

                head_optimizer.zero_grad()
                if backbone_optimizer is not None:
                    backbone_optimizer.zero_grad()
                loss.backward()
                head_optimizer.step()
                if backbone_optimizer is not None:
                    backbone_optimizer.step()

                used = int(len(selected))
                total_loss += float(loss.item()) * used
                n_frames_seen += used
                stats["batches"] += 1
                stats["frames_used"] += used
                stats["one_hot_positive_frames"] += int(one_hot_slice[selected].sum())
                stats["boundary_positive_frames"] += int(boundary_slice[selected].sum())

                if stats["batches"] == 1 or stats["batches"] % args.log_every_batches == 0:
                    running_loss = total_loss / max(n_frames_seen, 1)
                    print(
                        f"finetune epoch {epoch:03d}/{args.epochs} "
                        f"video {video_index:04d}/{len(work_keys):04d} "
                        f"batch {batch_index:04d} loss={running_loss:.6f}",
                        flush=True,
                    )

            if deadline_expired(args):
                avg_loss = total_loss / max(n_frames_seen, 1)
                losses.append(avg_loss)
                save_training_state(epoch)
                raise TimeBudgetExpired("Time budget reached after saving fine-tune checkpoint.")

        avg_loss = total_loss / max(n_frames_seen, 1)
        losses.append(avg_loss)
        print(f"finetune epoch {epoch:03d}/{args.epochs} done loss={avg_loss:.6f}", flush=True)
        save_training_state(epoch)
        if deadline_expired(args):
            raise TimeBudgetExpired("Time budget reached after saving fine-tune checkpoint.")

    stats["one_hot_positive_rate"] = (
        float(stats["one_hot_positive_frames"] / stats["frames_used"]) if stats["frames_used"] else 0.0
    )
    stats["boundary_positive_rate"] = (
        float(stats["boundary_positive_frames"] / stats["frames_used"]) if stats["frames_used"] else 0.0
    )
    return model, losses, stats


def merge_head_into_state(base_state: dict[str, torch.Tensor], head: ClassificationHead) -> dict[str, torch.Tensor]:
    full_sd = {k: v.detach().cpu().clone() for k, v in base_state.items()}
    for k, v in head.fc1.state_dict().items():
        full_sd[f"fc1_0.{k}"] = v.detach().cpu()
    for k, v in head.cls_layer1.state_dict().items():
        full_sd[f"cls_layer1.{k}"] = v.detach().cpu()
    for k, v in head.cls_layer2.state_dict().items():
        full_sd[f"cls_layer2.{k}"] = v.detach().cpu()
    return full_sd


def load_model_from_state(full_sd: dict[str, torch.Tensor]):
    from refineshot.model.supernet import TransNetV2Supernet

    model = TransNetV2Supernet().eval().to(device)
    model.load_state_dict(full_sd)
    return model


def predict_video_logits(model, video_path: str) -> np.ndarray:
    frames = get_frames(video_path)
    if len(frames) == 0:
        raise RuntimeError(f"No decoded frames: {video_path}")
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in get_batches(frames):
            t = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float().to(device)
            out = model(t)
            if isinstance(out, tuple):
                out = out[0]
            chunks.append(out[0].detach().cpu().numpy()[25:75])
    logits = np.concatenate(chunks, 0)[: len(frames)]
    if logits.ndim == 1:
        logits = logits[:, np.newaxis]
    return logits.astype(np.float32)


def load_or_run_logits(
    model,
    entries: dict[str, dict[str, Any]],
    keys: list[str],
    cache_path: str,
    cache_config: dict[str, Any],
    no_cache: bool,
) -> dict[str, np.ndarray]:
    if os.path.exists(cache_path) and not no_cache:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("config") == cache_config:
            print(f"Loading logits cache: {cache_path}")
            return cached["logits"]
        print(f"Logits cache config changed, rebuilding: {cache_path}")

    logits: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for i, key in enumerate(keys, 1):
        entry = entries[key]
        if not os.path.exists(entry["video_path"]):
            missing.append(key)
            continue
        pct = 100.0 * i / max(len(keys), 1)
        print(f"  inference {pct:6.2f}% [{i}/{len(keys)}] {key}", flush=True)
        logits[key] = predict_video_logits(model, entry["video_path"])

    if missing:
        print(f"WARNING: {len(missing)} videos missing.")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"config": cache_config, "logits": logits}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Logits cache saved -> {cache_path}")
    return logits


def logits_to_pred_dict(logits: dict[str, np.ndarray], temperature: float, sigma: float) -> dict[str, np.ndarray]:
    pred: dict[str, np.ndarray] = {}
    for key, arr in logits.items():
        probs = sigmoid_np(arr / temperature).squeeze()
        if sigma > 0:
            probs = gaussian_filter1d(probs, sigma=sigma)
        pred[key] = probs[:, np.newaxis].astype(np.float32)
    return pred


def gt_for_logits(entries: dict[str, dict[str, Any]], logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: transitions_to_scenes(entries[key]["transitions"], len(value))
        for key, value in logits.items()
        if key in entries
    }


def evaluate_best(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> dict[str, float]:
    common_pred = {k: v for k, v in pred.items() if k in gt}
    common_gt = {k: gt[k] for k in common_pred}
    _, f1, precision, recall, threshold, _ = mAP_f1_p_fix_r(common_pred, common_gt, fixed_r=-1)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "n_videos": len(common_pred),
    }


def evaluate_fixed(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray], threshold: float) -> dict[str, float]:
    tp = fp = fn = 0
    for key, value in pred.items():
        if key not in gt:
            continue
        binary = (value.squeeze() > threshold).astype(np.uint8)
        pred_scenes = predictions_to_scenes(binary)
        _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt[key], pred_scenes)
        tp += tp_
        fp += fp_
        fn += fn_
    f1, precision, recall = f1_pr(tp, fp, fn)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def find_temperature(logits: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> float:
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for key, arr in logits.items():
        if key not in gt:
            continue
        labels, _ = scenes2zero_one_representation(gt[key], len(arr))
        all_logits.append(torch.from_numpy(arr.astype(np.float32)))
        all_labels.append(torch.from_numpy(labels.astype(np.float32)).unsqueeze(1))

    if not all_logits:
        return 1.0

    logits_t = torch.cat(all_logits, 0)
    labels_t = torch.cat(all_labels, 0)

    def objective(temp: float) -> float:
        return float(F.binary_cross_entropy_with_logits(logits_t / temp, labels_t).item())

    result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def maybe_cap_keys(keys: list[str], limit: int) -> list[str]:
    if limit > 0:
        return keys[:limit]
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument("--base-ckpt", default=DEFAULT_BASE_CKPT)
    parser.add_argument("--out-ckpt", default=DEFAULT_OUT_CKPT)
    parser.add_argument("--sample-cache", default=DEFAULT_SAMPLE_CACHE)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--eval-cache-dir", default=DEFAULT_EVAL_CACHE_DIR)
    parser.add_argument("--resume-state", default=DEFAULT_RESUME_STATE)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--run-manifest", default="")
    parser.add_argument("--epochs", type=int, default=v8_config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--loss", choices=["bce", "focal"], default=v8_config.TRAIN_LOSS)
    parser.add_argument("--lr", type=float, default=v8_config.TRAIN_LEARNING_RATE)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--finetune-scope", choices=FINETUNE_SCOPE_CHOICES, default="head_only")
    parser.add_argument("--max-finetune-videos", type=int, default=0)
    parser.add_argument("--finetune-batches-per-video", type=int, default=0)
    parser.add_argument("--finetune-max-frames-per-batch", type=int, default=50)
    parser.add_argument("--finetune-min-neg-per-batch", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=v8_config.FOCAL_GAMMA)
    parser.add_argument("--alpha", type=float, default=v8_config.FOCAL_ALPHA)
    parser.add_argument("--weight-decay", type=float, default=v8_config.TRAIN_WEIGHT_DECAY)
    parser.add_argument("--manyhot-weight", type=float, default=v8_config.MANYHOT_WEIGHT)
    parser.add_argument("--sigma", type=float, default=v8_config.DEPLOY_SIGMA)
    parser.add_argument("--temperature-mode", choices=["off", "auto"], default="auto")
    parser.add_argument("--boundary-window", type=int, default=1)
    parser.add_argument("--max-samples-per-video", type=int, default=160)
    parser.add_argument("--max-total-samples", type=int, default=0)
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--min-neg-per-video", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-train-videos", type=int, default=0)
    parser.add_argument("--max-val-videos", type=int, default=200)
    parser.add_argument("--max-test-videos", type=int, default=0)
    parser.add_argument("--max-cache-video-frames", type=int, default=180000)
    parser.add_argument("--max-cache-video-seconds", type=float, default=7200.0)
    parser.add_argument("--save-every-videos", type=int, default=50)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--stop-after-minutes", type=float, default=0.0)
    parser.add_argument("--rebuild-sample-cache", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--ignore-resume-config", action="store_true")
    parser.add_argument("--no-eval-cache", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true")
    args = parser.parse_args()
    validate_training_mode_args(args)
    args._deadline = time.monotonic() + args.stop_after_minutes * 60.0 if args.stop_after_minutes > 0 else None

    set_global_seeds(args.seed)
    print(f"Device: {device}")

    meta = load_metadata(args.meta)
    entries = meta["entries"]
    train_keys = list(meta["train_keys"])
    val_keys = maybe_cap_keys(list(meta["val_keys"]), args.max_val_videos)
    shot_test_entries = meta["shot_test_entries"]
    shot_test_keys = maybe_cap_keys(sorted(shot_test_entries.keys()), args.max_test_videos)
    print(f"Train keys: {len(train_keys)}  Val keys: {len(val_keys)}  Shot test keys: {len(shot_test_keys)}")

    base_hash = sha256_file(args.base_ckpt)
    backbone = load_supernet(args.base_ckpt)

    if args.finetune_scope == "head_only":
        pretrained_head_cpu = {
            "fc1": {k: v.detach().cpu().clone() for k, v in backbone.fc1_0.state_dict().items()},
            "cls_layer1": {k: v.detach().cpu().clone() for k, v in backbone.cls_layer1.state_dict().items()},
            "cls_layer2": {k: v.detach().cpu().clone() for k, v in backbone.cls_layer2.state_dict().items()},
            "full_state": {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()},
        }

        try:
            features, one_hot, boundary, sample_stats = build_or_load_sample_cache(
                args.sample_cache,
                args.meta,
                entries,
                train_keys,
                backbone,
                base_hash,
                args,
            )
        except TimeBudgetExpired as exc:
            print(f"{exc} Resume by rerunning the same command with previous outputs restored.")
            backbone.cpu()
            if device == "cuda":
                torch.cuda.empty_cache()
            return

        backbone.cpu()
        del backbone
        if device == "cuda":
            torch.cuda.empty_cache()

        print("Sample cache stats:")
        print(f"  samples: {sample_stats['samples']}")
        print(f"  one-hot positive rate: {sample_stats['one_hot_positive_rate']:.6f}")
        print(f"  boundary positive rate: {sample_stats['boundary_positive_rate']:.6f}")
        if sample_stats["skipped"]:
            print(f"  skipped videos: {len(sample_stats['skipped'])}")
        data_manifest = None
        if args.data_manifest:
            data_manifest = write_training_data_manifest(
                args.data_manifest,
                args.meta,
                args.base_ckpt,
                entries,
                sample_stats,
                args.data_seed,
            )
            print(f"Training data manifest -> {args.data_manifest}")

        dataset = SampleFeatureDataset(features, one_hot, boundary)
        in_features = int(features.shape[1])
        head = ClassificationHead(in_features=in_features).to(device)
        head.fc1.load_state_dict(pretrained_head_cpu["fc1"])
        head.cls_layer1.load_state_dict(pretrained_head_cpu["cls_layer1"])
        head.cls_layer2.load_state_dict(pretrained_head_cpu["cls_layer2"])

        training_started = time.perf_counter()
        try:
            head, train_losses = train_head(head, dataset, args)
        except TimeBudgetExpired as exc:
            print(f"{exc} Resume by rerunning the same command with previous outputs restored.")
            return
        training_elapsed_seconds = time.perf_counter() - training_started
        state_fingerprint = hash_head_state(head)
        full_state = merge_head_into_state(pretrained_head_cpu["full_state"], head)
        eval_model = load_model_from_state(full_state)
    else:
        print(f"End-to-end fine-tune scope: {args.finetune_scope}")
        training_started = time.perf_counter()
        try:
            eval_model, train_losses, sample_stats = fine_tune_model(backbone, entries, train_keys, args)
        except TimeBudgetExpired as exc:
            print(f"{exc} Resume by rerunning the same command with previous outputs restored.")
            return
        training_elapsed_seconds = time.perf_counter() - training_started
        state_fingerprint = hash_state_dict(eval_model.state_dict())
        full_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        print("Fine-tune stats:")
        print(f"  videos seen: {sample_stats['videos_seen']}/{sample_stats['videos_requested']}")
        print(f"  batches: {sample_stats['batches']}")
        print(f"  frames used: {sample_stats['frames_used']}")
        print(f"  one-hot positive rate: {sample_stats['one_hot_positive_rate']:.6f}")
        print(f"  boundary positive rate: {sample_stats['boundary_positive_rate']:.6f}")
        if sample_stats["videos_skipped"]:
            print(f"  skipped videos: {len(sample_stats['videos_skipped'])}")
        data_manifest = None
    os.makedirs(args.eval_cache_dir, exist_ok=True)

    val_cache_config = {
        "state_fingerprint": state_fingerprint,
        "keys_hash": hash_keys(val_keys),
        "split": "combined_val",
    }
    val_logits = load_or_run_logits(
        eval_model,
        entries,
        val_keys,
        os.path.join(args.eval_cache_dir, "combined_val_logits.pkl"),
        val_cache_config,
        args.no_eval_cache,
    )
    val_gt = gt_for_logits(entries, val_logits)

    pred_no_temp = logits_to_pred_dict(val_logits, temperature=1.0, sigma=args.sigma)
    val_no_temp = evaluate_best(pred_no_temp, val_gt)

    if args.temperature_mode == "auto":
        temperature = find_temperature(val_logits, val_gt)
        pred_temp = logits_to_pred_dict(val_logits, temperature=temperature, sigma=args.sigma)
        val_temp = evaluate_best(pred_temp, val_gt)
        use_temperature = val_temp["f1"] >= val_no_temp["f1"]
        deploy_temperature = temperature if use_temperature else 1.0
        deploy_threshold = val_temp["threshold"] if use_temperature else val_no_temp["threshold"]
        deploy_val = val_temp if use_temperature else val_no_temp
    else:
        temperature = 1.0
        val_temp = None
        use_temperature = False
        deploy_temperature = 1.0
        deploy_threshold = val_no_temp["threshold"]
        deploy_val = val_no_temp

    print("\nValidation:")
    print(
        f"  no temp: F1={val_no_temp['f1']:.6f} P={val_no_temp['precision']:.6f} "
        f"R={val_no_temp['recall']:.6f} thr={val_no_temp['threshold']:.4f}"
    )
    if val_temp is not None:
        print(
            f"  temp T={temperature:.4f}: F1={val_temp['f1']:.6f} P={val_temp['precision']:.6f} "
            f"R={val_temp['recall']:.6f} thr={val_temp['threshold']:.4f}"
        )
    else:
        print("  temp: disabled")
    print(f"  deploy: T={deploy_temperature:.4f} thr={deploy_threshold:.4f}")

    torch.save(
        {
            "net": full_state,
            "phase2_config": {
                "gamma": args.gamma,
                "alpha": args.alpha,
                "loss": args.loss,
                "temperature": deploy_temperature,
                "temperature_mode": args.temperature_mode,
                "use_temperature": use_temperature,
                "sigma": args.sigma,
                "threshold": deploy_threshold,
                "manyhot_weight": args.manyhot_weight,
                "boundary_window": args.boundary_window,
                "val_f1": deploy_val["f1"],
                "training_data": "Shot train + ClipShots train",
                "metadata": os.path.abspath(args.meta),
                "sample_cache": os.path.abspath(args.sample_cache) if args.finetune_scope == "head_only" else None,
                "sample_cache_stats": sample_stats,
            },
        },
        args.out_ckpt,
    )
    print(f"\nCheckpoint saved -> {args.out_ckpt}")

    shot_test_best = None
    shot_test_deploy = None
    if not args.skip_test_eval:
        test_cache_config = {
            "state_fingerprint": state_fingerprint,
            "keys_hash": hash_keys(shot_test_keys),
            "split": "shot_test",
        }
        test_logits = load_or_run_logits(
            eval_model,
            shot_test_entries,
            shot_test_keys,
            os.path.join(args.eval_cache_dir, "shot_test_logits.pkl"),
            test_cache_config,
            args.no_eval_cache,
        )
        test_gt = gt_for_logits(shot_test_entries, test_logits)
        test_pred = logits_to_pred_dict(test_logits, temperature=deploy_temperature, sigma=args.sigma)
        shot_test_best = evaluate_best(test_pred, test_gt)
        shot_test_deploy = evaluate_fixed(test_pred, test_gt, deploy_threshold)
        print("\nShot test (200-video AutoShot test split):")
        print(
            f"  best sweep: F1={shot_test_best['f1']:.6f} P={shot_test_best['precision']:.6f} "
            f"R={shot_test_best['recall']:.6f} thr={shot_test_best['threshold']:.4f}"
        )
        print(
            f"  deploy thr={deploy_threshold:.4f}: F1={shot_test_deploy['f1']:.6f} "
            f"P={shot_test_deploy['precision']:.6f} R={shot_test_deploy['recall']:.6f} "
            f"TP={shot_test_deploy['tp']} FP={shot_test_deploy['fp']} FN={shot_test_deploy['fn']}"
        )

    results = {
        "train_losses": train_losses,
        "training_elapsed_seconds_this_invocation": training_elapsed_seconds,
        "sample_stats": sample_stats,
        "val_no_temp": val_no_temp,
        "val_temp": val_temp,
        "deploy": {
            "temperature": deploy_temperature,
            "threshold": deploy_threshold,
            "use_temperature": use_temperature,
        },
        "shot_test_best": shot_test_best,
        "shot_test_deploy": shot_test_deploy,
        "args": vars(args),
    }
    with open(args.results, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved -> {args.results}")

    if args.run_manifest:
        run_manifest = {
            "schema_version": 1,
            "training_seed": args.seed,
            "data_seed": args.data_seed,
            "selected_keys_hash": sample_stats["selected_keys_hash"],
            "state_fingerprint": state_fingerprint,
            "configuration": {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("_")
            },
            "artifacts": {
                "checkpoint": {
                    "path": os.path.abspath(args.out_ckpt),
                    "sha256": sha256_file(args.out_ckpt),
                },
                "results": {
                    "path": os.path.abspath(args.results),
                    "sha256": sha256_file(args.results),
                },
                "data_manifest": (
                    {
                        "path": os.path.abspath(args.data_manifest),
                        "sha256": sha256_file(args.data_manifest),
                    }
                    if data_manifest is not None
                    else None
                ),
            },
            "validation": {
                "videos": len(val_logits),
                "logits_keys_hash": hash_keys(list(val_logits)),
                "no_temperature": val_no_temp,
                "temperature_candidate": val_temp,
            },
            "training_elapsed_seconds_this_invocation": training_elapsed_seconds,
            "test_evaluated": not args.skip_test_eval,
        }
        os.makedirs(os.path.dirname(args.run_manifest) or ".", exist_ok=True)
        with open(args.run_manifest, "w", encoding="utf-8") as handle:
            json.dump(run_manifest, handle, indent=2)
            handle.write("\n")
        print(f"Run manifest -> {args.run_manifest}")


if __name__ == "__main__":
    main()
