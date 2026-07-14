"""Pinned configuration for the final RefineShot v8 submission."""

from __future__ import annotations

from typing import Final

RELEASE_NAME: Final = "RefineShot Final v8"
RELEASE_DATE: Final = "2026-07-09"

# Winning focal-loss configuration selected by the SHOT-first v8 sweep.
TRAIN_EPOCHS: Final = 20
TRAIN_LOSS: Final = "focal"
TRAIN_LEARNING_RATE: Final = 7e-6
TRAIN_WEIGHT_DECAY: Final = 1e-4
FOCAL_GAMMA: Final = 2.0
FOCAL_ALPHA: Final = 0.6
MANYHOT_WEIGHT: Final = 0.3

# Values embedded in the final v8 checkpoint after model selection.
DEPLOY_TEMPERATURE: Final = 0.24127588762141639
DEPLOY_SIGMA: Final = 2.0
DEPLOY_THRESHOLD: Final = 0.19

FINAL_CHECKPOINT_NAME: Final = "refineshot_v8_final.pth"
FINAL_CHECKPOINT_SHA256: Final = "DB57A3CC6EA1931C90085F8122D634697790A8248ABE776A93F17741895A9889"
BASE_CHECKPOINT_NAME: Final = "autoshot_base.pth"
BASE_CHECKPOINT_SHA256: Final = "3E85290546CE6D32F4A3581EC2CAE87AEDD2402246A0D46B4D361A330B4B1FA6"


def training_cli_args() -> list[str]:
    """Return the pinned CLI options used by the selected final v8 run."""

    return [
        "--epochs",
        str(TRAIN_EPOCHS),
        "--loss",
        TRAIN_LOSS,
        "--lr",
        str(TRAIN_LEARNING_RATE),
        "--weight-decay",
        str(TRAIN_WEIGHT_DECAY),
        "--gamma",
        str(FOCAL_GAMMA),
        "--alpha",
        str(FOCAL_ALPHA),
        "--manyhot-weight",
        str(MANYHOT_WEIGHT),
        "--sigma",
        str(DEPLOY_SIGMA),
        "--temperature-mode",
        "auto",
        "--finetune-scope",
        "head_only",
    ]
