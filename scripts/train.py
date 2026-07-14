"""Run Phase 2 training with the pinned final-v8 hyperparameters."""

from __future__ import annotations

import sys

from refineshot import v8_config
from refineshot.train_phase2 import main

if __name__ == "__main__":
    # Pinned options are appended so they take precedence over accidental
    # duplicate options supplied by the caller.
    sys.argv.extend(v8_config.training_cli_args())
    main()
