"""Dependency-light checks for the pinned final-v8 submission identity."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refineshot import v8_config  # noqa: E402


class FinalV8ConfigTests(unittest.TestCase):
    def test_training_config_matches_selection_summary(self) -> None:
        summary = json.loads(
            (ROOT / "models" / "metadata" / "final_selection_summary.json").read_text(encoding="utf-8")
        )
        selected = summary["best_config"]
        self.assertEqual(v8_config.FOCAL_GAMMA, selected["gamma"])
        self.assertEqual(v8_config.FOCAL_ALPHA, selected["alpha"])
        self.assertEqual(v8_config.MANYHOT_WEIGHT, selected["manyhot_weight"])
        self.assertEqual(v8_config.TRAIN_LEARNING_RATE, selected["lr"])
        self.assertEqual(v8_config.TRAIN_WEIGHT_DECAY, selected["weight_decay"])

    def test_deploy_config_matches_selection_summary(self) -> None:
        summary = json.loads(
            (ROOT / "models" / "metadata" / "final_selection_summary.json").read_text(encoding="utf-8")
        )
        validation = summary["best_validation_summary"]
        self.assertEqual(v8_config.DEPLOY_TEMPERATURE, validation["temperature"])
        self.assertEqual(v8_config.DEPLOY_SIGMA, validation["sigma"])
        self.assertEqual(v8_config.DEPLOY_THRESHOLD, summary["selected_threshold"])


if __name__ == "__main__":
    unittest.main()
