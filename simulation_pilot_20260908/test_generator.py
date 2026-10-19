"""Automated contract tests for gccp_sim_s1_s4_v1."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generator.generate_scenarios import (
    DEFAULT_CONFIG_PATH,
    SCENARIOS,
    load_config,
    run_pipeline,
    scale_function,
)


class GeneratorContractTests(unittest.TestCase):
    def test_scale_function_is_positive_and_bounded(self) -> None:
        grid = np.linspace(-100.0, 100.0, 10001)
        values = scale_function(grid)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values > 0.0).all())
        self.assertTrue((values >= 0.6).all())
        self.assertTrue((values <= 2.0).all())

    def test_full_saved_output_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gccp_generator_test_") as temporary_dir:
            output_root = Path(temporary_dir)
            report = run_pipeline(output_root, DEFAULT_CONFIG_PATH)
            self.assertEqual(report["overall_status"], "PASS")
            self.assertEqual(report["split_counts"], {
                "train": 1000,
                "local_calibration": 300,
                "validation": 200,
                "final_calibration": 200,
                "test": 300,
            })
            self.assertEqual(report["local_calibration_role_counts"], {
                "scale_fit": 200,
                "tuning_calibration": 100,
            })
            self.assertEqual(report["reproducibility_result"], "PASS")
            self.assertEqual(report["paired_design_result"], "PASS")
            self.assertEqual(report["leakage_check_result"], "PASS")
            for scenario in SCENARIOS:
                self.assertEqual(report["row_counts"][scenario], {"observed": 2000, "truth": 2000})
            saved_report = json.loads((output_root / "generator_check.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report["overall_status"], "PASS")

    def test_frozen_config_identity(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(config["generator_version"], "gccp_sim_s1_s4_v1")
        self.assertEqual(config["random_seeds"]["base_variables"], 20260907)
        self.assertEqual(config["random_seeds"]["outer_split"], 620907)
        self.assertEqual(config["random_seeds"]["local_calibration_role"], 620908)


if __name__ == "__main__":
    unittest.main(verbosity=2)
