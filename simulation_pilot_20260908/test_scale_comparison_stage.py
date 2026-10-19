"""Contract tests for the blocked residual-scale comparison preparation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


WORKING_DIR = Path(__file__).resolve().parent
if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))

from scale_comparison.fit_local_scales import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    FEATURES,
    ORACLE_COLUMNS,
    construct_symmetric_intervals,
    deterministic_neighbour_indices,
    load_json,
    run_stage,
    safe_correlation,
    select_validation_candidate,
    split_conformal_quantile,
)


class ScaleComparisonPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="gccp_scale_contract_")
        cls.output_root = Path(cls.temporary.name)
        cls.config = load_json(DEFAULT_CONFIG_PATH)
        cls.check = run_stage(WORKING_DIR, DEFAULT_CONFIG_PATH, cls.output_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_01_exact_subset_access(self) -> None:
        leakage = self.check["leakage_checks"]
        self.assertTrue(leakage["only_scale_fit_labels_accessed"])
        for entry in leakage["feature_access_log"] + leakage["target_access_log"]:
            self.assertEqual(entry["rows_loaded"], 200)
            self.assertEqual(entry["split_values"], ["local_calibration"])
            self.assertEqual(entry["local_role_values"], ["scale_fit"])

    def test_02_no_final_or_test_labels_loaded(self) -> None:
        leakage = self.check["leakage_checks"]
        self.assertFalse(leakage["tuning_calibration_labels_accessed"])
        self.assertFalse(leakage["validation_labels_accessed"])
        self.assertFalse(leakage["final_calibration_labels_accessed"])
        self.assertFalse(leakage["test_labels_accessed"])

    def test_03_no_oracle_columns_used(self) -> None:
        self.assertFalse(self.check["leakage_checks"]["oracle_files_read_or_hashed"])
        for scenario in ("S1", "S2", "S3", "S4"):
            frame = pd.read_csv(self.output_root / "scale_outputs" / f"{scenario}_scale_fit_preparation.csv")
            self.assertFalse(set(frame.columns).intersection(ORACLE_COLUMNS))
            self.assertNotIn("y", frame.columns)

    def test_04_scale_fit_only_preprocessing(self) -> None:
        self.assertTrue(self.check["configuration_checks"]["scale_fit_only_preprocessing"])
        for scenario in ("S1", "S2", "S3", "S4"):
            frame = pd.read_csv(self.output_root / "scale_outputs" / f"{scenario}_scale_fit_preparation.csv")
            z = frame[[f"z_{feature}" for feature in FEATURES]].to_numpy(dtype=float)
            np.testing.assert_allclose(np.mean(z, axis=0), 0.0, rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(np.std(z, axis=0), 1.0, rtol=0.0, atol=1e-14)

    def test_05_euc_and_gccm_search_budgets_match(self) -> None:
        self.assertTrue(self.check["configuration_checks"]["euc_gccm_search_budgets_identical"])
        self.assertEqual(
            self.config["methods"]["Normalized-EUC"]["candidate_k"],
            self.config["methods"]["Normalized-GCCM"]["candidate_k"],
        )

    def test_06_deterministic_tie_handling(self) -> None:
        reference = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        indices, distances = deterministic_neighbour_indices(
            np.array([[0.0, 0.0]]),
            reference,
            np.array([999]),
            np.array([20, 10, 30]),
            k=3,
            leave_one_out=False,
        )
        self.assertEqual(indices.tolist(), [[1, 0, 2]])
        np.testing.assert_array_equal(distances, np.ones((1, 3)))

    def test_07_all_prepared_scales_are_positive_finite(self) -> None:
        for scenario in ("S1", "S2", "S3", "S4"):
            frame = pd.read_csv(self.output_root / "scale_outputs" / f"{scenario}_scale_fit_preparation.csv")
            columns = [column for column in frame.columns if column.endswith(tuple(f"k{k}" for k in (10, 20, 30, 50, 75, 100)))]
            values = frame[columns].to_numpy(dtype=float)
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((values > 0.0).all())

    def test_08_deterministic_reruns(self) -> None:
        reproducibility = self.check["reproducibility"]
        self.assertEqual(reproducibility["status"], "PASS")
        self.assertTrue(all(reproducibility["csv_hash_match"].values()))
        self.assertTrue(all(reproducibility["scaler_hash_match"].values()))
        self.assertTrue(reproducibility["manifest_hash_match"])

    def test_09_conformal_quantile_rank(self) -> None:
        quantile, rank = split_conformal_quantile(np.arange(1.0, 101.0), alpha=0.10)
        self.assertEqual(rank, 91)
        self.assertEqual(quantile, 91.0)

    def test_10_alpha_interval_construction(self) -> None:
        quantile, _ = split_conformal_quantile(np.arange(1.0, 101.0), alpha=0.10)
        lower, upper = construct_symmetric_intervals([10.0, 20.0], [1.0, 2.0], quantile)
        np.testing.assert_array_equal(lower, np.array([-81.0, -162.0]))
        np.testing.assert_array_equal(upper, np.array([101.0, 202.0]))

    def test_11_s4_constant_sigma_handling(self) -> None:
        self.assertIsNone(safe_correlation([1.0, 2.0, 3.0], [1.3, 1.3, 1.3]))

    def test_12_coverage_constrained_selection(self) -> None:
        rows = [
            {"k": 10, "coverage": 0.87, "mean_interval_score": 3.0, "mean_width": 2.0},
            {"k": 20, "coverage": 0.89, "mean_interval_score": 2.5, "mean_width": 2.1},
            {"k": 50, "coverage": 0.91, "mean_interval_score": 2.5, "mean_width": 2.1},
        ]
        selected = select_validation_candidate(rows, target_coverage=0.90, tolerance=0.02)
        self.assertEqual(selected["k"], 50)
        self.assertEqual(selected["selection_mode"], "coverage_tolerance_feasible")


if __name__ == "__main__":
    unittest.main()
