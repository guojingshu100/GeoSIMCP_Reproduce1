"""Contract tests for the frozen shared-predictor stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

WORKING_DIR = Path(__file__).resolve().parent
if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))

from predictor.fit_predictors import (
    DEFAULT_CONFIG_PATH,
    META_COLUMNS,
    ORACLE_COLUMNS,
    SCENARIOS,
    load_json,
    run_stage,
)


class PredictorStageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="gccp_predictor_contract_")
        cls.output_root = Path(cls.temporary.name)
        cls.config = load_json(DEFAULT_CONFIG_PATH)
        cls.report = run_stage(WORKING_DIR, DEFAULT_CONFIG_PATH, cls.output_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_01_exact_feature_authorization(self) -> None:
        self.assertEqual(
            self.config["authorized_features"],
            ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"],
        )
        self.assertTrue(self.report["checks"]["feature_authorization"]["exact_match"])
        self.assertFalse(self.report["checks"]["feature_authorization"]["predictor_scaler_used"])

    def test_02_train_only_fitting(self) -> None:
        split_check = self.report["checks"]["split_and_row_counts"]["per_scenario"]
        for scenario in SCENARIOS:
            self.assertEqual(split_check[scenario]["train_rows_passed_to_fit"], 1000)
        leakage = self.report["checks"]["leakage_guards"]
        self.assertEqual(leakage["fit_split_only"], "train")
        self.assertEqual(leakage["prohibited_label_splits_loaded"], [])

    def test_03_prediction_and_residual_schemas(self) -> None:
        output_dir = self.output_root / "predictor_outputs"
        for scenario in SCENARIOS:
            prediction = pd.read_csv(output_dir / f"{scenario}_predictions.csv", keep_default_na=False)
            residual = pd.read_csv(output_dir / f"{scenario}_authorized_residuals.csv", keep_default_na=False)
            self.assertEqual(list(prediction.columns), self.config["prediction_schema"])
            self.assertEqual(list(residual.columns), self.config["authorized_residual_schema"])
            self.assertEqual(len(prediction), 2000)
            self.assertEqual(len(residual), 1500)

    def test_04_final_and_test_residuals_are_excluded(self) -> None:
        output_dir = self.output_root / "predictor_outputs"
        for scenario in SCENARIOS:
            residual = pd.read_csv(output_dir / f"{scenario}_authorized_residuals.csv", keep_default_na=False)
            self.assertFalse(set(residual["split"]) & {"final_calibration", "test"})

    def test_05_local_calibration_internal_roles(self) -> None:
        output_dir = self.output_root / "predictor_outputs"
        for scenario in SCENARIOS:
            residual = pd.read_csv(output_dir / f"{scenario}_authorized_residuals.csv", keep_default_na=False)
            local = residual[residual["split"] == "local_calibration"]
            self.assertEqual(local["local_calibration_role"].value_counts().to_dict(), {
                "scale_fit": 200,
                "tuning_calibration": 100,
            })

    def test_06_deterministic_predictions_and_residuals(self) -> None:
        prediction = self.report["checks"]["prediction_reproducibility"]
        residual = self.report["checks"]["residual_reproducibility"]
        self.assertEqual(prediction["status"], "PASS")
        self.assertTrue(all(prediction["hash_match"].values()))
        self.assertEqual(residual["status"], "PASS")
        self.assertTrue(all(residual["hash_match"].values()))

    def test_07_model_reload_consistency(self) -> None:
        reload_check = self.report["checks"]["model_reload_prediction_check"]
        self.assertEqual(reload_check["status"], "PASS")
        self.assertTrue(all(reload_check["per_scenario"].values()))

    def test_08_observed_oracle_separation(self) -> None:
        output_dir = self.output_root / "predictor_outputs"
        for scenario in SCENARIOS:
            prediction = pd.read_csv(output_dir / f"{scenario}_predictions.csv", keep_default_na=False)
            residual = pd.read_csv(output_dir / f"{scenario}_authorized_residuals.csv", keep_default_na=False)
            self.assertNotIn("y", prediction.columns)
            self.assertNotIn("y", residual.columns)
            self.assertFalse(set(prediction.columns) & ORACLE_COLUMNS)
            self.assertFalse(set(residual.columns) & ORACLE_COLUMNS)
            self.assertEqual(list(prediction.columns[:4]), META_COLUMNS)
        self.assertFalse(self.report["final_calibration_labels_accessed"])
        self.assertFalse(self.report["test_labels_accessed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
