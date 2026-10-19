"""Fit frozen per-scenario predictors and prepare authorized residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PREDICTOR_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PREDICTOR_DIR / "predictor_config_v1.json"
DEFAULT_WORKING_DIR = PREDICTOR_DIR.parent
SCENARIOS = ("S1", "S2", "S3", "S4")
META_COLUMNS = ["sample_id", "scenario", "split", "local_calibration_role"]
ORACLE_COLUMNS = {"mu_true", "sigma_true", "epsilon", "g_true", "eta"}
PROHIBITED_LABEL_SPLITS = {"final_calibration", "test"}


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_guarded(path: Path, payload: dict[str, Any], overwrite: bool = False) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == serialized:
            return
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite nonidentical file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def make_model(config: dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(**config["model_configuration"])


def predict_in_fixed_tree_order(model: RandomForestRegressor, features: np.ndarray) -> np.ndarray:
    """Average tree predictions in a fixed order for bytewise reproducibility."""
    assert model.estimators_
    tree_predictions = np.empty((len(model.estimators_), len(features)), dtype=np.float64)
    for index, estimator in enumerate(model.estimators_):
        tree_predictions[index] = estimator.predict(features)
    return np.mean(tree_predictions, axis=0, dtype=np.float64)


def verify_generator_inputs(
    working_dir: Path | str,
    predictor_config: dict[str, Any],
) -> dict[str, Any]:
    """Verify generator status and all eight frozen input hashes before fitting."""
    root = Path(working_dir).resolve()
    generator_config_path = root / "generator" / "config_v1.json"
    generator_check_path = root / "generator_check.json"
    generator_config = load_json(generator_config_path)
    generator_check = load_json(generator_check_path)
    assert generator_config["generator_version"] == predictor_config["generator_version"]
    assert generator_check["generator_version"] == predictor_config["generator_version"]
    assert generator_check["overall_status"] == "PASS"
    assert sha256_file(generator_config_path) == predictor_config["generator_config_sha256"]
    assert sha256_file(generator_check_path) == predictor_config["generator_check_sha256"]

    recomputed: dict[str, str] = {}
    for relative_name, expected_hash in predictor_config["input_file_hashes"].items():
        path = root / relative_name
        assert path.is_file(), f"Missing frozen generator output: {path}"
        actual_hash = sha256_file(path)
        assert actual_hash == expected_hash, f"Input hash mismatch: {relative_name}"
        assert generator_check["sha256"][relative_name] == expected_hash
        recomputed[relative_name] = actual_hash
    assert len(recomputed) == 8
    return {
        "status": "PASS",
        "generator_overall_status": generator_check["overall_status"],
        "generator_version": generator_config["generator_version"],
        "generator_config_sha256": sha256_file(generator_config_path),
        "generator_check_sha256": sha256_file(generator_check_path),
        "recomputed_input_hashes": recomputed,
        "truth_files_hashed_but_not_parsed": True,
    }


def create_frozen_input_manifest(
    predictor_config: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_version": "frozen_input_manifest_v1",
        "predictor_stage_version": predictor_config["predictor_stage_version"],
        "created_at": predictor_config["stage_timestamp"],
        "generator_version": predictor_config["generator_version"],
        "generator_status_required": "PASS",
        "generator_status_observed": verification["generator_overall_status"],
        "generator_config_sha256": verification["generator_config_sha256"],
        "generator_check_sha256": verification["generator_check_sha256"],
        "input_file_hashes": verification["recomputed_input_hashes"],
        "hash_verification": "PASS",
        "truth_files_hashed_but_not_parsed": True,
    }


def read_predictor_inputs(observed_path: Path, features: list[str]) -> pd.DataFrame:
    """Read metadata and predictor features for all rows, explicitly excluding y."""
    header = list(pd.read_csv(observed_path, nrows=0).columns)
    expected_observed = META_COLUMNS + features + ["y"]
    assert header == expected_observed, f"Unexpected observed schema: {observed_path}"
    frame = pd.read_csv(observed_path, usecols=META_COLUMNS + features, keep_default_na=False)
    assert list(frame.columns) == META_COLUMNS + features
    assert len(frame) == 2000
    assert frame["sample_id"].is_unique
    assert not (set(frame.columns) & ORACLE_COLUMNS)
    assert "y" not in frame.columns
    return frame


def load_authorized_labels(
    observed_path: Path,
    metadata: pd.DataFrame,
    authorized_splits: Iterable[str],
    label_access_log: list[dict[str, Any]],
) -> pd.Series:
    """Load y only for selected row numbers; prohibited split labels never enter a DataFrame."""
    requested = set(authorized_splits)
    assert requested
    assert not requested.intersection(PROHIBITED_LABEL_SPLITS)
    mask = metadata["split"].isin(requested).to_numpy()
    selected_positions = np.flatnonzero(mask)
    file_line_numbers = set((selected_positions + 1).tolist())

    def skip_unselected(line_number: int) -> bool:
        return line_number != 0 and line_number not in file_line_numbers

    labels = pd.read_csv(
        observed_path,
        usecols=["sample_id", "y"],
        skiprows=skip_unselected,
    )
    expected_ids = metadata.iloc[selected_positions]["sample_id"].reset_index(drop=True)
    pd.testing.assert_series_equal(labels["sample_id"], expected_ids, check_names=False, check_dtype=False)
    assert len(labels) == len(selected_positions)
    assert np.isfinite(labels["y"].to_numpy(dtype=float)).all()
    loaded_splits = sorted(metadata.iloc[selected_positions]["split"].unique().tolist())
    assert set(loaded_splits) == requested
    label_access_log.append(
        {
            "file": observed_path.name,
            "requested_splits": sorted(requested),
            "loaded_splits": loaded_splits,
            "row_count": len(labels),
            "prohibited_splits_loaded": [],
        }
    )
    return pd.Series(labels["y"].to_numpy(dtype=float), index=selected_positions, name="y")


def subset_diagnostics(
    scenario: str,
    subset: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    residual = y_true - y_pred
    absolute = np.abs(residual)
    return {
        "scenario": scenario,
        "subset": subset,
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_abs_residual": float(absolute.mean()),
        "median_abs_residual": float(np.median(absolute)),
        "p90_abs_residual": float(np.quantile(absolute, 0.90)),
    }


def _output_paths(output_root: Path) -> list[Path]:
    output_dir = output_root / "predictor_outputs"
    return [output_dir / f"{scenario}_{suffix}" for scenario in SCENARIOS for suffix in ("predictions.csv", "authorized_residuals.csv", "model.joblib")]


def fit_predictor_stage(
    working_dir: Path | str,
    output_root: Path | str,
    predictor_config: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the complete non-oracle predictor stage for all four scenarios."""
    input_root = Path(working_dir).resolve()
    output_root_path = Path(output_root).resolve()
    output_dir = output_root_path / "predictor_outputs"
    paths = _output_paths(output_root_path)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite existing predictor outputs: " + ", ".join(str(path) for path in existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    features = list(predictor_config["authorized_features"])
    assert features == ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"]
    model_template = make_model(predictor_config)
    assert model_template.get_params(deep=False) == predictor_config["model_configuration"]

    inputs: dict[str, pd.DataFrame] = {}
    for scenario in SCENARIOS:
        inputs[scenario] = read_predictor_inputs(input_root / "data" / f"{scenario}_observed.csv", features)
    common_columns = ["sample_id", "split", "local_calibration_role"] + features
    for scenario in SCENARIOS[1:]:
        pd.testing.assert_frame_equal(inputs["S1"][common_columns], inputs[scenario][common_columns], check_dtype=False)

    label_access_log: list[dict[str, Any]] = []
    non_oracle_diagnostics: list[dict[str, Any]] = []
    per_scenario: dict[str, Any] = {}
    train_responses: dict[str, np.ndarray] = {}
    # Keep strong references until the cross-scenario assertion.  Saving only
    # id(model) is unsafe because CPython may reuse an address after an object
    # from an earlier loop iteration is released.
    fitted_models: list[RandomForestRegressor] = []

    for scenario in SCENARIOS:
        observed_path = input_root / "data" / f"{scenario}_observed.csv"
        frame = inputs[scenario]
        split_counts = {key: int(value) for key, value in frame["split"].value_counts().to_dict().items()}
        assert split_counts == {
            "train": 1000,
            "local_calibration": 300,
            "validation": 200,
            "final_calibration": 200,
            "test": 300,
        }
        train_labels = load_authorized_labels(observed_path, frame, {"train"}, label_access_log)
        assert len(train_labels) == 1000
        train_positions = train_labels.index.to_numpy(dtype=int)
        assert set(frame.iloc[train_positions]["split"]) == {"train"}

        model = make_model(predictor_config)
        fitted_models.append(model)
        params_before = model.get_params(deep=False)
        assert params_before == predictor_config["model_configuration"]
        x_all = frame[features].to_numpy(dtype=float)
        x_train = x_all[train_positions]
        model.fit(x_train, train_labels.to_numpy(dtype=float))
        assert model.get_params(deep=False) == params_before
        # sklearn aggregates tree predictions in parallel when n_jobs=-1, so
        # thread scheduling can change the final few floating-point bits.  A
        # fixed estimator order preserves the same forest prediction while
        # producing bytewise-reproducible frozen CSV artifacts.
        assert predictor_config["prediction_aggregation"] == "fixed_estimators_order_mean"
        y_pred_all = predict_in_fixed_tree_order(model, x_all)
        assert len(y_pred_all) == 2000 and np.isfinite(y_pred_all).all()

        prediction = frame[META_COLUMNS].copy()
        prediction["y_pred"] = y_pred_all
        assert list(prediction.columns) == predictor_config["prediction_schema"]
        assert "y" not in prediction.columns and not (set(prediction.columns) & ORACLE_COLUMNS)

        model_path = output_dir / f"{scenario}_model.joblib"
        prediction_path = output_dir / f"{scenario}_predictions.csv"
        prediction.to_csv(prediction_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")
        joblib.dump(model, model_path, compress=3, protocol=5)

        tuning_labels = load_authorized_labels(
            observed_path,
            frame,
            {"local_calibration", "validation"},
            label_access_log,
        )
        authorized_labels = pd.concat([train_labels, tuning_labels]).sort_index()
        assert len(authorized_labels) == 1500
        authorized_positions = authorized_labels.index.to_numpy(dtype=int)
        authorized_split_values = set(frame.iloc[authorized_positions]["split"])
        assert authorized_split_values == {"train", "local_calibration", "validation"}
        assert not authorized_split_values.intersection(PROHIBITED_LABEL_SPLITS)

        residual_values = authorized_labels.to_numpy(dtype=float) - y_pred_all[authorized_positions]
        residual = frame.iloc[authorized_positions][META_COLUMNS].copy()
        residual["y_pred"] = y_pred_all[authorized_positions]
        residual["residual"] = residual_values
        residual["abs_residual"] = np.abs(residual_values)
        assert list(residual.columns) == predictor_config["authorized_residual_schema"]
        assert "y" not in residual.columns and not (set(residual.columns) & ORACLE_COLUMNS)
        assert not set(residual["split"]).intersection(PROHIBITED_LABEL_SPLITS)
        local_roles = residual.loc[residual["split"] == "local_calibration", "local_calibration_role"].value_counts().to_dict()
        assert local_roles == {"scale_fit": 200, "tuning_calibration": 100}
        residual_path = output_dir / f"{scenario}_authorized_residuals.csv"
        residual.to_csv(residual_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")

        diagnostic_masks = {
            "train": residual["split"] == "train",
            "scale_fit": (residual["split"] == "local_calibration") & (residual["local_calibration_role"] == "scale_fit"),
            "tuning_calibration": (residual["split"] == "local_calibration") & (residual["local_calibration_role"] == "tuning_calibration"),
            "validation": residual["split"] == "validation",
        }
        labels_by_id = pd.Series(authorized_labels.to_numpy(), index=frame.iloc[authorized_positions]["sample_id"].to_numpy())
        for subset, mask in diagnostic_masks.items():
            subset_frame = residual.loc[mask]
            y_true = labels_by_id.loc[subset_frame["sample_id"]].to_numpy(dtype=float)
            non_oracle_diagnostics.append(
                subset_diagnostics(scenario, subset, y_true, subset_frame["y_pred"].to_numpy(dtype=float))
            )

        reloaded = joblib.load(model_path)
        reloaded_predictions = reloaded.predict(x_all)
        np.testing.assert_allclose(reloaded_predictions, y_pred_all, rtol=0.0, atol=1e-12)
        assert reloaded.get_params(deep=False) == predictor_config["model_configuration"]
        train_responses[scenario] = train_labels.to_numpy(dtype=float)
        per_scenario[scenario] = {
            "model_path": model_path,
            "prediction_path": prediction_path,
            "residual_path": residual_path,
            "prediction_rows": len(prediction),
            "residual_rows": len(residual),
            "train_rows_passed_to_fit": len(train_positions),
            "fit_split_values": ["train"],
            "feature_columns": features,
            "model_configuration": model.get_params(deep=False),
            "model_reload_prediction_match": True,
            "split_counts": split_counts,
            "local_calibration_role_counts": {key: int(value) for key, value in local_roles.items()},
        }

    assert len(fitted_models) == 4
    assert len({id(model) for model in fitted_models}) == 4
    response_pair_checks: dict[str, bool] = {}
    for left_index, left in enumerate(SCENARIOS):
        for right in SCENARIOS[left_index + 1 :]:
            differs = not np.array_equal(train_responses[left], train_responses[right])
            assert differs
            response_pair_checks[f"{left}_vs_{right}"] = differs

    output_hashes: dict[str, str] = {}
    model_hashes: dict[str, str] = {}
    for scenario in SCENARIOS:
        for kind in ("predictions.csv", "authorized_residuals.csv"):
            path = output_dir / f"{scenario}_{kind}"
            output_hashes[path.name] = sha256_file(path)
        model_path = output_dir / f"{scenario}_model.joblib"
        model_hashes[model_path.name] = sha256_file(model_path)

    assert not any(PROHIBITED_LABEL_SPLITS.intersection(entry["loaded_splits"]) for entry in label_access_log)
    return {
        "per_scenario": per_scenario,
        "non_oracle_diagnostics": non_oracle_diagnostics,
        "label_access_log": label_access_log,
        "output_hashes": output_hashes,
        "model_hashes": model_hashes,
        "feature_matrices_identical_across_scenarios": True,
        "responses_differ_on_train": response_pair_checks,
        "distinct_model_objects": True,
        "truth_files_parsed": False,
    }


def predictor_reproducibility_check(
    working_dir: Path,
    predictor_config: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gccp_predictor_a_") as first_dir, tempfile.TemporaryDirectory(prefix="gccp_predictor_b_") as second_dir:
        verify_generator_inputs(working_dir, predictor_config)
        first = fit_predictor_stage(working_dir, Path(first_dir), predictor_config)
        verify_generator_inputs(working_dir, predictor_config)
        second = fit_predictor_stage(working_dir, Path(second_dir), predictor_config)
        prediction_hash_match = {
            name: first["output_hashes"][name] == second["output_hashes"][name]
            for name in first["output_hashes"]
            if name.endswith("_predictions.csv")
        }
        residual_hash_match = {
            name: first["output_hashes"][name] == second["output_hashes"][name]
            for name in first["output_hashes"]
            if name.endswith("_authorized_residuals.csv")
        }
        model_hash_match = {
            name: first["model_hashes"][name] == second["model_hashes"][name]
            for name in first["model_hashes"]
        }
        assert all(prediction_hash_match.values())
        assert all(residual_hash_match.values())
        warnings: list[str] = []
        if not all(model_hash_match.values()):
            warnings.append(
                "Joblib model bytes were not identical across runs; CSV predictions/residuals and reloaded predictions remained identical."
            )
    return {
        "status": "PASS",
        "separate_temporary_directories": True,
        "prediction_hash_match": prediction_hash_match,
        "residual_hash_match": residual_hash_match,
        "joblib_model_hash_match": model_hash_match,
        "joblib_hashes_all_identical": all(model_hash_match.values()),
        "warnings": warnings,
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(spearmanr(x, y).statistic)


def _load_truth_for_oracle_subsets(
    truth_path: Path,
    allowed_splits: set[str],
) -> pd.DataFrame:
    metadata = pd.read_csv(truth_path, usecols=META_COLUMNS, keep_default_na=False)
    assert allowed_splits == {"local_calibration", "validation"}
    mask = metadata["split"].isin(allowed_splits).to_numpy()
    positions = np.flatnonzero(mask)
    line_numbers = set((positions + 1).tolist())

    def skip_unselected(line_number: int) -> bool:
        return line_number != 0 and line_number not in line_numbers

    truth = pd.read_csv(
        truth_path,
        usecols=META_COLUMNS + ["mu_true", "sigma_true"],
        skiprows=skip_unselected,
        keep_default_na=False,
    )
    assert set(truth["split"]) == allowed_splits
    assert len(truth) == 500
    return truth


def run_oracle_diagnostics(
    working_dir: Path | str,
    output_root: Path | str,
    outputs_frozen: bool,
    truth_access_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read selected truth rows only after fitting artifacts are saved and hashed."""
    assert outputs_frozen, "Oracle diagnostics cannot run before fitting outputs are frozen."
    input_root = Path(working_dir).resolve()
    output_dir = Path(output_root).resolve() / "predictor_outputs"
    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for scenario in SCENARIOS:
        truth_path = input_root / "data" / f"{scenario}_truth.csv"
        truth = _load_truth_for_oracle_subsets(truth_path, {"local_calibration", "validation"})
        truth_access_log.append(
            {
                "file": truth_path.name,
                "loaded_splits": ["local_calibration", "validation"],
                "row_count": len(truth),
                "accessed_after_output_hash_freeze": True,
            }
        )
        residual = pd.read_csv(output_dir / f"{scenario}_authorized_residuals.csv", keep_default_na=False)
        residual = residual[residual["split"].isin({"local_calibration", "validation"})]
        merged = residual.merge(
            truth[["sample_id", "scenario", "split", "local_calibration_role", "mu_true", "sigma_true"]],
            on=META_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        assert len(merged) == 500
        for subset in ("local_calibration", "validation"):
            frame = merged[merged["split"] == subset]
            absolute_residual = frame["abs_residual"].to_numpy(dtype=float)
            residual_squared = frame["residual"].to_numpy(dtype=float) ** 2
            sigma = frame["sigma_true"].to_numpy(dtype=float)
            sigma_squared = sigma**2
            prediction_bias = np.abs(
                frame["y_pred"].to_numpy(dtype=float) - frame["mu_true"].to_numpy(dtype=float)
            )
            if np.std(sigma) == 0.0:
                warnings.append(
                    f"{scenario} {subset}: sigma_true is constant, so sigma-based oracle correlations are undefined and recorded as null."
                )
            results.append(
                {
                    "oracle_diagnostic_scenario": scenario,
                    "oracle_diagnostic_subset": subset,
                    "oracle_diagnostic_n": int(len(frame)),
                    "oracle_diagnostic_pearson_abs_residual_sigma_true": _safe_pearson(absolute_residual, sigma),
                    "oracle_diagnostic_spearman_abs_residual_sigma_true": _safe_spearman(absolute_residual, sigma),
                    "oracle_diagnostic_pearson_squared_residual_sigma_true_squared": _safe_pearson(residual_squared, sigma_squared),
                    "oracle_diagnostic_pearson_abs_prediction_bias_sigma_true": _safe_pearson(prediction_bias, sigma),
                    "oracle_diagnostic_mean_abs_prediction_bias": float(prediction_bias.mean()),
                    "oracle_diagnostic_rmse_y_pred_mu_true": float(
                        np.sqrt(np.mean((frame["y_pred"].to_numpy(dtype=float) - frame["mu_true"].to_numpy(dtype=float)) ** 2))
                    ),
                }
            )
    return results, warnings


def run_stage(
    working_dir: Path | str = DEFAULT_WORKING_DIR,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_root: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(working_dir).resolve()
    destination = Path(output_root).resolve() if output_root is not None else root
    config = load_json(config_path)
    check_path = destination / "predictor_check_v1.json"
    try:
        assert config["predictor_stage_version"] == "gccp_rf_predictor_v1"
        assert runtime_versions() == config["package_versions"]
        verification = verify_generator_inputs(root, config)
        manifest = create_frozen_input_manifest(config, verification)
        write_json_guarded(destination / "frozen_input_manifest_v1.json", manifest, overwrite=overwrite)
        fit_result = fit_predictor_stage(root, destination, config, overwrite=overwrite)
        outputs_frozen = bool(fit_result["output_hashes"] and fit_result["model_hashes"])
        assert outputs_frozen
        reproducibility = predictor_reproducibility_check(root, config)
        truth_access_log: list[dict[str, Any]] = []
        assert fit_result["truth_files_parsed"] is False and not truth_access_log
        oracle_diagnostics, oracle_warnings = run_oracle_diagnostics(
            root,
            destination,
            outputs_frozen=outputs_frozen,
            truth_access_log=truth_access_log,
        )
        warnings = list(reproducibility["warnings"]) + oracle_warnings

        checks = {
            "input_hash_verification": verification,
            "model_configuration_verification": {
                "status": "PASS",
                "identical_across_scenarios": True,
                "configuration": config["model_configuration"],
                "validation_metrics_changed_configuration": False,
            },
            "feature_authorization": {
                "status": "PASS",
                "authorized_features": config["authorized_features"],
                "exact_match": True,
                "predictor_scaler_used": False,
            },
            "split_and_row_counts": {
                "status": "PASS",
                "per_scenario": {
                    scenario: {
                        "train_rows_passed_to_fit": fit_result["per_scenario"][scenario]["train_rows_passed_to_fit"],
                        "prediction_rows": fit_result["per_scenario"][scenario]["prediction_rows"],
                        "authorized_residual_rows": fit_result["per_scenario"][scenario]["residual_rows"],
                        "split_counts": fit_result["per_scenario"][scenario]["split_counts"],
                        "local_calibration_role_counts": fit_result["per_scenario"][scenario]["local_calibration_role_counts"],
                    }
                    for scenario in SCENARIOS
                },
            },
            "leakage_guards": {
                "status": "PASS",
                "fit_split_only": "train",
                "label_access_log": fit_result["label_access_log"],
                "prohibited_label_splits_loaded": [],
                "prediction_files_exclude_y_and_oracle": True,
                "residual_files_exclude_y_and_oracle": True,
                "residual_files_exclude_final_calibration_and_test": True,
                "truth_parsed_before_output_freeze": False,
                "truth_access_log": truth_access_log,
            },
            "scenario_consistency": {
                "status": "PASS",
                "observed_predictor_matrices_identical": fit_result["feature_matrices_identical_across_scenarios"],
                "train_responses_differ": fit_result["responses_differ_on_train"],
                "distinct_model_objects": fit_result["distinct_model_objects"],
                "prediction_order_aligned_by_sample_id": True,
                "outputs_are_reusable_without_refit": True,
            },
            "prediction_reproducibility": {
                "status": "PASS",
                "hash_match": reproducibility["prediction_hash_match"],
            },
            "residual_reproducibility": {
                "status": "PASS",
                "hash_match": reproducibility["residual_hash_match"],
            },
            "model_reload_prediction_check": {
                "status": "PASS",
                "strict_tolerance": {"rtol": 0.0, "atol": 1e-12},
                "per_scenario": {
                    scenario: fit_result["per_scenario"][scenario]["model_reload_prediction_match"]
                    for scenario in SCENARIOS
                },
            },
            "joblib_reproducibility": {
                "status": "PASS",
                "hash_match": reproducibility["joblib_model_hash_match"],
                "all_identical": reproducibility["joblib_hashes_all_identical"],
                "nonidentical_joblib_bytes_are_warning_only": True,
            },
            "oracle_isolation": {
                "status": "PASS",
                "run_after_outputs_saved_and_hashed": True,
                "allowed_splits": ["local_calibration", "validation"],
                "used_for_fitting_or_selection": False,
            },
        }
        required_statuses = [item["status"] for item in checks.values()]
        overall_status = "PASS" if all(status == "PASS" for status in required_statuses) else "FAIL"
        report = {
            "predictor_stage_version": config["predictor_stage_version"],
            "generator_version": config["generator_version"],
            "stage_timestamp": config["stage_timestamp"],
            "overall_status": overall_status,
            "checks": checks,
            "output_file_hashes": fit_result["output_hashes"],
            "model_file_hashes": fit_result["model_hashes"],
            "non_oracle_diagnostics": fit_result["non_oracle_diagnostics"],
            "isolated_oracle_diagnostics": oracle_diagnostics,
            "warnings": warnings,
            "final_calibration_labels_accessed": False,
            "test_labels_accessed": False,
        }
        write_json_guarded(check_path, report, overwrite=overwrite)
        if overall_status != "PASS":
            raise RuntimeError(f"Predictor stage validation failed; see {check_path}")
        return report
    except Exception as error:
        failure = {
            "predictor_stage_version": config.get("predictor_stage_version", "unknown"),
            "overall_status": "FAIL",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        try:
            write_json_guarded(check_path, failure, overwrite=overwrite)
        except FileExistsError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = run_stage(args.working_dir, args.config, args.output_dir, overwrite=args.overwrite)
    print("Predictor stage:", report["predictor_stage_version"])
    print("Overall status:", report["overall_status"])
    print("Final-calibration labels accessed:", report["final_calibration_labels_accessed"])
    print("Test labels accessed:", report["test_labels_accessed"])
    print("Warnings:", len(report["warnings"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
