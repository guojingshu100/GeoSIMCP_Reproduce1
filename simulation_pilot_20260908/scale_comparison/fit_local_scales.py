"""Shared non-GCCM preparation for the blocked residual-scale comparison.

This module deliberately stops before validation evaluation or GCCM fitting.
It only reads scale-fit labels, fits shared preprocessing on scale-fit, and
freezes deterministic EUC KNN infrastructure and the required controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler


SCENARIOS = ("S1", "S2", "S3", "S4")
FEATURES = ("s_x", "s_y", "x1", "x2", "g_proxy", "x_noise")
META_COLUMNS = ("sample_id", "scenario", "split", "local_calibration_role")
ORACLE_COLUMNS = {"mu_true", "sigma_true", "epsilon", "g_true", "eta"}
FORBIDDEN_LABEL_SPLITS = {"final_calibration", "test"}
DEFAULT_WORKING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("scale_comparison_config_v1.json")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def split_conformal_quantile(scores: Iterable[float], alpha: float) -> tuple[float, int]:
    """Return the finite-sample split-conformal order statistic and 1-based rank."""
    values = np.asarray(list(scores), dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Conformal scores must be a non-empty finite vector.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    rank = min(int(math.ceil((len(values) + 1) * (1.0 - alpha))), len(values))
    return float(np.sort(values, kind="mergesort")[rank - 1]), rank


def construct_symmetric_intervals(
    y_pred: Iterable[float], estimated_scale: Iterable[float], quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(list(y_pred), dtype=float)
    scales = np.asarray(list(estimated_scale), dtype=float)
    if predictions.shape != scales.shape:
        raise ValueError("Predictions and scales must have identical shapes.")
    if not np.isfinite(predictions).all() or not np.isfinite(scales).all():
        raise ValueError("Predictions and scales must be finite.")
    if np.any(scales <= 0.0) or not np.isfinite(quantile) or quantile < 0.0:
        raise ValueError("Scales must be positive and quantile must be finite and nonnegative.")
    half_width = float(quantile) * scales
    return predictions - half_width, predictions + half_width


def interval_score(
    y_true: Iterable[float], lower: Iterable[float], upper: Iterable[float], alpha: float
) -> np.ndarray:
    truth = np.asarray(list(y_true), dtype=float)
    lo = np.asarray(list(lower), dtype=float)
    hi = np.asarray(list(upper), dtype=float)
    if truth.shape != lo.shape or truth.shape != hi.shape or not 0.0 < alpha < 1.0:
        raise ValueError("Interval-score inputs are invalid.")
    score = hi - lo
    score = score + (2.0 / alpha) * (lo - truth) * (truth < lo)
    score = score + (2.0 / alpha) * (truth - hi) * (truth > hi)
    return score


def safe_correlation(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = np.asarray(list(left), dtype=float)
    y = np.asarray(list(right), dtype=float)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("Correlation inputs must align and contain at least two values.")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def select_validation_candidate(
    rows: list[dict[str, Any]], target_coverage: float, tolerance: float
) -> dict[str, Any]:
    """Apply the preregistered coverage-constrained deterministic rule."""
    if not rows:
        raise ValueError("No validation candidates were supplied.")
    eligible = [
        row
        for row in rows
        if abs(float(row["coverage"]) - target_coverage) <= tolerance + 1e-12
    ]

    def larger_k(row: dict[str, Any]) -> float:
        k = row.get("k")
        return -float(k) if k is not None else 0.0

    if eligible:
        chosen = min(
            eligible,
            key=lambda row: (
                float(row["mean_interval_score"]),
                float(row["mean_width"]),
                larger_k(row),
            ),
        )
        result = dict(chosen)
        result["selection_mode"] = "coverage_tolerance_feasible"
        return result
    chosen = min(
        rows,
        key=lambda row: (
            abs(float(row["coverage"]) - target_coverage),
            float(row["mean_interval_score"]),
            float(row["mean_width"]),
            larger_k(row),
        ),
    )
    result = dict(chosen)
    result["selection_mode"] = "coverage_tolerance_fallback"
    return result


def deterministic_neighbour_indices(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    query_sample_ids: np.ndarray,
    reference_sample_ids: np.ndarray,
    k: int,
    leave_one_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Order by squared distance, then reference sample_id."""
    query = np.asarray(query_features, dtype=float)
    reference = np.asarray(reference_features, dtype=float)
    # Generator IDs are zero-padded strings.  String ordering is stable and
    # also gives one common tie rule for synthetic and small unit-test IDs.
    query_ids = np.asarray(query_sample_ids).astype(str)
    reference_ids = np.asarray(reference_sample_ids).astype(str)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("Query and reference feature matrices do not align.")
    if len(query_ids) != len(query) or len(reference_ids) != len(reference):
        raise ValueError("Sample IDs do not align with feature rows.")
    available = len(reference) - (1 if leave_one_out else 0)
    if not 1 <= k <= available:
        raise ValueError("k exceeds the available reference neighbours.")
    indices = np.empty((len(query), k), dtype=np.int64)
    squared_distances = np.empty((len(query), k), dtype=float)
    for row_index, (query_row, query_id) in enumerate(zip(query, query_ids, strict=True)):
        distances = np.sum((reference - query_row) ** 2, axis=1, dtype=np.float64)
        if leave_one_out:
            distances = distances.copy()
            distances[reference_ids == query_id] = np.inf
        order = np.lexsort((reference_ids, distances))
        selected = order[:k]
        if not np.isfinite(distances[selected]).all():
            raise AssertionError("Leave-one-out exclusion left too few neighbours.")
        indices[row_index] = selected
        squared_distances[row_index] = distances[selected]
    return indices, squared_distances


def estimate_knn_scale(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    reference_targets: np.ndarray,
    query_sample_ids: np.ndarray,
    reference_sample_ids: np.ndarray,
    k: int,
    scale_floor: float,
    leave_one_out: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("scale_floor must be finite and positive.")
    targets = np.asarray(reference_targets, dtype=float)
    if len(targets) != len(reference_features) or np.any(targets < 0.0) or not np.isfinite(targets).all():
        raise ValueError("Reference scale targets are invalid.")
    neighbour_indices, squared_distances = deterministic_neighbour_indices(
        query_features,
        reference_features,
        query_sample_ids,
        reference_sample_ids,
        k,
        leave_one_out,
    )
    raw_scales = np.mean(targets[neighbour_indices], axis=1, dtype=np.float64)
    scales = np.maximum(raw_scales, scale_floor)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise AssertionError("KNN scales must be finite and strictly positive.")
    return scales, neighbour_indices, squared_distances


def _skip_except_positions(positions: set[int]):
    return lambda line_number: line_number > 0 and (line_number - 1) not in positions


def load_scale_fit_features(
    observed_path: Path,
    prediction_path: Path,
    access_log: list[dict[str, Any]],
) -> pd.DataFrame:
    predictions = pd.read_csv(prediction_path, keep_default_na=False)
    expected_prediction_schema = list(META_COLUMNS) + ["y_pred"]
    if list(predictions.columns) != expected_prediction_schema:
        raise AssertionError(f"Unexpected prediction schema in {prediction_path.name}")
    mask = (predictions["split"] == "local_calibration") & (
        predictions["local_calibration_role"] == "scale_fit"
    )
    positions = set(np.flatnonzero(mask).tolist())
    if len(positions) != 200:
        raise AssertionError("Expected exactly 200 scale_fit feature rows.")
    use_columns = list(META_COLUMNS) + list(FEATURES)
    frame = pd.read_csv(
        observed_path,
        usecols=use_columns,
        skiprows=_skip_except_positions(positions),
        keep_default_na=False,
    )
    frame = frame[use_columns]
    expected_meta = predictions.loc[mask, list(META_COLUMNS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[list(META_COLUMNS)].reset_index(drop=True), expected_meta, check_dtype=False)
    if "y" in frame.columns or ORACLE_COLUMNS.intersection(frame.columns):
        raise AssertionError("A prohibited label or oracle column was loaded.")
    access_log.append(
        {
            "source": observed_path.name,
            "columns_loaded": use_columns,
            "label_columns_loaded": [],
            "rows_loaded": len(frame),
            "split_values": sorted(frame["split"].unique().tolist()),
            "local_role_values": sorted(frame["local_calibration_role"].unique().tolist()),
            "final_calibration_rows_loaded": 0,
            "test_rows_loaded": 0,
        }
    )
    return frame


def load_scale_fit_target(
    residual_path: Path,
    access_log: list[dict[str, Any]],
) -> pd.DataFrame:
    metadata = pd.read_csv(residual_path, usecols=list(META_COLUMNS), keep_default_na=False)
    mask = (metadata["split"] == "local_calibration") & (
        metadata["local_calibration_role"] == "scale_fit"
    )
    positions = set(np.flatnonzero(mask).tolist())
    if len(positions) != 200:
        raise AssertionError("Expected exactly 200 scale_fit target rows.")
    use_columns = list(META_COLUMNS) + ["abs_residual"]
    frame = pd.read_csv(
        residual_path,
        usecols=use_columns,
        skiprows=_skip_except_positions(positions),
        keep_default_na=False,
    )
    frame = frame[use_columns]
    if "y" in frame.columns or ORACLE_COLUMNS.intersection(frame.columns):
        raise AssertionError("A prohibited label or oracle column was loaded.")
    if set(frame["split"]) != {"local_calibration"} or set(frame["local_calibration_role"]) != {"scale_fit"}:
        raise AssertionError("Target access was not restricted to scale_fit.")
    target = frame["abs_residual"].to_numpy(dtype=float)
    if not np.isfinite(target).all() or np.any(target < 0.0):
        raise AssertionError("Scale targets must be finite and nonnegative.")
    access_log.append(
        {
            "source": residual_path.name,
            "columns_loaded": use_columns,
            "label_derived_columns_loaded": ["abs_residual"],
            "rows_loaded": len(frame),
            "split_values": ["local_calibration"],
            "local_role_values": ["scale_fit"],
            "tuning_calibration_rows_loaded": 0,
            "validation_rows_loaded": 0,
            "final_calibration_rows_loaded": 0,
            "test_rows_loaded": 0,
        }
    )
    return frame


def verify_upstream(working_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    generator_check = load_json(working_dir / "generator_check.json")
    predictor_check = load_json(working_dir / "predictor_check_v1.json")
    predictor_config = load_json(working_dir / "predictor" / "predictor_config_v1.json")
    if generator_check["overall_status"] != "PASS" or predictor_check["overall_status"] != "PASS":
        raise AssertionError("Upstream generator and predictor stages must both be PASS.")
    if predictor_check["final_calibration_labels_accessed"] or predictor_check["test_labels_accessed"]:
        raise AssertionError("Upstream prohibited-label access flag is not clean.")
    if predictor_config["authorized_features"] != list(FEATURES):
        raise AssertionError("Predictor feature contract changed.")
    recomputed: dict[str, str] = {}
    expected = predictor_check["output_file_hashes"]
    for scenario in SCENARIOS:
        for suffix in ("predictions.csv", "authorized_residuals.csv"):
            name = f"{scenario}_{suffix}"
            path = working_dir / "predictor_outputs" / name
            recomputed[name] = sha256_file(path)
            if recomputed[name] != expected[name]:
                raise AssertionError(f"Frozen predictor output hash changed: {name}")
    return {
        "status": "PASS",
        "generator_status": "PASS",
        "predictor_status": "PASS",
        "generator_version": generator_check["generator_version"],
        "predictor_stage_version": predictor_check["predictor_stage_version"],
        "predictor_output_hashes_recomputed": recomputed,
        "label_bearing_generator_csvs_read_or_hashed": False,
        "truth_files_read_or_hashed": False,
    }


def _expected_output_paths(output_root: Path) -> list[Path]:
    output_dir = output_root / "scale_outputs"
    paths: list[Path] = []
    for scenario in SCENARIOS:
        paths.extend(
            [
                output_dir / f"{scenario}_scale_fit_preparation.csv",
                output_dir / f"{scenario}_standard_scaler.joblib",
            ]
        )
    paths.append(output_dir / "non_gccm_preparation_manifest_v1.json")
    return paths


def prepare_non_gccm(
    working_dir: Path | str,
    output_root: Path | str,
    config: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(working_dir).resolve()
    destination = Path(output_root).resolve()
    output_dir = destination / "scale_outputs"
    expected_paths = _expected_output_paths(destination)
    existing = [path for path in expected_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite existing scale preparation: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)
    if config["gccm"]["execution_status"] != "BLOCKED" or config["gccm"]["library_sizes"] is not None:
        raise AssertionError("This entry point is restricted to the scientifically blocked preparation state.")
    if config["authorized_features"] != list(FEATURES):
        raise AssertionError("Feature authorization mismatch.")
    if config["search_budget"]["Normalized-EUC"] != config["search_budget"]["Normalized-GCCM"]:
        raise AssertionError("EUC and GCCM KNN search budgets must match.")

    feature_access_log: list[dict[str, Any]] = []
    target_access_log: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    output_hashes: dict[str, str] = {}
    scaler_hashes: dict[str, str] = {}
    standardized_reference: np.ndarray | None = None

    for scenario in SCENARIOS:
        feature_frame = load_scale_fit_features(
            root / "data" / f"{scenario}_observed.csv",
            root / "predictor_outputs" / f"{scenario}_predictions.csv",
            feature_access_log,
        )
        target_frame = load_scale_fit_target(
            root / "predictor_outputs" / f"{scenario}_authorized_residuals.csv",
            target_access_log,
        )
        pd.testing.assert_frame_equal(
            feature_frame[list(META_COLUMNS)].reset_index(drop=True),
            target_frame[list(META_COLUMNS)].reset_index(drop=True),
            check_dtype=False,
        )
        x_raw = feature_frame[list(FEATURES)].to_numpy(dtype=float)
        target = target_frame["abs_residual"].to_numpy(dtype=float)
        sample_ids = feature_frame["sample_id"].astype(str).to_numpy()
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x_raw)
        np.testing.assert_allclose(scaler.mean_, np.mean(x_raw, axis=0), rtol=0.0, atol=1e-14)
        if standardized_reference is None:
            standardized_reference = x_scaled
        else:
            np.testing.assert_allclose(x_scaled, standardized_reference, rtol=0.0, atol=1e-14)

        scale_floor = max(
            float(np.finfo(np.float64).eps),
            float(np.quantile(target, config["local_scale_estimator"]["scale_floor"]["quantile"], method="linear")),
        )
        global_scale = float(np.mean(target, dtype=np.float64))
        if not np.isfinite(scale_floor) or scale_floor <= 0.0 or not np.isfinite(global_scale) or global_scale <= 0.0:
            raise AssertionError("Prepared scale constants must be finite and positive.")

        output = feature_frame[list(META_COLUMNS)].copy()
        output["abs_residual"] = target
        output["raw_cp_scale"] = 1.0
        output["global_scale"] = global_scale
        output["scale_floor"] = scale_floor
        for column_index, feature in enumerate(FEATURES):
            output[f"z_{feature}"] = x_scaled[:, column_index]

        k_diagnostics: dict[str, Any] = {}
        for k in config["local_scale_estimator"]["candidate_k"]:
            scales, neighbour_indices, squared_distances = estimate_knn_scale(
                x_scaled,
                x_scaled,
                target,
                sample_ids,
                sample_ids,
                int(k),
                scale_floor,
                leave_one_out=True,
            )
            output[f"euc_loo_scale_k{k}"] = scales
            activation_count = int(np.count_nonzero(scales == scale_floor))
            k_diagnostics[str(k)] = {
                "actual_neighbour_count_min": int(neighbour_indices.shape[1]),
                "actual_neighbour_count_max": int(neighbour_indices.shape[1]),
                "finite_positive": bool(np.isfinite(scales).all() and np.all(scales > 0.0)),
                "scale_floor_activation_count": activation_count,
                "scale_min": float(np.min(scales)),
                "scale_median": float(np.median(scales)),
                "scale_max": float(np.max(scales)),
                "non_oracle_corr_loo_scale_abs_residual": safe_correlation(scales, target),
                "nearest_squared_distance_min": float(np.min(squared_distances)),
                "nearest_squared_distance_max": float(np.max(squared_distances)),
            }

        csv_path = output_dir / f"{scenario}_scale_fit_preparation.csv"
        scaler_path = output_dir / f"{scenario}_standard_scaler.joblib"
        output.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")
        joblib.dump(scaler, scaler_path, compress=3, protocol=5)
        reloaded = joblib.load(scaler_path)
        np.testing.assert_allclose(reloaded.transform(x_raw), x_scaled, rtol=0.0, atol=0.0)
        output_hashes[csv_path.name] = sha256_file(csv_path)
        scaler_hashes[scaler_path.name] = sha256_file(scaler_path)
        diagnostics[scenario] = {
            "scale_fit_rows": len(output),
            "feature_columns": list(FEATURES),
            "scaler_fit_subset": "scale_fit",
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "scale_floor": scale_floor,
            "global_scale": global_scale,
            "target_min": float(np.min(target)),
            "target_median": float(np.median(target)),
            "target_max": float(np.max(target)),
            "k_diagnostics": k_diagnostics,
            "validation_metrics_computed": False,
        }

    manifest = {
        "manifest_version": "non_gccm_preparation_manifest_v1",
        "scale_stage_version": config["scale_stage_version"],
        "status": "PASS",
        "primary_comparison_status": "BLOCKED",
        "output_hashes": output_hashes,
        "scaler_hashes": scaler_hashes,
        "feature_access_log": feature_access_log,
        "target_access_log": target_access_log,
        "validation_labels_accessed": False,
        "final_calibration_labels_accessed": False,
        "test_labels_accessed": False,
        "oracle_files_read_or_hashed": False,
    }
    manifest_path = output_dir / "non_gccm_preparation_manifest_v1.json"
    write_json(manifest_path, manifest)
    return {
        "status": "PASS",
        "diagnostics": diagnostics,
        "output_hashes": output_hashes,
        "scaler_hashes": scaler_hashes,
        "manifest_hash": sha256_file(manifest_path),
        "feature_access_log": feature_access_log,
        "target_access_log": target_access_log,
        "validation_labels_accessed": False,
        "final_calibration_labels_accessed": False,
        "test_labels_accessed": False,
        "oracle_files_read_or_hashed": False,
    }


def reproducibility_check(working_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gccp_scale_prep_a_") as first, tempfile.TemporaryDirectory(
        prefix="gccp_scale_prep_b_"
    ) as second:
        first_result = prepare_non_gccm(working_dir, Path(first), config)
        second_result = prepare_non_gccm(working_dir, Path(second), config)
        csv_matches = {
            name: digest == second_result["output_hashes"][name]
            for name, digest in first_result["output_hashes"].items()
        }
        scaler_matches = {
            name: digest == second_result["scaler_hashes"][name]
            for name, digest in first_result["scaler_hashes"].items()
        }
        manifest_match = first_result["manifest_hash"] == second_result["manifest_hash"]
        if not all(csv_matches.values()) or not all(scaler_matches.values()) or not manifest_match:
            raise AssertionError("Shared non-GCCM preparation is not bytewise reproducible.")
    return {
        "status": "PASS",
        "separate_temporary_directories": True,
        "csv_hash_match": csv_matches,
        "scaler_hash_match": scaler_matches,
        "manifest_hash_match": manifest_match,
    }


def run_stage(
    working_dir: Path | str = DEFAULT_WORKING_DIR,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_root: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(working_dir).resolve()
    destination = root if output_root is None else Path(output_root).resolve()
    config = load_json(config_path)
    check_path = destination / "scale_comparison_check_v1.json"
    if check_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing check file: {check_path}")
    upstream = verify_upstream(root, config)
    prepared = prepare_non_gccm(root, destination, config, overwrite=overwrite)
    reproducibility = reproducibility_check(root, config)
    check = {
        "overall_status": "BLOCKED",
        "scale_stage_version": config["scale_stage_version"],
        "gccm_applicability_status": "BLOCKED",
        "gccm_blockers": [
            "Rscript is not available in the current PATH; R packages and historical GCCM core cannot be executed.",
            "No n=200 GCCM library-size grid has been preregistered; historical grids cannot be reused unchanged.",
        ],
        "non_gccm_preparation_status": "PASS",
        "primary_validation_comparison_executed": False,
        "validation_results_inspected": False,
        "upstream_verification": upstream,
        "configuration_checks": {
            "feature_authorization": config["authorized_features"] == list(FEATURES),
            "scale_fit_only_preprocessing": config["preprocessing"]["fit_subset"] == "scale_fit",
            "identity_abs_residual_target": config["residual_target"]
            == {"column": "abs_residual", "transform": "identity", "fit_subset": "scale_fit"},
            "euc_gccm_search_budgets_identical": config["search_budget"]["Normalized-EUC"]
            == config["search_budget"]["Normalized-GCCM"],
            "coverage_tolerance_frozen": config["validation_selection"]["absolute_coverage_tolerance"],
            "gccm_substitute_used": False,
        },
        "leakage_checks": {
            "feature_access_log": prepared["feature_access_log"],
            "target_access_log": prepared["target_access_log"],
            "only_scale_fit_labels_accessed": True,
            "tuning_calibration_labels_accessed": False,
            "validation_labels_accessed": prepared["validation_labels_accessed"],
            "final_calibration_labels_accessed": prepared["final_calibration_labels_accessed"],
            "test_labels_accessed": prepared["test_labels_accessed"],
            "oracle_files_read_or_hashed": prepared["oracle_files_read_or_hashed"],
            "generator_label_bearing_csvs_hashed": False,
        },
        "preparation_diagnostics": prepared["diagnostics"],
        "output_file_hashes": prepared["output_hashes"],
        "scaler_file_hashes": prepared["scaler_hashes"],
        "preparation_manifest_hash": prepared["manifest_hash"],
        "reproducibility": reproducibility,
        "implementation_file_hashes_at_execution": {
            "scale_comparison/00_gccm_applicability_audit.md": sha256_file(
                root / "scale_comparison" / "00_gccm_applicability_audit.md"
            ),
            "scale_comparison/fit_local_scales.py": sha256_file(Path(__file__).resolve()),
            "scale_comparison/scale_comparison_config_v1.json": sha256_file(Path(config_path).resolve()),
            "scale_comparison/README.md": sha256_file(root / "scale_comparison" / "README.md"),
            "test_scale_comparison_stage.py": sha256_file(root / "test_scale_comparison_stage.py"),
        },
        "package_versions": package_versions(),
        "warnings": [
            "Raw-CP and Normalized-Global are algebraically equivalent for symmetric split conformal intervals when the global scale is one shared positive constant.",
            "No validation metrics or GCCM weights were produced because the primary branch is scientifically blocked.",
        ],
    }
    write_json(check_path, check)
    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        check = run_stage(args.working_dir, args.config, args.output_root, args.overwrite)
    except Exception:
        if args.output_root is None:
            failure_path = Path(args.working_dir).resolve() / "scale_comparison_check_v1.json"
            if args.overwrite or not failure_path.exists():
                write_json(
                    failure_path,
                    {
                        "overall_status": "FAIL",
                        "scale_stage_version": "gccp_residual_scale_v1_blocked",
                    },
                )
        raise
    print("Scale stage overall status:", check["overall_status"])
    print("Non-GCCM preparation:", check["non_gccm_preparation_status"])
    print("Primary validation comparison executed:", check["primary_validation_comparison_executed"])
    print("Final-calibration labels accessed:", check["leakage_checks"]["final_calibration_labels_accessed"])
    print("Test labels accessed:", check["leakage_checks"]["test_labels_accessed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
