"""Stage 5: official Tigramite shuffle significance for the fixed eight CMI scores."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from tigramite.independence_tests.cmiknn import CMIknn

from src.cmi_adapter_v1 import direct_score, standardize_selected
from run_stage3_cmi_v1 import audited_loaders
from run_stage4_oracle_cmi_v1 import load_oracle_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MASTER_SEED = 20260910
SIGNIFICANCE_SEED = 3
META = ["sample_id", "scenario", "split", "local_calibration_role"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def estimator() -> CMIknn:
    return CMIknn(knn=0.10, shuffle_neighbors=5, significance="shuffle_test", transform=False, workers=1, sig_samples=199, sig_blocklength=1, seed=SIGNIFICANCE_SEED)


def score_and_null(array: np.ndarray, xyz: np.ndarray) -> tuple[float, float, np.ndarray]:
    test = estimator()
    raw = float(test.get_dependence_measure(array.copy(), xyz.copy()))
    p_value, null_dist = test.get_shuffle_significance(array.copy(), xyz.copy(), raw, return_null_dist=True)
    return raw, float(p_value), np.asarray(null_dist, dtype=float)


def frozen_scores(path: Path) -> dict[tuple[str, str], float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {(row["scenario"], row["conditioning_set"]): float(row["raw_score"]) for row in csv.DictReader(handle)}


def model_frame(scenario: str, feature_access: list[dict], target_access: list[dict]) -> pd.DataFrame:
    pilot = ROOT / "simulation_pilot_20260908"
    load_features, load_target = audited_loaders()
    features = load_features(pilot / "data" / f"{scenario}_observed.csv", pilot / "predictor_outputs" / f"{scenario}_predictions.csv", feature_access)
    target = load_target(pilot / "predictor_outputs" / f"{scenario}_authorized_residuals.csv", target_access)
    if len(features) != 200 or len(target) != 200 or features.sample_id.duplicated().any() or target.sample_id.duplicated().any() or not features[META].equals(target[META]):
        raise AssertionError(f"{scenario} A_model: scale-fit metadata alignment failed")
    frame = features[META + ["x2", "s_x", "s_y"]].copy()
    frame["target"] = target["abs_residual"].to_numpy(float)
    return frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def prepared_cases(target_name: str, scenario: str, frame: pd.DataFrame) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    if len(frame) != 200 or frame.sample_id.duplicated().any() or set(frame["split"]) != {"local_calibration"} or set(frame["local_calibration_role"]) != {"scale_fit"}:
        raise AssertionError(f"{target_name}/{scenario}: expected 200 unique scale-fit rows")
    values, params = standardize_selected({name: frame[name].to_numpy(float) for name in ("x2", "s_x", "s_y", "target")})
    return ({"M0": (np.vstack([values["x2"], values["target"]]), np.array([0, 1], dtype=int)), "Mspace": (np.vstack([values["x2"], values["target"], values["s_x"], values["s_y"]]), np.array([0, 1, 2, 2], dtype=int))}, {"columns": ["x2", "s_x", "s_y", target_name], "mean": params.mean, "scale": params.scale})


def main() -> None:
    started = time.perf_counter()
    stage3_path, stage4_path = OUT / "stage3_scores.csv", OUT / "stage4_oracle_scores.csv"
    frozen = {"A_model": frozen_scores(stage3_path), "A_oracle": frozen_scores(stage4_path)}
    feature_access: list[dict] = []
    target_access: list[dict] = []
    inputs = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in (stage3_path, stage4_path)}
    datasets, preprocessing, identity = {}, {}, {}
    for scenario in ("S2", "S4"):
        datasets[("A_model", scenario)] = model_frame(scenario, feature_access, target_access)
        oracle, oracle_hashes = load_oracle_rows(scenario, feature_access)
        datasets[("A_oracle", scenario)] = oracle[META + ["x2", "s_x", "s_y"]].assign(target=oracle["A_oracle"].to_numpy(float))
        identity[scenario] = {"n": len(oracle), "max_abs_discrepancy": float(np.max(np.abs(oracle["A_oracle"].to_numpy(float) - oracle["sigma_true"].to_numpy(float) * np.abs(oracle["epsilon"].to_numpy(float)))))}
        for key, value in oracle_hashes.items():
            if key != "selected_id_source":
                inputs[key] = value
    rows, arrays = [], {}
    for target_name in ("A_model", "A_oracle"):
        for scenario in ("S2", "S4"):
            cases, params = prepared_cases(target_name, scenario, datasets[(target_name, scenario)])
            preprocessing[f"{target_name}/{scenario}"] = params
            for condition, (array, xyz) in cases.items():
                expected = frozen[target_name][(scenario, condition)]
                recomputed, _ = direct_score(array, xyz, MASTER_SEED)
                if abs(recomputed - expected) > 1e-12:
                    raise AssertionError(f"{target_name}/{scenario}/{condition}: frozen-score mismatch {recomputed} vs {expected}")
                raw, p_value, null_dist = score_and_null(array, xyz)
                if abs(raw - expected) > 1e-12:
                    raise AssertionError(f"{target_name}/{scenario}/{condition}: significance raw-score mismatch")
                arrays[(target_name, scenario, condition)] = (array, xyz)
                rows.append({"target": target_name, "scenario": scenario, "conditioning": condition, "n": 200, "k": 20, "knn_fraction": 0.10, "shuffle_neighbors": 5, "sig_samples": 199, "seed": SIGNIFICANCE_SEED, "raw_score": raw, "null_mean": float(null_dist.mean()), "null_q95": float(np.quantile(null_dist, 0.95, method="linear")), "p_value": p_value, "bh_q_value": None, "significant_raw_0_05": p_value <= 0.05, "significant_bh_0_05": None, "monte_carlo_flag": "BORDERLINE_MONTE_CARLO" if 0.025 <= p_value <= 0.10 else ""})
    p_values = np.asarray([row["p_value"] for row in rows], dtype=float)
    q_values = false_discovery_control(p_values, method="bh")
    for row, q_value in zip(rows, q_values):
        row["bh_q_value"] = float(q_value)
        row["significant_bh_0_05"] = bool(q_value <= 0.05)
    repeated = {}
    for condition in ("M0", "Mspace"):
        array, xyz = arrays[("A_oracle", "S2", condition)]
        one = score_and_null(array, xyz)
        two = score_and_null(array, xyz)
        if one[0] != two[0] or one[1] != two[1] or not np.array_equal(one[2], two[2]):
            raise AssertionError(f"A_oracle/S2/{condition}: fresh-object significance reproducibility failed")
        repeated[condition] = {"raw_score": one[0], "p_value": one[1], "null_distribution_exact_match": True}
    fieldnames = ["target", "scenario", "conditioning", "n", "k", "knn_fraction", "shuffle_neighbors", "sig_samples", "seed", "raw_score", "null_mean", "null_q95", "p_value", "bh_q_value", "significant_raw_0_05", "significant_bh_0_05", "monte_carlo_flag"]
    with (OUT / "stage5_cmi_significance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    metadata = {"stage": "Stage 5", "execution_status": "PASS", "interpreter": sys.executable, "packages": {name: importlib.metadata.version(name) for name in ("tigramite", "numpy", "scipy", "pandas", "scikit-learn")}, "input_paths_and_sha256": inputs, "selected_id_source": "audited Stage 3 scale-fit loaders; Stage 4 selected oracle rows use the same feature-loader IDs", "configuration": {"knn_fraction": 0.10, "actual_k": 20, "transform": False, "workers": 1, "significance": "shuffle_test", "sig_samples": 199, "shuffle_neighbors": 5, "sig_blocklength": 1, "seed": SIGNIFICANCE_SEED, "p_value": "one-sided empirical (count(null >= raw)+1)/(199+1)"}, "verified_tigramite_behavior": {"nonempty_Z": "CMIknn restricted nearest-neighbor shuffle of Y", "empty_Z": "parent block shuffle of X", "blocklength_1": "single-point blocks randomly permuted"}, "oracle_identity": identity, "preprocessing": preprocessing, "access_logs": {"feature": feature_access, "A_model_target": target_access}, "integration": {"all_eight_n_200_and_metadata_aligned": True, "all_recomputed_scores_match_frozen_within": 1e-12, "representative_fresh_object_reproducibility": repeated}, "bh_adjustment": "scipy.stats.false_discovery_control(method='bh')", "results": rows, "elapsed_time_seconds": time.perf_counter() - started}
    (OUT / "stage5_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
