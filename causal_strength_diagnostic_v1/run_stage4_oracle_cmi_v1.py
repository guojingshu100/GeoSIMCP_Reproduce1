"""Stage 4: four fixed CMIknn scores using only authorized oracle residual rows."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.cmi_adapter_v1 import direct_score, standardize_selected

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MASTER_SEED = 20260910
META = ["sample_id", "scenario", "split", "local_calibration_role"]
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_audited_feature_loader():
    path = ROOT / "simulation_pilot_20260908" / "scale_comparison" / "fit_local_scales.py"
    spec = importlib.util.spec_from_file_location("stage4_audited_loaders", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import audited loader: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_scale_fit_features


def read_selected(path: Path, columns: list[str], selected_ids: set[str]) -> pd.DataFrame:
    """Read non-target metadata first, then read target columns for selected file rows only."""
    metadata = pd.read_csv(path, usecols=META, keep_default_na=False)
    positions = set(np.flatnonzero(metadata["sample_id"].isin(selected_ids)).tolist())
    if len(positions) != 200 or metadata.iloc[list(positions)]["sample_id"].duplicated().any():
        raise AssertionError(f"{path.name}: selected IDs are not exactly 200 unique rows")
    def skip(row_number: int) -> bool:
        return row_number != 0 and (row_number - 1) not in positions
    frame = pd.read_csv(path, usecols=columns, skiprows=skip, keep_default_na=False)
    if len(frame) != 200 or set(frame["sample_id"]) != selected_ids:
        raise AssertionError(f"{path.name}: selected-row read mismatch")
    return frame


def load_oracle_rows(scenario: str, feature_access: list[dict]) -> tuple[pd.DataFrame, dict[str, str]]:
    pilot = ROOT / "simulation_pilot_20260908"
    observed = pilot / "data" / f"{scenario}_observed.csv"
    truth = pilot / "data" / f"{scenario}_truth.csv"
    predictions = pilot / "predictor_outputs" / f"{scenario}_predictions.csv"
    load_features = import_audited_feature_loader()
    features = load_features(observed, predictions, feature_access)
    if len(features) != 200 or features.sample_id.duplicated().any():
        raise AssertionError(f"{scenario}: audited scale-fit feature IDs invalid")
    selected_ids = set(features["sample_id"].astype(str))
    observed_rows = read_selected(observed, META + ["y"], selected_ids)
    truth_rows = read_selected(truth, META + ["mu_true", "sigma_true", "epsilon"], selected_ids)
    merged = features[META + ["x2", "s_x", "s_y"]].merge(observed_rows, on=META, how="inner", validate="one_to_one")
    merged = merged.merge(truth_rows, on=META, how="inner", validate="one_to_one")
    if len(merged) != 200 or merged.sample_id.duplicated().any() or set(merged.sample_id) != selected_ids:
        raise AssertionError(f"{scenario}: incomplete metadata join")
    if set(merged["scenario"]) != {scenario} or set(merged["split"]) != {"local_calibration"} or set(merged["local_calibration_role"]) != {"scale_fit"}:
        raise AssertionError(f"{scenario}: metadata labels not restricted to scale_fit")
    numeric = merged[["x2", "s_x", "s_y", "y", "mu_true", "sigma_true", "epsilon"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise AssertionError(f"{scenario}: nonfinite selected value")
    merged = merged.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    merged["A_oracle"] = np.abs(merged["y"].to_numpy(float) - merged["mu_true"].to_numpy(float))
    expected = merged["sigma_true"].to_numpy(float) * np.abs(merged["epsilon"].to_numpy(float))
    discrepancy = float(np.max(np.abs(merged["A_oracle"].to_numpy(float) - expected)))
    if not np.allclose(merged["A_oracle"].to_numpy(float), expected, rtol=TOLERANCE, atol=TOLERANCE):
        raise AssertionError(f"{scenario}: oracle identity failed; max discrepancy={discrepancy}")
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in (observed, truth, predictions)}
    return merged, hashes | {"selected_id_source": "load_scale_fit_features(observed,predictions): local_calibration + scale_fit"}


def integration_check() -> dict:
    seed = MASTER_SEED
    rng = np.random.default_rng(seed)
    array = np.vstack([rng.normal(size=32), rng.normal(size=32)])
    array[1] += array[0]
    xyz = np.array([0, 1], dtype=int)
    first, kwargs = direct_score(array, xyz, seed)
    repeat, _ = direct_score(array, xyz, seed)
    if first != repeat:
        raise AssertionError("identical-seed CMI adapter reproducibility failed")
    return {"identical_seed_reproducibility": True, "synthetic_shape": list(array.shape), "synthetic_xyz": xyz.tolist(), "constructor_kwargs": kwargs}


def main() -> None:
    started = time.perf_counter()
    verification = integration_check()
    feature_access: list[dict] = []
    rows, preprocessing, identities, hashes = [], {}, {}, {}
    for scenario_index, scenario in enumerate(("S2", "S4")):
        frame, scenario_hashes = load_oracle_rows(scenario, feature_access)
        hashes.update(scenario_hashes)
        values, params = standardize_selected({name: frame[name].to_numpy(float) for name in ("x2", "s_x", "s_y", "A_oracle")})
        preprocessing[scenario] = {"columns": ["x2", "s_x", "s_y", "A_oracle"], "mean": params.mean, "scale": params.scale}
        identities[scenario] = {"n": len(frame), "tolerance_atol_rtol": TOLERANCE, "status": "PASS", "max_abs_discrepancy": float(np.max(np.abs(frame["A_oracle"].to_numpy(float) - frame["sigma_true"].to_numpy(float) * np.abs(frame["epsilon"].to_numpy(float)))))}
        for case_index, (condition, names, xyz) in enumerate((("M0", ["x2", "A_oracle"], [0, 1]), ("Mspace", ["x2", "A_oracle", "s_x", "s_y"], [0, 1, 2, 2]))):
            seed = MASTER_SEED + 101 + scenario_index * 2 + case_index
            array = np.vstack([values[name] for name in names])
            score, _ = direct_score(array, np.asarray(xyz, dtype=int), seed)
            rows.append({"scenario": scenario, "target_name": "A_oracle_abs_y_minus_mu_true", "conditioning_set": condition, "n": int(array.shape[1]), "knn_fraction": 0.10, "actual_k": max(1, int(0.10 * array.shape[1])), "seed": seed, "raw_score": score, "status": "PASS"})
    with (OUT / "stage4_oracle_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "target_name", "conditioning_set", "n", "knn_fraction", "actual_k", "seed", "raw_score", "status"])
        writer.writeheader(); writer.writerows(rows)
    metadata = {"stage": "Stage 4", "execution_status": "PASS", "master_seed": MASTER_SEED, "interpreter": sys.executable, "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ("tigramite", "numpy", "scipy", "pandas", "scikit-learn")}, "input_paths_and_sha256": hashes, "selected_id_source": "audited load_scale_fit_features; its filter is split=local_calibration and local_calibration_role=scale_fit", "target_definition": "A_oracle = abs(y - mu_true)", "identity_validation": identities, "configuration": {"estimator": "tigramite.independence_tests.cmiknn.CMIknn", "knn_fraction": 0.10, "actual_k": 20, "transform": False, "workers": 1, "direct_method": "get_dependence_measure(array, xyz)", "xyz": {"M0": [0, 1], "Mspace": [0, 1, 2, 2]}}, "preprocessing": preprocessing, "feature_loader_access_log": feature_access, "reproducibility_checks": verification, "scores": rows, "elapsed_time_seconds": time.perf_counter() - started}
    (OUT / "stage4_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"identity": identities, "scores": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
