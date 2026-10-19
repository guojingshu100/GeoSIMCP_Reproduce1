"""Fixed four-score Stage 3 runner using audited scale-fit loaders only."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.cmi_adapter_v1 import direct_score, standardize_selected

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MASTER_SEED = 20260910
META = ["sample_id", "scenario", "split", "local_calibration_role"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audited_loaders():
    path = ROOT / "simulation_pilot_20260908" / "scale_comparison" / "fit_local_scales.py"
    spec = importlib.util.spec_from_file_location("stage3_audited_loaders", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import audited loaders from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_scale_fit_features, module.load_scale_fit_target


def official_direct(array: np.ndarray, xyz: np.ndarray, seed: int) -> float:
    from tigramite.independence_tests.cmiknn import CMIknn

    estimator = CMIknn(knn=0.10, transform=False, workers=1, seed=seed)
    return float(estimator.get_dependence_measure(array.copy(), xyz.copy()))


def main() -> None:
    started = time.perf_counter()
    load_features, load_target = audited_loaders()
    feature_access, target_access = [], []
    rows, preprocessing, input_hashes = [], {}, {}

    # Integration only: adapter vs a fresh direct official call, then repeat adapter.
    verify_seed = MASTER_SEED
    rng = np.random.default_rng(verify_seed)
    synthetic = np.vstack([rng.normal(size=32), rng.normal(size=32)])
    synthetic[1] += synthetic[0]
    xyz_synthetic = np.array([0, 1], dtype=int)
    adapter_value, constructor_kwargs = direct_score(synthetic, xyz_synthetic, verify_seed)
    direct_value = official_direct(synthetic, xyz_synthetic, verify_seed)
    repeat_value, _ = direct_score(synthetic, xyz_synthetic, verify_seed)
    if adapter_value != direct_value or adapter_value != repeat_value:
        raise AssertionError("official direct-call agreement or identical-seed reproducibility failed")

    for scenario_index, scenario in enumerate(("S2", "S4")):
        pilot = ROOT / "simulation_pilot_20260908"
        observed = pilot / "data" / f"{scenario}_observed.csv"
        predictions = pilot / "predictor_outputs" / f"{scenario}_predictions.csv"
        residuals = pilot / "predictor_outputs" / f"{scenario}_authorized_residuals.csv"
        for path in (observed, predictions, residuals):
            input_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
        features = load_features(observed, predictions, feature_access)
        target = load_target(residuals, target_access)
        if len(features) != 200 or len(target) != 200 or features.sample_id.duplicated().any() or target.sample_id.duplicated().any():
            raise AssertionError(f"{scenario}: expected 200 unique scale-fit rows")
        if not features[META].equals(target[META]):
            raise AssertionError(f"{scenario}: feature/target metadata mismatch")
        if set(features["split"]) != {"local_calibration"} or set(features["local_calibration_role"]) != {"scale_fit"}:
            raise AssertionError(f"{scenario}: non-scale-fit row loaded")
        selected = {name: features[name].to_numpy(dtype=float) for name in ("x2", "s_x", "s_y")}
        selected["abs_residual"] = target["abs_residual"].to_numpy(dtype=float)
        standardized, params = standardize_selected(selected)
        preprocessing[scenario] = {"columns": list(selected), "mean": params.mean, "scale": params.scale}
        cases = (("M0", ["x2", "abs_residual"], [0, 1]), ("Mspace", ["x2", "abs_residual", "s_x", "s_y"], [0, 1, 2, 2]))
        for case_index, (condition, names, labels) in enumerate(cases):
            seed = MASTER_SEED + 1 + scenario_index * 2 + case_index
            array = np.vstack([standardized[name] for name in names])
            xyz = np.asarray(labels, dtype=int)
            score, kwargs = direct_score(array, xyz, seed)
            rows.append({"scenario": scenario, "target_name": "A_model_abs_residual", "conditioning_set": condition, "n": array.shape[1], "knn_fraction": 0.10, "actual_k": max(1, int(0.10 * array.shape[1])), "seed": seed, "raw_score": score, "status": "PASS"})

    csv_columns = ["scenario", "target_name", "conditioning_set", "n", "knn_fraction", "actual_k", "seed", "raw_score", "status"]
    import csv
    with (OUT / "stage3_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "stage": "Stage 3", "execution_status": "PASS", "master_seed": MASTER_SEED,
        "interpreter": sys.executable, "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ("tigramite", "numpy", "scipy", "pandas", "scikit-learn")},
        "data_route": "audited load_scale_fit_features and load_scale_fit_target; no experiment pipeline invoked", "input_paths_and_sha256": input_hashes,
        "configuration": {"estimator": "CMIknn", "knn_fraction": 0.10, "transform": False, "workers": 1, "constructor_kwargs": constructor_kwargs, "array_convention": "(variables, samples)", "xyz": {"M0": [0, 1], "Mspace": [0, 1, 2, 2]}},
        "preprocessing": preprocessing, "access_logs": {"feature": feature_access, "target": target_access},
        "verification_results": {"synthetic_shape": list(synthetic.shape), "synthetic_xyz": xyz_synthetic.tolist(), "adapter_direct_official_agreement": True, "identical_seed_reproducibility": True, "project_metadata_alignment": True},
        "scores": rows, "elapsed_time_seconds": time.perf_counter() - started,
    }
    (OUT / "stage3_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
