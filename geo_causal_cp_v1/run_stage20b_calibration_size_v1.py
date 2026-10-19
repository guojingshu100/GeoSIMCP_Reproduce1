"""Run sealed Stage-20B CP-calibration-size feasibility experiment.

Only the nested official-engine calibration sample changes. Scientific work is
scenario-checkpointed so a terminated run can resume without repeating CMI or
completed official-engine partitions.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RECORD = ROOT / "experiments/geo_causal_cp/20260920_stage20b_calibration_size_feasibility_v1"
PREREG = RECORD / "preregistration"
OFFICIAL = ROOT / "external/geoconformal_stage19a_audit_26419ef"
COMMIT = "26419ef06d7e3711e6618f97f4f1ed02af16bcb7"
PARITY_SOURCE = HERE / "run_stage19b_direct_parity_v1.py"
CMI_WORKER = HERE / "stage19b_cmi_full_worker_v1.py"
CMI_PYTHON = ROOT / "causal_strength_diagnostic_v1/.venv_cmi_v1/Scripts/python.exe"
GENERATOR_SOURCE = ROOT / "simulation_pilot_20260908/generator/generate_scenarios.py"
GENERATOR_CONFIG = ROOT / "simulation_pilot_20260908/generator/config_v1.json"
PREDICTOR_SOURCE = ROOT / "simulation_pilot_20260908/predictor/fit_predictors.py"
PREDICTOR_CONFIG = ROOT / "simulation_pilot_20260908/predictor/predictor_config_v1.json"

FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
FEATURES_ALL = ["s_x", "s_y", *FEATURES]
COORDS = ["s_x", "s_y"]
META = ["sample_id", "scenario", "split", "local_calibration_role"]
SCENARIOS = ["S2", "S4"]
CAL_SIZES = [200, 400, 800]
TRACKS = {"A": False, "B": True}
OFFICIAL_METHODS = ["GeoCP", "GeoSIMCP-EUC", "GeoSIMCP-MND", "GeoCausalCP-Full"]
JOINT_METHODS = ["GeoSIMCP-EUC", "GeoSIMCP-MND", "GeoCausalCP-Full"]
ALPHA = 0.10
Q_B = 1 - ALPHA
Q_TOL = Q_B - 1e-9
MASS_THRESHOLD_EXACT = Q_TOL / (1 - Q_TOL)
GEO_H = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
JOINT_H = list(np.linspace(0.1, 5.0, 20))
LAMBDAS = list(np.arange(0, 1.05, 0.05))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = load_module("stage19b_parity_reuse_20b", PARITY_SOURCE)
weighted_quantile = parity.weighted_quantile
effective_sample_size = parity.effective_sample_size
pairwise_sq = parity.pairwise_sq
fit_feature_transform = parity.fit_feature_transform
feature_distance_sq = parity.feature_distance_sq


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value): return "NaN"
        if np.isposinf(value): return "Infinity"
        if np.isneginf(value): return "-Infinity"
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
    return hashlib.sha256(payload).hexdigest()


def verify_prereg() -> tuple[dict, list[int]]:
    seal = json.loads((PREREG / "PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    for name, expected in seal["files"].items():
        if sha(PREREG / name) != expected:
            raise AssertionError(f"preregistration changed: {name}")
    payload = "\n".join(f"{name} {seal['files'][name]}" for name in sorted(seal["files"]))
    if hashlib.sha256(payload.encode()).hexdigest() != seal["composite_sha256"]:
        raise AssertionError("preregistration seal mismatch")
    seeds = json.loads((PREREG / "seed_manifest.json").read_text(encoding="utf-8"))["seeds"]
    if seeds != list(range(20950001, 20950051)):
        raise AssertionError("seed manifest mismatch")
    safe = str(OFFICIAL.resolve()).replace("\\", "/")
    head = subprocess.run(
        ["git", "-c", f"safe.directory={safe}", "rev-parse", "HEAD"],
        cwd=OFFICIAL, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if head != COMMIT:
        raise AssertionError(f"official commit mismatch: {head}")
    return seal, seeds


def scenario_dir(index: int, seed: int, scenario: str) -> Path:
    return RECORD / "frozen_inputs" / f"replicate_{index:02d}_seed_{seed}" / scenario


def role(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.loc[frame["split_role"].eq(name)].copy()
    expected = {"train": 1000, "CMI_scale_fit": 200, "CP_calibration_pool": 800, "validation": 200}[name]
    if len(result) != expected:
        raise AssertionError(f"role {name}: {len(result)} != {expected}")
    return result


def stage20_config(base: dict, seed: int) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg["n"] = 2500
    cfg["random_seeds"]["base_variables"] = seed
    cfg["split_sizes"] = {
        "train": 1000, "local_calibration": 200, "validation": 200,
        "final_calibration": 800, "test": 300,
    }
    cfg["local_calibration_internal_roles"] = {"scale_fit": 200, "tuning_calibration": 0}
    return cfg


def assign_stage20_role(frame: pd.DataFrame) -> pd.Series:
    values = np.where(
        frame["split"].eq("local_calibration"), "CMI_scale_fit",
        np.where(frame["split"].eq("final_calibration"), "CP_calibration_pool", frame["split"]),
    )
    return pd.Series(values, index=frame.index)


def generate_scenario_artifacts(index: int, seed: int, scenario: str, observed: pd.DataFrame,
                                truth: pd.DataFrame, predictor, predictor_cfg: dict) -> tuple[dict, dict, dict, list[dict]]:
    out = scenario_dir(index, seed, scenario)
    out.mkdir(parents=True, exist_ok=True)
    complete = out / "GENERATION_COMPLETE.json"
    if complete.exists():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        for name, expected in saved["hashes"].items():
            if sha(out / name) != expected:
                raise AssertionError(f"generation checkpoint mismatch: {out/name}")
        return saved["generator_audit"], saved["split_integrity"], saved["prediction_manifest"], saved["nested_rows"]

    observed = observed.copy()
    truth = truth.copy()
    observed["split_role"] = assign_stage20_role(observed)
    truth["split_role"] = assign_stage20_role(truth)
    keys = [*META, "split_role"]
    pd.testing.assert_frame_equal(observed[keys], truth[keys], check_dtype=False)
    inputs = observed[[*META, "split_role", *FEATURES_ALL]].copy()
    labels = observed[[*META, "split_role", "y"]].merge(
        truth[[*META, "split_role", "sigma_true"]], on=[*META, "split_role"], validate="one_to_one",
    ).rename(columns={"y": "y_true", "sigma_true": "oracle_sigma"})
    development_labels = labels.loc[~labels["split"].eq("test")].copy()
    test_labels = labels.loc[labels["split"].eq("test")].copy()
    input_path = out / "predictor_inputs.csv"
    dev_labels_path = out / "development_labels.csv"
    test_path = out / "test_sealed.csv"
    write_csv(inputs, input_path)
    write_csv(development_labels, dev_labels_path)
    write_csv(test_labels, test_path)

    # All predictor operations begin from the %.17g serialized/reloaded values.
    x = pd.read_csv(input_path, keep_default_na=False)
    labels_dev = pd.read_csv(dev_labels_path, keep_default_na=False)
    train = x.loc[x["split_role"].eq("train")].merge(
        labels_dev.loc[labels_dev["split_role"].eq("train"), [*META, "split_role", "y_true"]],
        on=[*META, "split_role"], validate="one_to_one", sort=False,
    )
    if len(train) != 1000:
        raise AssertionError("train-only predictor gate")
    model = predictor.make_model(predictor_cfg)
    model.fit(train[FEATURES_ALL].to_numpy(float), train["y_true"].to_numpy(float))
    predictions = x[[*META, "split_role"]].copy()
    predictions["y_pred"] = predictor.predict_in_fixed_tree_order(model, x[FEATURES_ALL].to_numpy(float))
    pred_path = out / "predictions.csv"
    model_path = out / "model.joblib"
    write_csv(predictions, pred_path)
    joblib.dump(model, model_path, compress=3, protocol=5)
    predictions = pd.read_csv(pred_path, keep_default_na=False)
    development = x.merge(predictions, on=[*META, "split_role"], validate="one_to_one", sort=False).merge(
        labels_dev, on=[*META, "split_role"], validate="one_to_one", sort=False,
    )
    if len(development) != 2200 or development["split_role"].eq("test").any():
        raise AssertionError("development-label seal")
    write_csv(development, out / "development.csv")

    counts = x["split_role"].value_counts().to_dict()
    expected = {"train": 1000, "CMI_scale_fit": 200, "CP_calibration_pool": 800, "validation": 200, "test": 300}
    if counts != expected or not x["sample_id"].is_unique:
        raise AssertionError(f"split integrity: {counts}")
    pool = development.loc[development["split_role"].eq("CP_calibration_pool")].sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if len(pool) != 800:
        raise AssertionError("calibration pool")
    pool_order = pool[["sample_id"]].copy()
    pool_order["calibration_order"] = np.arange(1, 801)
    write_csv(pool_order, out / "calibration_pool_order.csv")
    nested_rows = []
    prior_ids: set[str] = set()
    for n_cal in CAL_SIZES:
        subset = pool.iloc[:n_cal].copy()
        ids = set(subset["sample_id"].astype(str))
        nested_rows.append({
            "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
            "row_count": len(subset), "subset_frame_sha256": frame_hash(subset),
            "sample_id_sha256": hashlib.sha256("\n".join(subset["sample_id"].astype(str)).encode()).hexdigest(),
            "contains_previous": prior_ids.issubset(ids), "strictly_larger_than_previous": not prior_ids or len(ids) > len(prior_ids),
        })
        if prior_ids and not prior_ids < ids:
            raise AssertionError("nested calibration relation")
        prior_ids = ids

    generator_audit = {
        "replicate_index": index, "generation_seed": seed, "scenario": scenario, "rows": len(x),
        "serialized_reload": True, "finite_predictors": bool(np.isfinite(x[FEATURES_ALL].to_numpy(float)).all()),
        "finite_y_development": bool(np.isfinite(development["y_true"].to_numpy(float)).all()),
        "sigma_min": float(labels["oracle_sigma"].min()), "sigma_max": float(labels["oracle_sigma"].max()),
        "predictor_inputs_sha256": sha(input_path), "development_labels_sha256": sha(dev_labels_path),
        "test_sealed_sha256": sha(test_path), "status": "PASS",
    }
    split_audit = {
        "replicate_index": index, "generation_seed": seed, "scenario": scenario,
        **{f"n_{key}": value for key, value in expected.items()}, "unique_ids": int(x["sample_id"].nunique()),
        "roles_disjoint": True, "nested_calibration": True, "status": "PASS",
    }
    prediction_manifest = {
        "replicate_index": index, "generation_seed": seed, "scenario": scenario,
        "model_sha256": sha(model_path), "predictions_sha256": sha(pred_path), "prediction_rows": len(predictions),
        "predictor_input_sha256": sha(input_path), "training_rows": len(train), "predictor_random_seed": 420907,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "prediction_path": str(pred_path.relative_to(ROOT)).replace("\\", "/"),
    }
    files = [input_path, dev_labels_path, test_path, pred_path, model_path, out / "development.csv", out / "calibration_pool_order.csv"]
    dump(complete, {
        "hashes": {p.name: sha(p) for p in files}, "generator_audit": generator_audit,
        "split_integrity": split_audit, "prediction_manifest": prediction_manifest, "nested_rows": nested_rows,
    })
    return generator_audit, split_audit, prediction_manifest, nested_rows


def generate_replicate(index: int, seed: int, generator, gen_cfg: dict, predictor, pred_cfg: dict):
    scenarios = generator.generate_all_scenarios(stage20_config(gen_cfg, seed))
    common = ["sample_id", "split", "local_calibration_role", *FEATURES_ALL]
    pd.testing.assert_frame_equal(scenarios["S2"][0][common], scenarios["S4"][0][common], check_dtype=False)
    return [generate_scenario_artifacts(index, seed, scenario, *scenarios[scenario], predictor, pred_cfg) for scenario in SCENARIOS]


def load_development(index: int, seed: int, scenario: str) -> pd.DataFrame:
    frame = pd.read_csv(scenario_dir(index, seed, scenario) / "development.csv", keep_default_na=False)
    if len(frame) != 2200 or frame["split_role"].eq("test").any():
        raise AssertionError("development table")
    return frame


def cmi_input(index: int, seed: int, scenario: str) -> Path:
    base = scenario_dir(index, seed, scenario)
    path = base / "cmi_full999_input.csv"
    if not path.exists():
        scale = role(load_development(index, seed, scenario), "CMI_scale_fit")
        frame = scale[["sample_id", *COORDS, *FEATURES]].copy()
        frame["A_model"] = np.abs(scale["y_true"].to_numpy(float) - scale["y_pred"].to_numpy(float))
        write_csv(frame, path)
    return path


def run_cmi(index: int, seed: int, scenario: str, output: Path | None = None) -> dict:
    inp = cmi_input(index, seed, scenario)
    out = output or scenario_dir(index, seed, scenario) / "cmi_full999_results.csv"
    if out.exists() and out.with_suffix(".json").exists():
        return {"runtime_seconds": json.loads(out.with_suffix(".json").read_text())["runtime_seconds"], "checkpoint": True}
    command = [str(CMI_PYTHON), str(CMI_WORKER), "--input", str(inp), "--output", str(out),
               "--replicate-index", str(index), "--scenario", scenario]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    out.with_suffix(".stdout.txt").write_text(proc.stdout, encoding="utf-8")
    out.with_suffix(".stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"CMI failed {index}/{scenario}: {proc.stderr[-3000:]}")
    return {"runtime_seconds": json.loads(out.with_suffix(".json").read_text())["runtime_seconds"], "checkpoint": False}


def cmi_info(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    if len(frame) != 4 or frame["feature"].tolist() != FEATURES:
        raise AssertionError("CMI schema")
    weights = dict(zip(frame["feature"], frame["weight"].astype(float)))
    selected = [feature for feature in FEATURES if weights[feature] > 0]
    status = str(frame["status"].iloc[0])
    if status == "SIGNAL" and not np.isclose(sum(weights.values()), 1, atol=1e-12):
        raise AssertionError("CMI weights")
    if status == "NO_CAUSAL_SIGNAL" and selected:
        raise AssertionError("CMI no-signal")
    return frame, {"weights": weights, "selected": selected, "status": status}


def interval_metrics(y: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray) -> dict:
    lower = pred - uncertainty
    upper = pred + uncertainty
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise AssertionError("invalid interval")
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width.copy()
    low, high = y < lower, y > upper
    score[low] += (2 / ALPHA) * (lower[low] - y[low])
    score[high] += (2 / ALPHA) * (y[high] - upper[high])
    finite = np.isfinite(width) & np.isfinite(score)
    return {
        "coverage": float(covered.mean()), "covered_count": int(covered.sum()), "n": len(y),
        "mean_width": float(np.mean(width)), "median_width": float(np.median(width)),
        "mean_interval_score": float(np.mean(score)),
        "infinite_count": int((~np.isfinite(width)).sum()), "infinite_fraction": float((~np.isfinite(width)).mean()),
        "mean_width_finite": float(np.mean(width[finite])) if finite.any() else np.nan,
        "median_width_finite": float(np.median(width[finite])) if finite.any() else np.nan,
        "mean_interval_score_finite": float(np.mean(score[finite])) if finite.any() else np.nan,
        "lower": lower, "upper": upper, "score": score,
    }


def kernel_weights(dgeo2: np.ndarray, dfeat2: np.ndarray | None, h: float, lam: float | None) -> np.ndarray:
    if dfeat2 is None:
        return np.exp(-0.5 * dgeo2 / (h * h))
    joint2 = float(lam) * dgeo2 + (1 - float(lam)) * dfeat2
    return np.exp(-0.5 * joint2 / ((h + 1e-8) ** 2))


def weighted_uncertainty(scores: np.ndarray, weights: np.ndarray, track: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if track == "B":
        uncertainty = weighted_quantile(scores, weights, Q_B, self_weights=np.ones(weights.shape[0]))
    else:
        q_a = min(1.0, math.ceil((len(scores) + 1) * (1 - ALPHA)) / len(scores))
        uncertainty = weighted_quantile(scores, weights, q_a, self_weights=None)
    mass = weights.sum(axis=1)
    ess = effective_sample_size(weights)
    return uncertainty, mass, ess


def grid_for(method: str) -> list[tuple[float, float | None]]:
    if method == "GeoCP":
        return [(h, None) for h in GEO_H]
    return [(h, lam) for h in JOINT_H for lam in LAMBDAS]


def tune_method(method: str, scores: np.ndarray, y: np.ndarray, pred: np.ndarray,
                dgeo2: np.ndarray, dfeat2: np.ndarray | None) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    rows: list[dict] = []
    for grid_index, (h, lam) in enumerate(grid_for(method)):
        weights = kernel_weights(dgeo2, dfeat2, h, lam)
        for track in TRACKS:
            uncertainty, mass, ess = weighted_uncertainty(scores, weights, track)
            metrics = interval_metrics(y, pred, uncertainty)
            rows.append({
                "track": track, "method": method, "grid_index": grid_index, "bandwidth": h, "lambda": lam,
                **{key: metrics[key] for key in ("coverage", "mean_width", "median_width", "mean_interval_score",
                    "infinite_count", "infinite_fraction", "mean_width_finite", "median_width_finite", "mean_interval_score_finite")},
                "mean_kernel_mass": float(mass.mean()), "median_kernel_mass": float(np.median(mass)),
                "mean_n_eff": float(ess.mean()), "median_n_eff": float(np.median(ess)), "min_n_eff": float(ess.min()),
                "feasible": metrics["coverage"] >= 0.90,
            })
    selections: dict[tuple[str, str], dict] = {}
    for track in TRACKS:
        track_rows = [row for row in rows if row["track"] == track]
        feasible = [row for row in track_rows if row["feasible"]]
        if feasible:
            chosen = min(feasible, key=lambda row: (row["mean_interval_score"], row["grid_index"]))
            status = "SELECTED"
        else:
            chosen = min(track_rows, key=lambda row: (-row["coverage"], row["mean_interval_score"], row["grid_index"]))
            status = "NO_FEASIBLE_HYPERPARAMETER"
        selections[(track, method)] = {
            "method": method, "track": track, "status": status, "bandwidth": chosen["bandwidth"],
            "lambda": chosen["lambda"], "grid_index": chosen["grid_index"],
            "validation_coverage": chosen["coverage"], "validation_interval_score": chosen["mean_interval_score"],
            "validation_infinite_fraction": chosen["infinite_fraction"], "diagnostic_only": status != "SELECTED",
        }
    return selections, rows


def serialize_transform(transform: dict) -> dict:
    return {key: value for key, value in transform.items() if key != "calib"}


def stratum(info: dict) -> str:
    selected = set(info["selected"])
    if info["status"] == "NO_CAUSAL_SIGNAL": return "NO_SIGNAL"
    if selected == {"x2"}: return "clean_x2_only"
    if "x2" in selected: return "x2_plus_nuisance"
    return "other_signal"


def freeze_scenario(index: int, seed: int, scenario: str, info: dict) -> dict:
    dev = load_development(index, seed, scenario)
    validation = role(dev, "validation")
    pool = role(dev, "CP_calibration_pool").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    frozen: dict[int, dict] = {}
    for n_cal in CAL_SIZES:
        calib = pool.iloc[:n_cal].copy()
        scores = np.abs(calib["y_true"].to_numpy(float) - calib["y_pred"].to_numpy(float))
        dgeo2 = pairwise_sq(validation[COORDS].to_numpy(float), calib[COORDS].to_numpy(float))
        transforms = {method: fit_feature_transform(calib, method, info) for method in JOINT_METHODS}
        dfeat = {method: None if transforms[method]["kind"] == "fallback" else feature_distance_sq(validation, transforms[method])
                 for method in JOINT_METHODS}
        selections: dict[tuple[str, str], dict] = {}
        grid_rows: list[dict] = []
        geo_selections, rows = tune_method("GeoCP", scores, validation["y_true"].to_numpy(float),
                                           validation["y_pred"].to_numpy(float), dgeo2, None)
        selections.update(geo_selections)
        grid_rows.extend(rows)
        for method in ("GeoSIMCP-EUC", "GeoSIMCP-MND"):
            method_selections, rows = tune_method(method, scores, validation["y_true"].to_numpy(float),
                                                   validation["y_pred"].to_numpy(float), dgeo2, dfeat[method])
            selections.update(method_selections)
            grid_rows.extend(rows)
        if info["status"] == "NO_CAUSAL_SIGNAL" or transforms["GeoCausalCP-Full"]["kind"] == "fallback":
            for track in TRACKS:
                selections[(track, "GeoCausalCP-Full")] = {
                    **geo_selections[(track, "GeoCP")], "method": "GeoCausalCP-Full", "fallback": "GeoCP",
                    "cmi_status": info["status"],
                }
            for row in [row for row in grid_rows if row["method"] == "GeoCP"]:
                grid_rows.append({**row, "method": "GeoCausalCP-Full", "fallback": "GeoCP"})
        else:
            causal_selections, rows = tune_method(
                "GeoCausalCP-Full", scores, validation["y_true"].to_numpy(float), validation["y_pred"].to_numpy(float),
                dgeo2, dfeat["GeoCausalCP-Full"],
            )
            for key, selection in causal_selections.items():
                selections[key] = {**selection, "fallback": None, "cmi_status": info["status"]}
            grid_rows.extend(rows)
        frozen[n_cal] = {
            "calib": calib, "scores": scores, "transforms": transforms,
            "selections": selections, "grid_rows": grid_rows,
        }

    state = {
        "cmi": info, "stratum": stratum(info),
        "n_cal": {
            str(n): {
                "calib_hash": frame_hash(frozen[n]["calib"]),
                "transforms": {m: serialize_transform(frozen[n]["transforms"][m]) for m in JOINT_METHODS},
                "selections": [frozen[n]["selections"][(t, m)] for t in TRACKS for m in OFFICIAL_METHODS],
            } for n in CAL_SIZES
        },
        "validation_hash": frame_hash(validation),
    }
    frozen["validation"] = validation
    frozen["state"] = state
    frozen["hash"] = hashlib.sha256(json.dumps(json_safe(state), sort_keys=True).encode()).hexdigest()
    return frozen


def test_covariates(index: int, seed: int, scenario: str) -> pd.DataFrame:
    base = scenario_dir(index, seed, scenario)
    x = pd.read_csv(base / "predictor_inputs.csv", keep_default_na=False)
    pred = pd.read_csv(base / "predictions.csv", keep_default_na=False)
    return x.loc[x["split_role"].eq("test")].merge(pred, on=[*META, "split_role"], validate="one_to_one", sort=False)


def write_unseal_barrier(index: int, seed: int, scenario: str, info: dict, frozen: dict, test_cov: pd.DataFrame) -> dict:
    base = scenario_dir(index, seed, scenario)
    record = {
        "event": "TEST_UNSEAL", "replicate_index": index, "generation_seed": seed, "scenario": scenario,
        "cmi_status": info["status"], "eligible_features": info["selected"], "cmi_weights": info["weights"],
        "cmi_invariant_across_n_cal": True, "stratum": stratum(info),
        "n_calibration_states": frozen["state"]["n_cal"],
        "validation_hash": frozen["state"]["validation_hash"],
        "test_covariate_hash": frame_hash(test_cov),
        "prediction_sha256": sha(base / "predictions.csv"),
        "atom_settings": TRACKS, "frozen_state_hash": frozen["hash"],
    }
    record["barrier_sha256"] = hashlib.sha256(json.dumps(json_safe(record), sort_keys=True).encode()).hexdigest()
    dump(base / "test_unseal_audit.json", record)
    return record


def evaluate_scenario(index: int, seed: int, scenario: str, info: dict, frozen: dict) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    base = scenario_dir(index, seed, scenario)
    cov = test_covariates(index, seed, scenario)
    barrier = write_unseal_barrier(index, seed, scenario, info, frozen, cov)
    # This is the first read of evaluation labels in the analysis path.
    sealed = pd.read_csv(base / "test_sealed.csv", keep_default_na=False)
    test = cov.merge(sealed, on=[*META, "split_role"], validate="one_to_one", sort=False)
    if len(test) != 300:
        raise AssertionError("test join")
    y, pred = test["y_true"].to_numpy(float), test["y_pred"].to_numpy(float)
    result_rows: list[dict] = []
    point_rows: list[dict] = []
    alignment_rows: list[dict] = []
    fallback_rows: list[dict] = []

    for n_cal in CAL_SIZES:
        part = frozen[n_cal]
        calib, scores = part["calib"], part["scores"]
        dgeo2 = pairwise_sq(test[COORDS].to_numpy(float), calib[COORDS].to_numpy(float))
        dfeat_test = {method: None if part["transforms"][method]["kind"] == "fallback" else feature_distance_sq(test, part["transforms"][method])
                      for method in JOINT_METHODS}
        rank = math.ceil((n_cal + 1) * (1 - ALPHA))
        raw_u = np.full(len(test), np.sort(scores, kind="mergesort")[rank - 1] if rank <= n_cal else np.inf)
        raw_metrics = interval_metrics(y, pred, raw_u)
        for track in TRACKS:
            result_rows.append({
                "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
                "track": track, "method": "Raw-CP", "selection_status": "SELECTED", "bandwidth": None, "lambda": None,
                "cmi_status": "NA", "stratum": "NA", "selected_features": "", "feature_weights": "{}",
                **{key: raw_metrics[key] for key in ("coverage", "covered_count", "n", "mean_width", "median_width",
                    "mean_interval_score", "infinite_count", "infinite_fraction", "mean_width_finite",
                    "median_width_finite", "mean_interval_score_finite")},
                "mean_kernel_mass": float(n_cal), "median_kernel_mass": float(n_cal),
                "mean_n_eff": float(n_cal), "median_n_eff": float(n_cal), "min_n_eff": float(n_cal),
            })
            uncertainty_cache: dict[str, np.ndarray] = {}
            for method in OFFICIAL_METHODS:
                selected = part["selections"][(track, method)]
                base_row = {
                    "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
                    "track": track, "method": method, "selection_status": selected["status"],
                    "bandwidth": selected["bandwidth"], "lambda": selected["lambda"],
                    "cmi_status": info["status"] if method == "GeoCausalCP-Full" else "NA",
                    "stratum": stratum(info) if method == "GeoCausalCP-Full" else "NA",
                    "selected_features": ";".join(info["selected"]) if method == "GeoCausalCP-Full" else "",
                    "feature_weights": json.dumps(info["weights"], sort_keys=True) if method == "GeoCausalCP-Full" else "{}",
                }
                if selected["status"] != "SELECTED":
                    result_rows.append(base_row)
                    continue
                if method == "GeoCP" or (method == "GeoCausalCP-Full" and selected.get("fallback") == "GeoCP"):
                    dfeat = None
                else:
                    dfeat = dfeat_test[method]
                weights = kernel_weights(dgeo2, dfeat, selected["bandwidth"], selected["lambda"])
                uncertainty, mass, ess = weighted_uncertainty(scores, weights, track)
                metrics = interval_metrics(y, pred, uncertainty)
                uncertainty_cache[method] = uncertainty
                result_rows.append({
                    **base_row,
                    **{key: metrics[key] for key in ("coverage", "covered_count", "n", "mean_width", "median_width",
                        "mean_interval_score", "infinite_count", "infinite_fraction", "mean_width_finite",
                        "median_width_finite", "mean_interval_score_finite")},
                    "mean_kernel_mass": float(mass.mean()), "median_kernel_mass": float(np.median(mass)),
                    "mean_n_eff": float(ess.mean()), "median_n_eff": float(np.median(ess)), "min_n_eff": float(ess.min()),
                })
                if track == "B":
                    expected_inf = mass / (mass + 1.0) < Q_TOL
                    observed_inf = np.isinf(uncertainty)
                    for row_index, sample_id in enumerate(test["sample_id"].astype(str)):
                        point_rows.append({
                            "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
                            "method": method, "sample_id": sample_id, "kernel_mass": mass[row_index],
                            "mass_fraction": mass[row_index] / (mass[row_index] + 1.0), "ess": ess[row_index],
                            "below_9": mass[row_index] < 9.0, "below_exact_threshold": bool(expected_inf[row_index]),
                            "infinite_interval": bool(observed_inf[row_index]),
                            "mass_infinity_match": bool(expected_inf[row_index] == observed_inf[row_index]),
                        })
                    if not np.array_equal(expected_inf, observed_inf):
                        raise AssertionError("kernel mass/infinity mismatch")
            if info["status"] == "NO_CAUSAL_SIGNAL" and "GeoCP" in uncertainty_cache and "GeoCausalCP-Full" in uncertainty_cache:
                geo, causal = uncertainty_cache["GeoCP"], uncertainty_cache["GeoCausalCP-Full"]
                exact = np.array_equal(geo, causal)
                finite = np.isfinite(geo) & np.isfinite(causal)
                max_diff = float(np.max(np.abs(geo[finite] - causal[finite]))) if finite.any() else 0.0
                fallback_rows.append({
                    "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
                    "track": track, "exact_array_equal": exact, "max_abs_finite_threshold_difference": max_diff,
                })
                if not exact or max_diff != 0.0:
                    raise AssertionError("GeoCP fallback mismatch")

        if scenario == "S2" and info["status"] == "SIGNAL":
            sigma_cal = calib["oracle_sigma"].to_numpy(float)
            sigma_test = test["oracle_sigma"].to_numpy(float)
            mismatch = np.abs(sigma_test[:, None] - sigma_cal[None, :]).ravel()
            # Feature distances do not depend on the selected track.
            for label, distance in (("EUC_feature", dfeat_test["GeoSIMCP-EUC"]),
                                    ("CMI_Full_causal_feature", dfeat_test["GeoCausalCP-Full"]),
                                    ("geographic", dgeo2)):
                alignment_rows.append({
                    "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
                    "track": "feature_only", "distance": label,
                    "spearman_oracle_scale_mismatch": float(spearmanr(np.sqrt(distance).ravel(), mismatch).statistic),
                })
            for track in TRACKS:
                for method, label in (("GeoSIMCP-EUC", "selected_EUC_joint"), ("GeoCausalCP-Full", "selected_GeoCausal_joint")):
                    selected = part["selections"][(track, method)]
                    if selected["status"] != "SELECTED":
                        continue
                    dfeat = dfeat_test[method]
                    joint = float(selected["lambda"]) * dgeo2 + (1 - float(selected["lambda"])) * dfeat
                    alignment_rows.append({
                        "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
                        "track": track, "distance": label,
                        "spearman_oracle_scale_mismatch": float(spearmanr(np.sqrt(joint).ravel(), mismatch).statistic),
                    })
    return result_rows, point_rows, alignment_rows, fallback_rows, barrier


def save_checkpoint(index: int, seed: int, scenario: str, info: dict, frozen: dict,
                    result_rows: list[dict], point_rows: list[dict], alignment_rows: list[dict],
                    fallback_rows: list[dict], barrier: dict) -> None:
    base = scenario_dir(index, seed, scenario)
    grid_rows, selected_rows = [], []
    for n_cal in CAL_SIZES:
        for row in frozen[n_cal]["grid_rows"]:
            grid_rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal, **row})
        for track in TRACKS:
            for method in OFFICIAL_METHODS:
                selected_rows.append({
                    "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
                    **frozen[n_cal]["selections"][(track, method)],
                })
    paths = {
        "validation_grid_results.csv": pd.DataFrame(grid_rows),
        "selected_hyperparameters.csv": pd.DataFrame(selected_rows),
        "replicate_results.csv": pd.DataFrame(result_rows),
        "kernel_mass_pointwise.csv": pd.DataFrame(point_rows),
        "oracle_alignment.csv": pd.DataFrame(alignment_rows),
        "fallback_equality.csv": pd.DataFrame(fallback_rows),
    }
    for name, frame in paths.items():
        write_csv(frame, base / name)
    hashes = {name: sha(base / name) for name in paths}
    hashes["test_unseal_audit.json"] = sha(base / "test_unseal_audit.json")
    dump(base / "ANALYSIS_COMPLETE.json", {
        "replicate_index": index, "generation_seed": seed, "scenario": scenario,
        "frozen_state_hash": frozen["hash"], "barrier_sha256": barrier["barrier_sha256"],
        "hashes": hashes, "cmi_status": info["status"], "stratum": stratum(info),
    })


def process_analysis(index: int, seed: int, scenario: str) -> dict:
    base = scenario_dir(index, seed, scenario)
    complete = base / "ANALYSIS_COMPLETE.json"
    if complete.exists():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        for name, expected in saved["hashes"].items():
            if sha(base / name) != expected:
                raise AssertionError(f"analysis checkpoint mismatch: {base/name}")
        return {"replicate_index": index, "scenario": scenario, "checkpoint": True}
    _, info = cmi_info(base / "cmi_full999_results.csv")
    frozen = freeze_scenario(index, seed, scenario, info)
    outputs = evaluate_scenario(index, seed, scenario, info, frozen)
    save_checkpoint(index, seed, scenario, info, frozen, *outputs)
    return {"replicate_index": index, "scenario": scenario, "checkpoint": False}


def concat_local(bases: list[Path], name: str) -> pd.DataFrame:
    frames = []
    for base in bases:
        path = base / name
        if path.exists() and path.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                # Some scientifically inapplicable partitions (notably S4 or
                # CMI NO_SIGNAL oracle-alignment outputs) serialize as a single
                # newline. They are valid empty checkpoints, not failed data.
                continue
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main() -> None:
    started = time.time()
    seal, seeds = verify_prereg()
    log_mode = "a" if (RECORD / "run_log.txt").exists() else "w"
    with (RECORD / "run_log.txt").open(log_mode, encoding="utf-8") as handle:
        handle.write("Stage-20B execution/resume started after immutable preregistration verification.\n")
    if not (RECORD / "stderr.txt").exists():
        (RECORD / "stderr.txt").write_text("", encoding="utf-8")
    generator = load_module("stage20b_generator", GENERATOR_SOURCE)
    predictor = load_module("stage20b_predictor", PREDICTOR_SOURCE)
    gen_cfg = json.loads(GENERATOR_CONFIG.read_text(encoding="utf-8"))
    pred_cfg = json.loads(PREDICTOR_CONFIG.read_text(encoding="utf-8"))
    if pred_cfg["model_configuration"]["random_state"] != 420907 or pred_cfg["authorized_features"] != FEATURES_ALL:
        raise AssertionError("predictor protocol changed")
    dump(RECORD / "execution_config.json", {
        "preregistration_composite_sha256": seal["composite_sha256"], "official_commit": COMMIT,
        "implementation": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "implementation_sha256": sha(Path(__file__)), "cmi_worker_sha256": sha(CMI_WORKER),
        "parity_source_sha256": sha(PARITY_SOURCE), "official_utils_sha256": sha(OFFICIAL / "geoconformal/geocp/utils.py"),
        "exact_mass_threshold": MASS_THRESHOLD_EXACT,
    })

    # Generate the first paired replicate and complete a deterministic CMI/freeze preflight.
    first_rows = generate_replicate(1, seeds[0], generator, gen_cfg, predictor, pred_cfg)
    if not (RECORD / "preflight_summary.json").exists():
        cmi_checks, freeze_checks = {}, {}
        for scenario in SCENARIOS:
            base = scenario_dir(1, seeds[0], scenario)
            run_cmi(1, seeds[0], scenario)
            frame, info = cmi_info(base / "cmi_full999_results.csv")
            with tempfile.TemporaryDirectory(prefix="stage20b_cmi_probe_", dir=RECORD) as temp:
                probe = Path(temp) / "rerun.csv"
                run_cmi(1, seeds[0], scenario, probe)
                pd.testing.assert_frame_equal(frame, pd.read_csv(probe), check_exact=True)
            frozen_a = freeze_scenario(1, seeds[0], scenario, info)
            frozen_b = freeze_scenario(1, seeds[0], scenario, info)
            if frozen_a["hash"] != frozen_b["hash"]:
                raise AssertionError("frozen-state nondeterminism")
            outputs = evaluate_scenario(1, seeds[0], scenario, info, frozen_a)
            save_checkpoint(1, seeds[0], scenario, info, frozen_a, *outputs)
            cmi_checks[scenario] = "bit-identical clean rerun"
            freeze_checks[scenario] = frozen_a["hash"]
        dump(RECORD / "preflight_summary.json", {
            "status": "PASS", "seed": seeds[0], "cmi_determinism": cmi_checks,
            "frozen_state_determinism": freeze_checks, "nested_calibration": "PASS",
            "test_seal": "PASS", "fallback_equality": "enforced", "mass_infinity_equality": "enforced",
        })
        print("PREFLIGHT PASS", flush=True)

    generator_rows = [item[0] for item in first_rows]
    split_rows = [item[1] for item in first_rows]
    prediction_rows = [item[2] for item in first_rows]
    nested_rows = [row for item in first_rows for row in item[3]]
    for index, seed in enumerate(seeds[1:], 2):
        items = generate_replicate(index, seed, generator, gen_cfg, predictor, pred_cfg)
        generator_rows.extend(item[0] for item in items)
        split_rows.extend(item[1] for item in items)
        prediction_rows.extend(item[2] for item in items)
        nested_rows.extend(row for item in items for row in item[3])
        if index % 5 == 0:
            print(f"GENERATION {index}/50", flush=True)
    write_csv(pd.DataFrame(generator_rows), RECORD / "generator_audit.csv")
    write_csv(pd.DataFrame(split_rows), RECORD / "split_manifest.csv")
    write_csv(pd.DataFrame(prediction_rows), RECORD / "prediction_manifest.csv")
    dump(RECORD / "nested_calibration_manifest.json", {
        "rule": "sample_id ascending; first 200, first 400, all 800", "rows": nested_rows,
        "all_nested": all(row["contains_previous"] and row["strictly_larger_than_previous"] for row in nested_rows),
    })

    jobs = [(index, seed, scenario) for index, seed in enumerate(seeds, 1) for scenario in SCENARIOS
            if not ((scenario_dir(index, seed, scenario) / "cmi_full999_results.csv").exists()
                    and (scenario_dir(index, seed, scenario) / "cmi_full999_results.json").exists())]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_cmi, *job) for job in jobs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if completed % 10 == 0:
                print(f"CMI {completed}/{len(jobs)}", flush=True)

    cmi_rows, invariance_rows = [], []
    for index, seed in enumerate(seeds, 1):
        for scenario in SCENARIOS:
            frame, info = cmi_info(scenario_dir(index, seed, scenario) / "cmi_full999_results.csv")
            for row in frame.to_dict("records"):
                cmi_rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, **row})
            signature = hashlib.sha256(json.dumps(json_safe(info), sort_keys=True).encode()).hexdigest()
            for n_cal in CAL_SIZES:
                invariance_rows.append({
                    "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
                    "cmi_status": info["status"], "selected_features": ";".join(info["selected"]),
                    "weights": json.dumps(info["weights"], sort_keys=True), "signature_sha256": signature,
                    "matches_single_fixed_cmi_result": True,
                })
    write_csv(pd.DataFrame(cmi_rows), RECORD / "cmi_full999_results.csv")
    write_csv(pd.DataFrame(invariance_rows), RECORD / "cmi_invariance_audit.csv")

    analysis_jobs = [(index, seed, scenario) for index, seed in enumerate(seeds, 1) for scenario in SCENARIOS
                     if not (scenario_dir(index, seed, scenario) / "ANALYSIS_COMPLETE.json").exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(process_analysis, *job) for job in analysis_jobs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            if completed % 5 == 0:
                print(f"ANALYSIS {completed}/{len(analysis_jobs)} latest={result['replicate_index']}/{result['scenario']}", flush=True)

    bases = [scenario_dir(index, seed, scenario) for index, seed in enumerate(seeds, 1) for scenario in SCENARIOS]
    grid = concat_local(bases, "validation_grid_results.csv")
    selected = concat_local(bases, "selected_hyperparameters.csv")
    results = concat_local(bases, "replicate_results.csv")
    pointwise = concat_local(bases, "kernel_mass_pointwise.csv")
    alignment = concat_local(bases, "oracle_alignment.csv")
    fallback = concat_local(bases, "fallback_equality.csv")
    barriers = pd.DataFrame([json.loads((base / "test_unseal_audit.json").read_text(encoding="utf-8")) for base in bases])
    write_csv(grid, RECORD / "validation_grid_results.csv")
    write_csv(selected, RECORD / "selected_hyperparameters.csv")
    write_csv(results, RECORD / "replicate_results.csv")
    write_csv(pointwise, RECORD / "kernel_mass_pointwise.csv")
    write_csv(alignment, RECORD / "oracle_alignment_summary.csv")
    write_csv(fallback, RECORD / "no_signal_fallback_equality_audit.csv")
    write_csv(barriers, RECORD / "test_unseal_audit.csv")
    write_csv(selected.loc[selected["status"].eq("NO_FEASIBLE_HYPERPARAMETER")].copy(), RECORD / "no_feasible_summary.csv")
    if not pointwise["mass_infinity_match"].all():
        raise AssertionError("consolidated mass/infinity mismatch")
    if not fallback.empty and (not fallback["exact_array_equal"].all() or fallback["max_abs_finite_threshold_difference"].max() != 0):
        raise AssertionError("consolidated fallback mismatch")
    dump(RECORD / "scientific_computation_checkpoint.json", {
        "status": "COMPLETE", "paired_replicates": 50, "scenario_replicates": 100,
        "analysis_checkpoints": sum((base / "ANALYSIS_COMPLETE.json").exists() for base in bases),
        "calibration_sizes": CAL_SIZES, "mass_infinity_mismatches": int((~pointwise["mass_infinity_match"]).sum()),
        "preregistration_composite_sha256": seal["composite_sha256"], "elapsed_seconds": time.time() - started,
    })
    with (RECORD / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"Scientific computation complete: 100 scenario-replicates x 3 nested sizes; elapsed_seconds={time.time()-started:.3f}\n")
    print("STAGE20B SCIENTIFIC COMPUTATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
