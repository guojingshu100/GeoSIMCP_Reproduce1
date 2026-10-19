"""Execute the sealed Stage-19B official GeoSIMCP parity experiment.

The runner is deterministic and scenario-checkpointed. It uses the audited
geoconformal weighted-quantile implementation directly; custom code only
constructs the preregistered CMI-weighted feature distance.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RECORD = ROOT / "experiments/geo_causal_cp/20260919_stage19b_direct_geosimcp_parity_v1"
PREREG = RECORD / "preregistration"
OFFICIAL = ROOT / "external/geoconformal_stage19a_audit_26419ef"
COMMIT = "26419ef06d7e3711e6618f97f4f1ed02af16bcb7"
STAGE15 = ROOT / "causal_only_cp_v1/run_stage15_fresh_confirmatory_v1.py"
CMI_WORKER = HERE / "stage19b_cmi_full_worker_v1.py"
CMI_PYTHON = ROOT / "causal_strength_diagnostic_v1/.venv_cmi_v1/Scripts/python.exe"
GENERATOR_SOURCE = ROOT / "simulation_pilot_20260908/generator/generate_scenarios.py"
GENERATOR_CONFIG = ROOT / "simulation_pilot_20260908/generator/config_v1.json"
PREDICTOR_SOURCE = ROOT / "simulation_pilot_20260908/predictor/fit_predictors.py"
PREDICTOR_CONFIG = ROOT / "simulation_pilot_20260908/predictor/predictor_config_v1.json"
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]
META = ["sample_id", "scenario", "split", "local_calibration_role"]
SCENARIOS = ["S2", "S4"]
TRACKS = {"A": False, "B": True}
JOINT_METHODS = ["GeoSIMCP-EUC", "GeoSIMCP-MND", "GeoCausalCP-Full"]
PRINCIPAL = ["Raw-CP", "GeoCP", *JOINT_METHODS]
ALL_METHODS = [*PRINCIPAL, "CausalOnly-Full"]
ALPHA = .10
Q_A = math.ceil((200 + 1) * (1 - ALPHA)) / 200
Q_B = 1 - ALPHA
GEO_H = [.1, .5, 1., 2., 3., 5.]
JOINT_H = list(np.linspace(.1, 5., 20))
LAMBDAS = list(np.arange(0, 1.05, .05))

_utils_spec = importlib.util.spec_from_file_location("audited_geoconformal_geocp_utils", OFFICIAL / "geoconformal/geocp/utils.py")
if _utils_spec is None or _utils_spec.loader is None:
    raise RuntimeError("cannot load audited geoconformal utilities")
_official_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_official_utils)
effective_sample_size = _official_utils.effective_sample_size
weighted_quantile = _official_utils.weighted_quantile


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value): return "NaN"
        if np.isposinf(value): return "Infinity"
        if np.isneginf(value): return "-Infinity"
        return float(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def json_default(value: Any):
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating):
        if np.isfinite(value): return float(value)
        return str(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def verify_prereg() -> tuple[dict, list[int]]:
    seal = json.loads((PREREG / "PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    for name, expected in seal["files"].items():
        if sha(PREREG / name) != expected: raise AssertionError(f"preregistration changed: {name}")
    payload = "\n".join(f"{n} {seal['files'][n]}" for n in sorted(seal["files"]))
    if hashlib.sha256(payload.encode()).hexdigest() != seal["composite_sha256"]: raise AssertionError("seal mismatch")
    seeds = json.loads((PREREG / "seed_manifest.json").read_text(encoding="utf-8"))["seeds"]
    if seeds != list(range(20840001, 20840051)): raise AssertionError("seed mismatch")
    # Managed sessions use a distinct Windows SID; authorize only this one
    # read-only invocation rather than modifying global Git configuration.
    safe = str(OFFICIAL.resolve()).replace("\\", "/")
    head = subprocess.run(["git", "-c", f"safe.directory={safe}", "rev-parse", "HEAD"],
                          cwd=OFFICIAL, capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT: raise AssertionError(f"official commit mismatch: {head}")
    return seal, seeds


def scenario_dir(index: int, seed: int, scenario: str) -> Path:
    return RECORD / "frozen_inputs" / f"replicate_{index:02d}_seed_{seed}" / scenario


def cmi_input(s15, index: int, seed: int, scenario: str) -> Path:
    path = scenario_dir(index, seed, scenario) / "cmi_full999_input.csv"
    if not path.exists():
        scale = s15.role(s15.load_development(index, seed, scenario), "scale_fit")
        frame = scale[["sample_id", *COORDS, *FEATURES]].copy()
        frame["A_model"] = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
        write_csv(frame, path)
    return path


def run_cmi(s15, index: int, seed: int, scenario: str, output: Path | None = None) -> dict:
    inp = cmi_input(s15, index, seed, scenario)
    out = output or scenario_dir(index, seed, scenario) / "cmi_full999_results.csv"
    if out.exists() and out.with_suffix(".json").exists():
        return {"runtime_seconds": json.loads(out.with_suffix(".json").read_text())["runtime_seconds"], "checkpoint": True}
    cmd = [str(CMI_PYTHON), str(CMI_WORKER), "--input", str(inp), "--output", str(out),
           "--replicate-index", str(index), "--scenario", scenario]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    out.with_suffix(".stdout.txt").write_text(proc.stdout, encoding="utf-8")
    out.with_suffix(".stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode: raise RuntimeError(f"CMI failed {index}/{scenario}: {proc.stderr[-3000:]}")
    return {"runtime_seconds": json.loads(out.with_suffix(".json").read_text())["runtime_seconds"], "checkpoint": False}


def cmi_info(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    if len(frame) != 4 or frame.feature.tolist() != FEATURES: raise AssertionError("CMI schema")
    weights = dict(zip(frame.feature, frame.weight.astype(float)))
    selected = [f for f in FEATURES if weights[f] > 0]
    status = str(frame.status.iloc[0])
    if status == "SIGNAL" and not np.isclose(sum(weights.values()), 1, atol=1e-12): raise AssertionError("CMI weight sum")
    if status == "NO_CAUSAL_SIGNAL" and selected: raise AssertionError("CMI no-signal")
    return frame, {"weights": weights, "selected": selected, "status": status}


def pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None, :] - 2*a@b.T, 0.)


def fit_feature_transform(calib: pd.DataFrame, method: str, info: dict) -> dict:
    x = calib[FEATURES].to_numpy(float)
    if method == "GeoSIMCP-MND":
        ranges = np.ptp(x, axis=0); ranges[ranges == 0] = 1e-8
        return {"kind": "mnd", "columns": FEATURES, "calib": x, "ranges": ranges}
    if method == "GeoSIMCP-EUC":
        sd_all = x.std(axis=0, ddof=0); valid = sd_all > 1e-8
        cols = [f for f, ok in zip(FEATURES, valid) if ok]
        raw = calib[cols].to_numpy(float); mu = raw.mean(axis=0); sd = raw.std(axis=0, ddof=0)
        return {"kind": "euc", "columns": cols, "calib": (raw-mu)/sd, "mean": mu, "sd": sd}
    selected = list(info["selected"])
    raw = calib[selected].to_numpy(float); sd_all = raw.std(axis=0, ddof=0) if selected else np.array([])
    keep = sd_all > 1e-8
    cols = [f for f, ok in zip(selected, keep) if ok]
    if not cols: return {"kind": "fallback", "columns": []}
    omega = np.array([info["weights"][f] for f in cols], float); omega /= omega.sum()
    raw = calib[cols].to_numpy(float); mu = raw.mean(axis=0); sd = raw.std(axis=0, ddof=0)
    return {"kind": "causal", "columns": cols, "calib": (raw-mu)/sd, "mean": mu, "sd": sd, "omega": omega}


def feature_distance_sq(query: pd.DataFrame, transform: dict) -> np.ndarray:
    kind = transform["kind"]
    if kind == "mnd":
        q = query[transform["columns"]].to_numpy(float)
        return (np.max(np.abs(q[:, None, :] - transform["calib"][None, :, :]) / transform["ranges"][None, None, :], axis=2) + 1e-8) ** 2
    q = query[transform["columns"]].to_numpy(float)
    q = (q-transform["mean"])/transform["sd"]
    if kind == "causal":
        # Weighted Euclidean distance, followed by the official +1e-8 distance offset.
        d2 = pairwise_sq(q*np.sqrt(transform["omega"]), transform["calib"]*np.sqrt(transform["omega"]))
    else:
        d2 = pairwise_sq(q, transform["calib"])
    return (np.sqrt(d2) + 1e-8) ** 2


def interval_metrics(y: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray, alpha: float = ALPHA) -> dict:
    lower, upper = pred-uncertainty, pred+uncertainty
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper): raise AssertionError("invalid interval")
    covered = (y >= lower) & (y <= upper); width = upper-lower
    score = width.copy()
    low = y < lower; high = y > upper
    score[low] += (2/alpha)*(lower[low]-y[low]); score[high] += (2/alpha)*(y[high]-upper[high])
    finite = np.isfinite(width)
    return {"coverage": float(covered.mean()), "covered_count": int(covered.sum()), "n": len(y),
            "mean_width": float(np.mean(width)), "median_width": float(np.median(width)),
            "mean_interval_score": float(np.mean(score)), "infinite_count": int((~finite).sum()),
            "infinite_fraction": float((~finite).mean()),
            "mean_width_finite": float(np.mean(width[finite])) if finite.any() else None,
            "median_width_finite": float(np.median(width[finite])) if finite.any() else None,
            "lower": lower, "upper": upper}


def weight_quantiles(scores: np.ndarray, dgeo2: np.ndarray, dfeat2: np.ndarray | None,
                     h: float, lam: float | None, include_atom: bool) -> tuple[np.ndarray, np.ndarray]:
    if dfeat2 is None:
        weights = np.exp(-.5*dgeo2/(h*h))
    else:
        joint2 = float(lam)*dgeo2 + (1-float(lam))*dfeat2
        weights = np.exp(-.5*joint2/((h+1e-8)**2))
    q = Q_B if include_atom else Q_A
    u = weighted_quantile(scores, weights, q, self_weights=np.ones(weights.shape[0]) if include_atom else None)
    return u, effective_sample_size(weights)


def tune_method(method: str, track: str, scores: np.ndarray, y: np.ndarray, pred: np.ndarray,
                dgeo2: np.ndarray, dfeat2: np.ndarray | None) -> tuple[dict, list[dict]]:
    rows = []
    grid = [(h, None) for h in GEO_H] if method == "GeoCP" else [(h, lam) for h in JOINT_H for lam in LAMBDAS]
    for grid_index, (h, lam) in enumerate(grid):
        uncertainty, neff = weight_quantiles(scores, dgeo2, dfeat2, h, lam, TRACKS[track])
        m = interval_metrics(y, pred, uncertainty)
        rows.append({"track": track, "method": method, "grid_index": grid_index, "bandwidth": h, "lambda": lam,
                     **{k: m[k] for k in ("coverage", "mean_width", "median_width", "mean_interval_score", "infinite_count", "infinite_fraction", "mean_width_finite", "median_width_finite")},
                     "mean_n_eff": float(np.mean(neff)), "median_n_eff": float(np.median(neff)), "min_n_eff": float(np.min(neff)),
                     "feasible": m["coverage"] >= .90})
    feasible = [r for r in rows if r["feasible"]]
    if feasible:
        chosen = min(feasible, key=lambda r: (r["mean_interval_score"], r["grid_index"]))
        status = "SELECTED"
    else:
        chosen = min(rows, key=lambda r: (-r["coverage"], r["mean_interval_score"], r["grid_index"]))
        status = "NO_FEASIBLE_HYPERPARAMETER"
    selected = {"method": method, "track": track, "status": status, "bandwidth": chosen["bandwidth"],
                "lambda": chosen["lambda"], "grid_index": chosen["grid_index"],
                "validation_coverage": chosen["coverage"], "validation_interval_score": chosen["mean_interval_score"],
                "diagnostic_only": status != "SELECTED"}
    return selected, rows


def raw_uncertainty(scores: np.ndarray, n_query: int) -> np.ndarray:
    rank = math.ceil((len(scores)+1)*(1-ALPHA))
    if rank > len(scores): return np.full(n_query, np.inf)
    return np.full(n_query, np.sort(scores, kind="mergesort")[rank-1])


def freeze_scenario(s15, index: int, seed: int, scenario: str, info: dict) -> dict:
    dev = s15.load_development(index, seed, scenario)
    scale = s15.role(dev, "scale_fit")
    tuning = s15.role(dev, "tuning_calibration")
    validation = s15.role(dev, "validation")
    calib = s15.role(dev, "final_calibration")
    scores = np.abs(calib.y_true.to_numpy(float)-calib.y_pred.to_numpy(float))
    ccoord = calib[COORDS].to_numpy(float); vcoord = validation[COORDS].to_numpy(float)
    dgeo2_val = pairwise_sq(vcoord, ccoord)
    transforms = {m: fit_feature_transform(calib, m, info) for m in JOINT_METHODS}
    dfeat_val = {m: None if transforms[m]["kind"] == "fallback" else feature_distance_sq(validation, transforms[m]) for m in JOINT_METHODS}
    selections, grid_rows = {}, []
    for track in TRACKS:
        geo_sel, rows = tune_method("GeoCP", track, scores, validation.y_true.to_numpy(float),
                                    validation.y_pred.to_numpy(float), dgeo2_val, None)
        selections[(track, "GeoCP")] = geo_sel; grid_rows += rows
        for method in ("GeoSIMCP-EUC", "GeoSIMCP-MND"):
            sel, rows = tune_method(method, track, scores, validation.y_true.to_numpy(float),
                                    validation.y_pred.to_numpy(float), dgeo2_val, dfeat_val[method])
            selections[(track, method)] = sel; grid_rows += rows
        if info["status"] == "NO_CAUSAL_SIGNAL" or transforms["GeoCausalCP-Full"]["kind"] == "fallback":
            selections[(track, "GeoCausalCP-Full")] = {**geo_sel, "method": "GeoCausalCP-Full",
                "fallback": "GeoCP", "cmi_status": info["status"]}
            for row in rows_for_method(grid_rows, track, "GeoCP"):
                grid_rows.append({**row, "method": "GeoCausalCP-Full", "fallback": "GeoCP"})
        else:
            sel, rows = tune_method("GeoCausalCP-Full", track, scores, validation.y_true.to_numpy(float),
                                    validation.y_pred.to_numpy(float), dgeo2_val, dfeat_val["GeoCausalCP-Full"])
            selections[(track, "GeoCausalCP-Full")] = {**sel, "fallback": None, "cmi_status": info["status"]}
            grid_rows += rows

    # Established Stage-18 CausalOnly-Full supporting shell; not an official-engine method.
    if info["status"] == "NO_CAUSAL_SIGNAL":
        chosen_k, causal_model, krows = s15.constant_model(scale, tuning, "CausalOnly-Full")
    else:
        chosen_k, causal_model, krows = s15.fit_local(scale, tuning, info["selected"],
            np.array([info["weights"][f] for f in info["selected"]]), "CausalOnly-Full")
    sigma_cal = s15.query_model(calib, causal_model, chosen_k)
    q_causal, rank = s15.quantile(scores/sigma_cal, ALPHA)
    base = scenario_dir(index, seed, scenario)
    x = pd.read_csv(base/"predictor_inputs.csv", keep_default_na=False)
    pred = pd.read_csv(base/"predictions.csv", keep_default_na=False)
    test_cov = x[x.split.eq("test")].merge(pred, on=META+["split_role"], validate="one_to_one", sort=False)
    sigma_test_causal = s15.query_model(test_cov, causal_model, chosen_k)
    serial_sel = [selections[(t, m)] for t in TRACKS for m in ["GeoCP", *JOINT_METHODS]]
    digest_payload = {"cmi": info, "selections": serial_sel, "causal_only_k": chosen_k,
                      "causal_only_q": q_causal, "calib_hash": s15.frame_hash(calib),
                      "validation_hash": s15.frame_hash(validation), "test_cov_hash": s15.frame_hash(test_cov)}
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, default=json_default).encode()).hexdigest()
    return {"dev": dev, "scale": scale, "tuning": tuning, "validation": validation, "calib": calib,
            "scores": scores, "transforms": transforms, "selections": selections, "grid_rows": grid_rows,
            "causal_only_k": chosen_k, "causal_only_model": causal_model, "causal_only_q": q_causal,
            "causal_only_rank": rank, "causal_only_krows": krows, "sigma_test_causal": sigma_test_causal,
            "test_cov": test_cov, "hash": digest}


def rows_for_method(rows: list[dict], track: str, method: str) -> list[dict]:
    return [r for r in rows if r["track"] == track and r["method"] == method]


def serial_transform(transform: dict) -> dict:
    return {k: v for k, v in transform.items() if k != "calib"}


def write_barrier(s15, index: int, seed: int, scenario: str, info: dict, frozen: dict) -> dict:
    base = scenario_dir(index, seed, scenario)
    record = {"event": "TEST_UNSEAL", "replicate_index": index, "generation_seed": seed, "scenario": scenario,
        "cmi_status": info["status"], "eligible_features": info["selected"], "cmi_weights": info["weights"],
        "feature_transforms": {m: serial_transform(frozen["transforms"][m]) for m in JOINT_METHODS},
        "selected_hyperparameters": [frozen["selections"][(t,m)] for t in TRACKS for m in ["GeoCP", *JOINT_METHODS]],
        "causal_only_selected_k": frozen["causal_only_k"], "causal_only_q_norm": frozen["causal_only_q"],
        "atom_settings": {t: TRACKS[t] for t in TRACKS}, "weighted_quantile": "official geoconformal.geocp.utils.weighted_quantile step",
        "scale_fit_hash": s15.frame_hash(frozen["scale"]), "tuning_hash": s15.frame_hash(frozen["tuning"]),
        "validation_hash": s15.frame_hash(frozen["validation"]), "cp_calibration_hash": s15.frame_hash(frozen["calib"]),
        "prediction_sha256": sha(base/"predictions.csv"), "test_covariate_hash": s15.frame_hash(frozen["test_cov"]),
        "frozen_state_hash": frozen["hash"]}
    record["barrier_sha256"] = hashlib.sha256(json.dumps(json_safe(record), sort_keys=True).encode()).hexdigest()
    dump(base/"test_unseal_audit.json", record)
    return record


def selected_uncertainty(frozen: dict, method: str, track: str, query: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    selected = frozen["selections"][(track, method)]
    if selected["status"] != "SELECTED": return None, None
    calib = frozen["calib"]
    dgeo2 = pairwise_sq(query[COORDS].to_numpy(float), calib[COORDS].to_numpy(float))
    if method == "GeoCP": dfeat = None
    elif method == "GeoCausalCP-Full" and selected.get("fallback") == "GeoCP": dfeat = None
    else: dfeat = feature_distance_sq(query, frozen["transforms"][method])
    return weight_quantiles(frozen["scores"], dgeo2, dfeat, selected["bandwidth"], selected["lambda"], TRACKS[track])


def evaluate_scenario(s15, index: int, seed: int, scenario: str, info: dict, frozen: dict) -> tuple[list[dict], list[dict], dict]:
    base = scenario_dir(index, seed, scenario)
    barrier = write_barrier(s15, index, seed, scenario, info, frozen)
    sealed = pd.read_csv(base/"test_sealed.csv", keep_default_na=False)
    test = frozen["test_cov"].merge(sealed, on=META+["split_role"], validate="one_to_one", sort=False)
    if len(test) != 300: raise AssertionError("test join")
    y, pred = test.y_true.to_numpy(float), test.y_pred.to_numpy(float)
    rows, inf_rows = [], []
    raw_u = raw_uncertainty(frozen["scores"], len(test)); raw_m = interval_metrics(y, pred, raw_u)
    for track in TRACKS:
        rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, "track": track,
            "method": "Raw-CP", "selection_status": "SELECTED", "bandwidth": None, "lambda": None,
            "cmi_status": "NA", "selected_features": "", "feature_weights": "{}", "selected_k": None,
            "q_norm": float(raw_u[0]), **{k: raw_m[k] for k in ("coverage","covered_count","n","mean_width","median_width","mean_interval_score","infinite_count","infinite_fraction","mean_width_finite","median_width_finite")},
            "mean_n_eff": 200., "median_n_eff": 200., "min_n_eff": 200.})
        cache = {}
        for method in ["GeoCP", *JOINT_METHODS]:
            selected = frozen["selections"][(track, method)]
            uncertainty, neff = selected_uncertainty(frozen, method, track, test)
            if uncertainty is None:
                rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, "track": track,
                    "method": method, "selection_status": selected["status"], "bandwidth": selected["bandwidth"],
                    "lambda": selected["lambda"], "cmi_status": info["status"] if method=="GeoCausalCP-Full" else "NA",
                    "selected_features": ";".join(info["selected"]) if method=="GeoCausalCP-Full" else "",
                    "feature_weights": json.dumps(info["weights"], sort_keys=True) if method=="GeoCausalCP-Full" else "{}"})
                continue
            m = interval_metrics(y, pred, uncertainty); cache[method] = (m, uncertainty)
            rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, "track": track,
                "method": method, "selection_status": selected["status"], "bandwidth": selected["bandwidth"], "lambda": selected["lambda"],
                "cmi_status": info["status"] if method=="GeoCausalCP-Full" else "NA",
                "selected_features": ";".join(info["selected"]) if method=="GeoCausalCP-Full" else "",
                "feature_weights": json.dumps(info["weights"], sort_keys=True) if method=="GeoCausalCP-Full" else "{}",
                "selected_k": None, "q_norm": None,
                **{k: m[k] for k in ("coverage","covered_count","n","mean_width","median_width","mean_interval_score","infinite_count","infinite_fraction","mean_width_finite","median_width_finite")},
                "mean_n_eff": float(np.mean(neff)), "median_n_eff": float(np.median(neff)), "min_n_eff": float(np.min(neff))})
            if track == "B": inf_rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                "method": method, "infinite_count": m["infinite_count"], "infinite_fraction": m["infinite_fraction"],
                "any_infinite": m["infinite_count"]>0, "mean_width_finite": m["mean_width_finite"],
                "median_width_finite": m["median_width_finite"], "mean_n_eff": float(np.mean(neff)),
                "median_n_eff": float(np.median(neff)), "min_n_eff": float(np.min(neff))})
        if info["status"] == "NO_CAUSAL_SIGNAL" and "GeoCP" in cache and "GeoCausalCP-Full" in cache:
            if not np.array_equal(cache["GeoCP"][1], cache["GeoCausalCP-Full"][1]):
                raise AssertionError("GeoCP fallback mismatch, including finite/infinite pattern")
        # Supporting Stage-18 normalized-CP causal-only ablation is identical across tracks.
        u = frozen["causal_only_q"]*frozen["sigma_test_causal"]
        m = interval_metrics(y, pred, u)
        rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, "track": track,
            "method": "CausalOnly-Full", "selection_status": "SELECTED", "bandwidth": None, "lambda": None,
            "cmi_status": info["status"], "selected_features": ";".join(info["selected"]),
            "feature_weights": json.dumps(info["weights"], sort_keys=True), "selected_k": frozen["causal_only_k"],
            "q_norm": frozen["causal_only_q"], **{k: m[k] for k in ("coverage","covered_count","n","mean_width","median_width","mean_interval_score","infinite_count","infinite_fraction","mean_width_finite","median_width_finite")},
            "mean_n_eff": None, "median_n_eff": None, "min_n_eff": None})
        if info["status"] == "NO_CAUSAL_SIGNAL":
            diff = max(np.max(np.abs(m["lower"]-raw_m["lower"])), np.max(np.abs(m["upper"]-raw_m["upper"])))
            if diff > 1e-12: raise AssertionError(f"CausalOnly fallback mismatch: {diff}")
    return rows, inf_rows, barrier


def save_analysis_checkpoint(s15, index: int, seed: int, scenario: str, info: dict, frozen: dict,
                             result_rows: list[dict], inf_rows: list[dict], barrier: dict) -> None:
    base = scenario_dir(index, seed, scenario)
    grid = pd.DataFrame([{**{"replicate_index": index, "generation_seed": seed, "scenario": scenario}, **r} for r in frozen["grid_rows"]])
    selected = pd.DataFrame([{**{"replicate_index": index, "generation_seed": seed, "scenario": scenario},
        **frozen["selections"][(track, method)]} for track in TRACKS for method in ["GeoCP", *JOINT_METHODS]])
    write_csv(grid, base/"validation_grid_results.csv")
    write_csv(selected, base/"selected_hyperparameters.csv")
    write_csv(pd.DataFrame(result_rows), base/"replicate_results.csv")
    write_csv(pd.DataFrame(inf_rows), base/"infinite_interval_diagnostics.csv")
    write_csv(pd.DataFrame([{**{"replicate_index": index, "generation_seed": seed, "scenario": scenario}, **r}
                            for r in frozen["causal_only_krows"]]), base/"causal_only_k_tuning.csv")
    hashes = {p.name: sha(p) for p in [base/"validation_grid_results.csv", base/"selected_hyperparameters.csv",
        base/"replicate_results.csv", base/"infinite_interval_diagnostics.csv", base/"causal_only_k_tuning.csv",
        base/"test_unseal_audit.json"]}
    dump(base/"ANALYSIS_COMPLETE.json", {"replicate_index": index, "generation_seed": seed, "scenario": scenario,
        "frozen_state_hash": frozen["hash"], "barrier_sha256": barrier["barrier_sha256"], "hashes": hashes,
        "cmi_status": info["status"]})


def process_analysis(s15, index: int, seed: int, scenario: str) -> dict:
    base = scenario_dir(index, seed, scenario); complete = base/"ANALYSIS_COMPLETE.json"
    if complete.exists():
        meta = json.loads(complete.read_text())
        for name, expected in meta["hashes"].items():
            if sha(base/name) != expected: raise AssertionError(f"analysis checkpoint mismatch {base/name}")
        return {"replicate_index": index, "scenario": scenario, "checkpoint": True, "cmi_status": meta["cmi_status"]}
    _, info = cmi_info(base/"cmi_full999_results.csv")
    frozen = freeze_scenario(s15, index, seed, scenario, info)
    result_rows, inf_rows, barrier = evaluate_scenario(s15, index, seed, scenario, info, frozen)
    save_analysis_checkpoint(s15, index, seed, scenario, info, frozen, result_rows, inf_rows, barrier)
    return {"replicate_index": index, "scenario": scenario, "checkpoint": False, "cmi_status": info["status"]}


def concat_local(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True, sort=False)


def main() -> None:
    started = time.time(); seal, seeds = verify_prereg()
    RECORD.mkdir(parents=True, exist_ok=True)
    (RECORD/"run_log.txt").write_text("Stage-19B execution started after preregistration seal verification.\n", encoding="utf-8")
    (RECORD/"stderr.txt").write_text("", encoding="utf-8")
    s15 = load_module("stage15_reused_19b", STAGE15); s15.RECORD = RECORD
    generator = s15.load_module("stage19b_generator", GENERATOR_SOURCE)
    predictor = s15.load_module("stage19b_predictor", PREDICTOR_SOURCE)
    gen_cfg = json.loads(GENERATOR_CONFIG.read_text()); pred_cfg = json.loads(PREDICTOR_CONFIG.read_text())
    if pred_cfg["model_configuration"]["random_state"] != 420907 or pred_cfg["authorized_features"] != s15.FEATURES_ALL:
        raise AssertionError("predictor protocol changed")
    dump(RECORD/"execution_config.json", {"preregistration_composite_sha256": seal["composite_sha256"],
        "official_commit": COMMIT, "implementation": str(Path(__file__).relative_to(ROOT)).replace("\\","/"),
        "implementation_sha256": sha(Path(__file__)), "cmi_worker_sha256": sha(CMI_WORKER),
        "stage15_reused_sha256": sha(STAGE15), "official_utils_sha256": sha(OFFICIAL/"geoconformal/geocp/utils.py"),
        "official_weights_sha256": sha(OFFICIAL/"geoconformal/geocp/weights.py")})

    # First-pair preflight before generation of the remaining 49 pairs.
    first = s15.generate_replicate(1, seeds[0], generator, gen_cfg, predictor, pred_cfg)
    cmi_determinism, freeze_hashes, runtimes = {}, {}, []
    if (RECORD/"preflight_summary.json").exists():
        preflight = json.loads((RECORD/"preflight_summary.json").read_text())
        if preflight.get("status") != "PASS": raise AssertionError("invalid preflight checkpoint")
        print("PREFLIGHT CHECKPOINT PASS", flush=True)
    else:
        generator_probe = s15.generate_reproducibility_probe(generator, gen_cfg, seeds[0])
        predictor_probe = {sc: s15.predictor_determinism_probe(1, seeds[0], sc, predictor, pred_cfg) for sc in SCENARIOS}
        for scenario in SCENARIOS:
            base = scenario_dir(1, seeds[0], scenario); out = base/"cmi_full999_results.csv"
            runtimes.append({"replicate_index": 1, "scenario": scenario, **run_cmi(s15, 1, seeds[0], scenario, out)})
            frame, info = cmi_info(out)
            with tempfile.TemporaryDirectory(prefix="stage19b_cmi_rerun_", dir=RECORD) as td:
                out2 = Path(td)/"rerun.csv"; run_cmi(s15, 1, seeds[0], scenario, out2)
                cols = ["feature","raw_cmi","p_value","bh_q","eligible","weight","status","rank"]
                pd.testing.assert_frame_equal(frame[cols], pd.read_csv(out2)[cols], check_exact=True)
            cmi_determinism[scenario] = "bit-identical"
            a = freeze_scenario(s15, 1, seeds[0], scenario, info)
            b = freeze_scenario(s15, 1, seeds[0], scenario, info)
            if a["hash"] != b["hash"]: raise AssertionError("freeze nondeterminism")
            freeze_hashes[scenario] = a["hash"]
            rr, ir, barrier = evaluate_scenario(s15, 1, seeds[0], scenario, info, a)
            save_analysis_checkpoint(s15, 1, seeds[0], scenario, info, a, rr, ir, barrier)
        dump(RECORD/"preflight_summary.json", {"status": "PASS", "seed": seeds[0], "generator": generator_probe,
            "predictor": predictor_probe, "cmi_clean_rerun": cmi_determinism, "analysis_clean_rerun_hashes": freeze_hashes,
            "official_commit": COMMIT, "coordinate_leakage": "PASS", "track_A_B_atom_settings": "PASS",
            "test_seal": "PASS", "fallback_equality": "enforced during evaluation"})
        print("PREFLIGHT PASS", flush=True)

    generator_rows = [x[0] for x in first]; split_rows = [x[1] for x in first]; prediction_rows = [x[2] for x in first]
    for index, seed in enumerate(seeds[1:], 2):
        rows = s15.generate_replicate(index, seed, generator, gen_cfg, predictor, pred_cfg)
        generator_rows += [x[0] for x in rows]; split_rows += [x[1] for x in rows]; prediction_rows += [x[2] for x in rows]
        if index % 5 == 0: print(f"GENERATION {index}/50", flush=True)
    write_csv(pd.DataFrame(generator_rows), RECORD/"generator_audit.csv")
    write_csv(pd.DataFrame(split_rows), RECORD/"split_manifest.csv")
    write_csv(pd.DataFrame(prediction_rows), RECORD/"prediction_manifest.csv")

    jobs = []
    for index, seed in enumerate(seeds, 1):
        for scenario in SCENARIOS:
            out = scenario_dir(index, seed, scenario)/"cmi_full999_results.csv"
            if not (out.exists() and out.with_suffix(".json").exists()): jobs.append((index, seed, scenario, out))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_cmi, s15, i, seed, sc, out):(i,sc) for i,seed,sc,out in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            i, sc = futures[future]; runtimes.append({"replicate_index": i, "scenario": sc, **future.result()})
            if done % 10 == 0: print(f"CMI {done}/{len(jobs)}", flush=True)

    cmi_rows, weight_rows = [], []
    for index, seed in enumerate(seeds, 1):
        for scenario in SCENARIOS:
            frame, info = cmi_info(scenario_dir(index,seed,scenario)/"cmi_full999_results.csv")
            for row in frame.to_dict("records"): cmi_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,**row})
            for f in FEATURES: weight_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,
                "feature":f,"selected":f in info["selected"],"weight":info["weights"][f],"status":info["status"]})
    write_csv(pd.DataFrame(cmi_rows), RECORD/"cmi_full999_results.csv")
    write_csv(pd.DataFrame(weight_rows), RECORD/"causal_feature_weight_audit.csv")
    runtimes = []
    for index, seed in enumerate(seeds, 1):
        for scenario in SCENARIOS:
            meta = json.loads((scenario_dir(index,seed,scenario)/"cmi_full999_results.json").read_text())
            runtimes.append({"replicate_index":index,"scenario":scenario,"runtime_seconds":meta["runtime_seconds"]})
    dump(RECORD/"cmi_runtime.json", {"jobs": runtimes, "total_reported_seconds": sum(x["runtime_seconds"] for x in runtimes)})

    analysis_jobs = [(i, seed, sc) for i,seed in enumerate(seeds,1) for sc in SCENARIOS
                     if not (scenario_dir(i,seed,sc)/"ANALYSIS_COMPLETE.json").exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(process_analysis, s15, *job) for job in analysis_jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            if done % 5 == 0: print(f"OFFICIAL ANALYSIS {done}/{len(analysis_jobs)} latest={result['replicate_index']}/{result['scenario']}", flush=True)

    bases = [scenario_dir(i,seed,sc) for i,seed in enumerate(seeds,1) for sc in SCENARIOS]
    grid = concat_local([b/"validation_grid_results.csv" for b in bases])
    selected = concat_local([b/"selected_hyperparameters.csv" for b in bases])
    results = concat_local([b/"replicate_results.csv" for b in bases])
    inf = concat_local([b/"infinite_interval_diagnostics.csv" for b in bases])
    barriers = pd.DataFrame([json.loads((b/"test_unseal_audit.json").read_text()) for b in bases])
    for track in TRACKS:
        write_csv(grid[grid.track.eq(track)].copy(), RECORD/f"validation_grid_results_track{track}.csv")
        write_csv(selected[selected.track.eq(track)].copy(), RECORD/f"selected_hyperparameters_track{track}.csv")
        write_csv(results[results.track.eq(track)].copy(), RECORD/f"replicate_results_track{track}.csv")
    write_csv(selected[selected.status.eq("NO_FEASIBLE_HYPERPARAMETER")].copy(), RECORD/"no_feasible_summary.csv")
    write_csv(barriers, RECORD/"test_unseal_audit.csv")
    write_csv(inf, RECORD/"infinite_interval_diagnostics_trackB.csv")
    fallback_rows = []
    metric_cols = ["coverage","mean_width","median_width","mean_interval_score","infinite_count","infinite_fraction",
                   "mean_width_finite","median_width_finite","mean_n_eff","median_n_eff","min_n_eff"]
    for (index, seed, scenario, track), group in results.groupby(["replicate_index","generation_seed","scenario","track"], sort=True):
        causal = group[group.method.eq("GeoCausalCP-Full")]
        if causal.empty or str(causal.iloc[0].get("cmi_status")) != "NO_CAUSAL_SIGNAL": continue
        geo = group[group.method.eq("GeoCP")].iloc[0]; causal = causal.iloc[0]
        param_equal = geo["selection_status"] == causal["selection_status"] and geo["bandwidth"] == causal["bandwidth"]
        values_equal = all((pd.isna(geo[c]) and pd.isna(causal[c])) or geo[c] == causal[c] for c in metric_cols)
        fallback_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,"track":track,
            "selection_status_equal":geo["selection_status"]==causal["selection_status"],
            "bandwidth_equal":geo["bandwidth"]==causal["bandwidth"],"all_reported_metrics_equal":values_equal,
            "exact_threshold_branch":"same selected GeoCP computation; per-partition np.array_equal assertion",
            "status":"PASS" if param_equal and values_equal else "FAIL"})
    fallback = pd.DataFrame(fallback_rows)
    if not fallback.empty and not fallback.status.eq("PASS").all(): raise AssertionError("consolidated fallback audit")
    write_csv(fallback, RECORD/"no_signal_fallback_equality_audit.csv")
    dump(RECORD/"scientific_computation_checkpoint.json", {"status":"COMPLETE", "scenario_replicates":len(bases),
        "paired_replicates":50, "analysis_checkpoints":sum((b/"ANALYSIS_COMPLETE.json").exists() for b in bases),
        "elapsed_seconds":time.time()-started, "preregistration_composite_sha256":seal["composite_sha256"]})
    with (RECORD/"run_log.txt").open("a",encoding="utf-8") as f:
        f.write(f"Scientific computation complete: 100 scenario-replicates; elapsed_seconds={time.time()-started:.3f}\n")
    print("STAGE19B SCIENTIFIC COMPUTATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
