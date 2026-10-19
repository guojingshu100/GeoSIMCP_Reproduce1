"""Stage 11 v2: frozen CausalOnly-CMI normalized split conformal prediction.

Only the input plumbing differs from v1: this runner consumes the exact
Stage-11A recovered split tables. It never reruns CMI, refits predictors, or
regenerates simulations.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECOVERED = HERE / "stage11_frozen_split_inputs_v1"
STAGE11A_RECORD = ROOT / "experiments" / "causal_only_cp" / "20260918_stage11a_frozen_input_recovery_v1"
RECORD = ROOT / "experiments" / "causal_only_cp" / "20260918_stage11_causalonly_cmi_normcp_v2"
STAGE9_WEIGHTS = ROOT / "causal_strength_diagnostic_v1" / "stage9_provisional_weights.csv"
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]
SCENARIOS = ["S2", "S4"]
METHODS = ["Raw-CP", "GeoOnly-NormCP", "CausalOnly-CMI-NormCP", "Oracle-NormCP"]
K_VALUES = [20, 40, 60, 80]
ALPHA = 0.10
FLOOR = 1e-12
TIE_TOL = 1e-12
META = ["sample_id", "scenario", "replicate_id", "seed", "split", "local_calibration_role", "split_role"]
UNSEAL_COLUMNS = ["phase", "replicate_index", "generation_seed", "scenario", "selected_features", "weights_json",
                  "causal_selected_k", "geo_selected_k", "q_raw", "q_causal", "q_geo", "q_oracle",
                  "scale_fit_hash", "tuning_hash", "final_calibration_hash", "barrier_row_sha256"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def provenance_gate() -> tuple[list[dict], dict]:
    summary = json.loads((STAGE11A_RECORD / "stage11a_summary.json").read_text(encoding="utf-8"))
    if summary["readiness"] != "READY_FOR_STAGE11_RERUN" or summary["paired_scenario_count"] != 40:
        raise AssertionError("Stage-11A readiness gate failed")
    expected = {"train": 1000, "scale_fit": 200, "tuning_calibration": 100,
                "validation": 200, "final_calibration": 200, "test": 300}
    if summary["split_counts"] != expected:
        raise AssertionError("Stage-11A split-count gate failed")
    output_manifest = json.loads((STAGE11A_RECORD / "output_manifest.json").read_text(encoding="utf-8"))
    for item in output_manifest["record_files"]:
        path = STAGE11A_RECORD / item["path"]
        if not path.is_file():
            raise AssertionError(f"Stage-11A record missing: {path}")
        actual = sha256(path)
        if actual != item["sha256"]:
            # The immutable Stage-11A manifest has one documented transcription
            # defect: oracle_sigma_audit.csv is missing its final hex character.
            # Accept only that exact malformed-prefix case and require identity
            # with the canonical recovery copy; no scientific hash is relaxed.
            canonical = RECOVERED / item["path"]
            typo_is_proven = (item["path"] == "oracle_sigma_audit.csv" and len(item["sha256"]) == 63
                              and actual.startswith(item["sha256"]) and canonical.is_file()
                              and sha256(canonical) == actual)
            if not typo_is_proven:
                raise AssertionError(f"Stage-11A record hash mismatch: {path}")
    for name, rows in (("frozen_input_equality_audit.csv", 80), ("prediction_join_audit.csv", 40),
                       ("split_integrity.csv", 40), ("oracle_sigma_audit.csv", 40)):
        frame = pd.read_csv(STAGE11A_RECORD / name)
        if len(frame) != rows or not frame["status"].eq("PASS").all():
            raise AssertionError(f"Stage-11A audit failed: {name}")
    equality = pd.read_csv(STAGE11A_RECORD / "frozen_input_equality_audit.csv")
    if float(equality["max_abs_diff"].max()) != 0.0 or int(equality["unequal_values"].sum()) != 0:
        raise AssertionError("Stage-11A exact equality gate failed")
    loader_repro = json.loads((STAGE11A_RECORD / "loader_reproducibility.json").read_text(encoding="utf-8"))
    if loader_repro["status"] != "PASS" or not loader_repro["all_first_second_hashes_equal"]:
        raise AssertionError("Stage-11A loader reproducibility gate failed")
    inventory = pd.read_csv(STAGE11A_RECORD / "split_inventory.csv")
    if len(inventory) != 240:
        raise AssertionError("Stage-11A inventory row count mismatch")
    seeds = summary["seeds"]
    if len(seeds) != 20 or seeds != list(range(20270908, 20270928)):
        raise AssertionError("unexpected frozen seed list")
    for index, seed in enumerate(seeds, 1):
        for scenario in SCENARIOS:
            group = inventory[(inventory.replicate_index == index) & (inventory.scenario == scenario)]
            counts = dict(zip(group.split_role, group.rows))
            if counts != expected:
                raise AssertionError(f"inventory mismatch: {index}/{scenario}")
            for _, row in group.drop_duplicates("path").iterrows():
                path = ROOT / row.path
                if sha256(path) != row.sha256:
                    raise AssertionError(f"recovered input hash mismatch: {path}")
    return [{"replicate_index": i, "generation_seed": seed} for i, seed in enumerate(seeds, 1)], summary


def frozen_weights() -> tuple[dict[tuple[int, str], dict], pd.DataFrame]:
    frame = pd.read_csv(STAGE9_WEIGHTS)
    frame = frame[(frame["condition"] == "Mspace") &
                  (frame["transform"] == "bh_gated_raw_positive")].copy()
    if len(frame) != 160 or set(frame.feature) != set(FEATURES):
        raise AssertionError("Stage-9 weight artifact mismatch")
    frame["stage11_status"] = frame.status.replace({"NO_SIGNAL": "NO_CAUSAL_SIGNAL"})
    result = {}
    for (rep, scenario), group in frame.groupby(["replicate_index", "scenario"], sort=True):
        weights = {row.feature: float(row.weight) for row in group.itertuples()}
        eligible = [f for f in FEATURES if weights[f] > 0]
        status = str(group.stage11_status.iloc[0])
        if status == "OK" and not np.isclose(sum(weights.values()), 1.0, atol=1e-15, rtol=0):
            raise AssertionError("weight normalization failure")
        if status == "NO_CAUSAL_SIGNAL" and any(weights.values()):
            raise AssertionError("nonzero no-signal weight")
        result[(int(rep), scenario)] = {"weights": weights, "selected": eligible, "status": status}
    signal = frame.groupby(["scenario", "replicate_index"]).stage11_status.first()
    s2 = frame[frame.scenario == "S2"].pivot(index="replicate_index", columns="feature", values="weight")[FEATURES]
    verified = {
        "S2_no_signal": int((signal.loc["S2"] == "NO_CAUSAL_SIGNAL").sum()),
        "S4_no_signal": int((signal.loc["S4"] == "NO_CAUSAL_SIGNAL").sum()),
        "S2_x2_selected": int((s2.x2 > 0).sum()),
        "S2_x2_largest": int(((s2.x2 > 0) & s2.x2.eq(s2.max(axis=1))).sum()),
    }
    if verified != {"S2_no_signal": 7, "S4_no_signal": 19, "S2_x2_selected": 13, "S2_x2_largest": 12}:
        raise AssertionError(f"Stage-9 context mismatch: {verified}")
    return result, frame


def standardizer(reference: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mean = reference[columns].to_numpy(float).mean(axis=0)
    sd = reference[columns].to_numpy(float).std(axis=0, ddof=1)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 1e-12):
        raise AssertionError("eligible feature has effectively zero scale-fit SD")
    return mean, sd


def scaled(frame: pd.DataFrame, columns: list[str], mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (frame[columns].to_numpy(float) - mean) / sd


def neighbor_scale(reference_z: np.ndarray, reference_a: np.ndarray, reference_ids: np.ndarray,
                   query_z: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    output = np.empty(len(query_z), dtype=float)
    for i, row in enumerate(query_z):
        distance = np.sqrt(np.sum(weights * (reference_z - row) ** 2, axis=1))
        order = np.lexsort((reference_ids, distance))
        output[i] = max(FLOOR, float(reference_a[order[:k]].mean()))
    return output


def tune_local(reference: pd.DataFrame, tuning: pd.DataFrame, columns: list[str], weights: np.ndarray,
               method: str, replicate: int, seed: int, scenario: str) -> tuple[int, dict, list[dict]]:
    mean, sd = standardizer(reference, columns)
    rz = scaled(reference, columns, mean, sd)
    tz = scaled(tuning, columns, mean, sd)
    residual_ref = np.abs(reference.y_true.to_numpy(float) - reference.y_pred.to_numpy(float))
    residual_tune = np.abs(tuning.y_true.to_numpy(float) - tuning.y_pred.to_numpy(float))
    rows, values = [], {}
    for k in K_VALUES:
        sigma = neighbor_scale(rz, residual_ref, reference.sample_id.to_numpy(str), tz, weights, k)
        mae = float(np.mean(np.abs(residual_tune - sigma)))
        values[k] = mae
    best = min(values.values())
    selected = max(k for k, value in values.items() if abs(value - best) <= TIE_TOL)
    for k in K_VALUES:
        rows.append({"replicate_index": replicate, "generation_seed": seed, "scenario": scenario,
                     "method": method, "k": k, "tuning_scale_mae": values[k], "selected": k == selected,
                     "reference_split": "scale_fit", "tuning_split": "tuning_calibration"})
    return selected, {"mean": mean, "sd": sd, "reference_z": rz, "weights": weights,
                      "reference_a": residual_ref, "reference_ids": reference.sample_id.to_numpy(str),
                      "tuning_mae": values[selected]}, rows


def conformal_quantile(scores: np.ndarray) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=float)
    rank = math.ceil((len(scores) + 1) * (1.0 - ALPHA))
    if rank != 181 or len(scores) != 200:
        raise AssertionError(f"unexpected conformal rank: n={len(scores)} rank={rank}")
    return float(np.sort(scores, kind="mergesort")[rank - 1]), rank


def metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    if not (np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower <= upper)):
        raise AssertionError("invalid interval bounds")
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width + (2 / ALPHA) * (lower - y) * (y < lower) + (2 / ALPHA) * (y - upper) * (y > upper)
    if not np.isfinite(score).all():
        raise AssertionError("nonfinite interval score")
    return {"coverage": float(covered.mean()), "covered_count": int(covered.sum()), "test_n": len(y),
            "mean_width": float(width.mean()), "median_width": float(np.median(width)),
            "mean_interval_score": float(score.mean()), "lower": lower, "upper": upper}


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.ptp(a) <= 1e-15 or np.ptp(b) <= 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def append_unseal(row: dict) -> None:
    path = RECORD / "test_unseal_audit.csv"
    base = {key: row[key] for key in UNSEAL_COLUMNS if key != "barrier_row_sha256"}
    row["barrier_row_sha256"] = hashlib.sha256(json_text(base).encode()).hexdigest()
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNSEAL_COLUMNS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_one(loader, replicate: int, seed: int, scenario: str, weight_info: dict, phase: str,
            write_barrier: bool = True) -> dict:
    # Test remains inaccessible through the loader throughout this block.
    scale = loader.load_split(replicate, seed, scenario, "scale_fit")
    tuning = loader.load_split(replicate, seed, scenario, "tuning_calibration")
    if len(scale) != 200 or len(tuning) != 100:
        raise AssertionError("development row-count mismatch")
    residual_scale = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
    residual_tune = np.abs(tuning.y_true.to_numpy(float) - tuning.y_pred.to_numpy(float))

    # Freeze practical scale models using scale_fit and tuning only.
    k_rows = []
    geo_k, geo_model, rows = tune_local(scale, tuning, COORDS, np.ones(2), "GeoOnly-NormCP", replicate, seed, scenario)
    k_rows.extend(rows)
    if weight_info["status"] == "NO_CAUSAL_SIGNAL":
        causal_k = None
        sigma_global = max(FLOOR, float(residual_scale.mean()))
        causal_model = {"constant": sigma_global, "tuning_mae": float(np.mean(np.abs(residual_tune - sigma_global)))}
        k_rows.append({"replicate_index": replicate, "generation_seed": seed, "scenario": scenario,
                       "method": "CausalOnly-CMI-NormCP", "k": None,
                       "tuning_scale_mae": causal_model["tuning_mae"], "selected": True,
                       "reference_split": "scale_fit", "tuning_split": "tuning_calibration"})
    else:
        selected = weight_info["selected"]
        if set(selected).intersection(COORDS):
            raise AssertionError("coordinate leakage into causal distance")
        weight_array = np.asarray([weight_info["weights"][f] for f in selected], dtype=float)
        causal_k, causal_model, rows = tune_local(scale, tuning, selected, weight_array,
                                                   "CausalOnly-CMI-NormCP", replicate, seed, scenario)
        k_rows.extend(rows)

    # Only now load final calibration. Validation is never loaded.
    final = loader.load_split(replicate, seed, scenario, "final_calibration")
    if len(final) != 200:
        raise AssertionError("final-calibration count mismatch")
    final_residual = np.abs(final.y_true.to_numpy(float) - final.y_pred.to_numpy(float))
    q_raw, rank = conformal_quantile(final_residual)

    geo_final_z = scaled(final, COORDS, geo_model["mean"], geo_model["sd"])
    geo_final_sigma = neighbor_scale(geo_model["reference_z"], geo_model["reference_a"],
                                     geo_model["reference_ids"], geo_final_z, geo_model["weights"], geo_k)
    q_geo, _ = conformal_quantile(final_residual / geo_final_sigma)
    if causal_k is None:
        causal_final_sigma = np.full(len(final), causal_model["constant"])
    else:
        cols = weight_info["selected"]
        causal_final_z = scaled(final, cols, causal_model["mean"], causal_model["sd"])
        causal_final_sigma = neighbor_scale(causal_model["reference_z"], causal_model["reference_a"],
                                            causal_model["reference_ids"], causal_final_z,
                                            causal_model["weights"], causal_k)
    q_causal, _ = conformal_quantile(final_residual / causal_final_sigma)
    oracle_final = final.oracle_sigma.to_numpy(float)
    if scenario == "S2" and np.unique(oracle_final).size <= 1:
        raise AssertionError("S2 oracle scale does not vary")
    if scenario == "S4" and not np.all(oracle_final == 1.3):
        raise AssertionError("S4 oracle scale is not exactly 1.3")
    q_oracle, _ = conformal_quantile(final_residual / oracle_final)

    # The immutable barrier is flushed before allow_test=True can be invoked.
    barrier = {"phase": phase, "replicate_index": replicate, "generation_seed": seed, "scenario": scenario,
               "selected_features": ";".join(weight_info["selected"]),
               "weights_json": json_text(weight_info["weights"]), "causal_selected_k": causal_k,
               "geo_selected_k": geo_k, "q_raw": q_raw, "q_causal": q_causal, "q_geo": q_geo,
               "q_oracle": q_oracle, "scale_fit_hash": frame_hash(scale), "tuning_hash": frame_hash(tuning),
               "final_calibration_hash": frame_hash(final)}
    if write_barrier:
        append_unseal(barrier)
    test = loader.load_split(replicate, seed, scenario, "test", allow_test=True)
    if len(test) != 300:
        raise AssertionError("test count mismatch")
    y, pred = test.y_true.to_numpy(float), test.y_pred.to_numpy(float)

    raw = metrics(y, pred - q_raw, pred + q_raw)
    geo_test_z = scaled(test, COORDS, geo_model["mean"], geo_model["sd"])
    geo_test_sigma = neighbor_scale(geo_model["reference_z"], geo_model["reference_a"], geo_model["reference_ids"],
                                    geo_test_z, geo_model["weights"], geo_k)
    geo = metrics(y, pred - q_geo * geo_test_sigma, pred + q_geo * geo_test_sigma)
    if causal_k is None:
        causal_test_sigma = np.full(len(test), causal_model["constant"])
    else:
        cols = weight_info["selected"]
        causal_test_z = scaled(test, cols, causal_model["mean"], causal_model["sd"])
        causal_test_sigma = neighbor_scale(causal_model["reference_z"], causal_model["reference_a"],
                                           causal_model["reference_ids"], causal_test_z,
                                           causal_model["weights"], causal_k)
    causal = metrics(y, pred - q_causal * causal_test_sigma, pred + q_causal * causal_test_sigma)
    oracle_test_sigma = test.oracle_sigma.to_numpy(float)
    oracle = metrics(y, pred - q_oracle * oracle_test_sigma, pred + q_oracle * oracle_test_sigma)
    fallback_diff = None
    if causal_k is None:
        fallback_diff = float(max(np.max(np.abs(causal["lower"] - raw["lower"])),
                                  np.max(np.abs(causal["upper"] - raw["upper"]))))
        if fallback_diff > 1e-12:
            raise AssertionError(f"NO_SIGNAL fallback differs from Raw-CP: {fallback_diff}")

    result_rows, scale_rows = [], []
    settings = {
        "Raw-CP": (raw, None, q_raw, None, None),
        "GeoOnly-NormCP": (geo, geo_k, q_geo, geo_model["tuning_mae"], geo_test_sigma),
        "CausalOnly-CMI-NormCP": (causal, causal_k, q_causal, causal_model["tuning_mae"], causal_test_sigma),
        "Oracle-NormCP": (oracle, None, q_oracle, None, oracle_test_sigma),
    }
    for method, (m, selected_k, q, tune_mae, sigma) in settings.items():
        result_rows.append({"replicate_index": replicate, "generation_seed": seed, "scenario": scenario,
                            "method": method, "coverage": m["coverage"], "covered_count": m["covered_count"],
                            "test_n": m["test_n"], "mean_width": m["mean_width"],
                            "median_width": m["median_width"], "mean_interval_score": m["mean_interval_score"],
                            "selected_k": selected_k, "q_norm": q,
                            "no_causal_signal": weight_info["status"] == "NO_CAUSAL_SIGNAL" if method == "CausalOnly-CMI-NormCP" else False,
                            "selected_feature_count": len(weight_info["selected"]) if method == "CausalOnly-CMI-NormCP" else 0,
                            "selected_features": ";".join(weight_info["selected"]) if method == "CausalOnly-CMI-NormCP" else "",
                            "feature_weights": json_text(weight_info["weights"]) if method == "CausalOnly-CMI-NormCP" else "{}",
                            "tuning_scale_mae": tune_mae})
        if sigma is not None:
            scale_rows.append({"replicate_index": replicate, "generation_seed": seed, "scenario": scenario,
                               "method": method, "tuning_scale_mae": tune_mae,
                               "test_oracle_correlation": safe_corr(np.asarray(sigma), oracle_test_sigma),
                               "test_oracle_mae": float(np.mean(np.abs(np.asarray(sigma) - oracle_test_sigma)))})
    interval_hashes = {method: hashlib.sha256(np.column_stack((settings[method][0]["lower"], settings[method][0]["upper"])).tobytes()).hexdigest() for method in METHODS}
    return {"result_rows": result_rows, "k_rows": k_rows, "scale_rows": scale_rows,
            "fallback_max_abs_bound_diff": fallback_diff, "rank": rank, "barrier": barrier,
            "interval_hashes": interval_hashes}


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, method), group in results.groupby(["scenario", "method"], sort=True):
        raw_mean = float(results[(results.scenario == scenario) & (results.method == "Raw-CP")].coverage.mean())
        mean_cov = float(group.coverage.mean())
        rows.append({"scenario": scenario, "method": method, "replicates": len(group),
                     "mean_coverage": mean_cov, "median_coverage": float(group.coverage.median()),
                     "coverage_sd": float(group.coverage.std(ddof=1)), "min_coverage": float(group.coverage.min()),
                     "max_coverage": float(group.coverage.max()), "pooled_coverage": float(group.covered_count.sum() / group.test_n.sum()),
                     "replicates_coverage_lt_0_90": int((group.coverage < .90).sum()),
                     "replicates_coverage_lt_0_89": int((group.coverage < .89).sum()),
                     "mean_width": float(group.mean_width.mean()),
                     "median_replicate_mean_width": float(group.mean_width.median()),
                     "sd_replicate_mean_width": float(group.mean_width.std(ddof=1)),
                     "mean_interval_score": float(group.mean_interval_score.mean()),
                     "median_interval_score": float(group.mean_interval_score.median()),
                     "material_undercoverage": bool(mean_cov < .89 or mean_cov < raw_mean - .01)})
    return pd.DataFrame(rows)


def paired(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        pivot = results[results.scenario == scenario].pivot(index="replicate_index", columns="method",
                                                             values=["coverage", "mean_width", "mean_interval_score"])
        for comparator in ("Raw-CP", "GeoOnly-NormCP"):
            for metric in ("coverage", "mean_width", "mean_interval_score"):
                diff = (pivot[metric]["CausalOnly-CMI-NormCP"] - pivot[metric][comparator]).to_numpy(float)
                mean, sd = float(diff.mean()), float(diff.std(ddof=1))
                half = float(stats.t.ppf(.975, len(diff) - 1) * sd / math.sqrt(len(diff)))
                rows.append({"scenario": scenario, "contrast": f"CausalOnly-CMI-NormCP minus {comparator}",
                             "metric": metric, "n_pairs": len(diff), "mean_difference": mean,
                             "median_difference": float(np.median(diff)), "sd_difference": sd,
                             "ci95_lower": mean - half, "ci95_upper": mean + half,
                             "ci_method": "Student-t CI for mean paired difference"})
    return pd.DataFrame(rows)


def main() -> None:
    generated = ["test_unseal_audit.csv", "replicate_results.csv", "weight_audit.csv", "k_tuning_results.csv",
                 "scale_diagnostics.csv", "scenario_summary.csv", "paired_comparisons.csv", "preflight_summary.json",
                 "stage11_summary.json"]
    if any((RECORD / name).exists() for name in generated):
        raise FileExistsError("refusing to overwrite Stage-11 v2 outputs")
    RECORD.mkdir(parents=True, exist_ok=True)
    reps, stage11a = provenance_gate()
    weights, weight_frame = frozen_weights()
    loader = load_module("stage11a_loader", RECOVERED / "loader.py")

    # One-replicate preflight and clean deterministic rerun.
    preflight_first, preflight_second = {}, {}
    for scenario in SCENARIOS:
        preflight_first[scenario] = run_one(loader, 1, 20270908, scenario, weights[(1, scenario)], "preflight_primary")
        preflight_second[scenario] = run_one(loader, 1, 20270908, scenario, weights[(1, scenario)], "preflight_clean_rerun")
        if preflight_first[scenario]["interval_hashes"] != preflight_second[scenario]["interval_hashes"]:
            raise AssertionError(f"preflight nondeterminism: {scenario}")
    preflight = {"status": "PASS", "replicate_index": 1, "generation_seed": 20270908,
                 "scenarios": {s: {"rank": preflight_first[s]["rank"],
                                   "interval_hashes": preflight_first[s]["interval_hashes"],
                                   "clean_rerun_identical": True,
                                   "fallback_max_abs_bound_diff": preflight_first[s]["fallback_max_abs_bound_diff"]}
                               for s in SCENARIOS},
                 "coordinate_leakage_audit": "PASS: causal columns are subset of x1,x2,g_proxy,x_noise",
                 "label_access_order": "PASS: tuning before final calibration; barrier flushed before test",
                 "validation_used": False}
    (RECORD / "preflight_summary.json").write_text(json.dumps(preflight, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    result_rows, k_rows, scale_rows, fallback = [], [], [], []
    for scenario in SCENARIOS:
        first = preflight_first[scenario]
        result_rows.extend(first["result_rows"]); k_rows.extend(first["k_rows"]); scale_rows.extend(first["scale_rows"])
        fallback.append({"replicate_index": 1, "scenario": scenario, "max_abs_bound_diff": first["fallback_max_abs_bound_diff"]})
    for rep in reps[1:]:
        index, seed = rep["replicate_index"], rep["generation_seed"]
        for scenario in SCENARIOS:
            outcome = run_one(loader, index, seed, scenario, weights[(index, scenario)], "production")
            result_rows.extend(outcome["result_rows"]); k_rows.extend(outcome["k_rows"]); scale_rows.extend(outcome["scale_rows"])
            fallback.append({"replicate_index": index, "scenario": scenario,
                             "max_abs_bound_diff": outcome["fallback_max_abs_bound_diff"]})

    results = pd.DataFrame(result_rows)
    if len(results) != 160:
        raise AssertionError("expected 160 method/replicate results")
    k_frame, scale_frame = pd.DataFrame(k_rows), pd.DataFrame(scale_rows)
    scenarios, pairs = summarize(results), paired(results)
    weight_frame.to_csv(RECORD / "weight_audit.csv", index=False, lineterminator="\n", float_format="%.17g")
    results.to_csv(RECORD / "replicate_results.csv", index=False, lineterminator="\n", float_format="%.17g")
    k_frame.to_csv(RECORD / "k_tuning_results.csv", index=False, lineterminator="\n", float_format="%.17g")
    scale_frame.to_csv(RECORD / "scale_diagnostics.csv", index=False, lineterminator="\n", float_format="%.17g")
    scenarios.to_csv(RECORD / "scenario_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pairs.to_csv(RECORD / "paired_comparisons.csv", index=False, lineterminator="\n", float_format="%.17g")
    summary = {"status": "PASS", "protocol": "unchanged Stage-11 v1 methodology", "input_gate": "PASS",
               "preflight": preflight, "replicate_results_rows": len(results), "scenario_summary": scenarios.to_dict("records"),
               "paired_comparisons": pairs.to_dict("records"),
               "k_distributions": {f"{s}/{m}": {str(k): int(v) for k, v in Counter(results[(results.scenario == s) & (results.method == m)].selected_k.dropna().astype(int)).items()}
                                   for s in SCENARIOS for m in ("CausalOnly-CMI-NormCP", "GeoOnly-NormCP")},
               "fallback_audit": fallback, "validation_used": False, "test_unseal_rows": int(sum(1 for _ in open(RECORD / "test_unseal_audit.csv", encoding="utf-8")) - 1),
               "gcmc_status": "SECONDARY / METHOD-DIAGNOSTIC ONLY",
               "gccm_status": "CausalOnly-GCCM remains a later required ablation"}
    (RECORD / "stage11_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "preflight": "PASS", "replicate_rows": len(results),
                      "unseal_rows": summary["test_unseal_rows"]}, indent=2))


if __name__ == "__main__":
    main()
