"""Stage 12: direct CausalOnly-GCCM normalized split-conformal ablation.

The GCCM estimator is the frozen historical point-graph implementation.  The
Stage-12 restriction is limited to the four eligible observed attributes;
coordinates construct the Delaunay graph but never enter the final distance.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORD = ROOT / "experiments" / "causal_only_cp" / "20260918_stage12_causalonly_gccm_ablation_v1"
RECOVERED = HERE / "stage11_frozen_split_inputs_v1"
STAGE11 = ROOT / "experiments" / "causal_only_cp" / "20260918_stage11_causalonly_cmi_normcp_v2"
STAGE11A = ROOT / "experiments" / "causal_only_cp" / "20260918_stage11a_frozen_input_recovery_v1"
PILOT = ROOT / "simulation_pilot_20260908"
HISTORICAL_OUTPUT = PILOT / "gccm_outputs"
HISTORICAL_RUNNER = PILOT / "gccm_runtime" / "run_gccm_scale_fit_v1.py"
HISTORICAL_WRAPPER = PILOT / "gccm_runtime" / "gccm_point_graph_wrapper_v1.R"
HISTORICAL_CONFIG = PILOT / "gccm_runtime" / "gccm_runtime_config_v1.json"
CORE = ROOT / "geoconformal-main" / "GCCP_Seattle" / "02_gccm" / "GCCM_updated 2" / "R" / "GCCM" / "GCCM4Lattice.r"
WRAPPER = HERE / "gccm_stage12_attribute_wrapper_v1.R"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")

FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]
SCENARIOS = ["S2", "S4"]
K_VALUES = [20, 40, 60, 80]
ALPHA = 0.10
FLOOR = 1e-12
TIE_TOL = 1e-12
CORE_HASH = "3552d871539dc682519da04d07306a97e9138f30b22ecdbfcc2cf61efa38b12d"
PARAMETERS = {"E": 3, "tau": 1, "b": 4, "seed": 42,
              "direction": "effect_xmap_cause", "library_sizes": [10, 18, 33, 60, 109, 198]}
UNSEAL_FIELDS = ["replicate_index", "generation_seed", "scenario", "selected_features", "weights_json",
                 "selected_k", "q_norm", "scale_fit_hash", "tuning_hash", "final_calibration_hash",
                 "barrier_row_sha256"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    data = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
    return hashlib.sha256(data).hexdigest()


def jdump(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def provenance_gate() -> tuple[list[dict], object]:
    summary = json.loads((STAGE11A / "stage11a_summary.json").read_text(encoding="utf-8"))
    if summary["readiness"] != "READY_FOR_STAGE11_RERUN" or summary["paired_scenario_count"] != 40:
        raise AssertionError("Stage-11A gate failed")
    for name, n in (("frozen_input_equality_audit.csv", 80), ("prediction_join_audit.csv", 40),
                    ("split_integrity.csv", 40)):
        f = pd.read_csv(STAGE11A / name)
        if len(f) != n or not f.status.eq("PASS").all():
            raise AssertionError(f"Stage-11A audit failure: {name}")
    loader_repro = json.loads((STAGE11A / "loader_reproducibility.json").read_text(encoding="utf-8"))
    if loader_repro["status"] != "PASS" or not loader_repro["all_first_second_hashes_equal"]:
        raise AssertionError("Stage-11A loader gate failed")
    seeds = summary["seeds"]
    if seeds != list(range(20270908, 20270928)):
        raise AssertionError("seed mismatch")
    loader = load_module("stage11a_loader_for_stage12", RECOVERED / "loader.py")
    return [{"replicate_index": i, "generation_seed": s} for i, s in enumerate(seeds, 1)], loader


def write_frozen_config() -> None:
    config = {
        "stage": "12", "readiness_class": "READY_RECOMPUTE_EXACT_ORIGINAL_GCCM_PROTOCOL",
        "scenarios": SCENARIOS, "seeds": list(range(20270908, 20270928)),
        "target": "abs(y_true-y_pred_frozen) on scale_fit only", "candidate_attributes": FEATURES,
        "coordinate_use": "Delaunay graph construction only; prohibited from final distance",
        "gccm": PARAMETERS, "convergence": {"gain_gt": 0, "spearman_gte": .60,
            "terminal_pearson_two_sided_p_lt": .05, "fisher_z_95_ci_lower_gt": 0},
        "strength": "terminal mean rho iff all convergence gates pass; otherwise zero",
        "weight": "positive usable attribute strength / sum usable strengths over four allowed attributes",
        "fallback": "NO_USABLE_GCCM_SIGNAL -> constant mean scale_fit absolute residual",
        "normcp": {"k_candidates": K_VALUES, "selection": "tuning-calibration scale MAE; larger-k tie",
                   "alpha": ALPHA, "final_calibration_n": 200, "rank": 181, "test_n": 300},
        "frozen_before_feature_results": True,
        "source_hashes": {"historical_core": sha256(CORE), "historical_runner": sha256(HISTORICAL_RUNNER),
                          "historical_wrapper": sha256(HISTORICAL_WRAPPER), "stage12_wrapper": sha256(WRAPPER)}
    }
    (RECORD / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def artifact_inventory(reps: list[dict]) -> pd.DataFrame:
    rows = []
    for rep in reps:
        for scenario in SCENARIOS:
            old = HISTORICAL_OUTPUT / scenario
            paths = [old / "weights.csv", old / "restricted_input.csv", old / "execution_config.json"]
            rows.append({"replicate_index": rep["replicate_index"], "seed": rep["generation_seed"],
                         "scenario": scenario, "available_seed_matched_gccm_artifact": False,
                         "available_historical_pilot_artifact": str(old.relative_to(ROOT)),
                         "historical_pilot_seed": 20260907, "input_rows": 200,
                         "target_definition": "abs_residual = abs(y-y_pred_frozen)",
                         "candidate_variables_historical": "s_x;s_y;x1;x2;g_proxy;x_noise",
                         "stage12_allowed_attributes": ";".join(FEATURES), "E": 3, "tau": 1, "b": 4,
                         "library_sizes": "10;18;33;60;109;198", "gccm_seed": 42,
                         "weight_fields": "feature;strength;weight;converged;gamma;status",
                         "weights_sha256": sha256(paths[0]), "restricted_input_sha256": sha256(paths[1]),
                         "execution_config_sha256": sha256(paths[2]),
                         "compatibility": "target/split/protocol compatible; realization not seed-matched; recomputation required"})
    return pd.DataFrame(rows)


def prepare_gccm_input(loader, rep: int, seed: int, scenario: str, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=False)
    frame = loader.load_split(rep, seed, scenario, "scale_fit").sort_values("sample_id").reset_index(drop=True)
    if len(frame) != 200 or not frame.sample_id.is_unique or set(frame.split_role) != {"scale_fit"}:
        raise AssertionError("invalid scale_fit input")
    residual = np.abs(frame.y_true.to_numpy(float) - frame.y_pred.to_numpy(float))
    x = frame[FEATURES].to_numpy(float)
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
    if np.any(sd <= 1e-12):
        raise AssertionError("zero feature SD")
    z = (x - mean) / sd
    xy = frame[COORDS].to_numpy(float)
    tri = Delaunay(xy)
    edges = sorted({tuple(sorted((int(t[i]) + 1, int(t[j]) + 1)))
                    for t in tri.simplices for i, j in ((0, 1), (1, 2), (0, 2))})
    inp = pd.DataFrame({"node_id": np.arange(1, 201), "sample_id": frame.sample_id,
                        "s_x": xy[:, 0], "s_y": xy[:, 1], "abs_residual": residual})
    for j, feature in enumerate(FEATURES):
        inp["z_" + feature] = z[:, j]
    inp.to_csv(out / "restricted_input.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(edges, columns=["u", "v"]).to_csv(out / "edges.csv", index=False, lineterminator="\n")
    shutil.copyfile(CORE, out / "historical_core.r")
    params = [("E", "3"), ("tau", "1"), ("b", "4"), ("seed", "42"),
              ("direction", "effect_xmap_cause"), ("library_sizes", "10;18;33;60;109;198")]
    pd.DataFrame(params, columns=["name", "value"]).to_csv(out / "parameters.csv", index=False, lineterminator="\n")
    return inp


def invoke_gccm(out: Path) -> dict:
    cmd = [str(RSCRIPT), "--vanilla", str(WRAPPER), str(out / "restricted_input.csv"),
           str(out / "edges.csv"), str(out / "historical_core.r"), str(out / "parameters.csv"), str(out), "estimate"]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=out, env={**os.environ, "LC_ALL": "C"}, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=1800)
    (out / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"GCCM failed in {out}: {proc.stderr}")
    weights = pd.read_csv(out / "weights.csv", keep_default_na=False)
    raw = pd.read_csv(out / "raw_curves.csv")
    if len(raw) != 3964 or weights.feature.tolist() != FEATURES:
        raise AssertionError("GCCM output schema/count mismatch")
    numeric = pd.to_numeric(weights.weight, errors="coerce")
    status = str(weights.status.iloc[0])
    if status == "WEIGHTS_AVAILABLE" and not np.isclose(numeric.sum(), 1, atol=1e-12, rtol=0):
        raise AssertionError("GCCM weights not normalized")
    if status == "NO_CONVERGED_FEATURES" and numeric.notna().any():
        raise AssertionError("unexpected all-zero fallback weight")
    meta = {"command": cmd, "returncode": proc.returncode, "runtime_seconds": time.perf_counter() - start,
            "hashes": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}}
    (out / "execution_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def read_weight_info(out: Path) -> tuple[dict, list[dict]]:
    weights = pd.read_csv(out / "weights.csv", keep_default_na=False)
    status = str(weights.status.iloc[0])
    if status == "NO_CONVERGED_FEATURES":
        mapping = {f: 0.0 for f in FEATURES}; selected = []
        stage_status = "NO_USABLE_GCCM_SIGNAL"
    else:
        mapping = {r.feature: float(r.weight) for r in weights.itertuples()}
        selected = [f for f in FEATURES if mapping[f] > 0]
        stage_status = "WEIGHTS_AVAILABLE"
    return {"weights": mapping, "selected": selected, "status": stage_status}, weights.to_dict("records")


def standardizer(reference: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = reference[columns].to_numpy(float)
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    if np.any(sd <= 1e-12):
        raise AssertionError("zero localization feature SD")
    return mean, sd


def scaled(frame: pd.DataFrame, columns: list[str], mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (frame[columns].to_numpy(float) - mean) / sd


def neighbor_scale(ref_z, ref_a, ref_ids, query_z, weights, k):
    answer = np.empty(len(query_z))
    for i, row in enumerate(query_z):
        distance = np.sqrt(np.sum(weights * (ref_z - row) ** 2, axis=1))
        order = np.lexsort((ref_ids, distance))
        answer[i] = max(FLOOR, float(ref_a[order[:k]].mean()))
    return answer


def tune(reference, tuning, columns, weights):
    mean, sd = standardizer(reference, columns)
    rz, tz = scaled(reference, columns, mean, sd), scaled(tuning, columns, mean, sd)
    ra = np.abs(reference.y_true.to_numpy(float) - reference.y_pred.to_numpy(float))
    ta = np.abs(tuning.y_true.to_numpy(float) - tuning.y_pred.to_numpy(float))
    values = {}
    for k in K_VALUES:
        sig = neighbor_scale(rz, ra, reference.sample_id.to_numpy(str), tz, weights, k)
        values[k] = float(np.mean(np.abs(ta - sig)))
    best = min(values.values())
    chosen = max(k for k, v in values.items() if abs(v - best) <= TIE_TOL)
    return chosen, {"mean": mean, "sd": sd, "rz": rz, "ra": ra,
                    "ids": reference.sample_id.to_numpy(str), "weights": weights,
                    "tuning_mae": values[chosen]}, values


def conformal_quantile(scores):
    rank = math.ceil((len(scores) + 1) * (1 - ALPHA))
    if len(scores) != 200 or rank != 181:
        raise AssertionError("finite-sample rank mismatch")
    return float(np.sort(np.asarray(scores), kind="mergesort")[rank - 1])


def interval_metrics(y, pred, q, sigma):
    lower, upper = pred - q * sigma, pred + q * sigma
    if not (np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower <= upper)):
        raise AssertionError("invalid intervals")
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width + 20 * (lower - y) * (y < lower) + 20 * (y - upper) * (y > upper)
    return {"coverage": float(covered.mean()), "covered_count": int(covered.sum()), "test_n": len(y),
            "mean_width": float(width.mean()), "median_width": float(np.median(width)),
            "mean_interval_score": float(score.mean()), "lower": lower, "upper": upper}


def append_unseal(row: dict) -> None:
    base = {k: row[k] for k in UNSEAL_FIELDS if k != "barrier_row_sha256"}
    row["barrier_row_sha256"] = hashlib.sha256(jdump(base).encode()).hexdigest()
    path = RECORD / "test_unseal_audit.csv"; exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNSEAL_FIELDS, lineterminator="\n")
        if not exists: writer.writeheader()
        writer.writerow(row); handle.flush(); os.fsync(handle.fileno())


def run_cp(loader, rep, seed, scenario, info, write_barrier=True):
    scale = loader.load_split(rep, seed, scenario, "scale_fit")
    tuning = loader.load_split(rep, seed, scenario, "tuning_calibration")
    residual_scale = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
    residual_tune = np.abs(tuning.y_true.to_numpy(float) - tuning.y_pred.to_numpy(float))
    if info["status"] == "NO_USABLE_GCCM_SIGNAL":
        selected_k = None; constant = max(FLOOR, float(residual_scale.mean()))
        model = {"constant": constant, "tuning_mae": float(np.mean(np.abs(residual_tune - constant)))}
        k_values = {None: model["tuning_mae"]}
    else:
        if set(info["selected"]) & set(COORDS):
            raise AssertionError("coordinate leakage")
        weights = np.array([info["weights"][f] for f in info["selected"]])
        selected_k, model, k_values = tune(scale, tuning, info["selected"], weights)
    final = loader.load_split(rep, seed, scenario, "final_calibration")
    final_a = np.abs(final.y_true.to_numpy(float) - final.y_pred.to_numpy(float))
    if selected_k is None:
        final_sigma = np.full(200, model["constant"])
    else:
        final_z = scaled(final, info["selected"], model["mean"], model["sd"])
        final_sigma = neighbor_scale(model["rz"], model["ra"], model["ids"], final_z, model["weights"], selected_k)
    q = conformal_quantile(final_a / final_sigma)
    barrier = {"replicate_index": rep, "generation_seed": seed, "scenario": scenario,
               "selected_features": ";".join(info["selected"]), "weights_json": jdump(info["weights"]),
               "selected_k": selected_k, "q_norm": q, "scale_fit_hash": frame_hash(scale),
               "tuning_hash": frame_hash(tuning), "final_calibration_hash": frame_hash(final)}
    if write_barrier: append_unseal(barrier)
    test = loader.load_split(rep, seed, scenario, "test", allow_test=True)
    if selected_k is None:
        test_sigma = np.full(300, model["constant"])
    else:
        test_z = scaled(test, info["selected"], model["mean"], model["sd"])
        test_sigma = neighbor_scale(model["rz"], model["ra"], model["ids"], test_z, model["weights"], selected_k)
    y, pred = test.y_true.to_numpy(float), test.y_pred.to_numpy(float)
    result = interval_metrics(y, pred, q, test_sigma)
    raw_q = conformal_quantile(final_a)
    raw = interval_metrics(y, pred, raw_q, np.ones(300))
    fallback_diff = None
    if selected_k is None:
        fallback_diff = float(max(np.max(np.abs(result["lower"] - raw["lower"])),
                                  np.max(np.abs(result["upper"] - raw["upper"]))))
        if fallback_diff > 1e-12:
            raise AssertionError(f"fallback mismatch: {fallback_diff}")
    row = {"replicate_index": rep, "generation_seed": seed, "scenario": scenario,
           "method": "CausalOnly-GCCM-NormCP", **{k: result[k] for k in
           ("coverage", "covered_count", "test_n", "mean_width", "median_width", "mean_interval_score")},
           "selected_k": selected_k, "q_norm": q,
           "no_usable_gccm_signal": info["status"] == "NO_USABLE_GCCM_SIGNAL",
           "selected_feature_count": len(info["selected"]), "selected_features": ";".join(info["selected"]),
           "feature_weights": jdump(info["weights"]), "tuning_scale_mae": model["tuning_mae"]}
    interval_hash = hashlib.sha256(np.column_stack((result["lower"], result["upper"])).tobytes()).hexdigest()
    return row, k_values, fallback_diff, interval_hash, barrier


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, method), g in results.groupby(["scenario", "method"], sort=True):
        raw_mean = float(results[(results.scenario == scenario) & (results.method == "Raw-CP")].coverage.mean())
        cov = float(g.coverage.mean())
        rows.append({"scenario": scenario, "method": method, "replicates": len(g), "mean_coverage": cov,
                     "median_coverage": float(g.coverage.median()), "coverage_sd": float(g.coverage.std(ddof=1)),
                     "min_coverage": float(g.coverage.min()), "max_coverage": float(g.coverage.max()),
                     "pooled_coverage": float(g.covered_count.sum() / g.test_n.sum()),
                     "replicates_coverage_lt_0_90": int((g.coverage < .90).sum()),
                     "replicates_coverage_lt_0_89": int((g.coverage < .89).sum()),
                     "mean_width": float(g.mean_width.mean()), "median_replicate_mean_width": float(g.mean_width.median()),
                     "sd_replicate_mean_width": float(g.mean_width.std(ddof=1)),
                     "mean_interval_score": float(g.mean_interval_score.mean()),
                     "median_interval_score": float(g.mean_interval_score.median()),
                     "material_undercoverage": bool(cov < .89 or cov < raw_mean - .01)})
    return pd.DataFrame(rows)


def paired(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        p = results[results.scenario == scenario].pivot(index="replicate_index", columns="method",
                                                        values=["coverage", "mean_width", "mean_interval_score"])
        for comparator in ("Raw-CP", "GeoOnly-NormCP", "CausalOnly-CMI-NormCP"):
            for metric in ("coverage", "mean_width", "mean_interval_score"):
                d = (p[metric]["CausalOnly-GCCM-NormCP"] - p[metric][comparator]).to_numpy(float)
                mean, sd = float(d.mean()), float(d.std(ddof=1))
                half = float(stats.t.ppf(.975, len(d)-1) * sd / math.sqrt(len(d)))
                base = p[metric][comparator].to_numpy(float)
                rel = float(np.mean(d / base)) if metric != "coverage" else None
                rows.append({"scenario": scenario, "contrast": f"CausalOnly-GCCM-NormCP minus {comparator}",
                             "metric": metric, "n_pairs": len(d), "mean_difference": mean,
                             "median_difference": float(np.median(d)), "sd_difference": sd,
                             "ci95_lower": mean-half, "ci95_upper": mean+half,
                             "mean_relative_difference": rel, "ci_method": "Student-t CI for mean paired difference"})
    return pd.DataFrame(rows)


def validate_stage11_compatibility(loader, reps):
    old = pd.read_csv(STAGE11 / "replicate_results.csv")
    if len(old) != 160 or set(old.generation_seed) != set(range(20270908, 20270928)):
        raise AssertionError("Stage-11 comparator rows/seeds mismatch")
    config = json.loads((STAGE11 / "config.json").read_text(encoding="utf-8"))
    if config["alpha"] != .1 or config["final_calibration_rows"] != 200 or config["test_rows"] != 300:
        raise AssertionError("Stage-11 alpha/test mismatch")
    unseal = pd.read_csv(STAGE11 / "test_unseal_audit.csv")
    for rep in reps:
        for scenario in SCENARIOS:
            subset = unseal[(unseal.replicate_index == rep["replicate_index"]) & (unseal.scenario == scenario)]
            if subset.empty: raise AssertionError("missing Stage-11 unseal evidence")
            scale = loader.load_split(rep["replicate_index"], rep["generation_seed"], scenario, "scale_fit")
            final = loader.load_split(rep["replicate_index"], rep["generation_seed"], scenario, "final_calibration")
            if not subset.scale_fit_hash.eq(frame_hash(scale)).all() or not subset.final_calibration_hash.eq(frame_hash(final)).all():
                raise AssertionError("Stage-11 split hash mismatch")
    return old


def main() -> None:
    RECORD.mkdir(parents=True, exist_ok=True)
    generated = ["gccm_artifact_inventory.csv", "preflight_summary.json", "test_unseal_audit.csv",
                 "gccm_weight_audit.csv", "replicate_results.csv", "scenario_summary.csv",
                 "paired_comparisons.csv", "stage12_summary.json"]
    if any((RECORD / p).exists() for p in generated):
        raise FileExistsError("refusing to overwrite Stage-12 outputs")
    if sha256(CORE) != CORE_HASH:
        raise AssertionError("historical GCCM core hash mismatch")
    reps, loader = provenance_gate()
    write_frozen_config()
    inventory = artifact_inventory(reps)
    inventory.to_csv(RECORD / "gccm_artifact_inventory.csv", index=False, lineterminator="\n")
    old_results = validate_stage11_compatibility(loader, reps)

    gccm_root = RECORD / "gccm_runs"; gccm_root.mkdir()
    weight_rows, run_metadata = [], []
    infos = {}
    # Preflight: first frozen pair, then independent clean rerun.
    preflight_hashes = {}
    for scenario in SCENARIOS:
        out = gccm_root / "replicate_01_seed_20270908" / scenario
        prepare_gccm_input(loader, 1, 20270908, scenario, out); meta = invoke_gccm(out)
        info, rows = read_weight_info(out); infos[(1, scenario)] = info
        for row in rows: weight_rows.append({"replicate_index": 1, "generation_seed": 20270908, "scenario": scenario, **row})
        run_metadata.append({"replicate_index": 1, "generation_seed": 20270908, "scenario": scenario, **meta})
        with tempfile.TemporaryDirectory(prefix="stage12_gccm_preflight_", dir=RECORD) as td:
            second = Path(td) / scenario
            prepare_gccm_input(loader, 1, 20270908, scenario, second); meta2 = invoke_gccm(second)
            keys = ["restricted_input.csv", "edges.csv", "parameters.csv", "embedding.csv", "raw_curves.csv",
                    "mean_curves.csv", "convergence.csv", "weights.csv"]
            matches = {k: sha256(out/k) == sha256(second/k) for k in keys}
            if not all(matches.values()): raise AssertionError(f"GCCM preflight nondeterminism: {scenario}")
            preflight_hashes[scenario] = matches
    cp_hashes = {}
    for scenario in SCENARIOS:
        first = run_cp(loader, 1, 20270908, scenario, infos[(1, scenario)], write_barrier=False)
        second = run_cp(loader, 1, 20270908, scenario, infos[(1, scenario)], write_barrier=False)
        if first[3] != second[3]: raise AssertionError("CP preflight nondeterminism")
        cp_hashes[scenario] = first[3]
    preflight = {"status": "PASS", "replicate_index": 1, "generation_seed": 20270908,
                 "gccm_clean_rerun_hashes_identical": preflight_hashes,
                 "cp_clean_rerun_interval_hashes": cp_hashes,
                 "coordinate_leakage": "PASS: final distance columns restricted to x1,x2,g_proxy,x_noise",
                 "test_seal": "PASS: q_norm frozen and audit barrier written before production test access"}
    (RECORD / "preflight_summary.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    # Complete GCCM estimation for remaining paired replicates.
    for rep in reps[1:]:
        idx, seed = rep["replicate_index"], rep["generation_seed"]
        for scenario in SCENARIOS:
            out = gccm_root / f"replicate_{idx:02d}_seed_{seed}" / scenario
            prepare_gccm_input(loader, idx, seed, scenario, out); meta = invoke_gccm(out)
            info, rows = read_weight_info(out); infos[(idx, scenario)] = info
            for row in rows: weight_rows.append({"replicate_index": idx, "generation_seed": seed, "scenario": scenario, **row})
            run_metadata.append({"replicate_index": idx, "generation_seed": seed, "scenario": scenario, **meta})
            print(f"GCCM {idx:02d}/20 {scenario} {info['status']} {meta['runtime_seconds']:.2f}s", flush=True)

    # Frozen weights feed the unchanged Stage-11 normalized-CP shell.
    gccm_rows, k_rows, fallback_rows = [], [], []
    for rep in reps:
        idx, seed = rep["replicate_index"], rep["generation_seed"]
        for scenario in SCENARIOS:
            row, kvals, diff, _, barrier = run_cp(loader, idx, seed, scenario, infos[(idx, scenario)], write_barrier=True)
            gccm_rows.append(row)
            for k, mae in kvals.items():
                k_rows.append({"replicate_index": idx, "generation_seed": seed, "scenario": scenario,
                               "k": k, "tuning_scale_mae": mae, "selected": k == row["selected_k"]})
            fallback_rows.append({"replicate_index": idx, "scenario": scenario,
                                  "no_usable_gccm_signal": infos[(idx, scenario)]["status"] == "NO_USABLE_GCCM_SIGNAL",
                                  "max_abs_bound_diff_vs_raw": diff})
    gccm = pd.DataFrame(gccm_rows)
    comparators = old_results[old_results.method.isin(["Raw-CP", "GeoOnly-NormCP", "CausalOnly-CMI-NormCP", "Oracle-NormCP"])].copy()
    combined = pd.concat([comparators, gccm], ignore_index=True, sort=False)
    scenarios = summarize(combined); pairs = paired(combined)
    weights_df = pd.DataFrame(weight_rows)
    weights_df.to_csv(RECORD / "gccm_weight_audit.csv", index=False, lineterminator="\n", float_format="%.17g")
    combined.to_csv(RECORD / "replicate_results.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(k_rows).to_csv(RECORD / "k_tuning_results.csv", index=False, lineterminator="\n", float_format="%.17g")
    scenarios.to_csv(RECORD / "scenario_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pairs.to_csv(RECORD / "paired_comparisons.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(run_metadata).to_json(RECORD / "gccm_run_metadata.json", orient="records", indent=2)

    s2w = weights_df[weights_df.scenario == "S2"].pivot(index="replicate_index", columns="feature", values="weight").apply(pd.to_numeric, errors="coerce").fillna(0)
    s4w = weights_df[weights_df.scenario == "S4"].pivot(index="replicate_index", columns="feature", values="weight").apply(pd.to_numeric, errors="coerce").fillna(0)
    freq = {"S2_x2_positive": int((s2w.x2 > 0).sum()),
            "S2_x2_largest": int(((s2w.x2 > 0) & s2w.x2.eq(s2w.max(axis=1))).sum()),
            "S2_g_proxy_largest": int(((s2w.g_proxy > 0) & s2w.g_proxy.eq(s2w.max(axis=1))).sum()),
            "S2_fallback": int(gccm[(gccm.scenario == "S2")].no_usable_gccm_signal.sum()),
            "S4_fallback": int(gccm[(gccm.scenario == "S4")].no_usable_gccm_signal.sum()),
            "S4_false_selection_replicates": int((s4w.sum(axis=1) > 0).sum()),
            "S4_dominant_counts": Counter(s4w.idxmax(axis=1)[s4w.max(axis=1) > 0]).most_common()}
    s2_summary = scenarios[(scenarios.scenario == "S2") & (scenarios.method == "CausalOnly-GCCM-NormCP")].iloc[0]
    raw_s2 = scenarios[(scenarios.scenario == "S2") & (scenarios.method == "Raw-CP")].iloc[0]
    coverage_failure = bool(s2_summary.material_undercoverage)
    width_gain = float(s2_summary.mean_width) < float(raw_s2.mean_width)
    score_gain = float(s2_summary.mean_interval_score) < float(raw_s2.mean_interval_score)
    scientific_class = ("GCCM_CAUSAL_ONLY_COVERAGE_FAILURE" if coverage_failure else
                        "GCCM_CAUSAL_ONLY_ALSO_IMPROVES_CP" if (width_gain or score_gain) else
                        "GCCM_SIGNAL_DOES_NOT_TRANSLATE_TO_CP_GAIN")
    summary = {"status": "PASS", "readiness_class": "READY_RECOMPUTE_EXACT_ORIGINAL_GCCM_PROTOCOL",
               "preflight": preflight, "gccm_protocol": PARAMETERS, "feature_frequencies": freq,
               "fallback_audit": fallback_rows,
               "selected_k_distributions": {s: {str(k): int(v) for k, v in Counter(gccm[gccm.scenario == s].selected_k.dropna().astype(int)).items()} for s in SCENARIOS},
               "scenario_summary": scenarios.to_dict("records"), "paired_comparisons": pairs.to_dict("records"),
               "scientific_class": scientific_class, "validation_used": False, "test_unseal_rows": 40,
               "historical_artifact_scope": "single pilot seed 20260907; all Stage-12 weights recomputed with frozen protocol",
               "method_implication": "direct comparison required; see scenario and paired tables"}
    (RECORD / "stage12_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "scientific_class": scientific_class, "frequencies": freq}, indent=2, default=str))


if __name__ == "__main__":
    main()
