"""Read-only Stage-20C audit of frozen Stage-20B coverage and abstention."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t as student_t


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/geo_causal_cp/20260920_stage20b_calibration_size_feasibility_v1"
RECORD = ROOT / "experiments/geo_causal_cp/20260920_stage20c_coverage_abstention_audit_v1"
RUNNER20B = ROOT / "geo_causal_cp_v1/run_stage20b_calibration_size_v1.py"
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]
META = ["sample_id", "scenario", "split", "local_calibration_role"]
METHODS = ["Raw-CP", "GeoCP", "GeoSIMCP-EUC", "GeoSIMCP-MND", "GeoCausalCP-Full"]
OFFICIAL_METHODS = METHODS[1:]
CAL_SIZES = [200, 400, 800]
TRACKS = ["A", "B"]
ALPHA = .10


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s20 = load_module("stage20b_readonly_for_20c", RUNNER20B)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(value: Any):
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return [safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value): return None
        if np.isposinf(value): return "Infinity"
        if np.isneginf(value): return "-Infinity"
        return float(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def scenario_dir(index: int, seed: int, scenario: str) -> Path:
    return SOURCE / "frozen_inputs" / f"replicate_{index:02d}_seed_{seed}" / scenario


def verify_source() -> dict:
    manifest = json.loads((SOURCE / "output_manifest.json").read_text(encoding="utf-8"))
    top_bad = []
    for item in manifest["files"]:
        path = SOURCE / item["path"]
        if not path.exists() or sha(path) != item["sha256"]:
            top_bad.append(item["path"])
    local_bad, generation_count, analysis_count = [], 0, 0
    for index, seed in enumerate(range(20950001, 20950051), 1):
        for scenario in ("S2", "S4"):
            base = scenario_dir(index, seed, scenario)
            for checkpoint_name, counter in (("GENERATION_COMPLETE.json", "generation"), ("ANALYSIS_COMPLETE.json", "analysis")):
                checkpoint = base / checkpoint_name
                if not checkpoint.exists():
                    local_bad.append(str(checkpoint.relative_to(SOURCE)))
                    continue
                meta = json.loads(checkpoint.read_text(encoding="utf-8"))
                for name, expected in meta["hashes"].items():
                    if not (base / name).exists() or sha(base / name) != expected:
                        local_bad.append(str((base / name).relative_to(SOURCE)))
                if counter == "generation": generation_count += 1
                else: analysis_count += 1
    prereg = json.loads((SOURCE / "preregistration/PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    prereg_bad = [name for name, expected in prereg["files"].items()
                   if sha(SOURCE / "preregistration" / name) != expected]
    status = "PASS" if not top_bad and not local_bad and not prereg_bad and generation_count == analysis_count == 100 else "FAIL"
    if status != "PASS":
        raise AssertionError({"top": top_bad, "local": local_bad[:10], "prereg": prereg_bad})
    return {"status": status, "top_level_manifest_files": len(manifest["files"]),
            "top_level_hash_mismatches": top_bad, "generation_checkpoints": generation_count,
            "analysis_checkpoints": analysis_count, "local_hash_mismatches": local_bad,
            "preregistration_hash_mismatches": prereg_bad,
            "preregistration_composite_sha256": prereg["composite_sha256"],
            "source_stderr_bytes": (SOURCE / "stderr.txt").stat().st_size}


def cmi_info(base: Path) -> dict:
    frame = pd.read_csv(base / "cmi_full999_results.csv")
    weights = dict(zip(frame.feature, frame.weight.astype(float)))
    selected = [feature for feature in FEATURES if weights[feature] > 0]
    return {"weights": weights, "selected": selected, "status": str(frame.status.iloc[0])}


def stratum(info: dict) -> str:
    chosen = set(info["selected"])
    if info["status"] == "NO_CAUSAL_SIGNAL": return "NO_SIGNAL"
    if chosen == {"x2"}: return "clean_x2_only"
    if "x2" in chosen: return "x2_plus_nuisance"
    return "other_signal"


def interval_arrays(y: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray) -> dict:
    lower, upper = pred - uncertainty, pred + uncertainty
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width.copy()
    lo, hi = y < lower, y > upper
    score[lo] += 20 * (lower[lo] - y[lo])
    score[hi] += 20 * (y[hi] - upper[hi])
    return {"lower": lower, "upper": upper, "covered": covered, "width": width, "score": score,
            "infinite": np.isinf(uncertainty)}


def reconstruct_partition(index: int, seed: int, scenario: str, n_cal: int, track: str,
                          selected_all: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = scenario_dir(index, seed, scenario)
    dev = pd.read_csv(base / "development.csv", keep_default_na=False)
    pool = dev.loc[dev.split_role.eq("CP_calibration_pool")].sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    calib = pool.iloc[:n_cal].copy()
    scores = np.abs(calib.y_true.to_numpy(float) - calib.y_pred.to_numpy(float))
    x = pd.read_csv(base / "predictor_inputs.csv", keep_default_na=False)
    pred_frame = pd.read_csv(base / "predictions.csv", keep_default_na=False)
    labels = pd.read_csv(base / "test_sealed.csv", keep_default_na=False)
    test_cov = x.loc[x.split_role.eq("test")].merge(pred_frame, on=[*META, "split_role"], validate="one_to_one", sort=False)
    test = test_cov.merge(labels, on=[*META, "split_role"], validate="one_to_one", sort=False)
    info = cmi_info(base)
    transforms = {method: s20.fit_feature_transform(calib, method, info) for method in s20.JOINT_METHODS}
    dgeo2 = s20.pairwise_sq(test[COORDS].to_numpy(float), calib[COORDS].to_numpy(float))
    dfeat = {method: None if transforms[method]["kind"] == "fallback" else s20.feature_distance_sq(test, transforms[method])
             for method in s20.JOINT_METHODS}
    y, pred = test.y_true.to_numpy(float), test.y_pred.to_numpy(float)
    arrays: dict[str, dict] = {}
    mass_ess: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rank = math.ceil((n_cal + 1) * .9)
    raw = np.full(len(test), np.sort(scores, kind="mergesort")[rank - 1])
    arrays["Raw-CP"] = interval_arrays(y, pred, raw)
    mass_ess["Raw-CP"] = (np.full(len(test), n_cal, float), np.full(len(test), n_cal, float))
    params = selected_all.loc[(selected_all.replicate_index == index) & (selected_all.scenario == scenario) &
                              (selected_all.n_cal == n_cal) & (selected_all.track == track)]
    for method in OFFICIAL_METHODS:
        row = params.loc[params.method.eq(method)]
        if len(row) != 1:
            raise AssertionError("selected parameter join")
        row = row.iloc[0]
        if row.status != "SELECTED":
            continue
        use_feature = None if method == "GeoCP" or (method == "GeoCausalCP-Full" and info["status"] == "NO_CAUSAL_SIGNAL") else dfeat[method]
        weights = s20.kernel_weights(dgeo2, use_feature, float(row.bandwidth), None if pd.isna(row["lambda"]) else float(row["lambda"]))
        uncertainty, mass, ess = s20.weighted_uncertainty(scores, weights, track)
        arrays[method] = interval_arrays(y, pred, uncertainty)
        mass_ess[method] = (mass, ess)

    rows = []
    for method, data in arrays.items():
        mass, ess = mass_ess[method]
        param = params.loc[params.method.eq(method)].iloc[0] if method != "Raw-CP" else None
        for position, sample in enumerate(test.itertuples(index=False)):
            rows.append({
                "replicate_index": index, "generation_seed": seed, "scenario": scenario, "n_cal": n_cal,
                "track": track, "method": method, "cmi_status": info["status"], "stratum": stratum(info),
                "sample_id": sample.sample_id, "covered": bool(data["covered"][position]),
                "infinite": bool(data["infinite"][position]), "width": data["width"][position],
                "interval_score": data["score"][position], "kernel_mass": mass[position], "ess": ess[position],
                "oracle_sigma": sample.oracle_sigma, "s_x": sample.s_x, "s_y": sample.s_y,
                "x1": sample.x1, "x2": sample.x2, "g_proxy": sample.g_proxy, "x_noise": sample.x_noise,
                "bandwidth": np.nan if param is None else param.bandwidth,
                "lambda": np.nan if param is None else param["lambda"],
            })
    frame = pd.DataFrame(rows)
    # Exact cross-check against frozen aggregate results.
    frozen = pd.read_csv(base / "replicate_results.csv")
    for method, group in frame.groupby("method"):
        expected = frozen.loc[(frozen.n_cal == n_cal) & (frozen.track == track) & (frozen.method == method)]
        if len(expected) != 1: raise AssertionError("aggregate result join")
        expected = expected.iloc[0]
        if not np.isclose(group.covered.mean(), expected.coverage, atol=1e-15): raise AssertionError("coverage reconstruction")
        if not np.isclose(group.infinite.mean(), expected.infinite_fraction, atol=1e-15): raise AssertionError("infinity reconstruction")
    return frame, info


def mean_ci(values: pd.Series | np.ndarray) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0: return {"n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(x.mean()); sd = float(x.std(ddof=1)) if n > 1 else 0.0
    half = float(student_t.ppf(.975, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "median": float(np.median(x)), "sd": sd, "ci_low": mean-half, "ci_high": mean+half}


def coverage_decomposition(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded = []
    for row in points.itertuples(index=False):
        expanded.append({**row._asdict(), "analysis_stratum": "all"})
        expanded.append({**row._asdict(), "analysis_stratum": "SIGNAL" if row.cmi_status == "SIGNAL" else "NO_SIGNAL"})
    data = pd.DataFrame(expanded)
    rows = []
    for keys, group in data.groupby(["scenario", "n_cal", "track", "method", "analysis_stratum"], sort=True):
        scenario, n_cal, track, method, audit_stratum = keys
        finite = ~group.infinite
        total = group.covered.mean(); p_inf = group.infinite.mean()
        cov_finite = group.loc[finite, "covered"].mean() if finite.any() else np.nan
        reconstructed = p_inf + (1-p_inf)*cov_finite if finite.any() else 1.0
        rows.append({"scenario": scenario, "n_cal": n_cal, "track": track, "method": method,
                     "stratum": audit_stratum, "point_count": len(group), "p_infinite": p_inf,
                     "p_finite": finite.mean(), "coverage_total": total, "coverage_finite": cov_finite,
                     "coverage_reconstructed": reconstructed, "identity_abs_error": abs(total-reconstructed)})
    result = pd.DataFrame(rows)
    if result.identity_abs_error.max() > 1e-14: raise AssertionError("coverage identity")
    return result, result[["scenario","n_cal","track","method","stratum","point_count","coverage_finite"]].copy()


def state_diagnostics(points_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal = points_b.loc[(points_b.scenario == "S2") & points_b.cmi_status.eq("SIGNAL")]
    eu = signal.loc[signal.method.eq("GeoSIMCP-EUC")]
    ca = signal.loc[signal.method.eq("GeoCausalCP-Full")]
    keys = ["replicate_index", "generation_seed", "scenario", "n_cal", "track", "sample_id"]
    paired = eu.merge(ca, on=keys, suffixes=("_euc", "_causal"), validate="one_to_one")
    paired["infinity_state"] = np.select(
        [~paired.infinite_euc & ~paired.infinite_causal,
         paired.infinite_euc & ~paired.infinite_causal,
         ~paired.infinite_euc & paired.infinite_causal],
        ["BOTH_FINITE", "EUC_INFINITY_CAUSAL_FINITE", "EUC_FINITE_CAUSAL_INFINITY"],
        default="BOTH_INFINITY")
    # Preserve the actual paired point classifications, not merely their counts.
    state_columns = [*keys, "infinity_state", "covered_euc", "covered_causal", "infinite_euc", "infinite_causal",
        "width_euc", "width_causal", "interval_score_euc", "interval_score_causal",
        "kernel_mass_euc", "kernel_mass_causal", "ess_euc", "ess_causal", "oracle_sigma_euc",
        "s_x_euc", "s_y_euc", "x1_euc", "x2_euc", "g_proxy_euc", "x_noise_euc",
        "bandwidth_euc", "lambda_euc", "bandwidth_causal", "lambda_causal"]
    states = paired[state_columns].copy()

    def subset_file(state: str, finite_method: str) -> pd.DataFrame:
        rows = []
        subset = paired.loc[paired.infinity_state.eq(state)]
        for (n_cal, rep), group in subset.groupby(["n_cal", "replicate_index"], sort=True):
            suffix = "causal" if finite_method == "causal" else "euc"
            rows.append({
                "aggregation": "replicate", "n_cal": n_cal, "replicate_index": rep, "point_count": len(group),
                "coverage": group[f"covered_{suffix}"].mean(), "mean_width": group[f"width_{suffix}"].mean(),
                "median_width": group[f"width_{suffix}"].median(), "mean_interval_score": group[f"interval_score_{suffix}"].mean(),
                "mean_oracle_sigma": group[f"oracle_sigma_{suffix}"].mean(), "median_oracle_sigma": group[f"oracle_sigma_{suffix}"].median(),
                "mean_kernel_mass": group[f"kernel_mass_{suffix}"].mean(), "median_kernel_mass": group[f"kernel_mass_{suffix}"].median(),
                "mean_ess": group[f"ess_{suffix}"].mean(), "median_ess": group[f"ess_{suffix}"].median(),
                "mean_s_x": group[f"s_x_{suffix}"].mean(), "mean_s_y": group[f"s_y_{suffix}"].mean(),
                "mean_x2": group[f"x2_{suffix}"].mean(), "mean_g_proxy": group[f"g_proxy_{suffix}"].mean(),
                "mean_lambda": group[f"lambda_{suffix}"].mean(), "mean_bandwidth": group[f"bandwidth_{suffix}"].mean(),
            })
        for n_cal, group in subset.groupby("n_cal", sort=True):
            suffix = "causal" if finite_method == "causal" else "euc"
            rows.append({
                "aggregation": "pooled", "n_cal": n_cal, "replicate_index": np.nan, "point_count": len(group),
                "coverage": group[f"covered_{suffix}"].mean(), "mean_width": group[f"width_{suffix}"].mean(),
                "median_width": group[f"width_{suffix}"].median(), "mean_interval_score": group[f"interval_score_{suffix}"].mean(),
                "mean_oracle_sigma": group[f"oracle_sigma_{suffix}"].mean(), "median_oracle_sigma": group[f"oracle_sigma_{suffix}"].median(),
                "mean_kernel_mass": group[f"kernel_mass_{suffix}"].mean(), "median_kernel_mass": group[f"kernel_mass_{suffix}"].median(),
                "mean_ess": group[f"ess_{suffix}"].mean(), "median_ess": group[f"ess_{suffix}"].median(),
                "mean_s_x": group[f"s_x_{suffix}"].mean(), "mean_s_y": group[f"s_y_{suffix}"].mean(),
                "mean_x2": group[f"x2_{suffix}"].mean(), "mean_g_proxy": group[f"g_proxy_{suffix}"].mean(),
                "mean_lambda": group[f"lambda_{suffix}"].mean(), "mean_bandwidth": group[f"bandwidth_{suffix}"].mean(),
            })
        return pd.DataFrame(rows)

    # BOTH_FINITE contains paired method outcomes and replicate-level differences.
    both_rows = []
    both = paired.loc[paired.infinity_state.eq("BOTH_FINITE")]
    for (n_cal, rep), group in both.groupby(["n_cal", "replicate_index"], sort=True):
        both_rows.append({
            "n_cal": n_cal, "replicate_index": rep, "point_count": len(group),
            "coverage_causal": group.covered_causal.mean(), "coverage_euc": group.covered_euc.mean(),
            "coverage_difference": (group.covered_causal.astype(int)-group.covered_euc.astype(int)).mean(),
            "mean_width_causal": group.width_causal.mean(), "mean_width_euc": group.width_euc.mean(),
            "mean_width_difference": (group.width_causal-group.width_euc).mean(),
            "median_width_causal": group.width_causal.median(), "median_width_euc": group.width_euc.median(),
            "mean_score_causal": group.interval_score_causal.mean(), "mean_score_euc": group.interval_score_euc.mean(),
            "mean_score_difference": (group.interval_score_causal-group.interval_score_euc).mean(),
            "mean_oracle_sigma": group.oracle_sigma_causal.mean(), "median_oracle_sigma": group.oracle_sigma_causal.median(),
            "mean_kernel_mass_causal": group.kernel_mass_causal.mean(), "mean_kernel_mass_euc": group.kernel_mass_euc.mean(),
            "mean_ess_causal": group.ess_causal.mean(), "mean_ess_euc": group.ess_euc.mean(),
        })
    both_frame = pd.DataFrame(both_rows)

    # Exact algebraic coverage-gap attribution per n_cal.
    attr_rows = []
    for n_cal, group in paired.groupby("n_cal", sort=True):
        n = len(group)
        e_only = group.infinity_state.eq("EUC_INFINITY_CAUSAL_FINITE")
        c_only = group.infinity_state.eq("EUC_FINITE_CAUSAL_INFINITY")
        both_f = group.infinity_state.eq("BOTH_FINITE")
        contrib_e = float((1 - group.loc[e_only, "covered_causal"].astype(int)).sum() / n)
        contrib_c = float(-(1 - group.loc[c_only, "covered_euc"].astype(int)).sum() / n)
        contrib_b = float((group.loc[both_f, "covered_euc"].astype(int)-group.loc[both_f, "covered_causal"].astype(int)).sum() / n)
        observed = float(group.covered_euc.mean()-group.covered_causal.mean())
        attr_rows.append({"n_cal": n_cal, "point_count": n,
            "euc_only_infinity_contribution": contrib_e, "causal_only_infinity_contribution": contrib_c,
            "both_finite_success_difference_contribution": contrib_b, "both_infinite_contribution": 0.0,
            "decomposed_coverage_euc_minus_causal": contrib_e+contrib_c+contrib_b,
            "observed_coverage_euc_minus_causal": observed,
            "absolute_decomposition_error": abs(observed-(contrib_e+contrib_c+contrib_b))})
    attribution = pd.DataFrame(attr_rows)
    if attribution.absolute_decomposition_error.max() > 1e-15: raise AssertionError("gap decomposition")
    return states, both_frame, subset_file("EUC_INFINITY_CAUSAL_FINITE", "causal"), subset_file("EUC_FINITE_CAUSAL_INFINITY", "euc"), attribution


def validation_audit(selected: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_clean = selected.drop(columns=[column for column in ("cmi_status", "stratum") if column in selected.columns])
    merged = selected_clean.merge(results[["replicate_index","generation_seed","scenario","n_cal","track","method","selection_status",
                                     "coverage","infinite_fraction"]],
                            left_on=["replicate_index","generation_seed","scenario","n_cal","track","method"],
                            right_on=["replicate_index","generation_seed","scenario","n_cal","track","method"],
                            suffixes=("_selected", "_result"), validate="one_to_one")
    merged = merged.loc[merged.status.eq("SELECTED") & merged.selection_status.eq("SELECTED")].copy()
    # Recover CMI status/stratum from GeoCausal result and attach to every method within a replicate.
    status = results.loc[results.method.eq("GeoCausalCP-Full"), ["replicate_index","generation_seed","scenario","n_cal","track","cmi_status","stratum"]]
    merged = merged.merge(status, on=["replicate_index","generation_seed","scenario","n_cal","track"], validate="many_to_one")
    merged["validation_margin"] = merged.validation_coverage - .90
    merged["test_minus_validation"] = merged.coverage - merged.validation_coverage
    values = merged.validation_coverage.to_numpy(float)
    merged["validation_boundary_bin"] = np.select(
        [np.isclose(values,.900), np.isclose(values,.905), np.isclose(values,.910), np.isclose(values,.915), values>=.920-1e-12],
        ["0.900","0.905","0.910","0.915","0.920+"], default="other")
    margin_cols = ["replicate_index","generation_seed","scenario","n_cal","track","method","cmi_status","stratum",
                   "bandwidth","lambda","validation_coverage","validation_margin","coverage","test_minus_validation",
                   "infinite_fraction","validation_boundary_bin"]
    margin = merged[margin_cols].copy()
    rows = []
    for keys, group in merged.groupby(["scenario","n_cal","track","method","cmi_status"], sort=True):
        scenario,n_cal,track,method,cmi_status = keys
        stats = mean_ci(group.test_minus_validation)
        corr = pearsonr(group.validation_margin, group.coverage).statistic if len(group)>2 and group.validation_margin.std()>0 and group.coverage.std()>0 else np.nan
        rows.append({"scenario":scenario,"n_cal":n_cal,"track":track,"method":method,"cmi_status":cmi_status,
                     **{f"coverage_change_{k}":v for k,v in stats.items()},
                     "correlation_validation_margin_test_coverage":corr,
                     "mean_validation_coverage":group.validation_coverage.mean(),"mean_test_coverage":group.coverage.mean(),
                     "at_0900_frequency":np.isclose(group.validation_coverage,.900).mean(),
                     "at_0905_frequency":np.isclose(group.validation_coverage,.905).mean(),
                     "at_0910_frequency":np.isclose(group.validation_coverage,.910).mean(),
                     "at_0915_frequency":np.isclose(group.validation_coverage,.915).mean(),
                     "at_0920plus_frequency":(group.validation_coverage>=.920-1e-12).mean()})
    generalization = pd.DataFrame(rows)

    diag_rows = []
    for keys, group in merged.loc[(merged.track=="B") & merged.method.isin(["GeoSIMCP-EUC","GeoCausalCP-Full"])].groupby(
            ["scenario","n_cal","method","cmi_status"], sort=True):
        scenario,n_cal,method,cmi_status = keys
        for predictor in ["lambda","bandwidth","validation_margin"]:
            for outcome in ["coverage","infinite_fraction","test_minus_validation"]:
                valid = group[[predictor,outcome]].dropna()
                pearson = pearsonr(valid[predictor],valid[outcome]).statistic if len(valid)>2 and valid[predictor].std()>0 and valid[outcome].std()>0 else np.nan
                spear = spearmanr(valid[predictor],valid[outcome]).statistic if len(valid)>2 and valid[predictor].nunique()>1 and valid[outcome].nunique()>1 else np.nan
                diag_rows.append({"scenario":scenario,"n_cal":n_cal,"track":"B","method":method,"cmi_status":cmi_status,
                                  "predictor":predictor,"outcome":outcome,"n":len(valid),"pearson":pearson,"spearman":spear})
    return margin, generalization, pd.DataFrame(diag_rows)


def track_comparison(selected: pd.DataFrame) -> pd.DataFrame:
    a = selected.loc[selected.track.eq("A")].copy(); b = selected.loc[selected.track.eq("B")].copy()
    keys = ["replicate_index","generation_seed","scenario","n_cal","method"]
    merged = a.merge(b, on=keys, suffixes=("_A","_B"), validate="one_to_one")
    def same(a, b):
        return (a.isna() & b.isna()) | np.isclose(a.fillna(0), b.fillna(0))
    merged["status_changed"] = merged.status_A != merged.status_B
    merged["bandwidth_changed"] = ~same(merged.bandwidth_A, merged.bandwidth_B)
    merged["lambda_changed"] = ~same(merged["lambda_A"], merged["lambda_B"])
    merged["localization_changed"] = merged.status_changed | merged.bandwidth_changed | merged.lambda_changed
    return merged


def score_semantics(grid: pd.DataFrame, selected: pd.DataFrame) -> tuple[str, dict]:
    feasible = grid.loc[grid.feasible.astype(str).str.lower().isin(["true","1"])]
    inf_feasible = np.isinf(feasible.mean_interval_score).sum()
    partitions = 0; no_finite_feasible = 0; grid_tie_active = 0; selected_inf = 0
    keys = ["replicate_index","generation_seed","scenario","n_cal","track","method"]
    for partition, chosen in selected.groupby(keys, sort=False):
        candidates = feasible
        for key, value in zip(keys, partition): candidates = candidates.loc[candidates[key] == value]
        if len(candidates):
            partitions += 1
            finite = candidates.loc[np.isfinite(candidates.mean_interval_score)]
            if len(finite)==0:
                no_finite_feasible += 1
                chosen_row = chosen.iloc[0]
                if chosen_row.status == "SELECTED" and np.isinf(chosen_row.validation_interval_score):
                    selected_inf += 1
                    if int(chosen_row.grid_index) == int(candidates.grid_index.min()): grid_tie_active += 1
    text = f"""# Validation interval-score semantics

The frozen Stage-20B `interval_metrics` function computes interval score from width first. An infinite uncertainty produces infinite width and therefore infinite mean validation interval score. No truncation or finite-only replacement occurs during tuning.

The frozen `tune_method` implementation first restricts candidates to validation coverage >= 0.90, then applies Python's deterministic `min((mean_interval_score, grid_index))`. Any finite-score feasible candidate strictly dominates an infinite-score candidate. If every feasible candidate has infinite score, all primary objective values tie at +infinity and the smallest frozen grid index is selected. If no candidate is coverage-feasible, the recorded candidate is diagnostic-only and test metrics are not evaluated.

Stored-grid verification:

- Feasible candidates with infinite validation score: {int(inf_feasible)}
- Method/replicate/size/track partitions containing feasible candidates: {partitions}
- Partitions with no finite-score feasible candidate: {no_finite_feasible}
- Selected settings with infinite validation score: {selected_inf}
- Cases where first-grid deterministic tie behavior was verified: {grid_tie_active}

Thus the test atom affects tuning twice: it can raise validation coverage through abstention, while the resulting +infinity score normally makes that candidate noncompetitive whenever a finite feasible alternative exists. Grid order becomes scientifically relevant only when all feasible candidates abstain somewhere.
"""
    return text, {"infinite_feasible_candidates":int(inf_feasible),"partitions_with_feasible":partitions,
                  "partitions_no_finite_feasible":no_finite_feasible,"selected_infinite_score":selected_inf,
                  "grid_order_tie_active":grid_tie_active}


def main() -> None:
    RECORD.mkdir(parents=True, exist_ok=True)
    with (RECORD / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write("Stage-20C read-only audit started/resumed.\n")
    if not (RECORD / "stderr.txt").exists():
        (RECORD / "stderr.txt").write_text("", encoding="utf-8")
    integrity = verify_source()
    selected = pd.read_csv(SOURCE / "selected_hyperparameters.csv")
    results = pd.read_csv(SOURCE / "replicate_results.csv")
    grid = pd.read_csv(SOURCE / "validation_grid_results.csv")

    point_frames = []
    for index, seed in enumerate(range(20950001,20950051),1):
        for scenario in ("S2","S4"):
            for n_cal in CAL_SIZES:
                for track in TRACKS:
                    frame, _ = reconstruct_partition(index,seed,scenario,n_cal,track,selected)
                    point_frames.append(frame)
        if index % 10 == 0:
            print(f"RECONSTRUCTION {index}/50", flush=True)
    points = pd.concat(point_frames, ignore_index=True, sort=False)
    decomposition, finite = coverage_decomposition(points)
    points_b = points.loc[points.track.eq("B")].copy()
    states,both,euc_inf,causal_inf,attribution = state_diagnostics(points_b)
    margin,generalization,hyper_diag = validation_audit(selected,results)
    track_compare = track_comparison(selected)
    semantics, semantics_counts = score_semantics(grid,selected)

    write_csv(decomposition, RECORD / "coverage_decomposition.csv")
    write_csv(finite, RECORD / "finite_only_coverage.csv")
    write_csv(states, RECORD / "pointwise_infinity_states.csv")
    write_csv(both, RECORD / "both_finite_comparison.csv")
    write_csv(euc_inf, RECORD / "euc_infinity_causal_finite.csv")
    write_csv(causal_inf, RECORD / "causal_infinity_euc_finite.csv")
    write_csv(attribution, RECORD / "coverage_gap_attribution.csv")
    write_csv(margin, RECORD / "validation_coverage_margin.csv")
    write_csv(generalization, RECORD / "validation_test_generalization.csv")
    write_csv(track_compare, RECORD / "trackA_trackB_hyperparameter_comparison.csv")
    write_csv(hyper_diag, RECORD / "hyperparameter_margin_diagnostics.csv")
    (RECORD / "validation_score_semantics.md").write_text(semantics, encoding="utf-8")

    # Compact facts for interpretation; classification is finalized only after exact diagnostics exist.
    n800 = decomposition.loc[(decomposition.scenario=="S2")&(decomposition.n_cal==800)&(decomposition.track=="B")&
                             (decomposition.stratum=="SIGNAL")&decomposition.method.isin(["GeoSIMCP-EUC","GeoCausalCP-Full"])]
    n800_map = {row.method: row._asdict() for row in n800.itertuples(index=False)}
    integrity["reconstructed_point_rows"] = len(points)
    integrity["coverage_identity_max_abs_error"] = decomposition.identity_abs_error.max()
    integrity["validation_semantics"] = semantics_counts
    dump(RECORD / "intermediate_facts.json", {"integrity":integrity,"n800_signal":n800_map,
                                               "coverage_attribution":attribution.to_dict("records")})
    with (RECORD / "run_log.txt").open("a",encoding="utf-8") as handle:
        handle.write(f"Pointwise reconstruction complete: rows={len(points)}; coverage_identity_error={decomposition.identity_abs_error.max():.3g}.\n")
    print("STAGE20C DERIVATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
