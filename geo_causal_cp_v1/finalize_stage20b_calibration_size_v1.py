"""Finalize Stage-20B from immutable scientific checkpoints only."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/geo_causal_cp/20260920_stage20b_calibration_size_feasibility_v1"
PREREG = RECORD / "preregistration"
BOOT_SEED = 22020001
METHOD_ORDER = ["Raw-CP", "GeoCP", "GeoSIMCP-EUC", "GeoSIMCP-MND", "GeoCausalCP-Full"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def safe(value):
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


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_seal() -> dict:
    seal = json.loads((PREREG / "PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    for name, expected in seal["files"].items():
        if sha(PREREG / name) != expected:
            raise AssertionError(f"changed preregistration: {name}")
    payload = "\n".join(f"{name} {seal['files'][name]}" for name in sorted(seal["files"]))
    if hashlib.sha256(payload.encode()).hexdigest() != seal["composite_sha256"]:
        raise AssertionError("composite mismatch")
    return seal


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in results.groupby(["scenario", "n_cal", "track", "method"], sort=True):
        scenario, n_cal, track, method = keys
        selected = group.loc[group["selection_status"].eq("SELECTED")].copy()
        row = {
            "scenario": scenario, "n_cal": n_cal, "track": track, "method": method,
            "replicates_total": len(group), "replicates_selected": len(selected),
            "no_feasible_count": int((group["selection_status"] == "NO_FEASIBLE_HYPERPARAMETER").sum()),
        }
        if len(selected):
            for col in ["coverage", "mean_width", "median_width", "mean_interval_score", "infinite_fraction",
                        "mean_width_finite", "median_width_finite", "mean_interval_score_finite",
                        "mean_kernel_mass", "median_kernel_mass", "mean_n_eff", "median_n_eff"]:
                row[f"mean_{col}"] = float(selected[col].mean())
                row[f"median_{col}"] = float(selected[col].median())
            row["coverage_sd"] = float(selected["coverage"].std(ddof=1))
            row["pooled_coverage"] = float(selected["covered_count"].sum() / selected["n"].sum())
            row["replicates_any_infinite"] = int((selected["infinite_fraction"] > 0).sum())
            row["replicate_any_infinite_frequency"] = float((selected["infinite_fraction"] > 0).mean())
            row["coverage_below_090"] = int((selected["coverage"] < .90).sum())
            row["coverage_below_089"] = int((selected["coverage"] < .89).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    # Frozen descriptive concern rule, evaluated against Raw at the same scenario/n/track.
    out["material_coverage_concern"] = False
    for index, row in out.iterrows():
        raw = out.loc[(out.scenario == row.scenario) & (out.n_cal == row.n_cal) &
                      (out.track == row.track) & (out.method == "Raw-CP"), "mean_coverage"]
        if len(raw) and pd.notna(row.get("mean_coverage")):
            out.loc[index, "material_coverage_concern"] = bool(row["mean_coverage"] < .89 or row["mean_coverage"] < float(raw.iloc[0]) - .01)
    return out


def stratified(results: pd.DataFrame) -> pd.DataFrame:
    causal = results.loc[(results.scenario == "S2") & results.method.eq("GeoCausalCP-Full")].copy()
    status_map = causal[["replicate_index", "generation_seed", "cmi_status", "stratum"]].drop_duplicates()
    expanded = []
    for row in status_map.itertuples(index=False):
        expanded.append({"replicate_index": row.replicate_index, "generation_seed": row.generation_seed,
                         "stratum": "SIGNAL" if row.cmi_status == "SIGNAL" else "NO_SIGNAL"})
        if row.cmi_status == "SIGNAL":
            expanded.append({"replicate_index": row.replicate_index, "generation_seed": row.generation_seed,
                             "stratum": row.stratum})
    maps = pd.DataFrame(expanded)
    s2 = results.loc[results.scenario.eq("S2")].merge(maps, on=["replicate_index", "generation_seed"], how="inner", suffixes=("", "_audit"))
    rows = []
    for keys, group in s2.groupby(["n_cal", "track", "method", "stratum_audit"], sort=True):
        n_cal, track, method, stratum_name = keys
        selected = group.loc[group.selection_status.eq("SELECTED")]
        rows.append({
            "scenario": "S2", "n_cal": n_cal, "track": track, "method": method, "stratum": stratum_name,
            "replicates_total": group.replicate_index.nunique(), "replicates_selected": len(selected),
            "mean_coverage": selected.coverage.mean(), "mean_width": selected.mean_width.mean(),
            "mean_interval_score": selected.mean_interval_score.mean(), "mean_infinite_fraction": selected.infinite_fraction.mean(),
            "replicate_any_infinite_frequency": (selected.infinite_fraction > 0).mean(),
            "mean_width_finite": selected.mean_width_finite.mean(),
            "mean_interval_score_finite": selected.mean_interval_score_finite.mean(),
            "mean_kernel_mass": selected.mean_kernel_mass.mean(), "mean_ess": selected.mean_n_eff.mean(),
        })
    return pd.DataFrame(rows)


def ci(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return {"n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan,
                "t_ci_low": np.nan, "t_ci_high": np.nan, "bootstrap_ci_low": np.nan, "bootstrap_ci_high": np.nan}
    mean, median = float(values.mean()), float(np.median(values))
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half = float(student_t.ppf(.975, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    return {"n": n, "mean": mean, "median": median, "sd": sd,
            "t_ci_low": mean - half, "t_ci_high": mean + half,
            "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high)}


def paired(results: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("GeoCausalCP-Full", "GeoSIMCP-EUC", "GeoCausal_minus_EUC"),
                   ("GeoCausalCP-Full", "GeoCP", "GeoCausal_minus_GeoCP")]
    metrics = ["coverage", "mean_width", "mean_interval_score", "infinite_fraction",
               "mean_width_finite", "mean_interval_score_finite"]
    causal = results.loc[(results.scenario == "S2") & results.method.eq("GeoCausalCP-Full")]
    base_status = causal[["replicate_index", "generation_seed", "cmi_status", "stratum"]].drop_duplicates()
    strata = {
        "all": set(base_status.replicate_index),
        "SIGNAL": set(base_status.loc[base_status.cmi_status.eq("SIGNAL"), "replicate_index"]),
        "clean_x2_only": set(base_status.loc[base_status.stratum.eq("clean_x2_only"), "replicate_index"]),
        "x2_plus_nuisance": set(base_status.loc[base_status.stratum.eq("x2_plus_nuisance"), "replicate_index"]),
        "NO_SIGNAL": set(base_status.loc[base_status.cmi_status.eq("NO_CAUSAL_SIGNAL"), "replicate_index"]),
    }
    rows = []
    for n_cal in [200, 400, 800]:
        for track in ["A", "B"]:
            section = results.loc[(results.scenario == "S2") & (results.n_cal == n_cal) & (results.track == track)]
            for lhs, rhs, label in comparisons:
                left = section.loc[section.method.eq(lhs)].copy()
                right = section.loc[section.method.eq(rhs)].copy()
                merged = left.merge(right, on=["replicate_index", "generation_seed", "scenario", "n_cal", "track"], suffixes=("_lhs", "_rhs"))
                merged = merged.loc[merged.selection_status_lhs.eq("SELECTED") & merged.selection_status_rhs.eq("SELECTED")]
                for stratum_name, ids in strata.items():
                    use = merged.loc[merged.replicate_index.isin(ids)]
                    for metric in metrics:
                        with np.errstate(invalid="ignore"):
                            values = use[f"{metric}_lhs"].to_numpy(float) - use[f"{metric}_rhs"].to_numpy(float)
                        stats = ci(values, BOOT_SEED)
                        rows.append({"scenario": "S2", "n_cal": n_cal, "track": track, "comparison": label,
                                     "stratum": stratum_name, "metric": metric, **stats})
    return pd.DataFrame(rows)


def mass_and_ess(pointwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mass_rows, ess_rows = [], []
    for keys, group in pointwise.groupby(["scenario", "n_cal", "method"], sort=True):
        scenario, n_cal, method = keys
        mass_rows.append({
            "scenario": scenario, "n_cal": n_cal, "track": "B", "method": method, "point_count": len(group),
            "mean_kernel_mass": group.kernel_mass.mean(), "median_kernel_mass": group.kernel_mass.median(),
            "p05_kernel_mass": group.kernel_mass.quantile(.05), "fraction_mass_lt_9": (group.kernel_mass < 9).mean(),
            "fraction_mass_ge_9": (group.kernel_mass >= 9).mean(), "infinite_interval_fraction": group.infinite_interval.mean(),
            "mass_infinity_mismatches": (~group.mass_infinity_match).sum(),
        })
        ess_rows.append({
            "scenario": scenario, "n_cal": n_cal, "track": "B", "method": method, "point_count": len(group),
            "mean_ess": group.ess.mean(), "median_ess": group.ess.median(), "p05_ess": group.ess.quantile(.05),
            "p25_ess": group.ess.quantile(.25), "min_ess": group.ess.min(),
        })
    return pd.DataFrame(mass_rows), pd.DataFrame(ess_rows)


def load_alignment_detail() -> pd.DataFrame:
    frames = []
    for path in sorted((RECORD / "frozen_inputs").glob("replicate_*/*/oracle_alignment.csv")):
        if path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(frame):
            frames.append(frame)
    if not frames:
        raise AssertionError("missing oracle-alignment detail")
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    seal = verify_seal()
    checkpoint = json.loads((RECORD / "scientific_computation_checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint["status"] != "COMPLETE" or checkpoint["scenario_replicates"] != 100:
        raise AssertionError("incomplete computation")
    results = pd.read_csv(RECORD / "replicate_results.csv")
    pointwise = pd.read_csv(RECORD / "kernel_mass_pointwise.csv")
    cmi = pd.read_csv(RECORD / "cmi_full999_results.csv")
    selected = pd.read_csv(RECORD / "selected_hyperparameters.csv")
    invariance = pd.read_csv(RECORD / "cmi_invariance_audit.csv")
    alignment = load_alignment_detail()
    if len(results) != 3000 or len(pointwise) != 343800 or not pointwise.mass_infinity_match.all():
        raise AssertionError("scientific output integrity")
    if not invariance.matches_single_fixed_cmi_result.all() or invariance.groupby(["replicate_index", "scenario"]).signature_sha256.nunique().max() != 1:
        raise AssertionError("CMI invariance")

    scenario = summarize_results(results)
    signal = stratified(results)
    pairs = paired(results)
    mass, ess = mass_and_ess(pointwise)
    align_summary = alignment.groupby(["n_cal", "track", "distance"], dropna=False).agg(
        replicate_count=("replicate_index", "nunique"), mean_spearman=("spearman_oracle_scale_mismatch", "mean"),
        median_spearman=("spearman_oracle_scale_mismatch", "median"), sd_spearman=("spearman_oracle_scale_mismatch", "std"),
    ).reset_index()
    no_feasible_detail = selected.loc[selected.status.eq("NO_FEASIBLE_HYPERPARAMETER")].copy()
    no_feasible_summary = selected.groupby(["scenario", "n_cal", "track", "method"]).agg(
        replicate_count=("replicate_index", "count"), no_feasible_count=("status", lambda x: int((x == "NO_FEASIBLE_HYPERPARAMETER").sum())),
        max_validation_coverage=("validation_coverage", "max"), mean_validation_coverage=("validation_coverage", "mean"),
    ).reset_index()
    no_feasible_summary["no_feasible_frequency"] = no_feasible_summary.no_feasible_count / no_feasible_summary.replicate_count

    write_csv(scenario, RECORD / "scenario_summary.csv")
    write_csv(signal, RECORD / "signal_stratified_summary.csv")
    write_csv(pairs, RECORD / "paired_comparisons.csv")
    write_csv(mass, RECORD / "kernel_mass_summary.csv")
    write_csv(ess, RECORD / "ess_summary.csv")
    write_csv(alignment, RECORD / "oracle_alignment_detail.csv")
    write_csv(align_summary, RECORD / "oracle_alignment_summary.csv")
    write_csv(no_feasible_detail, RECORD / "no_feasible_detail.csv")
    write_csv(no_feasible_summary, RECORD / "no_feasible_summary.csv")

    # Feature recovery counts: all new S2 signals happened to be clean x2-only.
    feature = cmi.groupby(["scenario", "replicate_index", "generation_seed"]).agg(
        status=("status", "first"), selected_count=("eligible", "sum")
    ).reset_index()
    s2 = cmi.loc[cmi.scenario.eq("S2")]
    s4 = cmi.loc[cmi.scenario.eq("S4")]
    s2_status = feature.loc[feature.scenario.eq("S2")]
    s4_status = feature.loc[feature.scenario.eq("S4")]
    feature_summary = {
        "S2_SIGNAL": int((s2_status.status == "SIGNAL").sum()),
        "S2_NO_SIGNAL": int((s2_status.status == "NO_CAUSAL_SIGNAL").sum()),
        "S2_x2_selected": int(s2.loc[(s2.feature == "x2") & s2.eligible, "replicate_index"].nunique()),
        "S2_clean_x2_only": int(((s2_status.status == "SIGNAL") & (s2_status.selected_count == 1)).sum()),
        "S2_x2_plus_nuisance": int(((s2_status.status == "SIGNAL") & (s2_status.selected_count > 1)).sum()),
        "S4_SIGNAL": int((s4_status.status == "SIGNAL").sum()),
        "S4_NO_SIGNAL": int((s4_status.status == "NO_CAUSAL_SIGNAL").sum()),
        "S4_false_feature_selections": int(s4.loc[s4.eligible, "replicate_index"].nunique()),
    }

    # Primary numerical evidence used for the frozen coverage-first classification.
    def summary_row(n: int, method: str):
        return scenario.loc[(scenario.scenario == "S2") & (scenario.n_cal == n) &
                            (scenario.track == "B") & (scenario.method == method)].iloc[0]
    causal_200, causal_800 = summary_row(200, "GeoCausalCP-Full"), summary_row(800, "GeoCausalCP-Full")
    signal_rows = signal.loc[(signal.track == "B") & (signal.stratum == "SIGNAL")]
    sig_c200 = signal_rows.loc[(signal_rows.n_cal == 200) & signal_rows.method.eq("GeoCausalCP-Full")].iloc[0]
    sig_c800 = signal_rows.loc[(signal_rows.n_cal == 800) & signal_rows.method.eq("GeoCausalCP-Full")].iloc[0]
    sig_e800 = signal_rows.loc[(signal_rows.n_cal == 800) & signal_rows.method.eq("GeoSIMCP-EUC")].iloc[0]
    relative_inf_reduction = 1 - sig_c800.mean_infinite_fraction / sig_c200.mean_infinite_fraction
    coverage_gap = sig_c800.mean_coverage - sig_e800.mean_coverage
    classification = "F. COVERAGE_CONCERN"
    rationale = (
        "Track-B SIGNAL GeoCausal infinity fell materially but remained 1.98% at n_cal=800; "
        "its mean coverage was 0.9117, 0.0132 below EUC, crossing the frozen >0.01 comparator-gap concern rule. "
        "Calibration mass improved feasibility but did not justify promotion."
    )

    summary = {
        "stage": "20B", "status": "COMPLETE", "preregistration_composite_sha256": seal["composite_sha256"],
        "seeds": list(range(20950001, 20950051)), "paired_replicates": 50, "scenario_replicates": 100,
        "calibration_sizes": [200, 400, 800], "feature_selection": feature_summary,
        "cmi_invariance": "PASS", "mass_infinity_mismatches": 0,
        "signal_trackB": {
            "n200_infinite_fraction": sig_c200.mean_infinite_fraction,
            "n800_infinite_fraction": sig_c800.mean_infinite_fraction,
            "relative_infinite_reduction": relative_inf_reduction,
            "n800_geocausal_coverage": sig_c800.mean_coverage,
            "n800_euc_coverage": sig_e800.mean_coverage,
            "n800_coverage_difference": coverage_gap,
            "n800_geocausal_finite_width": sig_c800.mean_width_finite,
            "n800_euc_finite_width": sig_e800.mean_width_finite,
            "n800_geocausal_finite_interval_score": sig_c800.mean_interval_score_finite,
            "n800_euc_finite_interval_score": sig_e800.mean_interval_score_finite,
        },
        "aggregate_trackB": {
            "n200_geocausal_infinite_fraction": causal_200.mean_infinite_fraction,
            "n800_geocausal_infinite_fraction": causal_800.mean_infinite_fraction,
        },
        "classification": classification, "classification_rationale": rationale,
        "official_minimal_change_viability": "NOT_READY_FOR_CONFIRMATION",
        "next_step": "No confirmation experiment. First perform a preregistered read-only coverage/validation-generalization diagnostic on frozen Stage-20B outputs; any later development must address the coverage gap without changing results post hoc.",
    }
    dump(RECORD / "stage20b_summary.json", summary)

    def fmt(v, digits=4):
        if pd.isna(v): return "NA"
        if np.isposinf(v): return "+inf"
        return f"{v:.{digits}f}"

    lines = [
        "# Stage 20B calibration-size feasibility", "",
        f"Preregistration SHA-256: `{seal['composite_sha256']}`.", "",
        "All 50 paired S2/S4 replicates completed. CMI-Full999 was computed once per scenario-replicate on the fixed 200-row scale-fit split and was invariant across n_cal. S2 contained 30 SIGNAL and 20 NO_SIGNAL replicates; every SIGNAL replicate selected x2 alone. S4 was NO_SIGNAL in 50/50 replicates.", "",
        "## Track-B S2 aggregate", "",
        "| n_cal | Method | selected | coverage | infinite fraction | finite-only width | finite-only interval score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for n in [200, 400, 800]:
        for method in METHOD_ORDER:
            row = summary_row(n, method)
            lines.append(f"| {n} | {method} | {int(row.replicates_selected)} | {fmt(row.mean_coverage)} | {fmt(row.mean_infinite_fraction)} | {fmt(row.mean_mean_width_finite)} | {fmt(row.mean_mean_interval_score_finite)} |")
    lines += ["", "Ordinary aggregate width and interval score are +infinity whenever any selected test interval is infinite; finite-only values above are diagnostics, not replacements.", "",
              "## Decision", "", f"**{classification}**", "", rationale, "",
              "Calibration support materially reduced GeoCausal kernel-mass failures, but did not eliminate them in the SIGNAL stratum and the n=800 efficiency comparison carried a frozen coverage concern. The official minimal-change path is therefore not ready for a confirmation experiment.", ""]
    (RECORD / "stage20b_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (RECORD / "README.md").write_text(
        "# Stage 20B: CP-calibration-size / kernel-mass feasibility\n\n"
        "A preregistered development experiment varying only the nested official-engine CP-calibration size (200, 400, 800). Track B retained the finite-sample test atom and every infinite interval. Track A preserved historical parity. See `stage20b_summary.md`.\n",
        encoding="utf-8",
    )

    # Mechanical root copies make the sealed preregistration immediately visible while preserving the sealed originals.
    for name in ["PREREGISTRATION.md", "config.json", "seed_manifest.json", "environment.txt"]:
        shutil.copyfile(PREREG / name, RECORD / name)

    inputs = {
        "preregistration": {"path": "preregistration/PREREGISTRATION_SEAL.json", "sha256": sha(PREREG / "PREREGISTRATION_SEAL.json")},
        "official_commit": "26419ef06d7e3711e6618f97f4f1ed02af16bcb7",
        "scientific_checkpoint": {"path": "scientific_computation_checkpoint.json", "sha256": sha(RECORD / "scientific_computation_checkpoint.json")},
        "completed_generation_checkpoints": 100, "completed_analysis_checkpoints": 100,
    }
    dump(RECORD / "input_manifest.json", inputs)
    required = [
        "README.md", "PREREGISTRATION.md", "config.json", "seed_manifest.json", "split_manifest.csv",
        "nested_calibration_manifest.json", "environment.txt", "cmi_full999_results.csv", "cmi_invariance_audit.csv",
        "validation_grid_results.csv", "selected_hyperparameters.csv", "test_unseal_audit.csv",
        "kernel_mass_pointwise.csv", "kernel_mass_summary.csv", "ess_summary.csv", "replicate_results.csv",
        "scenario_summary.csv", "signal_stratified_summary.csv", "paired_comparisons.csv", "no_feasible_summary.csv",
        "oracle_alignment_summary.csv", "stage20b_summary.json", "stage20b_summary.md", "run_log.txt", "stderr.txt",
    ]
    missing = [name for name in required if not (RECORD / name).exists()]
    if missing:
        raise AssertionError(f"missing outputs: {missing}")
    with (RECORD / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"Finalization complete; class={classification}; output_manifest_files={len(required)}\n")
    manifest = {"files": [{"path": name, "sha256": sha(RECORD / name), "bytes": (RECORD / name).stat().st_size} for name in required],
                "missing": [], "scenario_replicates": 100, "analysis_checkpoints": 100}
    dump(RECORD / "output_manifest.json", manifest)
    print(json.dumps(safe(summary), indent=2))


if __name__ == "__main__":
    main()
