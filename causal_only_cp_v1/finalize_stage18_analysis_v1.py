"""Derive preregistered Stage-18 summaries without changing scientific outputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/causal_only_cp/20260919_stage18_cmi_full_cp_translation_v1"
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
CMI_METHODS = ["CMI_MSPACE_999", "CMI_FULL_999"]
METHODS = ["Raw-CP", "GeoOnly-NormCP", "CausalOnly-CMI-Mspace999-NormCP", "CausalOnly-CMI-Full999-NormCP", "Oracle-NormCP"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RECORD/name, index=False, lineterminator="\n", float_format="%.17g")


def feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in ("S2", "S4"):
        for method in CMI_METHODS:
            d = frame[(frame.scenario == scenario) & (frame.method == method)]
            grouped = d.groupby("replicate_index")
            no_signal = grouped.status.first().eq("NO_CAUSAL_SIGNAL")
            x2 = d[d.feature.eq("x2")]
            selected_any_nuisance = grouped.apply(lambda g: bool(((g.feature != "x2") & g.eligible.astype(bool)).any()), include_groups=False)
            row = {"scenario": scenario, "cmi_method": method, "replicates": 50,
                   "x2_raw_rank1_count": int(x2["rank"].eq(1).sum()), "x2_selected_count": int(x2.eligible.astype(bool).sum()),
                   "x2_largest_weight_count": 0, "no_signal_count": int(no_signal.sum()),
                   "any_nuisance_selected_replicates": int(selected_any_nuisance.sum())}
            for feature in FEATURES:
                row[f"{feature}_selected_count"] = int(d.loc[d.feature.eq(feature), "eligible"].astype(bool).sum())
            largest = 0
            for _, g in grouped:
                gx = g.set_index("feature")
                if float(gx.loc["x2", "weight"]) > 0 and np.isclose(float(gx.loc["x2", "weight"]), float(g.weight.max()), atol=1e-15): largest += 1
            row["x2_largest_weight_count"] = largest
            rows.append(row)
    return pd.DataFrame(rows)


def scenario_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw_mean = results[results.method.eq("Raw-CP")].groupby("scenario").coverage.mean().to_dict()
    for (scenario, method), d in results.groupby(["scenario", "method"], sort=True):
        mean_cov = float(d.coverage.mean())
        rows.append({"scenario": scenario, "method": method, "replicates": len(d),
          "mean_coverage": mean_cov, "median_coverage": float(d.coverage.median()), "coverage_sd": float(d.coverage.std(ddof=1)),
          "min_coverage": float(d.coverage.min()), "max_coverage": float(d.coverage.max()),
          "pooled_coverage": float(d.covered_count.sum()/d.test_n.sum()), "coverage_below_0_90_count": int((d.coverage<.90).sum()),
          "coverage_below_0_89_count": int((d.coverage<.89).sum()), "mean_width": float(d.mean_width.mean()),
          "median_replicate_mean_width": float(d.mean_width.median()), "width_sd": float(d.mean_width.std(ddof=1)),
          "mean_interval_score": float(d.mean_interval_score.mean()), "median_interval_score": float(d.mean_interval_score.median()),
          "material_coverage_concern": bool(mean_cov < .89 or mean_cov < raw_mean[scenario]-.01)})
    return pd.DataFrame(rows)


def paired(results: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("Full_minus_Raw", "CausalOnly-CMI-Full999-NormCP", "Raw-CP"),
                   ("Full_minus_GeoOnly", "CausalOnly-CMI-Full999-NormCP", "GeoOnly-NormCP"),
                   ("Full_minus_Mspace", "CausalOnly-CMI-Full999-NormCP", "CausalOnly-CMI-Mspace999-NormCP")]
    metrics = ["coverage", "mean_width", "mean_interval_score"]
    rows = []; rng = np.random.default_rng(21918001)
    for scenario in ("S2", "S4"):
        d = results[results.scenario.eq(scenario)]
        for label, left, right in comparisons:
            a = d[d.method.eq(left)].set_index("replicate_index")
            b = d[d.method.eq(right)].set_index("replicate_index")
            if not a.index.equals(b.index) or len(a) != 50: raise AssertionError("paired alignment")
            for metric in metrics:
                diff = a[metric].to_numpy(float) - b[metric].to_numpy(float)
                se = diff.std(ddof=1)/math.sqrt(len(diff)); crit = stats.t.ppf(.975, len(diff)-1)
                draws = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
                right_mean = float(b[metric].mean())
                rows.append({"scenario": scenario, "comparison": label, "left_method": left, "right_method": right,
                    "metric": metric, "n": len(diff), "mean_difference": float(diff.mean()), "median_difference": float(np.median(diff)),
                    "sd_difference": float(diff.std(ddof=1)), "student_t_ci_lower": float(diff.mean()-crit*se),
                    "student_t_ci_upper": float(diff.mean()+crit*se), "bootstrap_ci_lower": float(np.quantile(draws,.025)),
                    "bootstrap_ci_upper": float(np.quantile(draws,.975)),
                    "relative_percent_difference": float(100*diff.mean()/right_mean) if right_mean else np.nan})
    return pd.DataFrame(rows)


def tradeoff(features: pd.DataFrame, results: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    fs = feature_summary(features)
    rows = []
    mapping = {"CMI_MSPACE_999": "CausalOnly-CMI-Mspace999-NormCP", "CMI_FULL_999": "CausalOnly-CMI-Full999-NormCP"}
    for scenario in ("S2", "S4"):
        for cmi, cp in mapping.items():
            f = fs[(fs.scenario==scenario)&(fs.cmi_method==cmi)].iloc[0]
            r = results[(results.scenario==scenario)&(results.method==cp)]
            sd = scales[(scales.method==cp)] if scenario=="S2" else pd.DataFrame()
            signal_sd = sd[sd.selection_stratum.ne("NO_SIGNAL")] if len(sd) else sd
            rows.append({"scenario": scenario, "cmi_method": cmi, "cp_method": cp,
                "x2_selection_rate": f.x2_selected_count/50, "x2_top_rate": f.x2_raw_rank1_count/50,
                "g_proxy_selection_rate": f.g_proxy_selected_count/50, "x1_selection_rate": f.x1_selected_count/50,
                "x_noise_selection_rate": f.x_noise_selected_count/50,
                "any_nuisance_selection_rate": f.any_nuisance_selected_replicates/50, "no_signal_rate": f.no_signal_count/50,
                "mean_oracle_correlation_signal": float(signal_sd.test_oracle_correlation.mean()) if len(signal_sd) else np.nan,
                "mean_oracle_mae_signal": float(signal_sd.test_oracle_mae.mean()) if len(signal_sd) else np.nan,
                "mean_coverage": float(r.coverage.mean()), "mean_width": float(r.mean_width.mean()),
                "mean_interval_score": float(r.mean_interval_score.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    seal = json.loads((RECORD/"PREREGISTRATION_SEAL.json").read_text())
    for name, expected in seal["files"].items():
        if sha(RECORD/name) != expected: raise AssertionError(f"sealed file changed: {name}")
    checkpoint = json.loads((RECORD/"execution_checkpoint.json").read_text())
    if checkpoint["status"] != "SCIENTIFIC_COMPUTATION_COMPLETE" or checkpoint["scenario_replicates"] != 100: raise AssertionError("checkpoint")
    mspace = pd.read_csv(RECORD/"cmi_mspace999_results.csv"); full = pd.read_csv(RECORD/"cmi_full999_results.csv")
    features = pd.concat([mspace,full],ignore_index=True); results = pd.read_csv(RECORD/"replicate_results.csv"); scales = pd.read_csv(RECORD/"scale_diagnostics.csv")
    if len(mspace)!=400 or len(full)!=400 or len(results)!=500 or len(scales)!=200: raise AssertionError("output completeness")
    fs = feature_summary(features); ss = scenario_summary(results); pc = paired(results); tq = tradeoff(features,results,scales)
    write_csv(fs,"feature_selection_summary.csv"); write_csv(ss,"scenario_summary.csv"); write_csv(pc,"paired_comparisons.csv"); write_csv(tq,"causal_quality_cp_tradeoff.csv")
    strata = scales.groupby(["method","selection_stratum"],dropna=False).agg(n=("replicate_index","size"),mean_oracle_correlation=("test_oracle_correlation","mean"),mean_oracle_mae=("test_oracle_mae","mean")).reset_index()
    write_csv(strata,"scale_stratified_summary.csv")
    analysis = {"status":"DERIVED_ANALYSIS_COMPLETE","preregistration_composite_sha256":seal["composite_sha256"],
        "scenario_replicates":100,"method_result_rows":500,"cmi_feature_rows_each":400,"test_unseal_rows":len(pd.read_csv(RECORD/"test_unseal_audit.csv")),
        "bootstrap_seed":21918001,"bootstrap_resamples":10000}
    (RECORD/"stage18_analysis.json").write_text(json.dumps(analysis,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(analysis,indent=2))


if __name__ == "__main__": main()
