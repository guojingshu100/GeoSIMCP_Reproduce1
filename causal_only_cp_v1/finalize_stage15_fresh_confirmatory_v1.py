"""Finalize the immutable Stage-15 fresh confirmatory simulation record.

This script performs only the analyses frozen in Stage-15A. It does not
generate data, refit predictors, rerun causal estimators, or alter intervals.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "experiments/confirmatory_simulation/20260918_stage15_fresh_confirmatory_v1"
PREREG = ROOT / "experiments/confirmatory_simulation/20260918_stage15_confirmatory_preregistration_v1"
RUNNER = ROOT / "causal_only_cp_v1/run_stage15_fresh_confirmatory_v1.py"
WORKER = ROOT / "causal_only_cp_v1/stage15_cmi_worker_v1.py"
METHODS = ["Raw-CP", "GeoOnly-NormCP", "CausalOnly-GCCM-NormCP",
           "CausalOnly-CMI-NormCP", "GeoCMI-Balanced-NormCP", "Oracle-NormCP"]
METRICS = ["coverage", "mean_width", "mean_interval_score"]
BOOT_SEED = 20260918
BOOT_N = 10000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(RECORD / name, index=False, lineterminator="\n", float_format="%.17g")


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def dump(name: str, value) -> None:
    (RECORD / name).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def scenario_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    nominal = 1 - alpha
    for scenario in ["S2", "S4"]:
        raw_cov = float(df[(df.scenario == scenario) & (df.method == "Raw-CP")].coverage.mean())
        for method in METHODS:
            x = df[(df.scenario == scenario) & (df.method == method)]
            rows.append({
                "alpha": alpha, "nominal_coverage": nominal, "scenario": scenario, "method": method,
                "replicates": len(x), "mean_coverage": x.coverage.mean(), "median_coverage": x.coverage.median(),
                "coverage_sd": x.coverage.std(ddof=1), "min_coverage": x.coverage.min(), "max_coverage": x.coverage.max(),
                "pooled_coverage": x.covered_count.sum() / x.test_n.sum(),
                "count_coverage_below_nominal": int((x.coverage < nominal).sum()),
                "count_coverage_below_nominal_minus_001": int((x.coverage < nominal - .01).sum()),
                "mean_width": x.mean_width.mean(), "median_replicate_mean_width": x.mean_width.median(),
                "width_sd": x.mean_width.std(ddof=1), "mean_interval_score": x.mean_interval_score.mean(),
                "median_interval_score": x.mean_interval_score.median(),
                "coverage_difference_vs_raw": x.coverage.mean() - raw_cov,
                "material_undercoverage": bool((x.coverage.mean() < nominal - .01) or
                                                (x.coverage.mean() < raw_cov - .01)),
            })
    return pd.DataFrame(rows)


def paired_ci(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, float)
    n = len(values); mean = float(values.mean()); sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(.975, n - 1) * sd / math.sqrt(n))
    # Fixed-seed percentile paired bootstrap of the replicate differences.
    means = values[rng.integers(0, n, size=(BOOT_N, n))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return {"n": n, "mean_difference": mean, "median_difference": float(np.median(values)),
            "sd_difference": sd, "t95_lower": mean - half, "t95_upper": mean + half,
            "bootstrap95_lower": float(lo), "bootstrap95_upper": float(hi)}


def paired_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    contrasts = [
        ("CausalOnly-CMI-NormCP", "Raw-CP"), ("CausalOnly-CMI-NormCP", "GeoOnly-NormCP"),
        ("GeoCMI-Balanced-NormCP", "CausalOnly-CMI-NormCP"),
        ("CausalOnly-GCCM-NormCP", "Raw-CP"), ("CausalOnly-GCCM-NormCP", "GeoOnly-NormCP"),
    ]
    rows = []; rng = np.random.default_rng(BOOT_SEED + int(alpha * 1000))
    for scenario in ["S2", "S4"]:
        sub = df[df.scenario == scenario]
        for method, comparator in contrasts:
            a = sub[sub.method == method].sort_values("replicate_index")
            b = sub[sub.method == comparator].sort_values("replicate_index")
            if not np.array_equal(a.replicate_index, b.replicate_index): raise AssertionError("pairing")
            for metric in METRICS:
                values = a[metric].to_numpy(float) - b[metric].to_numpy(float)
                row = {"alpha": alpha, "scenario": scenario, "method": method, "comparator": comparator,
                       "metric": metric, **paired_ci(values, rng)}
                base = float(b[metric].mean())
                row["relative_mean_difference_percent"] = 100 * row["mean_difference"] / base if base else None
                rows.append(row)
    return pd.DataFrame(rows)


def feature_recovery(cmi: pd.DataFrame, weights: pd.DataFrame, gccm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in ["S2", "S4"]:
        ws = weights[weights.scenario == scenario]
        per = ws.pivot(index="replicate_index", columns="feature", values="weight")
        status = ws.groupby("replicate_index").status.first()
        selected = per > 0
        top = per.idxmax(axis=1).where(per.max(axis=1) > 0, "NO_SIGNAL")
        rows.append({"estimator": "CMI99", "scenario": scenario, "replicates": len(per),
                     "x2_selected": int(selected.x2.sum()), "x2_top": int((top == "x2").sum()),
                     "no_signal": int((status == "NO_CAUSAL_SIGNAL").sum()),
                     "any_false_feature_selected": int(selected[["x1", "g_proxy", "x_noise"]].any(axis=1).sum()),
                     "x1_selected": int(selected.x1.sum()), "g_proxy_selected": int(selected.g_proxy.sum()),
                     "x_noise_selected": int(selected.x_noise.sum())})
        gs = gccm[gccm.scenario == scenario]
        gp = gs.pivot(index="replicate_index", columns="feature", values="normalized_weight").fillna(0)
        usable = gs.groupby("replicate_index").usable.sum()
        gtop = gp.idxmax(axis=1).where(gp.max(axis=1) > 0, "NO_SIGNAL")
        rows.append({"estimator": "GCCM", "scenario": scenario, "replicates": len(gp),
                     "x2_selected": int((gp.x2 > 0).sum()), "x2_top": int((gtop == "x2").sum()),
                     "no_signal": int((usable == 0).sum()),
                     "any_false_feature_selected": int((gp[["x1", "g_proxy", "x_noise"]] > 0).any(axis=1).sum()),
                     "x1_selected": int((gp.x1 > 0).sum()), "g_proxy_selected": int((gp.g_proxy > 0).sum()),
                     "x_noise_selected": int((gp.x_noise > 0).sum())})
    return pd.DataFrame(rows)


def stability(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    s2 = primary[primary.scenario == "S2"]
    for label, ids in [("first25", range(1, 26)), ("second25", range(26, 51))]:
        sub = s2[s2.replicate_index.isin(ids)]
        for comparator in ["Raw-CP", "GeoOnly-NormCP"]:
            a = sub[sub.method == "CausalOnly-CMI-NormCP"].sort_values("replicate_index")
            b = sub[sub.method == comparator].sort_values("replicate_index")
            rng = np.random.default_rng(BOOT_SEED + (0 if label == "first25" else 1))
            for metric in METRICS:
                rows.append({"subgroup": label, "scenario": "S2", "method": "CausalOnly-CMI-NormCP",
                             "comparator": comparator, "metric": metric,
                             **paired_ci(a[metric].to_numpy() - b[metric].to_numpy(), rng)})
    return pd.DataFrame(rows)


def signal_strata(primary: pd.DataFrame) -> pd.DataFrame:
    rows=[]; s2=primary[primary.scenario=="S2"]
    cmi=s2[s2.method=="CausalOnly-CMI-NormCP"].sort_values("replicate_index")
    for signal, label in [(True,"CMI_SIGNAL"),(False,"CMI_NO_SIGNAL")]:
        ids=cmi[cmi.signal_status.eq("SIGNAL") == signal].replicate_index
        a=cmi[cmi.replicate_index.isin(ids)].sort_values("replicate_index")
        for comparator in ["Raw-CP","GeoOnly-NormCP"]:
            b=s2[(s2.method==comparator)&s2.replicate_index.isin(ids)].sort_values("replicate_index")
            for metric in ["mean_width","mean_interval_score","coverage"]:
                d=a[metric].to_numpy()-b[metric].to_numpy()
                rows.append({"stratum":label,"n":len(d),"method":"CausalOnly-CMI-NormCP","comparator":comparator,
                             "metric":metric,"mean_difference":d.mean(),"median_difference":np.median(d),
                             "sd_difference":np.std(d,ddof=1) if len(d)>1 else None})
    return pd.DataFrame(rows)


def scale_summary(scale: pd.DataFrame) -> list[dict]:
    rows=[]
    sig=scale[(scale.scenario=="S2") & scale.cmi_signal.astype(bool)]
    for metric in ["test_oracle_correlation","test_oracle_mae"]:
        for method in ["GeoOnly-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP","Oracle-NormCP"]:
            x=sig[sig.method==method][metric]
            rows.append({"method":method,"metric":metric,"signal_replicates":len(x),"mean":x.mean(),"median":x.median(),"sd":x.std(ddof=1)})
        for method, comparator in [("CausalOnly-CMI-NormCP","GeoOnly-NormCP"),("GeoCMI-Balanced-NormCP","CausalOnly-CMI-NormCP")]:
            a=sig[sig.method==method].sort_values("replicate_index"); b=sig[sig.method==comparator].sort_values("replicate_index")
            d=a[metric].to_numpy()-b[metric].to_numpy(); n=len(d); m=d.mean(); sd=d.std(ddof=1); h=stats.t.ppf(.975,n-1)*sd/math.sqrt(n)
            rows.append({"method":method,"comparator":comparator,"metric":metric,"signal_replicates":n,
                         "mean_difference":m,"t95_lower":m-h,"t95_upper":m+h})
    return rows


def cmi999_summary(compare: pd.DataFrame, cp: pd.DataFrame) -> dict:
    feature=[]
    for scenario in ["S2","S4"]:
        x=compare[compare.scenario==scenario]
        by=x.pivot(index="replicate_index",columns="feature",values=["selected99","selected999","raw_cmi_99","raw_cmi_999"])
        s99=by["selected99"].astype(bool); s999=by["selected999"].astype(bool)
        feature.append({"scenario":scenario,"pairs":len(by),"x2_selected99":int(s99.x2.sum()),"x2_selected999":int(s999.x2.sum()),
                        "x2_top99":int((by["raw_cmi_99"].idxmax(axis=1)=="x2").sum()),
                        "x2_top999":int((by["raw_cmi_999"].idxmax(axis=1)=="x2").sum()),
                        "no_signal99":int((~s99.any(axis=1)).sum()),"no_signal999":int((~s999.any(axis=1)).sum()),
                        "selection_cells_changed":int((s99!=s999).to_numpy().sum()),
                        "max_raw_cmi_abs_difference":float(x.raw_abs_difference.max())})
    cps=[]
    for scenario in ["S2","S4"]:
        x=cp[cp.scenario==scenario]
        a=x[x.method=="CausalOnly-CMI999-NormCP"].sort_values("replicate_index")
        for method in ["CausalOnly-CMI-NormCP","Raw-CP","GeoOnly-NormCP","Oracle-NormCP"]:
            b=x[x.method==method].sort_values("replicate_index")
            for metric in METRICS:
                d=a[metric].to_numpy()-b[metric].to_numpy()
                cps.append({"scenario":scenario,"method":"CausalOnly-CMI999-NormCP","comparator":method,"metric":metric,
                            "n":len(d),"mean_difference":d.mean(),"median_difference":np.median(d),"sd_difference":d.std(ddof=1)})
    return {"feature_selection":feature,"cp_sensitivity":cps}


def markdown_table(df: pd.DataFrame, scenario: str) -> str:
    x=df[df.scenario==scenario]
    lines=["| Method | Coverage | Width | Interval score | Material undercoverage |","|---|---:|---:|---:|:---:|"]
    for m in METHODS:
        r=x[x.method==m].iloc[0]
        lines.append(f"| {m} | {r.mean_coverage:.4f} | {r.mean_width:.4f} | {r.mean_interval_score:.4f} | {'YES' if r.material_undercoverage else 'no'} |")
    return "\n".join(lines)


def main() -> None:
    checkpoint=json.loads((RECORD/"execution_checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint["status"]!="SCIENTIFIC_COMPUTATION_COMPLETE": raise RuntimeError("scientific execution incomplete")
    seal=json.loads((PREREG/"PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    primary=pd.read_csv(RECORD/"replicate_results_alpha010.csv")
    a005=pd.read_csv(RECORD/"replicate_results_alpha005.csv"); a020=pd.read_csv(RECORD/"replicate_results_alpha020.csv")
    if len(primary)!=600 or len(a005)!=600 or len(a020)!=600: raise AssertionError("result row count")
    cmi=pd.read_csv(RECORD/"cmi99_feature_results.csv"); weights=pd.read_csv(RECORD/"cmi99_weight_audit.csv")
    gccm=pd.read_csv(RECORD/"gccm_weight_audit.csv"); scale=pd.read_csv(RECORD/"scale_diagnostics.csv")
    summaries=[]; pairs=[]
    for alpha,frame,tag in [(.10,primary,"010"),(.05,a005,"005"),(.20,a020,"020")]:
        ss=scenario_summary(frame,alpha); pp=paired_summary(frame,alpha); summaries.append(ss); pairs.append(pp)
        write_csv(ss,f"scenario_summary_alpha{tag}.csv")
        if alpha==.10: write_csv(pp,"paired_comparisons_alpha010.csv")
    robust=pd.concat(summaries,ignore_index=True); write_csv(robust,"alpha_robustness_summary.csv")
    feature=feature_recovery(cmi,weights,gccm); write_csv(feature,"feature_recovery_summary.csv")
    stab=stability(primary); write_csv(stab,"stability_first25_second25.csv")
    strata=signal_strata(primary); write_csv(strata,"signal_stratified_summary.csv")
    c999=cmi999_summary(pd.read_csv(RECORD/"cmi999_comparison.csv"),pd.read_csv(RECORD/"cmi999_cp_sensitivity.csv"))
    p010=pairs[0]
    s2sum=summaries[0][summaries[0].scenario=="S2"].set_index("method")
    cm=s2sum.loc["CausalOnly-CMI-NormCP"]; raw=s2sum.loc["Raw-CP"]; geo=s2sum.loc["GeoOnly-NormCP"]
    coverage_fail=bool(cm.material_undercoverage)
    def p(metric, comp):
        return p010[(p010.scenario=="S2")&(p010.method=="CausalOnly-CMI-NormCP")&(p010.comparator==comp)&(p010.metric==metric)].iloc[0]
    eff=[p("mean_width","Raw-CP"),p("mean_interval_score","Raw-CP"),p("mean_width","GeoOnly-NormCP"),p("mean_interval_score","GeoOnly-NormCP")]
    directions=sum(float(x.mean_difference)<0 for x in eff)
    ci_raw=any((p(m,"Raw-CP").t95_upper<0) for m in ["mean_width","mean_interval_score"])
    ci_geo=any((p(m,"GeoOnly-NormCP").t95_upper<0) for m in ["mean_width","mean_interval_score"])
    practical_raw=(-p("mean_width","Raw-CP").mean_difference/raw.mean_width>=.01) or (-p("mean_interval_score","Raw-CP").mean_difference/raw.mean_interval_score>=.02)
    practical_geo=(-p("mean_width","GeoOnly-NormCP").mean_difference/geo.mean_width>=.01) or (-p("mean_interval_score","GeoOnly-NormCP").mean_difference/geo.mean_interval_score>=.02)
    if coverage_fail: classification="D. CONFIRMATORY_COVERAGE_CONCERN"
    elif directions==4 and ci_raw and ci_geo and practical_raw and practical_geo: classification="A. CONFIRMATORY_SUPPORT"
    elif directions>=3: classification="B. PARTIAL_CONFIRMATORY_SUPPORT"
    else: classification="C. CONFIRMATORY_EFFICIENCY_NOT_REPLICATED"
    runtime=json.loads((RECORD/"cmi999_runtime_benchmark.json").read_text(encoding="utf-8"))
    scale_rows=scale_summary(scale)
    summary={"status":"COMPLETE","preregistration_composite_sha256":seal["composite_sha256"],
      "fresh_seed_range":[20320001,20320050],"successful_paired_replicates":50,"successful_scenario_replicates":100,
      "preflight":json.loads((RECORD/"preflight_summary.json").read_text(encoding="utf-8")),
      "classification":classification,"scenario_summaries":clean(robust.to_dict("records")),
      "primary_paired_comparisons":clean(p010.to_dict("records")),"feature_recovery":clean(feature.to_dict("records")),
      "stability":clean(stab.to_dict("records")),"signal_strata":clean(strata.to_dict("records")),
      "scale_recovery":clean(scale_rows),"cmi999_runtime_decision":runtime,"cmi999":clean(c999),
      "stage14_status":"PM2.5 retrospective exploratory result = NO_CAUSAL_SIGNAL_DETECTED; not rerun.",
      "supported_method_family":METHODS,"primary_practical_method":"CausalOnly-CMI-NormCP" if classification.startswith(("A.","B.")) else "not newly confirmed"}
    dump("confirmatory_summary.json",summary)
    text=f"""# Stage 15 fresh confirmatory simulation\n\n**Classification: {classification}**\n\nThe experiment used 50 newly generated paired replicates (seeds 20320001–20320050) that were absent from prior project manifests. The method family and analysis rules were sealed first under preregistration composite SHA-256 `{seal['composite_sha256']}`.\n\n## Alpha 0.10 results\n\n### S2\n\n{markdown_table(summaries[0],'S2')}\n\n### S4\n\n{markdown_table(summaries[0],'S4')}\n\n## Feature recovery\n\n{feature.to_markdown(index=False)}\n\n## Interpretation\n\nCoverage was evaluated before efficiency using the preregistered threshold. Paired method-minus-comparator effects, both prespecified Student-t intervals and fixed-seed 10,000-resample bootstrap intervals, are in `paired_comparisons_alpha010.csv`. Alpha-level robustness, stability halves, signal strata, oracle-scale diagnostics, and the CMI-999 Monte-Carlo-resolution sensitivity are retained in separate machine-readable files.\n\nThe Stage-14 PM2.5 result was not rerun: it remains a retrospective exploratory NO_CAUSAL_SIGNAL result. No real-data confirmatory claim is made.\n"""
    (RECORD/"confirmatory_summary.md").write_text(text,encoding="utf-8")
    (RECORD/"README.md").write_text("# Stage 15 fresh confirmatory record\n\nThis immutable record executes the separately sealed Stage-15A preregistration on 50 fresh paired S2/S4 simulation replicates. See `confirmatory_summary.md` for results and `METHOD.md` for the frozen implementation.\n",encoding="utf-8")
    (RECORD/"METHOD.md").write_text("# Frozen method\n\nThe run used the unmodified S2/S4 generator, %.17g serialize/reload boundary, train-only historical 300-tree RF, Mspace CMIknn (k=20; 99 surrogates; four-feature BH), original GCCM protocol, and the six frozen conformal methods. Local k was selected from {20,40,60,80} using tuning residual-scale MAE. Final calibration alone set the finite-sample quantile. Test labels were unsealed only after all parameters were recorded. Alpha .05/.20 changed only calibration quantiles and intervals.\n",encoding="utf-8")
    (RECORD/"CLAIM_BOUNDARY.md").write_text("# Claim boundary\n\nThese are fresh confirmatory simulation replicates for a method family frozen before generation. They support claims under the audited simulation assumptions, not universal causal identification or real-data generalization. CMI is described as causally motivated conditional-dependence feature strength. Historical real-data labels remain retrospective/exploratory.\n",encoding="utf-8")
    manuscript=f"""# Manuscript notes\n\n## Experimental design\n\nWe evaluated six frozen split-conformal methods on 50 new paired S2 heteroscedastic and S4 homoscedastic replicates (seeds 20320001–20320050). Before generating outcomes, the generator, predictor, causal estimators, localization rules, conformal protocol, hypotheses, and analyses were sealed (SHA-256 `{seal['composite_sha256']}`). Predictors used training observations only; causally motivated conditional-dependence feature strengths used scale-fit observations only; neighborhood size used tuning calibration only; conformal quantiles used final calibration only; test labels remained sealed until all method parameters were fixed.\n\n## Main confirmatory result\n\n{classification}. At alpha=.10, CausalOnly-CMI-NormCP had S2 coverage {cm.mean_coverage:.4f}, mean width {cm.mean_width:.4f}, and mean interval score {cm.mean_interval_score:.4f}, versus Raw-CP ({raw.mean_coverage:.4f}, {raw.mean_width:.4f}, {raw.mean_interval_score:.4f}) and GeoOnly-NormCP ({geo.mean_coverage:.4f}, {geo.mean_width:.4f}, {geo.mean_interval_score:.4f}). Interpretation is coverage-first under the preregistered material-undercoverage rule.\n\n## Feature recovery\n\nSee `feature_recovery_summary.csv`. CMI feature selection was fixed at BH q<.05 and positive raw CMI; no feature was forced when the gate returned no signal.\n\n## Ablations\n\nThe complete table retains GCCM and balanced geography-plus-CMI as negative or mechanistic ablations, avoiding selective removal based on performance.\n\n## Coverage-level robustness\n\nResults at nominal 95%, 90%, and 80% coverage reused the identical fitted scale functions and selected k; only conformal quantiles and test intervals changed (`alpha_robustness_summary.csv`).\n\n## CMI-999 sensitivity\n\nThe preregistered runtime gate selected {runtime['selected_pair_count']} paired replicates. This analysis is a Monte-Carlo p-value-resolution sensitivity and does not replace the primary 99-surrogate result.\n\n## Real-data claim boundary\n\nThe existing PM2.5 analysis was retrospective and exploratory and detected no causal signal, correctly triggering Raw-CP fallback. It is not independent validation; a meaningful next real-data test requires genuinely future data or a demonstrably unused holdout.\n\n## Limitations\n\nThe conclusions are specific to the frozen simulation mechanisms, predictor, candidate features, sample sizes, and local mean-residual scale estimator. CMI provides conditional-dependence evidence under simulation assumptions rather than universal causal identification. The CMI-999 analysis is secondary, and real-data transport remains unresolved.\n"""
    (RECORD/"MANUSCRIPT_NOTES.md").write_text(manuscript,encoding="utf-8")
    (RECORD/"environment.txt").write_text(f"Python executable: {sys.executable}\nPython: {platform.python_version()}\nPlatform: {platform.platform()}\nnumpy: {np.__version__}\npandas: {pd.__version__}\nscipy: {stats.__version__ if hasattr(stats,'__version__') else 'see scipy package'}\n",encoding="utf-8")
    inputs=[]
    for p in [PREREG/"PREREGISTRATION_SEAL.json",RUNNER,WORKER,ROOT/"simulation_pilot_20260908/generator/generate_scenarios.py",ROOT/"simulation_pilot_20260908/predictor/fit_predictors.py"]:
        inputs.append({"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size})
    dump("input_manifest.json",{"inputs":inputs,"frozen_scenario_artifacts":"frozen_inputs/ (100 scenario directories; individually hashed in COMPLETE.json files)"})
    # Output manifest last: intentionally excludes itself and mutable run/stderr while finalization is running.
    outputs=[]
    for p in sorted(RECORD.glob("*")):
        if p.is_file() and p.name not in {"output_manifest.json","run_log.txt","stderr.txt"}:
            outputs.append({"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size})
    dump("output_manifest.json",{"outputs":outputs,"large_frozen_artifacts_referenced_by_nested_COMPLETE_manifests":True})
    print(json.dumps({"classification":classification,"rows":len(primary),"cmi999_pairs":runtime["selected_pair_count"]},indent=2))


if __name__ == "__main__": main()
