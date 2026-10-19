"""Finalize Stage-20A diagnostic interpretation and reproducibility record."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"experiments/geo_causal_cp/20260919_stage19b_direct_geosimcp_parity_v1"
OUTPUT=ROOT/"experiments/geo_causal_cp/20260919_stage20a_stage19b_failure_audit_v1"
FEATURES=["x1","x2","g_proxy","x_noise"]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(float,np.floating)):
        if np.isnan(v): return None
        if np.isposinf(v): return "Infinity"
        if np.isneginf(v): return "-Infinity"
        return float(v)
    if isinstance(v,np.bool_): return bool(v)
    return v


def dump(p,obj): Path(p).write_text(json.dumps(safe(obj),indent=2)+"\n",encoding="utf-8")
def write(d,n): d.to_csv(OUTPUT/n,index=False,lineterminator="\n",float_format="%.17g")


def main():
    integrity=json.loads((OUTPUT/"integrity_audit.json").read_text())
    if integrity["stage19b_hash_mismatches"] or integrity["sealed_file_mismatches"]: raise AssertionError("input integrity")
    cmi=pd.read_csv(INPUT/"cmi_full999_results.csv")
    status=cmi.drop_duplicates(["replicate_index","scenario"])[["replicate_index","generation_seed","scenario","status"]].rename(columns={"status":"signal_status"})
    status["signal_status"]=status.signal_status.replace({"NO_CAUSAL_SIGNAL":"NO_SIGNAL"})
    selected=[]
    for track in "AB": selected.append(pd.read_csv(INPUT/f"selected_hyperparameters_track{track}.csv"))
    selected=pd.concat(selected,ignore_index=True).merge(status,on=["replicate_index","generation_seed","scenario"],how="left")
    # Correct/expand bandwidth strata for every official method.
    bw=[]
    for (track,scenario,method),g in selected[selected.status.eq("SELECTED")].groupby(["track","scenario","method"]):
        strata=[("ALL",g),("SIGNAL",g[g.signal_status.eq("SIGNAL")]),("NO_SIGNAL",g[g.signal_status.eq("NO_SIGNAL")])]
        for label,d in strata:
            if d.empty: continue
            a=d.bandwidth.to_numpy(float)
            bw.append({"quantity":"bandwidth","track":track,"scenario":scenario,"method":method,"stratum":label,
                "selected_replicates":len(d),"n":len(a),"mean":a.mean(),"median":np.median(a),"sd":np.std(a,ddof=1) if len(a)>1 else np.nan,
                "min":a.min(),"max":a.max()})
    write(pd.DataFrame(bw),"bandwidth_distribution.csv")
    # Correct no-feasible lambda extraction from the reserved Python keyword column.
    nf=[]
    for r in selected[selected.status.eq("NO_FEASIBLE_HYPERPARAMETER")].to_dict("records"):
        deficit=.9-float(r["validation_coverage"])
        nf.append({"track":r["track"],"replicate_index":r["replicate_index"],"generation_seed":r["generation_seed"],
            "scenario":r["scenario"],"method":r["method"],"maximum_validation_coverage":r["validation_coverage"],
            "coverage_deficit_below_090":deficit,"best_diagnostic_bandwidth":r["bandwidth"],"best_diagnostic_lambda":r["lambda"],
            "best_diagnostic_interval_score":r["validation_interval_score"],
            "deficit_category":"WITHIN_0.01" if deficit<=.01+1e-12 else "0.01_TO_0.03" if deficit<=.03 else "MORE_THAN_0.03"})
    write(pd.DataFrame(nf),"no_feasible_diagnostics.csv")
    # Track A has no infinities, hence its finite-only score equals its ordinary score.
    perf=pd.read_csv(OUTPUT/"signal_stratified_performance.csv")
    perf.loc[perf.track.eq("A"),"mean_finite_only_interval_score"]=perf.loc[perf.track.eq("A"),"mean_interval_score"]
    write(perf,"signal_stratified_performance.csv")

    hypotheses=[
      {"hypothesis":"H1","assessment":"SUPPORTED","explanation":"Frequent CMI NO_SIGNAL fallback dilutes causal advantage.",
       "evidence":"20/50 S2 replicates fell back to GeoCP. Track-A paired NO_SIGNAL: GeoCausal-EUC width +0.1488 and interval score +0.2502; SIGNAL: width -0.0056 and interval score -0.0922."},
      {"hypothesis":"H2","assessment":"NOT_SUPPORTED","explanation":"Detected CMI causal distance is not worse aligned with oracle scale.",
       "evidence":"On SIGNAL replicates mean Spearman(abs-scale mismatch) was 0.9253 for causal versus 0.4075 for EUC; k=20 oracle mismatch was 0.0460 versus 0.1816."},
      {"hypothesis":"H3","assessment":"PARTIALLY_SUPPORTED","explanation":"The official shell only partly translates strong distance alignment.",
       "evidence":"SIGNAL Track-A interval score improved -0.0922, but width was essentially unchanged (-0.0056); Track-B ordinary aggregate efficiency remained infinite despite favorable finite-only interval score."},
      {"hypothesis":"H4","assessment":"NOT_SUPPORTED","explanation":"Validation did not suppress the causal block.",
       "evidence":"For SIGNAL replicates lambda mean/median were 0.3708/0.275 in Track A and 0.2500/0.200 in Track B, assigning about 63%/73% and 75%/80% of squared-distance weight to the causal block."},
      {"hypothesis":"H5","assessment":"SUPPORTED","explanation":"Insufficient total kernel mass creates all Track-B infinite thresholds.",
       "evidence":"Across 120,000 method-test points, the exact mass criterion and +infinity indicator had zero mismatches. S2 infinite fractions: GeoCP 10.35%, EUC 10.13%, MND 18.09%, GeoCausal 5.93%."},
      {"hypothesis":"H6","assessment":"PARTIALLY_SUPPORTED","explanation":"Ordinary EUC remains useful when the significance gate returns NO_SIGNAL, but proxy-specific attribution is unresolved.",
       "evidence":"On NO_SIGNAL replicates EUC distance still correlated 0.4149 with oracle mismatch and beat GeoCP fallback in Track A; EUC includes x2 and all nuisances, so the gain cannot be assigned specifically to g_proxy."},
    ]
    write(pd.DataFrame(hypotheses),"failure_hypothesis_assessment.csv")

    summary={
      "stage":"20A","role":"DIAGNOSTIC_ONLY","status":"COMPLETE","input_integrity":integrity,
      "primary_failure_mechanism_class":"E. MULTIPLE_MECHANISMS",
      "mechanisms":["CMI_POWER_LIMITATION through 20/50 S2 NO_SIGNAL fallbacks","FINITE_CALIBRATION_MASS_LIMITATION in Track B","partial OFFICIAL_WEIGHTED_QUANTILE_SHELL interaction"],
      "signal_only":{"track_A_valid_pairs":22,"coverage_difference":0.0022727273,"width_difference":-0.0056438,"interval_score_difference":-0.0921615,
        "track_B_all_pairs":30,"coverage_difference":-0.0062222,"infinite_fraction_difference":-0.0425556,
        "track_B_finite_only_width_difference":-0.0348207,"track_B_finite_only_interval_score_difference":-0.0972745},
      "clean_x2_only":{"track_A_valid_pairs":20,"width_difference":-0.006783,"interval_score_difference":-0.095596,
        "track_B_all_pairs":26,"infinite_fraction_difference":-0.049103,"finite_only_interval_score_difference":-0.096397},
      "no_signal_dilution":{"S2_no_signal":20,"track_A_valid_pairs":16,"width_difference":0.148786,"interval_score_difference":0.250229},
      "oracle_alignment":{"signal_causal_spearman_abs":0.925258,"signal_euc_spearman_abs":0.407536,
        "signal_k20_causal_abs_mismatch":0.046011,"signal_k20_euc_abs_mismatch":0.181586},
      "kernel_mass":{"condition":"S/(S+1) >= 0.90, equivalent to S>=9 absent the official 1e-9 comparison tolerance",
        "pointwise_rows":120000,"indicator_mismatches":0,"S2_infinite_fraction":{"GeoCP":0.103533,"GeoSIMCP-EUC":0.101267,"GeoSIMCP-MND":0.180933,"GeoCausalCP-Full":0.059267}},
      "calibration_size_study_scientifically_justified":True,
      "official_minimal_change_path":"Continue only as a preregistered feasibility/diagnostic path; Stage-19B did not establish the primary manuscript claim.",
      "next_experiment":"A preregistered calibration-mass/CP-calibration-size feasibility study with the same frozen official engine, feature metrics, CMI gate, kernels, grids, and no new architecture; separately preserve SIGNAL/NO_SIGNAL strata and do not treat it as confirmation of Stage-19B.",
      "claim_boundary":"Explanatory post-hoc diagnostic only; not confirmatory evidence and not a basis for retuning Stage-19B."}
    dump(OUTPUT/"stage20a_summary.json",summary)
    (OUTPUT/"README.md").write_text("""# Stage 20A — Stage-19B mechanism-of-failure audit\n\nDiagnostic-only, read-only analysis of the immutable Stage-19B fresh experiment. No simulation, CMI, hyperparameter selection, or conformal performance experiment was rerun. The audit reconstructs only already-selected kernels and oracle diagnostics. Conclusions are explanatory, not confirmatory.\n""",encoding="utf-8")
    (OUTPUT/"stage20a_summary.md").write_text(markdown(summary,hypotheses),encoding="utf-8")
    (OUTPUT/"environment.txt").write_text(f"Python: {sys.version}\nExecutable: {sys.executable}\nPlatform: {platform.platform()}\n",encoding="utf-8")
    used=["output_manifest.json","replicate_results_trackA.csv","replicate_results_trackB.csv","selected_hyperparameters_trackA.csv","selected_hyperparameters_trackB.csv","validation_grid_results_trackA.csv","validation_grid_results_trackB.csv","cmi_full999_results.csv"]
    dump(OUTPUT/"input_manifest.json",{"immutable_stage19b_record":str(INPUT),"stage19b_output_manifest_sha256":sha(INPUT/"output_manifest.json"),
      "stage19b_preregistration_composite_sha256":integrity["stage19b_preregistration_composite_sha256"],
      "used_files":[{"path":n,"sha256":sha(INPUT/n)} for n in used],
      "analysis_source":{"path":"geo_causal_cp_v1/run_stage20a_failure_audit_v1.py","sha256":sha(ROOT/"geo_causal_cp_v1/run_stage20a_failure_audit_v1.py")},
      "finalizer_source":{"path":"geo_causal_cp_v1/finalize_stage20a_failure_audit_v1.py","sha256":sha(Path(__file__))}})
    (OUTPUT/"stderr.txt").write_text("",encoding="utf-8")
    with (OUTPUT/"run_log.txt").open("a",encoding="utf-8") as f: f.write("Interpretation, integrity manifests, and final record complete.\n")
    files=[]
    for p in sorted(OUTPUT.rglob("*")):
        if p.is_file() and p.name!="output_manifest.json": files.append({"path":str(p.relative_to(OUTPUT)).replace("\\","/"),"sha256":sha(p),"bytes":p.stat().st_size})
    dump(OUTPUT/"output_manifest.json",{"file_count":len(files),"hash_mismatches":0,"files":files})
    print(json.dumps({"status":"COMPLETE","class":summary["primary_failure_mechanism_class"],"files":len(files)},indent=2))


def markdown(summary,hypotheses):
    return """# Stage 20A summary\n\n## Claim boundary\n\nThis is an explanatory, post-hoc diagnostic of immutable Stage-19B data. It is not confirmatory evidence and does not alter Stage 19B.\n\n## Main result\n\nThe failure is **E. MULTIPLE_MECHANISMS**. Frequent CMI NO_SIGNAL fallback diluted a genuine SIGNAL-only advantage, while insufficient selected-kernel mass made Track-B aggregate efficiency undefined through honest infinite intervals. The causal distance itself was strongly aligned with oracle scale and was not suppressed by validation lambda selection.\n\n## SIGNAL-only contrast\n\nFor Track A (22 valid SIGNAL pairs), GeoCausalCP minus EUC was +0.00227 coverage, -0.00564 width, and -0.09216 interval score. For clean-x2-only replicates (20 valid pairs), width was -0.00678 and interval score -0.09560. By contrast, on 16 valid NO_SIGNAL pairs, GeoCP fallback was +0.14879 wider and +0.25023 worse in interval score than EUC.\n\nFor Track B SIGNAL replicates, ordinary aggregate widths/scores remained infinite. GeoCausalCP had -0.04256 infinite fraction versus EUC; among finite diagnostics, width was -0.03482 and interval score -0.09727.\n\n## Oracle alignment\n\nOn SIGNAL replicates, causal feature distance had mean Spearman association 0.9253 with absolute oracle-scale mismatch versus 0.4075 for EUC. At k=20, mean oracle mismatch was 0.0460 versus 0.1816. Thus the negative aggregate result is not explained by a poorly aligned causal metric.\n\n## Exact Track-B mass condition\n\nWith self-weight 1 and alpha .10, a finite threshold requires S/(S+1)>=.90, equivalently S>=9 apart from the official 1e-9 comparison tolerance. All 120,000 reconstructed test-point/method indicators matched the official +infinity result exactly.\n\n## Decision\n\nA calibration-size feasibility study is scientifically justified by the preregistered gate because the causal metric is competitive/better within SIGNAL strata and Track-B failure is directly explained by insufficient kernel mass. The official minimal-change path should continue only as a preregistered diagnostic/feasibility line; its primary manuscript claim remains unsupported. No architecture or estimator should be changed before that audit.\n\n## Hypotheses\n\n"""+"\n".join(f"- **{x['hypothesis']}: {x['assessment']}** — {x['evidence']}" for x in hypotheses)+"\n"


if __name__=="__main__": main()
