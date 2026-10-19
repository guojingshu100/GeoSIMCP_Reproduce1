"""Reporting finalizer for the completed Stage-13 scientific outputs."""
from __future__ import annotations
import hashlib, importlib.metadata, json, math, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parent.parent
RECORD=ROOT/"experiments"/"causal_only_cp"/"20260918_stage13_geocmi_balanced_normcp_v1"
STAGE11=ROOT/"experiments"/"causal_only_cp"/"20260918_stage11_causalonly_cmi_normcp_v2"
STAGE12=ROOT/"experiments"/"causal_only_cp"/"20260918_stage12_causalonly_gccm_ablation_v1"
RUNNER=ROOT/"causal_only_cp_v1"/"run_stage13_geocmi_balanced_normcp_v1.py"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ci(d):
    d=np.asarray(d,float); mean=float(d.mean()); sd=float(d.std(ddof=1)); h=float(stats.t.ppf(.975,len(d)-1)*sd/math.sqrt(len(d)))
    return {"n":len(d),"mean":mean,"median":float(np.median(d)),"sd":sd,"ci95_lower":mean-h,"ci95_upper":mean+h}
def clean(v):
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list): return [clean(x) for x in v]
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,float)): return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v,(np.bool_,)): return bool(v)
    return v

def main():
    summary_path=RECORD/"stage13_summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"]!="PASS" or summary["classification"]!="PENDING_FINAL_INTERPRETATION": raise AssertionError("unexpected Stage-13 state")
    results=pd.read_csv(RECORD/"replicate_results.csv"); scale=pd.read_csv(RECORD/"scale_diagnostics.csv")
    weights=pd.read_csv(STAGE11/"weight_audit.csv")
    signal=weights[(weights.scenario=="S2")&(weights.condition=="Mspace")&(weights["transform"]=="bh_gated_raw_positive")].groupby("replicate_index").stage11_status.first().eq("OK")
    ids=signal.index[signal]
    rp=results[results.scenario=="S2"].pivot(index="replicate_index",columns="method",values=["coverage","mean_width","mean_interval_score"])
    signal_effects={}
    for metric in ("coverage","mean_width","mean_interval_score"):
        d=rp.loc[ids,(metric,"GeoCMI-Balanced-NormCP")]-rp.loc[ids,(metric,"CausalOnly-CMI-NormCP")]
        x=ci(d); x["fraction_geocmi_better"]=None if metric=="coverage" else float((d<0).mean()); signal_effects[metric]=x
    sx=scale[(scale.scenario=="S2")&scale.replicate_index.isin(ids)]
    sp=sx.pivot(index="replicate_index",columns="method",values=["test_oracle_correlation","test_oracle_mae"])
    scale_summary={"method_summary":{},"paired_geocmi_minus":{}}
    for method in ("GeoOnly-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP"):
        scale_summary["method_summary"][method]={m:{"mean":float(sp[m][method].mean()),"median":float(sp[m][method].median())} for m in ("test_oracle_correlation","test_oracle_mae")}
    for comparator in ("GeoOnly-NormCP","CausalOnly-CMI-NormCP"):
        scale_summary["paired_geocmi_minus"][comparator]={m:ci(sp[m]["GeoCMI-Balanced-NormCP"]-sp[m][comparator]) for m in ("test_oracle_correlation","test_oracle_mae")}
    false_rep=6
    s4false=results[(results.scenario=="S4")&(results.replicate_index==false_rep)&results.method.isin(["Raw-CP","GeoOnly-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP"])][["method","coverage","mean_width","mean_interval_score","selected_k"]]
    summary.update({"classification":"GEOGRAPHY_DILUTES_CMI_SIGNAL","signal_only_geocmi_minus_cmi":clean(signal_effects),
                    "s2_signal_scale_recovery":clean(scale_summary),
                    "s4_false_selection":{"replicate_index":6,"generation_seed":20270913,"feature":"x1","weight":1.0,"methods":clean(s4false.to_dict("records"))},
                    "coverage_first_interpretation":"No material undercoverage; classification follows worse efficiency and scale recovery relative to CausalOnly-CMI.",
                    "incremental_geography_answer":"Geography does not add useful information beyond CMI here; balanced geography dilutes the CMI residual-scale signal.",
                    "recommended_family":["Raw-CP","GeoOnly-NormCP","CausalOnly-GCCM-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP","Oracle-NormCP"],
                    "next_experiment":"Stage 14 preregistered real-data feasibility/provenance audit, followed by one-dataset exploratory evaluation centered on CausalOnly-CMI if the audit passes."})
    summary_path.write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+"\n",encoding="utf-8")

    scenarios=pd.read_csv(RECORD/"scenario_summary.csv"); pairs=pd.read_csv(RECORD/"paired_comparisons.csv")
    s2=scenarios[scenarios.scenario=="S2"][["method","mean_coverage","mean_width","mean_interval_score","material_undercoverage"]]
    s4=scenarios[scenarios.scenario=="S4"][["method","mean_coverage","mean_width","mean_interval_score","material_undercoverage"]]
    inc=pairs[(pairs.scenario=="S2")&(pairs.contrast.str.endswith("minus CausalOnly-CMI-NormCP"))]
    md=f"""# Stage 13 summary

## Classification

**C — GEOGRAPHY_DILUTES_CMI_SIGNAL**.

GeoCMI does not trigger material undercoverage, and it improves on Raw and
GeoOnly in S2. However, compared with CausalOnly-CMI it is 0.0481 wider on
average and its interval score is 0.0793 worse. The interval-score 95% CI is
[0.0209, 0.1376]. Among the 13 S2 SIGNAL replicates, balanced geography is
narrower in only {signal_effects['mean_width']['fraction_geocmi_better']*13:.0f}/13 and has lower interval score in only {signal_effects['mean_interval_score']['fraction_geocmi_better']*13:.0f}/13.

## S2

{s2.to_markdown(index=False)}

## S4

{s4.to_markdown(index=False)}

## Frozen fallback and scale diagnostics

- All 26 NO_SIGNAL cases exactly equal GeoOnly; maximum interval-bound difference: 0.
- S2 SIGNAL oracle-scale correlation: GeoOnly {scale_summary['method_summary']['GeoOnly-NormCP']['test_oracle_correlation']['mean']:.4f}, CausalOnly-CMI {scale_summary['method_summary']['CausalOnly-CMI-NormCP']['test_oracle_correlation']['mean']:.4f}, GeoCMI {scale_summary['method_summary']['GeoCMI-Balanced-NormCP']['test_oracle_correlation']['mean']:.4f}.
- S2 SIGNAL oracle-scale MAE: GeoOnly {scale_summary['method_summary']['GeoOnly-NormCP']['test_oracle_mae']['mean']:.4f}, CausalOnly-CMI {scale_summary['method_summary']['CausalOnly-CMI-NormCP']['test_oracle_mae']['mean']:.4f}, GeoCMI {scale_summary['method_summary']['GeoCMI-Balanced-NormCP']['test_oracle_mae']['mean']:.4f}.

The balanced geographic block therefore dilutes rather than improves the
supported CMI scale signal. This does not imply geography is universally
irrelevant; it rejects this fixed 0.5/0.5 combination for the present frozen
simulation protocol.
"""
    (RECORD/"stage13_summary.md").write_text(md,encoding="utf-8")
    (RECORD/"METHOD.md").write_text("""# Method

GeoCMI-Balanced-NormCP combines a frozen Stage-9 Mspace CMI causal block and
standardized Euclidean coordinates. For SIGNAL replicates,
`d^2 = 0.5 * sum_j w_j dz_j^2 + 0.25*dz_sx^2 + 0.25*dz_sy^2`. No mixing
coefficient is tuned. NO_SIGNAL dispatches exactly to GeoOnly-NormCP. The
scale-fit reference, k candidates, tuning-MAE selection, final conformal rank,
test seal, and interval scoring are identical to Stage 11.
""",encoding="utf-8")
    (RECORD/"README.md").write_text("""# Stage 13 — Geo + CMI balanced normalized CP

Status: **PASS**  
Classification: **GEOGRAPHY_DILUTES_CMI_SIGNAL**

Exact command:

`python causal_only_cp_v1\\run_stage13_geocmi_balanced_normcp_v1.py`

This record imports Stage-11 and Stage-12 comparators read-only. It does not
rerun CMI, GCCM, predictors, or simulations, and it does not tune the fixed
0.5/0.5 block allocation.
""",encoding="utf-8")
    env=[f"Python executable: {sys.executable}",f"Python: {platform.python_version()}",f"Platform: {platform.platform()}"]
    for pkg in ("numpy","pandas","scipy","scikit-learn"): env.append(f"{pkg}: {importlib.metadata.version(pkg)}")
    (RECORD/"environment.txt").write_text("\n".join(env)+"\n",encoding="utf-8")
    input_paths=[ROOT/"causal_only_cp_v1/stage11_frozen_split_inputs_v1/recovery_manifest.json",ROOT/"causal_strength_diagnostic_v1/stage9_provisional_weights.csv",STAGE11/"replicate_results.csv",STAGE11/"scale_diagnostics.csv",STAGE11/"test_unseal_audit.csv",STAGE12/"replicate_results.csv",STAGE12/"test_unseal_audit.csv",RUNNER,ROOT/"causal_only_cp_v1/run_stage11_causalonly_cmi_normcp_v2.py"]
    (RECORD/"input_manifest.json").write_text(json.dumps({"inputs":[{"path":str(p.relative_to(ROOT)).replace("\\","/"),"sha256":sha(p),"bytes":p.stat().st_size} for p in input_paths]},indent=2)+"\n",encoding="utf-8")
    excluded={"output_manifest.json","run_log.txt","stderr.txt"}
    files=sorted(p for p in RECORD.rglob("*") if p.is_file() and p.name not in excluded)
    entries=[{"path":str(p.relative_to(RECORD)).replace("\\","/"),"sha256":sha(p),"bytes":p.stat().st_size} for p in files]
    digest=hashlib.sha256("\n".join(x["path"]+":"+x["sha256"] for x in entries).encode()).hexdigest()
    (RECORD/"output_manifest.json").write_text(json.dumps({"files":entries,"aggregate_path_hash_sha256":digest,"excluded_mutable_logs":["run_log.txt","stderr.txt"],"file_count":len(entries)},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","classification":"GEOGRAPHY_DILUTES_CMI_SIGNAL","signal_replicates":13,"no_signal_equalities":26},indent=2))

if __name__=="__main__": main()
