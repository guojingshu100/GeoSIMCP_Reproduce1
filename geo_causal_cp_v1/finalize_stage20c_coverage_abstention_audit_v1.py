"""Finalize the read-only Stage-20C diagnostic record."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/geo_causal_cp/20260920_stage20b_calibration_size_feasibility_v1"
RECORD = ROOT / "experiments/geo_causal_cp/20260920_stage20c_coverage_abstention_audit_v1"


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    if isinstance(value,np.ndarray): return [safe(v) for v in value.tolist()]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,float)):
        if np.isnan(value): return None
        if np.isposinf(value): return "Infinity"
        if np.isneginf(value): return "-Infinity"
        return float(value)
    if isinstance(value,np.bool_): return bool(value)
    return value


def dump(path:Path,value)->None:
    path.write_text(json.dumps(safe(value),indent=2,allow_nan=False)+"\n",encoding="utf-8")


def mean_ci(values)->dict:
    x=np.asarray(values,float); x=x[np.isfinite(x)]; n=len(x)
    if not n: return {"n":0,"mean":np.nan,"median":np.nan,"sd":np.nan,"ci_low":np.nan,"ci_high":np.nan}
    mean=float(x.mean()); sd=float(x.std(ddof=1)) if n>1 else 0.; half=float(student_t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else 0.
    return {"n":n,"mean":mean,"median":float(np.median(x)),"sd":sd,"ci_low":mean-half,"ci_high":mean+half}


def pooled_subset(frame:pd.DataFrame,n:int)->pd.Series:
    return frame.loc[(frame.aggregation=="pooled")&(frame.n_cal==n)].iloc[0]


def main()->None:
    facts=json.loads((RECORD/"intermediate_facts.json").read_text(encoding="utf-8"))
    integrity=facts["integrity"]
    if integrity["status"]!="PASS" or integrity["coverage_identity_max_abs_error"]>1e-14: raise AssertionError("integrity")
    decomp=pd.read_csv(RECORD/"coverage_decomposition.csv")
    points=pd.read_csv(RECORD/"pointwise_infinity_states.csv")
    both=pd.read_csv(RECORD/"both_finite_comparison.csv")
    euc_inf=pd.read_csv(RECORD/"euc_infinity_causal_finite.csv")
    causal_inf=pd.read_csv(RECORD/"causal_infinity_euc_finite.csv")
    attribution=pd.read_csv(RECORD/"coverage_gap_attribution.csv")
    margin=pd.read_csv(RECORD/"validation_coverage_margin.csv")
    general=pd.read_csv(RECORD/"validation_test_generalization.csv")
    track=pd.read_csv(RECORD/"trackA_trackB_hyperparameter_comparison.csv")
    selected=pd.read_csv(SOURCE/"selected_hyperparameters.csv")
    results=pd.read_csv(SOURCE/"replicate_results.csv")

    def dec(method:str,n:int=800):
        return decomp.loc[(decomp.scenario=="S2")&(decomp.n_cal==n)&(decomp.track=="B")&
                          (decomp.stratum=="SIGNAL")&(decomp.method==method)].iloc[0]
    c800,e800=dec("GeoCausalCP-Full"),dec("GeoSIMCP-EUC")
    state_summary=points.groupby(["n_cal","infinity_state"]).size().rename("count").reset_index()
    state_summary["fraction"]=state_summary["count"]/(30*300)

    both_summary=[]
    for n,g in both.groupby("n_cal"):
        point_count=int(g.point_count.sum())
        pooled={
            "coverage_causal":float((g.coverage_causal*g.point_count).sum()/point_count),
            "coverage_euc":float((g.coverage_euc*g.point_count).sum()/point_count),
            "width_causal":float((g.mean_width_causal*g.point_count).sum()/point_count),
            "width_euc":float((g.mean_width_euc*g.point_count).sum()/point_count),
            "score_causal":float((g.mean_score_causal*g.point_count).sum()/point_count),
            "score_euc":float((g.mean_score_euc*g.point_count).sum()/point_count),
            "oracle_sigma":float((g.mean_oracle_sigma*g.point_count).sum()/point_count),
            "causal_mass":float((g.mean_kernel_mass_causal*g.point_count).sum()/point_count),
            "euc_mass":float((g.mean_kernel_mass_euc*g.point_count).sum()/point_count),
            "causal_ess":float((g.mean_ess_causal*g.point_count).sum()/point_count),
            "euc_ess":float((g.mean_ess_euc*g.point_count).sum()/point_count),
        }
        both_summary.append({"n_cal":int(n),"point_count":point_count,**pooled,
            "replicate_coverage_difference":mean_ci(g.coverage_difference),
            "replicate_width_difference":mean_ci(g.mean_width_difference),
            "replicate_score_difference":mean_ci(g.mean_score_difference)})

    # Validation origin of the n=800 gap.
    q=margin.loc[(margin.scenario=="S2")&(margin.n_cal==800)&(margin.track=="B")&(margin.cmi_status=="SIGNAL")]
    qc=q.loc[q.method=="GeoCausalCP-Full"]; qe=q.loc[q.method=="GeoSIMCP-EUC"]
    val_gap=float(qc.validation_coverage.mean()-qe.validation_coverage.mean())
    test_gap=float(qc.coverage.mean()-qe.coverage.mean())
    generalization_gap=float(qc.test_minus_validation.mean()-qe.test_minus_validation.mean())
    bins={}
    for method,g in q.groupby("method"):
        bins[method]={name:int((g.validation_boundary_bin==name).sum()) for name in ["0.900","0.905","0.910","0.915","0.920+"]}

    # Track A/B changes among jointly selected S2 SIGNAL settings.
    signal_ids=set(results.loc[(results.scenario=="S2")&(results.method=="GeoCausalCP-Full")&(results.cmi_status=="SIGNAL"),"replicate_index"])
    track_signal=track.loc[(track.scenario=="S2")&track.replicate_index.isin(signal_ids)]
    track_summary=[]
    for (n,method),g in track_signal.groupby(["n_cal","method"]):
        joint=g.loc[g.status_A.eq("SELECTED")&g.status_B.eq("SELECTED")]
        track_summary.append({"n_cal":int(n),"method":method,"jointly_selected":len(joint),
            "localization_changed_frequency":float(joint.localization_changed.mean()) if len(joint) else np.nan,
            "bandwidth_changed_frequency":float(joint.bandwidth_changed.mean()) if len(joint) else np.nan,
            "lambda_changed_frequency":float(joint.lambda_changed.mean()) if len(joint) else np.nan,
            "status_changed_frequency":float(g.status_changed.mean())})

    # Selected infinite-score counts, especially the primary n=800 SIGNAL comparison.
    status=results.loc[results.method.eq("GeoCausalCP-Full"),["replicate_index","scenario","n_cal","track","cmi_status"]]
    selected_status=selected.drop(columns=["cmi_status"],errors="ignore").merge(status,on=["replicate_index","scenario","n_cal","track"],how="left")
    inf_selected=selected_status.loc[(selected_status.track=="B")&selected_status.status.eq("SELECTED")&
                                     np.isinf(selected_status.validation_interval_score)]
    inf_selected_counts=inf_selected.groupby(["scenario","n_cal","method"]).size().rename("count").reset_index().to_dict("records")

    euc800=pooled_subset(euc_inf,800); reverse800=pooled_subset(causal_inf,800)
    euc_rep=euc_inf.loc[(euc_inf.aggregation=="replicate")&(euc_inf.n_cal==800)]
    reverse_rep=causal_inf.loc[(causal_inf.aggregation=="replicate")&(causal_inf.n_cal==800)]
    classification="B. PRIMARILY_DIFFERENTIAL_ABSTENTION"
    rationale=("At n_cal=800, 0.009000 of the 0.013222 EUC-minus-GeoCausal coverage gap came from EUC-only infinities, "
               "-0.000333 from causal-only infinities, and 0.004556 from both-finite success differences. "
               "GeoCausal finite-only coverage was 0.9099 overall and 0.9107 on jointly finite points; validation-to-test change was positive, not optimistic. ")

    summary={
        "stage":"20C","status":"COMPLETE","source_integrity":integrity,
        "stage20b_class_preserved":"F. COVERAGE_CONCERN",
        "n800_signal":{
            "GeoCausalCP":{"total_coverage":c800.coverage_total,"infinite_fraction":c800.p_infinite,"finite_only_coverage":c800.coverage_finite},
            "GeoSIMCP_EUC":{"total_coverage":e800.coverage_total,"infinite_fraction":e800.p_infinite,"finite_only_coverage":e800.coverage_finite},
            "GeoCausal_minus_EUC":{"total_coverage":c800.coverage_total-e800.coverage_total,
                "infinite_fraction":c800.p_infinite-e800.p_infinite,"finite_only_coverage":c800.coverage_finite-e800.coverage_finite}},
        "coverage_gap_attribution":attribution.to_dict("records"),"infinity_states":state_summary.to_dict("records"),
        "both_finite":both_summary,
        "n800_euc_infinity_causal_finite":{"points":int(euc800.point_count),"pooled_causal_coverage":euc800.coverage,
            "pooled_causal_width":euc800.mean_width,"pooled_causal_score":euc800.mean_interval_score,
            "mean_oracle_sigma":euc800.mean_oracle_sigma,"mean_causal_mass":euc800.mean_kernel_mass,"mean_causal_ess":euc800.mean_ess,
            "replicate_coverage":mean_ci(euc_rep.coverage)},
        "n800_causal_infinity_euc_finite":{"points":int(reverse800.point_count),"pooled_euc_coverage":reverse800.coverage,
            "pooled_euc_width":reverse800.mean_width,"pooled_euc_score":reverse800.mean_interval_score,
            "mean_oracle_sigma":reverse800.mean_oracle_sigma,"mean_euc_mass":reverse800.mean_kernel_mass,"mean_euc_ess":reverse800.mean_ess,
            "replicate_coverage":mean_ci(reverse_rep.coverage)},
        "n800_validation":{"causal_mean":float(qc.validation_coverage.mean()),"euc_mean":float(qe.validation_coverage.mean()),
            "causal_minus_euc_gap":val_gap,"test_gap":test_gap,"relative_generalization_change":generalization_gap,"boundary_counts":bins},
        "validation_score_semantics":integrity["validation_semantics"],"selected_infinite_score_counts":inf_selected_counts,
        "trackA_trackB_changes_S2_SIGNAL":track_summary,
        "diagnostic_classification":classification,"diagnostic_rationale":rationale,
        "causal_estimator_assessment":"CMI-Full is not implicated: its distance remained oracle-aligned and EUC-only abstention points received finite GeoCausal intervals with pooled coverage near 0.90.",
        "next_scientific_question":"On new development seeds, can a preregistered validation/selection rule explicitly balance finite-interval validity and abstention/informativeness without changing CMI-Full, the kernel, or the official weighted-conformal engine?",
    }
    dump(RECORD/"diagnostic_classification.json",{"stage20b_class":"F. COVERAGE_CONCERN","stage20c_diagnostic_class":classification,
        "does_not_replace_stage20b":True,"rationale":rationale})
    dump(RECORD/"stage20c_summary.json",summary)

    lines=["# Stage 20C: coverage, abstention, and validation-generalization audit","",
        "This was a read-only diagnostic of frozen Stage-20B outputs. Stage-20B remains **F. COVERAGE_CONCERN**.","",
        "## Primary n_cal=800 S2 SIGNAL result","",
        "| Method | Total coverage | Infinite fraction | Coverage conditional on finite |","|---|---:|---:|---:|",
        f"| GeoCausalCP-Full | {c800.coverage_total:.4f} | {c800.p_infinite:.4f} | {c800.coverage_finite:.4f} |",
        f"| GeoSIMCP-EUC | {e800.coverage_total:.4f} | {e800.p_infinite:.4f} | {e800.coverage_finite:.4f} |",
        f"| GeoCausal - EUC | {c800.coverage_total-e800.coverage_total:+.4f} | {c800.p_infinite-e800.p_infinite:+.4f} | {c800.coverage_finite-e800.coverage_finite:+.4f} |","",
        "The exact EUC-minus-GeoCausal total coverage gap (0.013222) decomposed into +0.009000 from EUC-only infinities, -0.000333 from causal-only infinities, and +0.004556 from different outcomes when both intervals were finite.","",
        "On BOTH_FINITE points, GeoCausal coverage was 0.9107 versus 0.9158 for EUC. The replicate-level difference was -0.0050 (95% t CI -0.0102 to 0.0003), while GeoCausal width and interval score remained lower.","",
        f"Among the 828 EUC-infinity/GeoCausal-finite points, GeoCausal pooled coverage was {euc800.coverage:.4f}, mean width {euc800.mean_width:.4f}, and interval score {euc800.mean_interval_score:.4f}. These intervals were informative but supported by lower mass/ESS than ordinary both-finite cases.","",
        "## Validation origin","",
        f"At n=800, mean validation coverage was {qc.validation_coverage.mean():.4f} for GeoCausal and {qe.validation_coverage.mean():.4f} for EUC (gap {val_gap:+.4f}); the test gap was {test_gap:+.4f}. Test-minus-validation changes were +{qc.test_minus_validation.mean():.4f} and +{qe.test_minus_validation.mean():.4f}, respectively. The gap therefore existed at validation selection and was not produced by test optimism.","",
        "Candidates containing any infinity had infinite validation interval score. When every feasible candidate had infinite score, the frozen grid-index tie rule selected the first candidate. This occurred in 253 partitions and materially changed Track-B localization relative to Track A.","",
        "## Diagnostic classification","",f"**{classification}**","",rationale,"",
        "The next question concerns validation selection under abstention—not CMI power. No new experiment was run or authorized by this audit.",""]
    (RECORD/"stage20c_summary.md").write_text("\n".join(lines),encoding="utf-8")
    (RECORD/"README.md").write_text("# Stage 20C read-only diagnostic\n\nThis record decomposes frozen Stage-20B coverage into finite prediction and +infinity abstention, audits validation-to-test generalization, and preserves Stage-20B class F unchanged.\n",encoding="utf-8")

    input_manifest={"source_record":str(SOURCE),"source_output_manifest_sha256":sha(SOURCE/"output_manifest.json"),
        "source_selected_hyperparameters_sha256":sha(SOURCE/"selected_hyperparameters.csv"),
        "source_replicate_results_sha256":sha(SOURCE/"replicate_results.csv"),
        "source_validation_grid_sha256":sha(SOURCE/"validation_grid_results.csv"),"integrity":integrity}
    dump(RECORD/"input_manifest.json",input_manifest)
    required=["README.md","coverage_decomposition.csv","finite_only_coverage.csv","pointwise_infinity_states.csv",
        "both_finite_comparison.csv","euc_infinity_causal_finite.csv","causal_infinity_euc_finite.csv","coverage_gap_attribution.csv",
        "validation_coverage_margin.csv","validation_test_generalization.csv","validation_score_semantics.md",
        "trackA_trackB_hyperparameter_comparison.csv","hyperparameter_margin_diagnostics.csv","diagnostic_classification.json",
        "stage20c_summary.json","stage20c_summary.md","input_manifest.json","run_log.txt","stderr.txt"]
    missing=[name for name in required if not (RECORD/name).exists()]
    if missing: raise AssertionError(missing)
    with (RECORD/"run_log.txt").open("a",encoding="utf-8") as f: f.write(f"Finalization complete; diagnostic_class={classification}.\n")
    dump(RECORD/"output_manifest.json",{"files":[{"path":name,"sha256":sha(RECORD/name),"bytes":(RECORD/name).stat().st_size} for name in required],
        "missing":[],"scientific_recomputation":False,"stage20b_modified":False})
    print(json.dumps(safe(summary),indent=2))


if __name__=="__main__": main()
