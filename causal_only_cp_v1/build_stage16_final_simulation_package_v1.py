"""Build the Stage-16 manuscript package from frozen Stage 9--15 records.

This is a consolidation-only program. It does not run a generator, predictor,
causal estimator, conformal method, or real-data evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"experiments/manuscript_package/20260918_stage16_final_simulation_package_v1"
S9=ROOT/"experiments/causal_strength_diagnostic/20260917_stage9_cmi_feature_recovery_v1"
S10=ROOT/"experiments/causal_strength_diagnostic/20260917_stage10b3_gcmc_mechanism_null_v1"
S11=ROOT/"experiments/causal_only_cp/20260918_stage11_causalonly_cmi_normcp_v2"
S12=ROOT/"experiments/causal_only_cp/20260918_stage12_causalonly_gccm_ablation_v1"
S13=ROOT/"experiments/causal_only_cp/20260918_stage13_geocmi_balanced_normcp_v1"
S14A=ROOT/"experiments/real_data/20260918_stage14_realdata_feasibility_v1"
S14=ROOT/"experiments/real_data/20260918_stage14_pm25_causalonly_cmi_exploratory_v1"
S15P=ROOT/"experiments/confirmatory_simulation/20260918_stage15_confirmatory_preregistration_v1"
S15=ROOT/"experiments/confirmatory_simulation/20260918_stage15_fresh_confirmatory_v1"
METHODS=["Raw-CP","GeoOnly-NormCP","CausalOnly-GCCM-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP","Oracle-NormCP"]


def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def dump(path:Path,obj)->None:
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:float(x) if isinstance(x,np.floating) else int(x)),encoding="utf-8")


def write_csv(df:pd.DataFrame,path:Path)->None:
    df.to_csv(path,index=False,lineterminator="\n",float_format="%.17g")


def md_table(df:pd.DataFrame,columns:list[str],labels:dict[str,str]|None=None,digits=4)->str:
    labels=labels or {}; head=[labels.get(c,c) for c in columns]
    lines=["| "+" | ".join(head)+" |","|"+"|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in columns)+"|"]
    for row in df[columns].itertuples(index=False,name=None):
        vals=[]
        for v in row:
            if isinstance(v,(float,np.floating)): vals.append(f"{v:.{digits}f}")
            elif isinstance(v,(bool,np.bool_)): vals.append("Yes" if v else "No")
            else: vals.append(str(v))
        lines.append("| "+" | ".join(vals)+" |")
    return "\n".join(lines)


def verify_stage15()->dict:
    seal=json.loads((S15P/"PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    prereg_bad=[]
    for name,expected in seal["files"].items():
        if sha(S15P/name)!=expected: prereg_bad.append(name)
    payload="\n".join(f"{name} {seal['files'][name]}" for name in sorted(seal["files"]))
    composite=hashlib.sha256(payload.encode()).hexdigest()
    seed=json.loads((S15P/"seed_manifest.json").read_text(encoding="utf-8"))
    seeds=[int(x["generation_seed"]) for x in seed["paired_seed_records"]]
    out=json.loads((S15/"output_manifest.json").read_text(encoding="utf-8"))
    output_bad=[]
    for row in out["outputs"]:
        p=ROOT/row["path"]
        if not p.exists() or sha(p)!=row["sha256"] or p.stat().st_size!=row["bytes"]: output_bad.append(row["path"])
    nested_bad=[]; completes=list((S15/"frozen_inputs").glob("replicate_*/*/COMPLETE.json"))
    for complete in completes:
        obj=json.loads(complete.read_text(encoding="utf-8"))
        for name,expected in obj["hashes"].items():
            if sha(complete.parent/name)!=expected: nested_bad.append(str((complete.parent/name).relative_to(ROOT)))
    unseal=pd.read_csv(S15/"test_unseal_audit.csv",keep_default_na=False)
    fields=[x for x in unseal.columns if x!="barrier_row_sha256"]; barrier_bad=[]
    for i,row in unseal.iterrows():
        obj={k:row[k] for k in fields}; obj["replicate_index"]=int(obj["replicate_index"]); obj["generation_seed"]=int(obj["generation_seed"])
        got=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if got!=row.barrier_row_sha256: barrier_bad.append(int(i))
    gen=pd.read_csv(S15/"generator_audit.csv"); split=pd.read_csv(S15/"split_integrity.csv"); pred=pd.read_csv(S15/"prediction_manifest.csv")
    audit={
      "status":"PASS" if not(prereg_bad or output_bad or nested_bad or barrier_bad) else "FAIL",
      "audited_utc":pd.Timestamp.now(tz="UTC").isoformat(),
      "preregistration_composite_expected":seal["composite_sha256"],"preregistration_composite_recomputed":composite,
      "preregistration_file_count":len(seal["files"]),"preregistration_hash_mismatches":prereg_bad,
      "seed_count":len(seeds),"seed_list":seeds,"seed_list_exact_contiguous_block":seeds==list(range(20320001,20320051)),
      "paired_replicates_completed":int(gen.generation_seed.nunique()),"scenario_replicates_completed":len(gen),
      "generator_serialization_pass":int(gen.serialized_reload.sum()),"split_integrity_pass":int((split.status=="PASS").sum()),
      "prediction_manifest_rows":len(pred),"nested_COMPLETE_manifests":len(completes),"nested_hash_mismatches":nested_bad,
      "output_manifest_entries":len(out["outputs"]),"output_manifest_hash_mismatches":output_bad,
      "final_stderr_bytes":(S15/"stderr.txt").stat().st_size,
      "test_unseal_rows":len(unseal),"primary_test_unseal_rows":int((unseal.phase=="primary").sum()),
      "cmi999_test_unseal_rows":int((unseal.phase=="cmi999").sum()),"test_unseal_barrier_hash_mismatches":barrier_bad,
      "scientific_outcomes_recomputed":False}
    if audit["status"]!="PASS" or composite!=seal["composite_sha256"] or audit["final_stderr_bytes"]!=0 or len(unseal)!=140:
        raise AssertionError(audit)
    return audit


def build_tables():
    main=pd.read_csv(S15/"scenario_summary_alpha010.csv")
    primary=main[["scenario","method","mean_coverage","coverage_sd","mean_width","width_sd","mean_interval_score",
                  "count_coverage_below_nominal","material_undercoverage"]].copy()
    rr=pd.read_csv(S15/"replicate_results_alpha010.csv")
    is_sd=rr.groupby(["scenario","method"],sort=False).mean_interval_score.std(ddof=1).rename("interval_score_sd").reset_index()
    primary=primary.merge(is_sd,on=["scenario","method"],validate="one_to_one")
    primary["method_order"]=primary.method.map({m:i+1 for i,m in enumerate(METHODS)}); primary=primary.sort_values(["scenario","method_order"]).drop(columns="method_order")
    write_csv(primary,OUT/"table_main_confirmatory.csv")
    labels={"mean_coverage":"Mean coverage","coverage_sd":"Coverage SD","mean_width":"Mean width","width_sd":"Width SD",
            "mean_interval_score":"Mean interval score","interval_score_sd":"Interval-score SD",
            "count_coverage_below_nominal":"Replicates coverage < .90","material_undercoverage":"Material undercoverage"}
    (OUT/"table_main_confirmatory.md").write_text("# Primary confirmatory results (fresh Stage 15, alpha=.10)\n\n"+md_table(primary,list(primary.columns),labels)+"\n",encoding="utf-8")

    pairs=pd.read_csv(S15/"paired_comparisons_alpha010.csv")
    wanted=[("CausalOnly-CMI-NormCP","Raw-CP","CMI minus Raw"),("CausalOnly-CMI-NormCP","GeoOnly-NormCP","CMI minus GeoOnly"),
            ("CausalOnly-GCCM-NormCP","Raw-CP","GCCM minus Raw"),("GeoCMI-Balanced-NormCP","CausalOnly-CMI-NormCP","GeoCMI minus CMI")]
    chunks=[]
    for method,comp,label in wanted:
        x=pairs[(pairs.scenario=="S2")&(pairs.method==method)&(pairs.comparator==comp)].copy(); x.insert(0,"contrast",label); chunks.append(x)
    paired=pd.concat(chunks,ignore_index=True)[["contrast","metric","n","mean_difference","median_difference","sd_difference","t95_lower","t95_upper","bootstrap95_lower","bootstrap95_upper","relative_mean_difference_percent"]]
    write_csv(paired,OUT/"table_paired_effects.csv")
    (OUT/"table_paired_effects.md").write_text("# Fresh S2 paired effects (method minus comparator)\n\n"+md_table(paired,list(paired.columns),digits=4)+"\n",encoding="utf-8")

    cmi=pd.read_csv(S15/"cmi99_feature_results.csv"); w=pd.read_csv(S15/"cmi99_weight_audit.csv")
    rows=[]
    for scenario in ["S2","S4"]:
        x=cmi[cmi.scenario==scenario]; raw=x.pivot(index="replicate_index",columns="feature",values="raw_cmi")
        ww=w[w.scenario==scenario].pivot(index="replicate_index",columns="feature",values="weight")
        n=len(raw)
        rows += [
          {"analysis":"PRIMARY CMI99","scenario":scenario,"diagnostic":"x2 raw rank #1","count":int((raw.idxmax(axis=1)=="x2").sum()),"denominator":n},
          {"analysis":"PRIMARY CMI99","scenario":scenario,"diagnostic":"x2 BH-gated selected","count":int((ww.x2>0).sum()),"denominator":n},
          {"analysis":"PRIMARY CMI99","scenario":scenario,"diagnostic":"x2 largest final weight","count":int((ww.idxmax(axis=1)=="x2").sum() if scenario=="S2" else (ww.x2>0).sum()),"denominator":n},
          {"analysis":"PRIMARY CMI99","scenario":scenario,"diagnostic":"NO_SIGNAL","count":int((ww.max(axis=1)<=0).sum()),"denominator":n},
          {"analysis":"PRIMARY CMI99","scenario":scenario,"diagnostic":"nuisance/false selection","count":int(((ww[["x1","g_proxy","x_noise"]]>0).any(axis=1) if scenario=="S2" else (ww>0).any(axis=1)).sum()),"denominator":n}]
    comp=pd.read_csv(S15/"cmi999_comparison.csv")
    for scenario in ["S2","S4"]:
        x=comp[comp.scenario==scenario]; n=x.replicate_index.nunique()
        for suffix,label in [("99","CMI99 ON CMI999 SUBSET"),("999","CMI999 MONTE-CARLO RESOLUTION SENSITIVITY")]:
            sel=x.pivot(index="replicate_index",columns="feature",values=f"selected{suffix}").astype(bool)
            raw=x.pivot(index="replicate_index",columns="feature",values=f"raw_cmi_{suffix}")
            rows += [
              {"analysis":label,"scenario":scenario,"diagnostic":"x2 raw rank #1","count":int((raw.idxmax(axis=1)=="x2").sum()),"denominator":n},
              {"analysis":label,"scenario":scenario,"diagnostic":"x2 BH-gated selected","count":int(sel.x2.sum()),"denominator":n},
              {"analysis":label,"scenario":scenario,"diagnostic":"NO_SIGNAL","count":int((~sel.any(axis=1)).sum()),"denominator":n},
              {"analysis":label,"scenario":scenario,"diagnostic":"false selection","count":int((sel.any(axis=1) if scenario=="S4" else sel[["x1","g_proxy","x_noise"]].any(axis=1)).sum()),"denominator":n}]
    feature=pd.DataFrame(rows); feature["proportion"]=feature["count"]/feature["denominator"]; write_csv(feature,OUT/"table_feature_recovery.csv")

    scale=pd.read_csv(S15/"scale_diagnostics.csv"); sig=scale[(scale.scenario=="S2")&scale.cmi_signal.astype(bool)]
    srows=[]
    for m in ["GeoOnly-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP","Oracle-NormCP"]:
        x=sig[sig.method==m]
        srows.append({"row_type":"method summary","method":m,"comparator":"","n":len(x),"mean_oracle_correlation":x.test_oracle_correlation.mean(),
                      "mean_oracle_scale_mae":x.test_oracle_mae.mean(),"correlation_mean_difference":None,"correlation_t95_lower":None,"correlation_t95_upper":None,
                      "mae_mean_difference":None,"mae_t95_lower":None,"mae_t95_upper":None})
    summary=json.loads((S15/"confirmatory_summary.json").read_text(encoding="utf-8"))["scale_recovery"]
    for method,comp in [("GeoCMI-Balanced-NormCP","CausalOnly-CMI-NormCP")]:
        corr=next(x for x in summary if x.get("method")==method and x.get("comparator")==comp and x["metric"]=="test_oracle_correlation")
        mae=next(x for x in summary if x.get("method")==method and x.get("comparator")==comp and x["metric"]=="test_oracle_mae")
        srows.append({"row_type":"paired difference","method":method,"comparator":comp,"n":corr["signal_replicates"],"mean_oracle_correlation":None,"mean_oracle_scale_mae":None,
          "correlation_mean_difference":corr["mean_difference"],"correlation_t95_lower":corr["t95_lower"],"correlation_t95_upper":corr["t95_upper"],
          "mae_mean_difference":mae["mean_difference"],"mae_t95_lower":mae["t95_lower"],"mae_t95_upper":mae["t95_upper"]})
    scale_table=pd.DataFrame(srows); write_csv(scale_table,OUT/"table_scale_recovery.csv")

    alpha=pd.read_csv(S15/"alpha_robustness_summary.csv"); alpha=alpha[alpha.scenario=="S2"][["alpha","nominal_coverage","method","replicates","mean_coverage","mean_width","mean_interval_score","material_undercoverage"]]
    alpha["frozen_components"]="predictor; causal scores; weights; k; scale estimator"
    alpha["changed_component"]="finite-sample conformal quantile and resulting intervals only"
    write_csv(alpha,OUT/"table_alpha_robustness.csv")
    return primary,paired,feature,scale_table,alpha,rr,scale


def figure_data(primary,paired,feature,scale_table,alpha,rr,scale):
    fd=OUT/"figure_data"; fd.mkdir(exist_ok=True)
    s2=rr[rr.scenario=="S2"]
    wide=s2.pivot(index=["replicate_index","generation_seed"],columns="method",values=["mean_width","mean_interval_score","coverage"]).reset_index()
    A=pd.DataFrame({"replicate_index":wide.replicate_index,"generation_seed":wide.generation_seed,
      "cmi_minus_raw_width":wide[("mean_width","CausalOnly-CMI-NormCP")]-wide[("mean_width","Raw-CP")],
      "cmi_minus_geoonly_width":wide[("mean_width","CausalOnly-CMI-NormCP")]-wide[("mean_width","GeoOnly-NormCP")]})
    B=pd.DataFrame({"replicate_index":wide.replicate_index,"generation_seed":wide.generation_seed,
      "cmi_minus_raw_interval_score":wide[("mean_interval_score","CausalOnly-CMI-NormCP")]-wide[("mean_interval_score","Raw-CP")],
      "cmi_minus_geoonly_interval_score":wide[("mean_interval_score","CausalOnly-CMI-NormCP")]-wide[("mean_interval_score","GeoOnly-NormCP")]})
    C=scale[(scale.scenario=="S2")&scale.cmi_signal.astype(bool)&scale.method.isin(["GeoOnly-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP"])][["replicate_index","generation_seed","method","test_oracle_correlation","test_oracle_mae"]]
    D=feature[((feature.analysis=="PRIMARY CMI99")&(((feature.scenario=="S2")&feature.diagnostic.isin(["x2 raw rank #1","x2 BH-gated selected","NO_SIGNAL"]))|((feature.scenario=="S4")&(feature.diagnostic=="nuisance/false selection"))))].copy()
    # Wilson intervals give replicate-level uncertainty for frequency points.
    z=1.959963984540054
    n=D.denominator.to_numpy(float); ph=D.proportion.to_numpy(float); den=1+z*z/n
    ctr=(ph+z*z/(2*n))/den; half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    D["wilson95_lower"]=ctr-half; D["wilson95_upper"]=ctr+half
    E=primary[["scenario","method","mean_coverage","coverage_sd","mean_width","width_sd"]].copy()
    write_csv(A,fd/"figure_A_paired_width_differences.csv"); write_csv(B,fd/"figure_B_paired_interval_score_differences.csv")
    write_csv(C,fd/"figure_C_oracle_scale_correlation.csv"); write_csv(D,fd/"figure_D_feature_selection_frequencies.csv")
    write_csv(E,fd/"figure_E_coverage_width_tradeoff.csv")


def evidence_chain():
    rows=[
      {"stage":"9","scientific_question":"Can CMI recover the residual-scale driver among observed attributes?","data_role":"20 development paired simulation replicates","main_result":"Primary Mspace gate passed; x2 top 17/20, BH-selected 13/20; S4 had 19/20 NO_SIGNAL.","claim_type":"METHOD DEVELOPMENT","record_path":str(S9.relative_to(ROOT))},
      {"stage":"10","scientific_question":"Is GCMC compatible with the same-location residual-scale mechanism and calibrated on the fixed graph?","data_role":"method diagnostics and controlled synthetic mechanisms","main_result":"General signal response but null calibration/directionality remained unsuitable for primary residual-scale weights.","claim_type":"METHOD DIAGNOSTIC","record_path":str(S10.relative_to(ROOT))},
      {"stage":"11","scientific_question":"Does CausalOnly-CMI improve normalized split CP?","data_role":"20 development paired simulation replicates","main_result":"S2 efficiency improved versus Raw and GeoOnly without material undercoverage; S4 fallback controlled the null.","claim_type":"METHOD DEVELOPMENT","record_path":str(S11.relative_to(ROOT))},
      {"stage":"12","scientific_question":"Does the original GCCM estimator help when geography is removed from the final distance?","data_role":"same 20 development replicates","main_result":"GCCM signal did not translate to CP gain.","claim_type":"ABLATION","record_path":str(S12.relative_to(ROOT))},
      {"stage":"13","scientific_question":"Does a fixed balanced geographic block add value beyond CMI?","data_role":"same 20 development replicates","main_result":"Balanced geography diluted CMI scale recovery and worsened efficiency.","claim_type":"ABLATION","record_path":str(S13.relative_to(ROOT))},
      {"stage":"14","scientific_question":"Is frozen CausalOnly-CMI feasible on an existing real dataset?","data_role":"historically exposed PM2.5 evaluation","main_result":"NO_CAUSAL_SIGNAL_DETECTED; exact Raw-CP fallback.","claim_type":"RETROSPECTIVE EXPLORATORY","record_path":str(S14.relative_to(ROOT))},
      {"stage":"15","scientific_question":"Do the frozen simulation findings replicate on never-used seeds?","data_role":"50 fresh preregistered paired simulation replicates","main_result":"CONFIRMATORY_SUPPORT: CMI improved S2 width and interval score versus Raw and GeoOnly without material undercoverage.","claim_type":"FRESH CONFIRMATORY SIMULATION","record_path":str(S15.relative_to(ROOT))}]
    write_csv(pd.DataFrame(rows),OUT/"evidence_chain.csv")


def write_docs(primary,paired,feature,scale_table,alpha,integrity):
    s2=primary[primary.scenario=="S2"].set_index("method"); s4=primary[primary.scenario=="S4"].set_index("method")
    def pe(label,metric): return paired[(paired.contrast==label)&(paired.metric==metric)].iloc[0]
    wr=pe("CMI minus Raw","mean_width"); sr=pe("CMI minus Raw","mean_interval_score"); cr=pe("CMI minus Raw","coverage")
    wg=pe("CMI minus GeoOnly","mean_width"); sg=pe("CMI minus GeoOnly","mean_interval_score"); cg=pe("CMI minus GeoOnly","coverage")
    gg=pe("GCCM minus Raw","mean_width"); gc=pe("GeoCMI minus CMI","mean_width"); gs=pe("GeoCMI minus CMI","mean_interval_score")
    method=f"""# Manuscript method draft\n\n## CausalOnly-CMI-NormCP\n\nFor each replicate, a frozen base predictor was trained using training observations only. On the disjoint scale-fit set, residual magnitude was defined as $A_i=|Y_i-\\hat f(X_i)|$. For each eligible non-coordinate attribute $X_j$, Tigramite CMIknn estimated the causally motivated conditional-dependence strength $I(X_j;A\\mid s_x,s_y)$ with nearest-neighbor parameter 20 and 99 surrogates. Benjamini-Hochberg correction was applied across the four candidate attributes. Attribute $j$ was eligible only when $q_j<0.05$ and its raw CMI was positive; eligible weights were $w_j=\\mathrm{{CMI}}_j/\\sum_\\ell\\mathrm{{CMI}}_\\ell$.\n\nCoordinates conditioned feature-strength estimation but did **not** enter the final causal-only localization distance. Selected attributes were standardized using scale-fit means and standard deviations, and distance was\n\n$$d_{{\\rm causal}}(a,b)=\\left[\\sum_j w_j(z_{{aj}}-z_{{bj}})^2\\right]^{{1/2}}.$$\n\nFor query $x$, the local scale was the mean scale-fit residual magnitude among its $k$ nearest causal neighbors. The candidate grid $k\\in\\{{20,40,60,80\\}}$ was fixed; tuning calibration selected the minimum residual-scale MAE, with larger $k$ breaking ties. Final-calibration scores were $R_i=|Y_i-\\hat f(X_i)|/\\hat\\sigma(X_i)$. With $n_{{cal}}$ final-calibration cases and miscoverage $\\alpha$, the quantile was the order statistic\n\n$$q_\\alpha=R_{{(r)}},\\qquad r=\\left\\lceil(n_{{cal}}+1)(1-\\alpha)\\right\\rceil,$$\n\nand intervals were $[\\hat f(x)-q_\\alpha\\hat\\sigma(x),\\hat f(x)+q_\\alpha\\hat\\sigma(x)]$. This is the standard finite-sample split-conformal correction, equivalent to including the test-point infinity atom.\n\nIf no attribute passed the gate, the method declared `NO_CAUSAL_SIGNAL`, used the mean scale-fit residual as a constant scale, and reproduced Raw-CP exactly. Scale fitting, $k$ selection, final calibration, and evaluation used disjoint roles.\n"""
    (OUT/"MANUSCRIPT_METHOD_DRAFT.md").write_text(method,encoding="utf-8")
    results=f"""# Manuscript results draft\n\n## 1. Feature-strength validation\n\nDevelopment Stage 9 established that Mspace CMI recovered x2 as the highest raw-CMI attribute in 17/20 S2 replicates and selected it after BH correction in 13/20, while 19/20 S4 replicates returned no signal. This was method-development evidence. In the separately preregistered Stage-15 confirmation, x2 had the largest raw CMI in 45/50 fresh S2 replicates, passed the BH gate in 20/50, and was the largest final weight in all 20 signal replicates. No nuisance attribute passed the S2 gate. S4 returned no signal in 49/50 replicates.\n\n## 2. CausalOnly-CMI conformal performance\n\nOn 50 fresh S2 replicates at nominal 90% coverage, CausalOnly-CMI achieved coverage {s2.loc['CausalOnly-CMI-NormCP','mean_coverage']:.4f}, mean width {s2.loc['CausalOnly-CMI-NormCP','mean_width']:.4f}, and interval score {s2.loc['CausalOnly-CMI-NormCP','mean_interval_score']:.4f}. Raw-CP values were {s2.loc['Raw-CP','mean_coverage']:.4f}, {s2.loc['Raw-CP','mean_width']:.4f}, and {s2.loc['Raw-CP','mean_interval_score']:.4f}. Paired CMI-minus-Raw differences were {cr.mean_difference:.4f} in coverage (95% CI {cr.t95_lower:.4f} to {cr.t95_upper:.4f}), {wr.mean_difference:.4f} in width ({wr.t95_lower:.4f} to {wr.t95_upper:.4f}), and {sr.mean_difference:.4f} in interval score ({sr.t95_lower:.4f} to {sr.t95_upper:.4f}). CMI did not trigger the preregistered material-undercoverage rule.\n\n## 3. Geography-only comparison\n\nGeoOnly coverage, width, and interval score were {s2.loc['GeoOnly-NormCP','mean_coverage']:.4f}, {s2.loc['GeoOnly-NormCP','mean_width']:.4f}, and {s2.loc['GeoOnly-NormCP','mean_interval_score']:.4f}. CMI-minus-GeoOnly paired differences were {cg.mean_difference:.4f}, {wg.mean_difference:.4f}, and {sg.mean_difference:.4f}, respectively. Thus geography alone did not explain the efficiency gain.\n\n## 4. GCCM ablation\n\nGCCM selected x2 in only 1/50 fresh S2 replicates and never assigned it the largest weight; g_proxy dominated 32/50. GCCM was {gg.mean_difference:.4f} wider than Raw on average (95% CI {gg.t95_lower:.4f} to {gg.t95_upper:.4f}), replicating the negative ablation.\n\n## 5. Geo+CMI ablation\n\nThe balanced GeoCMI method was {gc.mean_difference:.4f} wider than CMI (95% CI {gc.t95_lower:.4f} to {gc.t95_upper:.4f}) and its interval score was {gs.mean_difference:.4f} higher. Fixed balanced geography therefore diluted rather than enhanced the supported CMI localization in this simulation.\n\n## 6. Oracle-scale diagnostics\n\nAmong the 20 CMI-signal replicates, mean oracle-scale correlations were -0.0235 for GeoOnly, 0.9129 for CMI, and 0.8778 for GeoCMI. Corresponding MAEs were 0.3240, 0.2106, and 0.2235.\n\n## 7. Coverage-level robustness\n\nPredictors, causal scores, weights, selected $k$, and scale estimators were frozen when nominal coverage changed to 95% and 80%; only conformal calibration changed. The favorable CMI efficiency direction persisted at all three levels without an alpha-specific coverage concern.\n\n## 8. CMI surrogate-resolution sensitivity\n\nThe runtime gate permitted all 20 preregistered CMI-999 pairs. Increasing from 99 to 999 surrogates changed S2 x2 selection from 10/20 to 13/20, left raw CMI and raw ranking unchanged, and left S4 selection unchanged at one false x2 selection. On this subset, CMI999-minus-CMI99 differences were +0.0020 coverage, +0.0013 width, and -0.0616 interval score. This is a Monte-Carlo p-value-resolution sensitivity, not a replacement primary method.\n\n## 9. Fresh confirmatory simulation\n\nThe method family and hypotheses were sealed before generating seeds 20320001--20320050. All 50 paired replicates completed. The preregistered final classification was **A. CONFIRMATORY_SUPPORT**: CMI maintained acceptable coverage and improved both width and interval score against Raw and GeoOnly.\n\n## 10. Real-data limitation\n\nThe PM2.5 analysis remains retrospective and exploratory because its historical evaluation labels had previously been accessed. It detected no causal signal and therefore reproduced Raw-CP. It is not confirmatory real-data evidence.\n"""
    (OUT/"MANUSCRIPT_RESULTS_DRAFT.md").write_text(results,encoding="utf-8")
    ablation="""# Ablation interpretation\n\n- **Raw-CP** has no localization and establishes the baseline.\n- **GeoOnly-NormCP** applies the identical normalized-scale shell using only coordinates, testing whether geographical locality alone explains gains.\n- **CausalOnly-GCCM-NormCP** retains the original GCCM causal-strength mechanism while excluding geography from the final metric, directly testing whether that estimator is sufficient. Its negative Stage-12 result replicated in Stage 15.\n- **CausalOnly-CMI-NormCP** is the supported primary practical method. It localizes only in BH-gated, causally motivated conditional-dependence attributes.\n- **GeoCMI-Balanced-NormCP** tests whether a fixed 0.5/0.5 geographic block adds value beyond CMI. Its Stage-13 dilution result replicated: it was wider than CMI and recovered oracle scale less accurately.\n- **Oracle-NormCP** uses generator scale and provides an efficiency ceiling, not a deployable method.\n\nNegative ablations remain essential: they show that neither removing geography in the abstract nor adding geography indiscriminately accounts for the CMI gain.\n"""
    (OUT/"ABLATION_INTERPRETATION.md").write_text(ablation,encoding="utf-8")
    claims="""# Final claim boundary\n\n## Supported\n\n- Fresh preregistered simulation evidence supports improved interval efficiency for CausalOnly-CMI relative to Raw and GeoOnly under the frozen S2 mechanism.\n- Mean coverage remained within the preregistered tolerance.\n- Causally motivated conditional-dependence feature strength frequently ranked the residual-scale driver first, although the conservative BH gate selected it in 20/50 fresh replicates.\n- The exact NO_SIGNAL fallback controlled S4: 49/50 fresh replicates reverted to Raw-CP.\n- GCCM's negative ablation and the dilution from fixed balanced geography replicated.\n\n## Not supported\n\n- Universal causal identification.\n- Superiority on every spatial dataset or over every possible causal estimator.\n- Confirmatory real-data performance.\n- Local or conditional coverage guarantees. Split conformal supplies marginal coverage under its exchangeability conditions.\n- A claim that geography is universally irrelevant.\n\nPM2.5 evidence remains retrospective/exploratory and produced `NO_CAUSAL_SIGNAL_DETECTED`.\n"""
    (OUT/"FINAL_CLAIM_BOUNDARY.md").write_text(claims,encoding="utf-8")
    family="""# Final method family\n\nRetain all six methods in the manuscript simulation table: Raw-CP, GeoOnly-NormCP, CausalOnly-GCCM-NormCP, CausalOnly-CMI-NormCP, GeoCMI-Balanced-NormCP, and Oracle-NormCP.\n\n**CausalOnly-CMI-NormCP is the supported primary practical method.** Raw establishes the nonlocalized baseline; GeoOnly rules out geography-only benefit; GCCM is the direct negative causal-estimator ablation; GeoCMI tests incremental geography; and Oracle provides the scale-efficiency ceiling. Negative methods should not be removed because their role is inferential, not promotional.\n"""
    (OUT/"FINAL_METHOD_FAMILY.md").write_text(family,encoding="utf-8")
    future="""# Future never-accessed holdout protocol template\n\n## Claim status\n\nThis template is for a genuinely new, never-accessed spatial cross-section or future data batch. Before any outcome is read, record hashes, timestamps, sample IDs, exposure history, predictor provenance, candidate attributes, and disjoint role assignments. If prior label exposure is uncertain, classify the analysis as exploratory.\n\n## Frozen method\n\n- Base predictor and predictions frozen before residual-scale analysis.\n- Tigramite CMIknn with nearest-neighbor parameter 20.\n- Mspace conditioning on coordinates $(s_x,s_y)$.\n- Candidate non-coordinate attributes declared before labels are opened.\n- 99 surrogates and the established deterministic seed policy.\n- BH $q<0.05$ plus positive raw CMI eligibility.\n- Eligible raw CMI normalized to sum to one.\n- Exact constant-scale Raw-CP fallback when no feature is eligible.\n- Final distance contains eligible attributes only; coordinates never enter it.\n- Local scale is the mean scale-fit residual among $k$ nearest neighbors.\n- Fixed $k$ grid {20,40,60,80}; tuning-calibration residual-scale MAE only; larger-$k$ tie rule.\n- Separate final calibration and standard finite-sample normalized split CP.\n- Evaluation labels sealed until weights, standardization, $k$, and conformal quantile are recorded.\n\nFuture outcomes must not change any of these choices. If sample size cannot support the grid or disjoint roles, stop and preregister a separate adaptation rather than modifying this template after inspection.\n"""
    (OUT/"FUTURE_HOLDOUT_PROTOCOL_TEMPLATE.md").write_text(future,encoding="utf-8")
    handoff=f"""# Final simulation handoff\n\n## Final method\n\nCausalOnly-CMI-NormCP uses coordinate-conditioned CMI only to select and weight non-coordinate residual-scale attributes. Its final localization distance is strictly causal-feature-only; NO_SIGNAL produces exact Raw-CP.\n\n## Evidence roles\n\nStages 9 and 11 are method development. Stages 12 and 13 are development-set ablations. Stage 10 is a method diagnostic. Stage 14 is retrospective exploratory real-data evidence. Stage 15 is the sole fresh preregistered confirmatory simulation and must remain the primary results source. Never pool Stage-11--13 replicates with Stage 15.\n\n## Primary numbers\n\nAt alpha=.10 in S2, Raw/GeoOnly/CMI coverage was {s2.loc['Raw-CP','mean_coverage']:.4f}/{s2.loc['GeoOnly-NormCP','mean_coverage']:.4f}/{s2.loc['CausalOnly-CMI-NormCP','mean_coverage']:.4f}; widths were {s2.loc['Raw-CP','mean_width']:.4f}/{s2.loc['GeoOnly-NormCP','mean_width']:.4f}/{s2.loc['CausalOnly-CMI-NormCP','mean_width']:.4f}; interval scores were {s2.loc['Raw-CP','mean_interval_score']:.4f}/{s2.loc['GeoOnly-NormCP','mean_interval_score']:.4f}/{s2.loc['CausalOnly-CMI-NormCP','mean_interval_score']:.4f}. The final class is A. CONFIRMATORY_SUPPORT.\n\n## Canonical locations\n\n- Stage-15 preregistration: `{S15P.relative_to(ROOT)}`\n- Stage-15 confirmatory record: `{S15.relative_to(ROOT)}`\n- This package: `{OUT.relative_to(ROOT)}`\n- Earlier records: see `evidence_chain.csv`.\n\n## Do not rerun\n\nDo not retune or redesign the simulation family, merge development and confirmation samples, rerun historical real data as if independent, or reinterpret CMI as universal causal identification. Artifact hashes may be rechecked read-only.\n\n## Unresolved limitations and next legitimate experiment\n\nFeature gating remains conservative and surrogate resolution affects selection frequency. The simulations cover a particular residual-scale mechanism and predictor. There is no confirmatory real-data result. The next legitimate experiment is a separately preregistered application to a genuinely new, never-accessed spatial/future holdout using `FUTURE_HOLDOUT_PROTOCOL_TEMPLATE.md`; no outcome may be used to alter the frozen choices.\n"""
    (OUT/"FINAL_SIMULATION_HANDOFF.md").write_text(handoff,encoding="utf-8")


def write_plot_script():
    script=r'''"""Render Stage-16 figures from frozen figure-source CSVs only."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; DATA=HERE.parent/"figure_data"
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
COLORS={"Raw-CP":"#555555","GeoOnly-NormCP":"#4C78A8","CausalOnly-GCCM-NormCP":"#E45756","CausalOnly-CMI-NormCP":"#2A9D8F","GeoCMI-Balanced-NormCP":"#F2A541","Oracle-NormCP":"#7A5195"}

def save(fig,name):
    fig.tight_layout(); fig.savefig(HERE/f"{name}.png"); fig.savefig(HERE/f"{name}.pdf"); plt.close(fig)

for letter,filename,cols,ylabel in [
 ("A","figure_A_paired_width_differences.csv",["cmi_minus_raw_width","cmi_minus_geoonly_width"],"Paired mean-width difference"),
 ("B","figure_B_paired_interval_score_differences.csv",["cmi_minus_raw_interval_score","cmi_minus_geoonly_interval_score"],"Paired interval-score difference")]:
    d=pd.read_csv(DATA/filename); fig,ax=plt.subplots(figsize=(6.2,3.5)); x=np.arange(1,len(d)+1)
    labels=["CMI - Raw","CMI - GeoOnly"]
    for j,(c,label) in enumerate(zip(cols,labels)):
        ax.scatter(x+(j-.5)*.16,d[c],s=12,alpha=.65,label=label)
    ax.axhline(0,color="black",lw=.8); ax.set(xlabel="Fresh paired replicate",ylabel=ylabel); ax.legend(frameon=False,ncol=2); save(fig,f"figure_{letter}")

d=pd.read_csv(DATA/"figure_C_oracle_scale_correlation.csv")
order=["GeoOnly-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP"]
fig,ax=plt.subplots(figsize=(5.8,3.8)); vals=[d[d.method==m].test_oracle_correlation.to_numpy() for m in order]
parts=ax.violinplot(vals,showmeans=False,showmedians=True); [b.set_alpha(.35) for b in parts["bodies"]]
for i,v in enumerate(vals,1): ax.scatter(np.full(len(v),i)+np.linspace(-.07,.07,len(v)),v,s=12,alpha=.65)
ax.set_xticks(range(1,4),["GeoOnly","CausalOnly-CMI","GeoCMI"]); ax.set_ylabel("Correlation with oracle scale"); save(fig,"figure_C")

d=pd.read_csv(DATA/"figure_D_feature_selection_frequencies.csv"); fig,ax=plt.subplots(figsize=(6.4,3.5)); y=np.arange(len(d));
labels=[f"{r.scenario}: {r.diagnostic}" for r in d.itertuples()]; ax.errorbar(d.proportion,y,xerr=[d.proportion-d.wilson95_lower,d.wilson95_upper-d.proportion],fmt="o",capsize=3,color="#2A9D8F")
ax.set_yticks(y,labels); ax.set(xlim=(0,1),xlabel="Replicate proportion (Wilson 95% CI)"); save(fig,"figure_D")

d=pd.read_csv(DATA/"figure_E_coverage_width_tradeoff.csv"); fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
for ax,scenario in zip(axes,["S2","S4"]):
    x=d[d.scenario==scenario]
    for r in x.itertuples(): ax.errorbar(r.mean_width,r.mean_coverage,xerr=r.width_sd,yerr=r.coverage_sd,fmt="o",capsize=2,label=r.method,color=COLORS[r.method])
    ax.axhline(.9,color="black",ls="--",lw=.8); ax.set(title=scenario,xlabel="Mean interval width"); ax.grid(alpha=.15)
axes[0].set_ylabel("Mean replicate coverage"); axes[1].legend(frameon=False,fontsize=7,bbox_to_anchor=(1.02,1)); save(fig,"figure_E")
'''
    (OUT/"figures/render_figures.py").write_text(script,encoding="utf-8")


def manifests(integrity):
    sources=[S9/"stage9_summary.md",S10/"stage10b3_summary.json",S11/"stage11_summary.json",S12/"stage12_summary.json",S13/"stage13_summary.json",
             S14A/"feasibility_decision.json",S14/"stage14_summary.json",S15P/"PREREGISTRATION_SEAL.json",S15/"output_manifest.json",S15/"confirmatory_summary.json",
             S15/"scenario_summary_alpha010.csv",S15/"paired_comparisons_alpha010.csv",S15/"cmi99_feature_results.csv",S15/"cmi99_weight_audit.csv",
             S15/"cmi999_comparison.csv",S15/"cmi999_cp_sensitivity.csv",S15/"scale_diagnostics.csv",S15/"alpha_robustness_summary.csv"]
    dump(OUT/"input_manifest.json",{"sources":[{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for p in sources],
      "builder":{"path":str(Path(__file__).resolve().relative_to(ROOT)),"sha256":sha(Path(__file__).resolve())},"scientific_outcomes_recomputed":False})
    outputs=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"output_manifest.json","run_log.txt","stderr.txt"}:
            outputs.append({"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size})
    dump(OUT/"output_manifest.json",{"outputs":outputs,"stage15_integrity_status":integrity["status"]})


def main():
    if OUT.exists() and (OUT/"output_manifest.json").exists(): raise FileExistsError("completed Stage-16 package exists")
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"figures").mkdir(exist_ok=True)
    integrity=verify_stage15(); dump(OUT/"final_integrity_audit.json",integrity)
    evidence_chain(); primary,paired,feature,scale_table,alpha,rr,scale=build_tables(); figure_data(primary,paired,feature,scale_table,alpha,rr,scale)
    write_docs(primary,paired,feature,scale_table,alpha,integrity); write_plot_script()
    (OUT/"README.md").write_text("# Stage 16 final simulation manuscript package\n\nConsolidation-only package derived from frozen Stages 9--15. Stage 15 is the primary fresh confirmatory evidence; development and ablation records remain separate. No scientific outcome was recomputed.\n",encoding="utf-8")
    (OUT/"environment.txt").write_text(f"Python executable: {sys.executable}\nPython: {platform.python_version()}\nPlatform: {platform.platform()}\nnumpy: {np.__version__}\npandas: {pd.__version__}\n",encoding="utf-8")
    manifests(integrity)
    print(json.dumps({"status":"PASS","stage15_integrity":integrity["status"],"output":str(OUT),"files":len(list(OUT.rglob('*')))},indent=2))


if __name__=="__main__": main()
