"""Diagnostic-only mechanism-of-failure audit of immutable Stage-19B outputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments/geo_causal_cp/20260919_stage19b_direct_geosimcp_parity_v1"
OUTPUT = ROOT / "experiments/geo_causal_cp/20260919_stage20a_stage19b_failure_audit_v1"
STAGE19_CODE = ROOT / "geo_causal_cp_v1/run_stage19b_direct_parity_v1.py"
FEATURES = ["x1","x2","g_proxy","x_noise"]
COORDS = ["s_x","s_y"]
META = ["sample_id","scenario","split","local_calibration_role"]
OFFICIAL_METHODS = ["GeoCP","GeoSIMCP-EUC","GeoSIMCP-MND","GeoCausalCP-Full"]
COMPARE_METHODS = ["GeoCP","GeoSIMCP-EUC","GeoCausalCP-Full"]
BOOT_SEED = 22020001


def load_module(name: str, path: Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


s19=load_module("stage19b_audited_functions",STAGE19_CODE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,np.ndarray): return safe(v.tolist())
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(float,np.floating)):
        if np.isnan(v): return None
        if np.isposinf(v): return "Infinity"
        if np.isneginf(v): return "-Infinity"
        return float(v)
    if isinstance(v,np.bool_): return bool(v)
    return v


def dump(path: Path,obj) -> None:
    path.write_text(json.dumps(safe(obj),indent=2)+"\n",encoding="utf-8")


def write(frame: pd.DataFrame,name: str) -> None:
    frame.to_csv(OUTPUT/name,index=False,lineterminator="\n",float_format="%.17g")


def verify_input() -> dict:
    manifest=json.loads((INPUT/"output_manifest.json").read_text())
    bad=[]
    for row in manifest["files"]:
        p=INPUT/row["path"]
        if not p.exists() or sha(p)!=row["sha256"] or p.stat().st_size!=row["bytes"]: bad.append(row["path"])
    seal=json.loads((INPUT/"preregistration/PREREGISTRATION_SEAL.json").read_text())
    sealed_bad=[n for n,h in seal["files"].items() if sha(INPUT/"preregistration"/n)!=h]
    if bad or sealed_bad: raise AssertionError((bad[:5],sealed_bad))
    return {"stage19b_manifest_files":manifest["file_count"],"stage19b_hash_mismatches":0,
            "stage19b_preregistration_composite_sha256":seal["composite_sha256"],"sealed_file_mismatches":0}


def scenario_dir(index,seed,scenario):
    return INPUT/"frozen_inputs"/f"replicate_{int(index):02d}_seed_{int(seed)}"/scenario


def classify_selection(group: pd.DataFrame) -> dict:
    selected=group.loc[group.eligible.astype(bool),"feature"].tolist(); status=str(group.status.iloc[0])
    if status=="NO_CAUSAL_SIGNAL": exclusive="NO_SIGNAL"
    elif selected==["x2"]: exclusive="CLEAN_X2_ONLY"
    elif "x2" in selected: exclusive="X2_PLUS_NUISANCE"
    else: exclusive="OTHER_ONLY"
    return {"signal_status":"SIGNAL" if status=="SIGNAL" else "NO_SIGNAL","exclusive_stratum":exclusive,
            "selected_features":";".join(selected),"x2_selected":"x2" in selected}


def interval_point_score(y,pred,u,alpha=.1):
    lo,hi=pred-u,pred+u; width=hi-lo; score=width.copy()
    low=y<lo; high=y>hi
    score[low]+=(2/alpha)*(lo[low]-y[low]); score[high]+=(2/alpha)*(y[high]-hi[high])
    return width,score,np.isinf(u)


def weights_for(method,selected,calib,test,info):
    h=float(selected.bandwidth); lam=float(selected["lambda"]) if pd.notna(selected["lambda"]) else None
    dgeo2=s19.pairwise_sq(test[COORDS].to_numpy(float),calib[COORDS].to_numpy(float))
    transform=s19.fit_feature_transform(calib,method,info) if method!="GeoCP" else None
    fallback=method=="GeoCausalCP-Full" and (info["status"]=="NO_CAUSAL_SIGNAL" or transform["kind"]=="fallback")
    if method=="GeoCP" or fallback:
        w=np.exp(-.5*dgeo2/(h*h)); dfeat2=None
    else:
        dfeat2=s19.feature_distance_sq(test,transform)
        w=np.exp(-.5*(lam*dgeo2+(1-lam)*dfeat2)/((h+1e-8)**2))
    return w,dgeo2,dfeat2,transform,fallback


def describe(values):
    a=np.asarray(values,float); a=a[np.isfinite(a)]
    if not len(a): return {"n":0,"mean":np.nan,"median":np.nan,"sd":np.nan,"min":np.nan,"max":np.nan}
    return {"n":len(a),"mean":np.mean(a),"median":np.median(a),"sd":np.std(a,ddof=1) if len(a)>1 else np.nan,"min":np.min(a),"max":np.max(a)}


def distribution_table(selected,kind,selection_map):
    rows=[]
    methods=["GeoSIMCP-EUC","GeoSIMCP-MND","GeoCausalCP-Full"] if kind=="lambda" else OFFICIAL_METHODS
    value_col="lambda" if kind=="lambda" else "bandwidth"
    selected=selected[selected.status.eq("SELECTED")].merge(selection_map,on=["replicate_index","scenario"],how="left")
    for (track,scenario,method),g in selected[selected.method.isin(methods)].groupby(["track","scenario","method"]):
        strata=[("ALL",g)]
        if method=="GeoCausalCP-Full":
            strata += [(s,g[g.signal_status.eq(s)]) for s in ["SIGNAL","NO_SIGNAL"]]
            strata += [(s,g[g.exclusive_stratum.eq(s)]) for s in ["CLEAN_X2_ONLY","X2_PLUS_NUISANCE","OTHER_ONLY"]]
        for stratum,d in strata:
            vals=d[value_col].to_numpy(float); desc=describe(vals)
            row={"quantity":kind,"track":track,"scenario":scenario,"method":method,"stratum":stratum,
                 "selected_replicates":len(d),"not_applicable_count":int(pd.isna(d[value_col]).sum()),**desc}
            if kind=="lambda":
                finite=vals[np.isfinite(vals)]
                row.update({"lambda_eq_0":int(np.sum(np.isclose(finite,0))),"lambda_0_to_lt_025":int(np.sum((finite>0)&(finite<.25))),
                  "lambda_025_to_lt_050":int(np.sum((finite>=.25)&(finite<.5))),"lambda_050_to_lt_075":int(np.sum((finite>=.5)&(finite<.75))),
                  "lambda_075_to_lt_1":int(np.sum((finite>=.75)&(finite<1))),"lambda_eq_1":int(np.sum(np.isclose(finite,1)))})
            rows.append(row)
    return pd.DataFrame(rows)


def ci_row(values,rng):
    v=np.asarray(values,float); n=len(v)
    if n<2: return {"n":n,"mean":np.mean(v) if n else np.nan,"median":np.median(v) if n else np.nan,"sd":np.nan,
                    "student_t_ci_low":np.nan,"student_t_ci_high":np.nan,"bootstrap_ci_low":np.nan,"bootstrap_ci_high":np.nan}
    mean=np.mean(v); half=t.ppf(.975,n-1)*np.std(v,ddof=1)/np.sqrt(n)
    boot=np.mean(v[rng.integers(0,n,size=(10000,n))],axis=1); lo,hi=np.quantile(boot,[.025,.975])
    return {"n":n,"mean":mean,"median":np.median(v),"sd":np.std(v,ddof=1),"student_t_ci_low":mean-half,
            "student_t_ci_high":mean+half,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi}


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    integrity=verify_input()
    (OUTPUT/"run_log.txt").write_text("Stage-20A diagnostic-only audit started after Stage-19B hash verification.\n",encoding="utf-8")
    (OUTPUT/"stderr.txt").write_text("",encoding="utf-8")
    cmi=pd.read_csv(INPUT/"cmi_full999_results.csv")
    map_rows=[]
    infos={}
    for (idx,seed,scenario),g in cmi.groupby(["replicate_index","generation_seed","scenario"],sort=True):
        cls=classify_selection(g); map_rows.append({"replicate_index":idx,"generation_seed":seed,"scenario":scenario,**cls})
        weights=dict(zip(g.feature,g.weight.astype(float))); infos[(int(idx),scenario)]={"status":str(g.status.iloc[0]),
            "selected":[f for f in FEATURES if weights[f]>0],"weights":weights}
    selection_map=pd.DataFrame(map_rows)
    selected=pd.concat([pd.read_csv(INPUT/f"selected_hyperparameters_track{x}.csv") for x in "AB"],ignore_index=True)
    results=pd.concat([pd.read_csv(INPUT/f"replicate_results_track{x}.csv") for x in "AB"],ignore_index=True)
    results=results.merge(selection_map[["replicate_index","scenario","signal_status","exclusive_stratum"]],on=["replicate_index","scenario"],how="left")
    lambdas=distribution_table(selected,"lambda",selection_map); bandwidths=distribution_table(selected,"bandwidth",selection_map)
    write(lambdas,"lambda_distribution.csv"); write(bandwidths,"bandwidth_distribution.csv")

    point_rows=[]; alignment_rows=[]; neighbor_rows=[]; finite_score_rows=[]; weight_rows=[]
    seeds=sorted(cmi[["replicate_index","generation_seed"]].drop_duplicates().itertuples(index=False,name=None))
    selected_b=selected[(selected.track=="B")&selected.status.eq("SELECTED")]
    for pos,(index,seed) in enumerate(seeds,1):
      for scenario in ["S2","S4"]:
        base=scenario_dir(index,seed,scenario); dev=pd.read_csv(base/"development.csv",keep_default_na=False)
        calib=dev[dev.split_role.eq("final_calibration")].copy(); x=pd.read_csv(base/"predictor_inputs.csv",keep_default_na=False)
        pred=pd.read_csv(base/"predictions.csv",keep_default_na=False); cov=x[x.split.eq("test")].merge(pred,on=META+["split_role"],validate="one_to_one",sort=False)
        sealed=pd.read_csv(base/"test_sealed.csv",keep_default_na=False); test=cov.merge(sealed,on=META+["split_role"],validate="one_to_one",sort=False)
        scores=np.abs(calib.y_true.to_numpy(float)-calib.y_pred.to_numpy(float)); y=test.y_true.to_numpy(float); yp=test.y_pred.to_numpy(float)
        info=infos[(int(index),scenario)]; cls=selection_map[(selection_map.replicate_index==index)&(selection_map.scenario==scenario)].iloc[0]
        method_cache={}
        for method in OFFICIAL_METHODS:
            sel=selected_b[(selected_b.replicate_index==index)&(selected_b.scenario==scenario)&(selected_b.method==method)].iloc[0]
            w,dgeo2,dfeat2,transform,fallback=weights_for(method,sel,calib,test,info)
            mass=w.sum(axis=1); ess=(mass*mass)/(np.sum(w*w,axis=1)+1e-14)
            u=s19.weighted_quantile(scores,w,.9,self_weights=np.ones(len(test)))
            width,iscore,infmask=interval_point_score(y,yp,u); exact=mass/(mass+1)>=.9-1e-9
            if not np.array_equal(~exact,infmask): raise AssertionError(f"mass/infinity mismatch {index} {scenario} {method}")
            method_cache[method]={"w":w,"dgeo2":dgeo2,"dfeat2":dfeat2,"transform":transform,"fallback":fallback}
            for j in range(len(test)):
                point_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,"method":method,
                    "signal_status":cls.signal_status,"exclusive_stratum":cls.exclusive_stratum,"test_sample_id":test.sample_id.iloc[j],
                    "kernel_mass":mass[j],"mass_below_9":mass[j]<9,"mass_meets_exact_tolerance":exact[j],"ess":ess[j],
                    "interval_infinite":infmask[j],"finite_width":width[j] if np.isfinite(width[j]) else np.nan,
                    "finite_interval_score":iscore[j] if np.isfinite(iscore[j]) else np.nan})
            finite_score_rows.append({"replicate_index":index,"scenario":scenario,"method":method,
                "finite_only_interval_score":np.mean(iscore[np.isfinite(iscore)]) if np.isfinite(iscore).any() else np.nan})

        if scenario=="S2":
            sigma_c=calib.oracle_sigma.to_numpy(float); sigma_t=test.oracle_sigma.to_numpy(float)
            mismatch=np.abs(sigma_t[:,None]-sigma_c[None,:]); logmis=np.abs(np.log(sigma_t)[:,None]-np.log(sigma_c)[None,:])
            dgeo=np.sqrt(method_cache["GeoCP"]["dgeo2"])
            euc_feat=np.sqrt(method_cache["GeoSIMCP-EUC"]["dfeat2"])
            distances=[("Geographic",None,dgeo),("GeoSIMCP-EUC-feature",None,euc_feat)]
            if info["status"]=="SIGNAL":
                causal_feat=np.sqrt(method_cache["GeoCausalCP-Full"]["dfeat2"])
                distances.append(("GeoCausalCP-Full-feature",None,causal_feat))
            for track in "AB":
                for method,label,feat in [("GeoSIMCP-EUC","GeoSIMCP-EUC-joint",euc_feat)]:
                    sel=selected[(selected.track==track)&(selected.replicate_index==index)&(selected.scenario==scenario)&(selected.method==method)]
                    if len(sel) and sel.status.iloc[0]=="SELECTED":
                        lam=float(sel["lambda"].iloc[0]); distances.append((label,track,np.sqrt(lam*dgeo*dgeo+(1-lam)*feat*feat)))
                if info["status"]=="SIGNAL":
                    sel=selected[(selected.track==track)&(selected.replicate_index==index)&(selected.scenario==scenario)&(selected.method=="GeoCausalCP-Full")]
                    if len(sel) and sel.status.iloc[0]=="SELECTED":
                        lam=float(sel["lambda"].iloc[0]); distances.append(("GeoCausalCP-Full-joint",track,np.sqrt(lam*dgeo*dgeo+(1-lam)*causal_feat*causal_feat)))
            for method,track,dist in distances:
                alignment_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,"signal_status":cls.signal_status,
                    "exclusive_stratum":cls.exclusive_stratum,"distance":method,"track":track,
                    "spearman_oracle_abs":spearmanr(dist.ravel(),mismatch.ravel()).statistic,
                    "spearman_oracle_log":spearmanr(dist.ravel(),logmis.ravel()).statistic})
                order=np.argsort(dist,axis=1,kind="stable")
                for k in [20,40,80]:
                    ix=order[:,:k]
                    neighbor_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,"signal_status":cls.signal_status,
                      "exclusive_stratum":cls.exclusive_stratum,"distance":method,"track":track,"k":k,
                      "mean_oracle_abs_mismatch":np.take_along_axis(mismatch,ix,axis=1).mean(),
                      "mean_oracle_log_mismatch":np.take_along_axis(logmis,ix,axis=1).mean()})
            if info["status"]=="SIGNAL":
                nuisance=sum(info["weights"][f] for f in FEATURES if f!="x2")
                weight_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,
                    "exclusive_stratum":cls.exclusive_stratum,"x2_weight":info["weights"]["x2"],
                    "nuisance_total_weight":nuisance,"eligible_feature_count":len(info["selected"]),"selected_features":";".join(info["selected"])})
      if pos%10==0: print(f"DIAGNOSTIC {pos}/50",flush=True)

    point=pd.DataFrame(point_rows); finite_score=pd.DataFrame(finite_score_rows)
    write(point,"kernel_mass_pointwise.csv")
    # Kernel mass and ESS summaries, with S2 signal/fallback stratification.
    mass_rows=[]; ess_rows=[]
    for (scenario,method),g in point.groupby(["scenario","method"]):
        strata=[("ALL",g)]
        if scenario=="S2": strata += [(s,g[g.signal_status==s]) for s in ["SIGNAL","NO_SIGNAL"]]
        for stratum,d in strata:
            if d.empty: continue
            a=d.kernel_mass.to_numpy(float); e=d.ess.to_numpy(float)
            mass_rows.append({"scenario":scenario,"method":method,"stratum":stratum,"test_points":len(d),
                "mass_mean":a.mean(),"mass_median":np.median(a),"mass_p05":np.quantile(a,.05),"mass_p25":np.quantile(a,.25),"mass_min":a.min(),
                "fraction_mass_below_9":np.mean(a<9),"fraction_mass_exactly_infinite":d.interval_infinite.mean(),
                "mass_infinity_indicator_mismatches":int(np.sum((~d.mass_meets_exact_tolerance)!=d.interval_infinite))})
            ess_rows.append({"scenario":scenario,"method":method,"stratum":stratum,"test_points":len(d),"ess_mean":e.mean(),
                "ess_median":np.median(e),"ess_p05":np.quantile(e,.05),"ess_p25":np.quantile(e,.25),"ess_min":e.min()})
    write(pd.DataFrame(mass_rows),"kernel_mass_summary.csv"); write(pd.DataFrame(ess_rows),"ess_summary.csv")
    alignment=pd.DataFrame(alignment_rows); neighbors=pd.DataFrame(neighbor_rows)
    write(alignment,"oracle_distance_alignment.csv"); write(neighbors,"oracle_neighbor_alignment.csv")
    write(pd.DataFrame(weight_rows),"cmi_weight_diagnostics.csv")

    # Signal-stratified performance and paired contrasts.
    results=results.merge(finite_score,on=["replicate_index","scenario","method"],how="left")
    perf=[]
    strata=["NO_SIGNAL","SIGNAL","CLEAN_X2_ONLY","X2_PLUS_NUISANCE","OTHER_ONLY"]
    for track in "AB":
      for stratum in strata:
        key="signal_status" if stratum in ["SIGNAL","NO_SIGNAL"] else "exclusive_stratum"
        for method in COMPARE_METHODS:
            d=results[(results.track==track)&(results.scenario=="S2")&(results.method==method)&(results.selection_status=="SELECTED")&(results[key]==stratum)]
            if d.empty: continue
            perf.append({"track":track,"stratum":stratum,"method":method,"selected_replicates":len(d),
                "mean_coverage":d.coverage.mean(),"mean_width":d.mean_width.mean(),"median_replicate_width":d.median_width.mean(),
                "mean_interval_score":d.mean_interval_score.mean(),"mean_infinite_fraction":d.infinite_fraction.mean(),
                "mean_finite_only_width":d.mean_width_finite.mean(),"mean_finite_only_interval_score":d.finite_only_interval_score.mean()})
    write(pd.DataFrame(perf),"signal_stratified_performance.csv")
    pair_rows=[]; counter=0
    for track in "AB":
      for stratum in strata:
        key="signal_status" if stratum in ["SIGNAL","NO_SIGNAL"] else "exclusive_stratum"
        a=results[(results.track==track)&(results.scenario=="S2")&(results.method=="GeoCausalCP-Full")&(results.selection_status=="SELECTED")&(results[key]==stratum)]
        b=results[(results.track==track)&(results.scenario=="S2")&(results.method=="GeoSIMCP-EUC")&(results.selection_status=="SELECTED")&(results[key]==stratum)]
        m=a.merge(b,on=["replicate_index","generation_seed","scenario","track",key],suffixes=("_causal","_euc"))
        if m.empty: continue
        status={"selected_pairs":len(m),"both_any_infinite":int(((m.infinite_count_causal>0)&(m.infinite_count_euc>0)).sum()),
          "causal_only_any_infinite":int(((m.infinite_count_causal>0)&(m.infinite_count_euc==0)).sum()),
          "euc_only_any_infinite":int(((m.infinite_count_causal==0)&(m.infinite_count_euc>0)).sum()),
          "neither_any_infinite":int(((m.infinite_count_causal==0)&(m.infinite_count_euc==0)).sum())}
        outcomes=["coverage","mean_width","mean_interval_score"]
        if track=="B": outcomes += ["infinite_fraction","mean_width_finite","finite_only_interval_score"]
        for outcome in outcomes:
            x=m[f"{outcome}_causal"].to_numpy(float); yv=m[f"{outcome}_euc"].to_numpy(float); keep=np.isfinite(x)&np.isfinite(yv); diff=x[keep]-yv[keep]
            row=ci_row(diff,np.random.default_rng(BOOT_SEED+counter)); counter+=1
            pair_rows.append({"track":track,"stratum":stratum,"comparison":"GeoCausalCP-Full minus GeoSIMCP-EUC","outcome":outcome,
                              **status,"finite_pairs":len(diff),**row})
    write(pd.DataFrame(pair_rows),"signal_only_paired_comparisons.csv")

    # Validation-to-test generalization and no-feasible diagnostics.
    gen=[]
    for track in "AB":
        sel=pd.read_csv(INPUT/f"selected_hyperparameters_track{track}.csv")
        testres=pd.read_csv(INPUT/f"replicate_results_track{track}.csv")
        m=sel[sel.status.eq("SELECTED")].merge(testres,on=["replicate_index","generation_seed","scenario","track","method"],suffixes=("_validation","_test"))
        for r in m.itertuples():
            gen.append({"track":track,"replicate_index":r.replicate_index,"generation_seed":r.generation_seed,"scenario":r.scenario,"method":r.method,
                "validation_coverage":r.validation_coverage,"test_coverage":r.coverage,"coverage_optimism_validation_minus_test":r.validation_coverage-r.coverage,
                "validation_interval_score":r.validation_interval_score,"test_interval_score":r.mean_interval_score,
                "interval_score_optimism_test_minus_validation":r.mean_interval_score-r.validation_interval_score})
    write(pd.DataFrame(gen),"validation_test_generalization.csv")
    nf=[]
    for track in "AB":
        sel=pd.read_csv(INPUT/f"selected_hyperparameters_track{track}.csv")
        for r in sel[sel.status.eq("NO_FEASIBLE_HYPERPARAMETER")].itertuples():
            deficit=.9-r.validation_coverage
            nf.append({"track":track,"replicate_index":r.replicate_index,"generation_seed":r.generation_seed,"scenario":r.scenario,"method":r.method,
                "maximum_validation_coverage":r.validation_coverage,"coverage_deficit_below_090":deficit,
                "best_diagnostic_bandwidth":r.bandwidth,"best_diagnostic_lambda":getattr(r,"_7",np.nan),
                "best_diagnostic_interval_score":r.validation_interval_score,
                "deficit_category":"WITHIN_0.01" if deficit<=.01+1e-12 else "0.01_TO_0.03" if deficit<=.03 else "MORE_THAN_0.03"})
    write(pd.DataFrame(nf),"no_feasible_diagnostics.csv")
    dump(OUTPUT/"integrity_audit.json",integrity)
    dump(OUTPUT/"analysis_checkpoint.json",{"status":"DIAGNOSTIC_COMPUTATION_COMPLETE","input_scenario_replicates":100,
         "kernel_mass_point_rows":len(point),"oracle_alignment_rows":len(alignment),"no_new_simulation":True,"no_cmi_rerun":True,"no_retuning":True})
    (OUTPUT/"run_log.txt").write_text((OUTPUT/"run_log.txt").read_text()+"Diagnostic computations complete.\n",encoding="utf-8")
    print(json.dumps({"status":"COMPUTATION_COMPLETE","integrity":integrity,"point_rows":len(point),"alignment_rows":len(alignment),"neighbor_rows":len(neighbors)},indent=2))


if __name__=="__main__": main()
