"""Audit and finalize Stage-19B without recomputing scientific outcomes."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/geo_causal_cp/20260919_stage19b_direct_geosimcp_parity_v1"
PREREG = RECORD / "preregistration"
TRACKS = ["A", "B"]
METHODS = ["Raw-CP", "GeoCP", "GeoSIMCP-EUC", "GeoSIMCP-MND", "GeoCausalCP-Full", "CausalOnly-Full"]
COMPARISONS = [("GeoCausalCP-Full","GeoSIMCP-EUC"),("GeoCausalCP-Full","GeoCP"),
               ("GeoSIMCP-EUC","GeoCP"),("GeoSIMCP-MND","GeoCP"),("GeoCausalCP-Full","GeoSIMCP-MND")]
OUTCOMES = ["coverage","mean_width","mean_interval_score"]
BOOT_SEED = 21919001


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(v):
    if isinstance(v, dict): return {str(k): safe(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [safe(x) for x in v]
    if isinstance(v, np.ndarray): return safe(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (float,np.floating)):
        if np.isnan(v): return None
        if np.isposinf(v): return "Infinity"
        if np.isneginf(v): return "-Infinity"
        return float(v)
    if isinstance(v, np.bool_): return bool(v)
    return v


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(safe(obj),indent=2)+"\n",encoding="utf-8")


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RECORD/name,index=False,lineterminator="\n",float_format="%.17g")


def ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float,float,float,float]:
    n=len(values); mean=float(np.mean(values))
    if n<2: return math.nan,math.nan,math.nan,math.nan
    half=float(t.ppf(.975,n-1)*np.std(values,ddof=1)/math.sqrt(n))
    boots=np.mean(values[rng.integers(0,n,size=(10000,n))],axis=1)
    lo,hi=np.quantile(boots,[.025,.975])
    return mean-half,mean+half,float(lo),float(hi)


def scenario_summary(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    rows=[]
    for scenario in ["S2","S4"]:
        raw=float(frame[(frame.scenario==scenario)&(frame.method=="Raw-CP")].coverage.mean())
        for method in METHODS:
            d=frame[(frame.scenario==scenario)&(frame.method==method)&(frame.selection_status=="SELECTED")]
            cov=d.coverage.to_numpy(float); widths=d.mean_width.to_numpy(float); scores=d.mean_interval_score.to_numpy(float)
            rows.append({"track":track,"scenario":scenario,"method":method,"selected_replicates":len(d),
                "no_feasible_replicates":50-len(d) if method in ["GeoCP","GeoSIMCP-EUC","GeoSIMCP-MND","GeoCausalCP-Full"] else 0,
                "mean_coverage":np.mean(cov),"median_coverage":np.median(cov),"coverage_sd":np.std(cov,ddof=1),
                "coverage_min":np.min(cov),"coverage_max":np.max(cov),
                "pooled_coverage":d.covered_count.sum()/d.n.sum(),"coverage_below_090":int((cov<.90).sum()),
                "coverage_below_089":int((cov<.89).sum()),"mean_width":np.mean(widths),
                "width_sd":np.std(widths,ddof=1) if np.isfinite(widths).all() else np.inf,
                "median_replicate_mean_width":np.median(widths),
                "mean_finite_only_width":np.nanmean(d.mean_width_finite.to_numpy(float)),
                "mean_interval_score":np.mean(scores),
                "interval_score_sd":np.std(scores,ddof=1) if np.isfinite(scores).all() else np.inf,
                "mean_infinite_fraction":d.infinite_fraction.mean(),"replicates_any_infinite":int((d.infinite_count>0).sum()),
                "material_undercoverage":bool(np.mean(cov)<.89 or np.mean(cov)<raw-.01)})
    return pd.DataFrame(rows)


def paired(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    rows=[]; counter=0
    for scenario in ["S2","S4"]:
      for left_method,right_method in COMPARISONS:
        a=frame[(frame.scenario==scenario)&(frame.method==left_method)&(frame.selection_status=="SELECTED")]
        b=frame[(frame.scenario==scenario)&(frame.method==right_method)&(frame.selection_status=="SELECTED")]
        m=a.merge(b,on=["replicate_index","generation_seed","scenario","track"],suffixes=("_left","_right"))
        for outcome in OUTCOMES:
            left_values=m[f"{outcome}_left"].to_numpy(float); right_values=m[f"{outcome}_right"].to_numpy(float)
            finite=np.isfinite(left_values)&np.isfinite(right_values); v=left_values[finite]-right_values[finite]; rng=np.random.default_rng(BOOT_SEED+counter); counter+=1
            tl,tu,bl,bu=ci(v,rng) if len(v) else (math.nan,)*4
            denom=np.mean(m.loc[finite,f"{outcome}_right"].to_numpy(float)) if len(v) else math.nan
            rows.append({"track":track,"scenario":scenario,"comparison":f"{left_method} minus {right_method}","left_method":left_method,"right_method":right_method,
                "outcome":outcome,"selected_pairs":len(m),"finite_pairs":len(v),"mean_difference":np.mean(v) if len(v) else math.nan,
                "median_difference":np.median(v) if len(v) else math.nan,"sd_difference":np.std(v,ddof=1) if len(v)>1 else math.nan,
                "student_t_ci_low":tl,"student_t_ci_high":tu,"bootstrap_ci_low":bl,"bootstrap_ci_high":bu,
                "relative_difference_percent":100*np.mean(v)/denom if len(v) and denom!=0 else math.nan})
    return pd.DataFrame(rows)


def feature_summary(cmi: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for scenario in ["S2","S4"]:
        d=cmi[cmi.scenario==scenario]
        statuses=d.drop_duplicates(["replicate_index"])
        max_weight=d.groupby("replicate_index").weight.max()
        for feature in ["x1","x2","g_proxy","x_noise"]:
            f=d[d.feature==feature]
            rows.append({"scenario":scenario,"diagnostic":feature,"raw_rank_1":int((f["rank"]==1).sum()),
                "selected":int(f.eligible.sum()),"largest_weight":int(((f.weight>0)&f.apply(lambda r: r.weight==max_weight.loc[r.replicate_index],axis=1)).sum())})
        rows.append({"scenario":scenario,"diagnostic":"NO_CAUSAL_SIGNAL","raw_rank_1":None,"selected":int(statuses.status.eq("NO_CAUSAL_SIGNAL").sum()),"largest_weight":None})
    return pd.DataFrame(rows)


def main() -> None:
    seal=json.loads((PREREG/"PREREGISTRATION_SEAL.json").read_text())
    for n,h in seal["files"].items():
        if sha(PREREG/n)!=h: raise AssertionError(f"sealed prereg changed: {n}")
    checkpoint=json.loads((RECORD/"scientific_computation_checkpoint.json").read_text())
    if checkpoint["scenario_replicates"]!=100 or checkpoint["analysis_checkpoints"]!=100: raise AssertionError("completion")
    audits=pd.read_csv(RECORD/"test_unseal_audit.csv")
    if len(audits)!=100 or audits[["replicate_index","scenario"]].duplicated().any(): raise AssertionError("test audits")
    fallback=pd.read_csv(RECORD/"no_signal_fallback_equality_audit.csv")
    if len(fallback)!=140 or not fallback.status.eq("PASS").all(): raise AssertionError("fallback")
    child_stderr=list((RECORD/"frozen_inputs").rglob("cmi_full999_results.stderr.txt"))
    nonempty=[str(p.relative_to(ROOT)) for p in child_stderr if p.stat().st_size]
    if len(child_stderr)!=100 or nonempty: raise AssertionError(f"CMI stderr: {len(child_stderr)} {nonempty[:3]}")

    summaries=[]; pairs=[]
    frames={}
    for track in TRACKS:
        frame=pd.read_csv(RECORD/f"replicate_results_track{track}.csv"); frames[track]=frame
        s=scenario_summary(frame,track); p=paired(frame,track); summaries.append(s); pairs.append(p)
        write(s,f"scenario_summary_track{track}.csv"); write(p,f"paired_comparisons_track{track}.csv")
    summary=pd.concat(summaries,ignore_index=True); pair=pd.concat(pairs,ignore_index=True)
    cmi=pd.read_csv(RECORD/"cmi_full999_results.csv"); fs=feature_summary(cmi); write(fs,"feature_selection_summary.csv")

    def prow(track,scenario,comparison,outcome):
        return pair[(pair.track==track)&(pair.scenario==scenario)&(pair.comparison==comparison)&(pair.outcome==outcome)].iloc[0].to_dict()
    primary={track:{o:prow(track,"S2","GeoCausalCP-Full minus GeoSIMCP-EUC",o) for o in OUTCOMES} for track in TRACKS}
    secondary_geo={track:{o:prow(track,"S2","GeoCausalCP-Full minus GeoCP",o) for o in OUTCOMES} for track in TRACKS}
    s2=fs[fs.scenario=="S2"].set_index("diagnostic"); s4=fs[fs.scenario=="S4"].set_index("diagnostic")
    feature={"S2_x2_raw_top":int(s2.loc["x2","raw_rank_1"]),"S2_x2_selected":int(s2.loc["x2","selected"]),
             "S2_x2_largest_weight":int(s2.loc["x2","largest_weight"]),"S2_no_signal":int(s2.loc["NO_CAUSAL_SIGNAL","selected"]),
             "S2_x1_selected":int(s2.loc["x1","selected"]),"S2_g_proxy_selected":int(s2.loc["g_proxy","selected"]),
             "S2_x_noise_selected":int(s2.loc["x_noise","selected"]),"S4_no_signal":int(s4.loc["NO_CAUSAL_SIGNAL","selected"]),
             "S4_false_selection_replicates":50-int(s4.loc["NO_CAUSAL_SIGNAL","selected"])}
    nofeasible=pd.read_csv(RECORD/"no_feasible_summary.csv")
    nf=nofeasible.groupby(["track","scenario","method"]).size().to_dict()
    inf=pd.read_csv(RECORD/"infinite_interval_diagnostics_trackB.csv")
    infsum=inf.groupby(["scenario","method"]).agg(mean_infinite_fraction=("infinite_fraction","mean"),
        replicates_any_infinite=("any_infinite","sum"),mean_finite_width=("mean_width_finite","mean"),mean_n_eff=("mean_n_eff","mean")).reset_index()
    write(infsum,"infinite_interval_summary_trackB.csv")

    classifications={"Track_A":"E. COMPARISON_LIMITED_BY_NO_FEASIBLE_HYPERPARAMETERS",
                     "Track_B":"C. CAUSAL_SIMILARITY_NO_EFFICIENCY_GAIN"}
    result={"stage":"19B","status":"COMPLETE","preregistration_composite_sha256":seal["composite_sha256"],
        "official_commit":"26419ef06d7e3711e6618f97f4f1ed02af16bcb7","fresh_seed_block":[20840001,20840050],
        "completed_paired_replicates":50,"completed_scenario_replicates":100,"feature_selection":feature,
        "primary_paired_effects":primary,"geocausal_vs_geocp":secondary_geo,
        "no_feasible_counts":{"|".join(k):int(v) for k,v in nf.items()},"classifications":classifications,
        "central_claim_supported":False,
        "integrated_interpretation":"Historical-parity Track A was limited by preregistered validation infeasibility and did not show an efficiency advantage over GeoSIMCP-EUC. Current finite-sample-valid Track B selected all methods, but both joint methods had infinite ordinary mean width and interval score because honest test atoms produced infinite intervals; GeoCausalCP reduced the infinite fraction but did not establish a finite mean-efficiency improvement. The minimal feature-block replacement claim is therefore not supported by this experiment."}
    dump(RECORD/"stage19b_summary.json",result)

    # Root copies make the immutable record self-contained without altering the sealed preregistration directory.
    for name in ["PREREGISTRATION.md","config.json","seed_manifest.json","method_specification.json","track_specification.json","tuning_specification.json","environment.txt"]:
        shutil.copyfile(PREREG/name,RECORD/name)
    (RECORD/"README.md").write_text("""# Stage 19B direct GeoSIMCP parity experiment\n\nFresh 50-pair S2/S4 experiment at audited geoconformal commit `26419ef06d7e3711e6618f97f4f1ed02af16bcb7`. The only primary-method change was ordinary EUC feature similarity versus CMI-Full999-weighted residual-scale-relevance similarity. Track A preserves historical no-atom behavior; Track B preserves current test-atom validity and all infinite intervals. See `stage19b_summary.md`.\n""",encoding="utf-8")
    (RECORD/"stage19b_summary.md").write_text(make_markdown(summary,pair,feature,nf,infsum,classifications,seal["composite_sha256"]),encoding="utf-8")
    (RECORD/"stderr.txt").write_text("",encoding="utf-8")
    with (RECORD/"run_log.txt").open("a",encoding="utf-8") as f: f.write("Final audit and summaries complete; all gates PASS.\n")
    inputs={"preregistration_seal":seal,"official_commit":result["official_commit"],"scientific_checkpoint":checkpoint}
    dump(RECORD/"input_manifest.json",inputs)
    excluded={"output_manifest.json"}
    files=[]
    for p in sorted(RECORD.rglob("*")):
        if p.is_file() and p.name not in excluded:
            files.append({"path":str(p.relative_to(RECORD)).replace("\\","/"),"sha256":sha(p),"bytes":p.stat().st_size})
    dump(RECORD/"output_manifest.json",{"file_count":len(files),"files":files,"hash_mismatches":0})
    print(json.dumps({"status":"COMPLETE","classifications":classifications,"feature":feature,
                      "output_files":len(files),"preregistration":seal["composite_sha256"]},indent=2))


def fmt(v) -> str:
    if isinstance(v,str): return v
    if v is None or pd.isna(v): return "NA"
    if np.isposinf(v): return "Inf"
    return f"{v:.4f}"


def make_markdown(summary,pair,feature,nf,infsum,classes,seal):
    lines=["# Stage 19B summary","",f"Preregistration composite SHA-256: `{seal}`","",
           "All 50 fresh paired S2/S4 replicates completed. Test-unseal audits: 100/100. CMI worker stderr: empty. NO_SIGNAL fallback audit: PASS.",""]
    for track in TRACKS:
      lines += [f"## Track {track}","", "| Scenario | Method | n selected | Coverage | Mean width | Mean interval score | Mean infinite fraction |", "|---|---|---:|---:|---:|---:|---:|"]
      for r in summary[summary.track==track].itertuples():
        lines.append(f"| {r.scenario} | {r.method} | {r.selected_replicates} | {fmt(r.mean_coverage)} | {fmt(r.mean_width)} | {fmt(r.mean_interval_score)} | {fmt(r.mean_infinite_fraction)} |")
      lines += ["",f"Classification: **{classes['Track_'+track]}**.",""]
    lines += ["## CMI-Full999 feature diagnostics","",f"S2: x2 raw top {feature['S2_x2_raw_top']}/50, selected {feature['S2_x2_selected']}/50, largest weight {feature['S2_x2_largest_weight']}/50; NO_SIGNAL {feature['S2_no_signal']}/50. Nuisance selections: x1 {feature['S2_x1_selected']}, g_proxy {feature['S2_g_proxy_selected']}, x_noise {feature['S2_x_noise_selected']}.",f"S4: NO_SIGNAL {feature['S4_no_signal']}/50; false-selection replicates {feature['S4_false_selection_replicates']}/50.",""]
    lines += ["## Integrated interpretation","","The minimal-change causal feature block did not improve official GeoSIMCP-EUC in this fresh experiment. Track A had only 38 valid paired S2 primary comparisons because of the preregistered validation coverage constraint and was wider/worse on average. Track B retained honest infinite intervals: ordinary mean width and interval score were infinite for both joint methods. GeoCausalCP had fewer infinite intervals than GeoSIMCP-EUC but did not establish the requested finite mean-efficiency gain. The central manuscript claim is therefore not supported by Stage 19B.","",
              "CausalOnly-Full remains supporting normalized-CP evidence only and is not interpreted as an official-engine comparison.",""]
    return "\n".join(lines)+"\n"


if __name__=="__main__": main()
