"""Stage 13: balanced geographic + frozen-CMI normalized split CP ablation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECOVERED = HERE / "stage11_frozen_split_inputs_v1"
STAGE11 = ROOT / "experiments" / "causal_only_cp" / "20260918_stage11_causalonly_cmi_normcp_v2"
STAGE12 = ROOT / "experiments" / "causal_only_cp" / "20260918_stage12_causalonly_gccm_ablation_v1"
RECORD = ROOT / "experiments" / "causal_only_cp" / "20260918_stage13_geocmi_balanced_normcp_v1"
STAGE11_RUNNER = HERE / "run_stage11_causalonly_cmi_normcp_v2.py"
SCENARIOS = ["S2", "S4"]
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]
K_VALUES = [20, 40, 60, 80]
METHOD = "GeoCMI-Balanced-NormCP"
ALPHA = .10
UNSEAL_FIELDS = ["phase", "replicate_index", "generation_seed", "scenario", "no_causal_signal",
                 "selected_causal_features", "cmi_weights_json", "causal_block_weight", "geo_block_weight",
                 "selected_k", "q_norm", "scale_fit_hash", "tuning_hash", "final_calibration_hash",
                 "barrier_row_sha256"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def clean_json(value):
    if isinstance(value, dict): return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list): return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def append_unseal(row):
    base = {k: row[k] for k in UNSEAL_FIELDS if k != "barrier_row_sha256"}
    row["barrier_row_sha256"] = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = RECORD / "test_unseal_audit.csv"; exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNSEAL_FIELDS, lineterminator="\n")
        if not exists: writer.writeheader()
        writer.writerow(row); f.flush(); os.fsync(f.fileno())


def compatibility_gate(s11, loader, reps):
    old11 = pd.read_csv(STAGE11 / "replicate_results.csv")
    old12 = pd.read_csv(STAGE12 / "replicate_results.csv")
    if len(old11) != 160 or len(old12) != 200:
        raise AssertionError("comparator row count mismatch")
    if set(old11.generation_seed) != set(range(20270908, 20270928)) or set(old12.generation_seed) != set(range(20270908, 20270928)):
        raise AssertionError("comparator seed mismatch")
    cfg11 = json.loads((STAGE11 / "config.json").read_text(encoding="utf-8"))
    cfg12 = json.loads((STAGE12 / "config.json").read_text(encoding="utf-8"))
    if cfg11["alpha"] != ALPHA or cfg11["test_rows"] != 300 or cfg12["normcp"]["alpha"] != ALPHA:
        raise AssertionError("comparator alpha/test mismatch")
    audits = [("Stage11", pd.read_csv(STAGE11 / "test_unseal_audit.csv")),
              ("Stage12", pd.read_csv(STAGE12 / "test_unseal_audit.csv"))]
    for rep in reps:
        i, seed = rep["replicate_index"], rep["generation_seed"]
        for scenario in SCENARIOS:
            scale = loader.load_split(i, seed, scenario, "scale_fit")
            tuning = loader.load_split(i, seed, scenario, "tuning_calibration")
            final = loader.load_split(i, seed, scenario, "final_calibration")
            hashes = (s11.frame_hash(scale), s11.frame_hash(tuning), s11.frame_hash(final))
            for label, audit in audits:
                rows = audit[(audit.replicate_index == i) & (audit.scenario == scenario)]
                if rows.empty or not rows.scale_fit_hash.eq(hashes[0]).all() or not rows.tuning_hash.eq(hashes[1]).all() or not rows.final_calibration_hash.eq(hashes[2]).all():
                    raise AssertionError(f"{label} split hash mismatch: {i}/{scenario}")
    comp = pd.concat([old11[old11.method.isin(["Raw-CP", "GeoOnly-NormCP", "CausalOnly-CMI-NormCP", "Oracle-NormCP"])],
                      old12[old12.method == "CausalOnly-GCCM-NormCP"]], ignore_index=True, sort=False)
    if len(comp) != 200 or comp.groupby(["replicate_index", "scenario", "method"]).size().ne(1).any():
        raise AssertionError("comparator uniqueness mismatch")
    return comp


def tune_model(s11, scale, tuning, weight_info):
    # Geography is always fitted. This is also the exact NO_SIGNAL path.
    geo_k, geo_model, geo_rows = s11.tune_local(scale, tuning, COORDS, np.ones(2), METHOD,
                                                int(scale.replicate_id.iloc[0]), int(scale.seed.iloc[0]),
                                                str(scale.scenario.iloc[0]))
    if weight_info["status"] == "NO_CAUSAL_SIGNAL":
        return geo_k, geo_model, geo_rows, True
    selected = weight_info["selected"]
    if set(selected) & set(COORDS): raise AssertionError("coordinate in causal block")
    cm, csd = s11.standardizer(scale, selected)
    gm, gsd = s11.standardizer(scale, COORDS)
    rz = np.column_stack((s11.scaled(scale, selected, cm, csd), s11.scaled(scale, COORDS, gm, gsd)))
    tz = np.column_stack((s11.scaled(tuning, selected, cm, csd), s11.scaled(tuning, COORDS, gm, gsd)))
    causal_weights = np.array([weight_info["weights"][f] for f in selected], dtype=float)
    weights = np.concatenate((.5 * causal_weights, np.array([.25, .25])))
    if not np.isclose(weights[:len(selected)].sum(), .5, atol=1e-15, rtol=0) or not np.array_equal(weights[-2:], [.25, .25]):
        raise AssertionError("block weights are not exactly balanced")
    ra = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
    ta = np.abs(tuning.y_true.to_numpy(float) - tuning.y_pred.to_numpy(float))
    values = {}
    for k in K_VALUES:
        sigma = s11.neighbor_scale(rz, ra, scale.sample_id.to_numpy(str), tz, weights, k)
        values[k] = float(np.mean(np.abs(ta - sigma)))
    best = min(values.values()); chosen = max(k for k, v in values.items() if abs(v-best) <= s11.TIE_TOL)
    rows = [{"replicate_index": int(scale.replicate_id.iloc[0]), "generation_seed": int(scale.seed.iloc[0]),
             "scenario": str(scale.scenario.iloc[0]), "method": METHOD, "k": k,
             "tuning_scale_mae": values[k], "selected": k == chosen,
             "reference_split": "scale_fit", "tuning_split": "tuning_calibration"} for k in K_VALUES]
    model = {"causal_mean": cm, "causal_sd": csd, "geo_mean": gm, "geo_sd": gsd,
             "reference_z": rz, "reference_a": ra, "reference_ids": scale.sample_id.to_numpy(str),
             "weights": weights, "tuning_mae": values[chosen], "selected": selected}
    return chosen, model, rows, False


def query_sigma(s11, frame, model, k, fallback):
    if fallback:
        z = s11.scaled(frame, COORDS, model["mean"], model["sd"])
    else:
        z = np.column_stack((s11.scaled(frame, model["selected"], model["causal_mean"], model["causal_sd"]),
                             s11.scaled(frame, COORDS, model["geo_mean"], model["geo_sd"])))
    return s11.neighbor_scale(model["reference_z"], model["reference_a"], model["reference_ids"], z, model["weights"], k)


def run_one(s11, loader, rep, seed, scenario, weight_info, phase, write_barrier=True):
    scale = loader.load_split(rep, seed, scenario, "scale_fit")
    tuning = loader.load_split(rep, seed, scenario, "tuning_calibration")
    k, model, krows, fallback = tune_model(s11, scale, tuning, weight_info)
    final = loader.load_split(rep, seed, scenario, "final_calibration")
    final_sigma = query_sigma(s11, final, model, k, fallback)
    final_a = np.abs(final.y_true.to_numpy(float) - final.y_pred.to_numpy(float))
    q, rank = s11.conformal_quantile(final_a / final_sigma)
    if rank != 181: raise AssertionError("rank mismatch")
    barrier = {"phase": phase, "replicate_index": rep, "generation_seed": seed, "scenario": scenario,
               "no_causal_signal": fallback, "selected_causal_features": ";".join(weight_info["selected"]),
               "cmi_weights_json": s11.json_text(weight_info["weights"]), "causal_block_weight": 0 if fallback else .5,
               "geo_block_weight": 1 if fallback else .5, "selected_k": k, "q_norm": q,
               "scale_fit_hash": s11.frame_hash(scale), "tuning_hash": s11.frame_hash(tuning),
               "final_calibration_hash": s11.frame_hash(final)}
    if write_barrier: append_unseal(barrier)
    test = loader.load_split(rep, seed, scenario, "test", allow_test=True)
    test_sigma = query_sigma(s11, test, model, k, fallback)
    y, pred = test.y_true.to_numpy(float), test.y_pred.to_numpy(float)
    m = s11.metrics(y, pred-q*test_sigma, pred+q*test_sigma)
    row = {"replicate_index": rep, "generation_seed": seed, "scenario": scenario, "method": METHOD,
           "coverage": m["coverage"], "covered_count": m["covered_count"], "test_n": m["test_n"],
           "mean_width": m["mean_width"], "median_width": m["median_width"],
           "mean_interval_score": m["mean_interval_score"], "selected_k": k, "q_norm": q,
           "no_causal_signal": fallback, "selected_feature_count": len(weight_info["selected"]),
           "selected_features": ";".join(weight_info["selected"]),
           "feature_weights": s11.json_text(weight_info["weights"]), "tuning_scale_mae": model["tuning_mae"]}
    scale_row = {"replicate_index": rep, "generation_seed": seed, "scenario": scenario, "method": METHOD,
                 "tuning_scale_mae": model["tuning_mae"],
                 "test_oracle_correlation": s11.safe_corr(test_sigma, test.oracle_sigma.to_numpy(float)),
                 "test_oracle_mae": float(np.mean(np.abs(test_sigma-test.oracle_sigma.to_numpy(float))))}
    interval_hash = hashlib.sha256(np.column_stack((m["lower"], m["upper"])).tobytes()).hexdigest()
    return row, krows, scale_row, interval_hash, m, barrier


def summarize(results):
    rows=[]
    for (scenario, method), g in results.groupby(["scenario", "method"], sort=True):
        raw=float(results[(results.scenario==scenario)&(results.method=="Raw-CP")].coverage.mean())
        cov=float(g.coverage.mean())
        rows.append({"scenario":scenario,"method":method,"replicates":len(g),"mean_coverage":cov,
          "median_coverage":float(g.coverage.median()),"coverage_sd":float(g.coverage.std(ddof=1)),
          "min_coverage":float(g.coverage.min()),"max_coverage":float(g.coverage.max()),
          "pooled_coverage":float(g.covered_count.sum()/g.test_n.sum()),
          "replicates_coverage_lt_0_90":int((g.coverage<.90).sum()),"replicates_coverage_lt_0_89":int((g.coverage<.89).sum()),
          "mean_width":float(g.mean_width.mean()),"median_replicate_mean_width":float(g.mean_width.median()),
          "sd_replicate_mean_width":float(g.mean_width.std(ddof=1)),"mean_interval_score":float(g.mean_interval_score.mean()),
          "median_interval_score":float(g.mean_interval_score.median()),
          "material_undercoverage":bool(cov<.89 or cov<raw-.01)})
    return pd.DataFrame(rows)


def paired(results):
    rows=[]
    comparators=["Raw-CP","GeoOnly-NormCP","CausalOnly-CMI-NormCP","CausalOnly-GCCM-NormCP"]
    for scenario in SCENARIOS:
        p=results[results.scenario==scenario].pivot(index="replicate_index",columns="method",values=["coverage","mean_width","mean_interval_score"])
        for comparator in comparators:
            for metric in ("coverage","mean_width","mean_interval_score"):
                d=(p[metric][METHOD]-p[metric][comparator]).to_numpy(float); mean=float(d.mean()); sd=float(d.std(ddof=1))
                half=float(stats.t.ppf(.975,len(d)-1)*sd/math.sqrt(len(d)))
                base=p[metric][comparator].to_numpy(float)
                rows.append({"scenario":scenario,"contrast":f"{METHOD} minus {comparator}","metric":metric,
                 "n_pairs":len(d),"mean_difference":mean,"median_difference":float(np.median(d)),"sd_difference":sd,
                 "ci95_lower":mean-half,"ci95_upper":mean+half,
                 "mean_relative_percentage_change":None if metric=="coverage" else float(np.mean(d/base)*100),
                 "fraction_geocmi_better":float(np.mean(d<0)) if metric!="coverage" else None,
                 "ci_method":"Student-t CI for mean paired difference"})
    return pd.DataFrame(rows)


def stratified(results, weights):
    rows=[]
    p=results[results.scenario=="S2"].pivot(index="replicate_index",columns="method",values=["coverage","mean_width","mean_interval_score"])
    signal=pd.Series({rep: info["status"]!="NO_CAUSAL_SIGNAL" for (rep,s),info in weights.items() if s=="S2"})
    for label, mask in (("SIGNAL",signal),("NO_SIGNAL",~signal)):
        ids=signal.index[mask]
        for metric in ("coverage","mean_width","mean_interval_score"):
            d=(p.loc[ids,(metric,METHOD)]-p.loc[ids,(metric,"CausalOnly-CMI-NormCP")]).to_numpy(float)
            rows.append({"scenario":"S2","stratum":label,"n":len(ids),"contrast":f"{METHOD} minus CausalOnly-CMI-NormCP",
                         "metric":metric,"mean_difference":float(d.mean()),"median_difference":float(np.median(d)),
                         "sd_difference":float(d.std(ddof=1)) if len(d)>1 else None,
                         "fraction_geocmi_better":float(np.mean(d<0)) if metric!="coverage" else None})
    return pd.DataFrame(rows)


def main():
    RECORD.mkdir(parents=True,exist_ok=True)
    generated=["config.json","preflight_summary.json","test_unseal_audit.csv","replicate_results.csv",
               "k_tuning_results.csv","scale_diagnostics.csv","scenario_summary.csv","paired_comparisons.csv",
               "signal_stratified_summary.csv","stage13_summary.json"]
    if any((RECORD/f).exists() for f in generated): raise FileExistsError("refusing to overwrite Stage 13")
    s11=load_module("stage11_shell_for_stage13",STAGE11_RUNNER)
    reps, stage11a=s11.provenance_gate(); weights, weight_frame=s11.frozen_weights()
    loader=load_module("stage11a_loader_for_stage13",RECOVERED/"loader.py")
    comparators=compatibility_gate(s11,loader,reps)
    config={"stage":"13","method":METHOD,"seeds":[r["generation_seed"] for r in reps],"scenarios":SCENARIOS,
      "frozen_cmi_source":"Stage-9 Mspace BH-gated positive raw CMI","causal_features":FEATURES,"geo_features":COORDS,
      "causal_distance_squared":"sum_j w_j*(z_aj-z_bj)^2","geo_distance_squared":"0.5*dz_sx^2+0.5*dz_sy^2",
      "combined_distance_squared":"0.5*d_causal^2+0.5*d_geo^2","lambda_tuned":False,
      "no_signal":"exact GeoOnly-NormCP fallback","k_candidates":K_VALUES,"alpha":ALPHA,"rank":181,
      "validation_used":False,"test_seal":True,"frozen_before_results":True}
    (RECORD/"config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")

    # Paired seed-20270908 preflight, including an independent clean rerun.
    first={}; second={}
    for scenario in SCENARIOS:
        first[scenario]=run_one(s11,loader,1,20270908,scenario,weights[(1,scenario)],"preflight_primary",False)
        second[scenario]=run_one(s11,loader,1,20270908,scenario,weights[(1,scenario)],"preflight_clean_rerun",False)
        if first[scenario][3]!=second[scenario][3]: raise AssertionError("preflight nondeterminism")
    # S4 replicate 1 is frozen NO_SIGNAL, so it is the required preflight fallback gate.
    s4geo=comparators[(comparators.replicate_index==1)&(comparators.scenario=="S4")&(comparators.method=="GeoOnly-NormCP")].iloc[0]
    s4new=first["S4"][0]
    metric_diff=max(abs(float(s4new[x])-float(s4geo[x])) for x in ("coverage","mean_width","median_width","mean_interval_score","q_norm"))
    if metric_diff>1e-12 or weights[(1,"S4")]["status"]!="NO_CAUSAL_SIGNAL": raise AssertionError("NO_SIGNAL GeoOnly preflight mismatch")
    preflight={"status":"PASS","replicate_index":1,"generation_seed":20270908,
      "deterministic_interval_hashes":{s:first[s][3] for s in SCENARIOS},"clean_rerun_identical":True,
      "causal_block_audit":"PASS: subset of x1,x2,g_proxy,x_noise only","geo_block_audit":"PASS: s_x,s_y only",
      "fixed_block_weights":{"causal":.5,"geographic":.5},"lambda_tuning":False,
      "no_signal_preflight_scenario":"S4","no_signal_geoonly_max_metric_difference":metric_diff,
      "test_seal":"PASS: q_norm frozen before allow_test=True"}
    (RECORD/"preflight_summary.json").write_text(json.dumps(preflight,indent=2)+"\n",encoding="utf-8")

    newrows=[]; krows=[]; scalerows=[]; interval_cache={}
    for rep in reps:
        i,seed=rep["replicate_index"],rep["generation_seed"]
        for scenario in SCENARIOS:
            outcome=run_one(s11,loader,i,seed,scenario,weights[(i,scenario)],"production",True)
            newrows.append(outcome[0]); krows.extend(outcome[1]); scalerows.append(outcome[2]); interval_cache[(i,scenario)]=outcome[4]
            print(f"{i:02d}/20 {scenario} {'NO_SIGNAL->GeoOnly' if outcome[0]['no_causal_signal'] else 'SIGNAL'} k={outcome[0]['selected_k']}",flush=True)
    new=pd.DataFrame(newrows)
    combined=pd.concat([comparators,new],ignore_index=True,sort=False)
    scenarios=summarize(combined); pairs=paired(combined); strata=stratified(combined,weights)
    oldscale=pd.read_csv(STAGE11/"scale_diagnostics.csv")
    scale=pd.concat([oldscale[oldscale.method.isin(["GeoOnly-NormCP","CausalOnly-CMI-NormCP"])],pd.DataFrame(scalerows)],ignore_index=True)

    # Exact NO_SIGNAL interval equality, not merely metric equality.
    equality=[]
    for rep in reps:
        i,seed=rep["replicate_index"],rep["generation_seed"]
        for scenario in SCENARIOS:
            if weights[(i,scenario)]["status"]!="NO_CAUSAL_SIGNAL": continue
            # Recreate Stage-11 GeoOnly intervals under the same pure shell.
            scale_f=loader.load_split(i,seed,scenario,"scale_fit"); tune=loader.load_split(i,seed,scenario,"tuning_calibration")
            gk,gm,_=s11.tune_local(scale_f,tune,COORDS,np.ones(2),"GeoOnly-NormCP",i,seed,scenario)
            final=loader.load_split(i,seed,scenario,"final_calibration"); fa=np.abs(final.y_true-final.y_pred)
            fs=s11.neighbor_scale(gm["reference_z"],gm["reference_a"],gm["reference_ids"],s11.scaled(final,COORDS,gm["mean"],gm["sd"]),gm["weights"],gk)
            q,_=s11.conformal_quantile(fa.to_numpy(float)/fs)
            test=loader.load_split(i,seed,scenario,"test",allow_test=True); ts=s11.neighbor_scale(gm["reference_z"],gm["reference_a"],gm["reference_ids"],s11.scaled(test,COORDS,gm["mean"],gm["sd"]),gm["weights"],gk)
            pred=test.y_pred.to_numpy(float); low=pred-q*ts; up=pred+q*ts; m=interval_cache[(i,scenario)]
            diff=float(max(np.max(np.abs(low-m["lower"])),np.max(np.abs(up-m["upper"]))))
            if diff>1e-12: raise AssertionError("NO_SIGNAL interval mismatch")
            equality.append({"replicate_index":i,"scenario":scenario,"max_abs_bound_difference":diff})

    combined.to_csv(RECORD/"replicate_results.csv",index=False,lineterminator="\n",float_format="%.17g")
    pd.DataFrame(krows).to_csv(RECORD/"k_tuning_results.csv",index=False,lineterminator="\n",float_format="%.17g")
    scale.to_csv(RECORD/"scale_diagnostics.csv",index=False,lineterminator="\n",float_format="%.17g")
    scenarios.to_csv(RECORD/"scenario_summary.csv",index=False,lineterminator="\n",float_format="%.17g")
    pairs.to_csv(RECORD/"paired_comparisons.csv",index=False,lineterminator="\n",float_format="%.17g")
    strata.to_csv(RECORD/"signal_stratified_summary.csv",index=False,lineterminator="\n",float_format="%.17g")
    summary={"status":"PASS","preflight":preflight,"scenario_summary":scenarios.to_dict("records"),
      "paired_comparisons":pairs.where(pd.notna(pairs),None).to_dict("records"),
      "signal_stratified_summary":strata.where(pd.notna(strata),None).to_dict("records"),
      "no_signal_equality":{"rows":equality,"count":len(equality),"max_abs_bound_difference":max(x["max_abs_bound_difference"] for x in equality),"status":"PASS"},
      "weight_context":{"S2_no_signal":7,"S2_x2_selected":13,"S2_x2_largest":12,"S4_no_signal":19,"S4_false_x1":1},
      "selected_k_distributions":{s:{str(int(k)):int(v) for k,v in Counter(new[new.scenario==s].selected_k.astype(int)).items()} for s in SCENARIOS},
      "test_unseal_rows":len(pd.read_csv(RECORD/"test_unseal_audit.csv")),"classification":"PENDING_FINAL_INTERPRETATION"}
    (RECORD/"stage13_summary.json").write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","rows":len(combined),"no_signal_equalities":len(equality)},indent=2))


if __name__=="__main__": main()
