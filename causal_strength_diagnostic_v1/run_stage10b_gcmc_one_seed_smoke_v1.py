"""Stage 10B one-paired-seed GCMC smoke test using frozen Stage-8/9 scale-fit artifacts."""
from __future__ import annotations

import csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = HERE / "stage8_frozen_replicates_v1"
RECORD = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b_gcmc_one_seed_smoke_v1"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")
RLIB = ROOT / "environment/gcmc_spedm_1.12_r4.6.1/library"
META = ["sample_id", "scenario", "split", "local_calibration_role"]
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stable_hash(lines) -> str:
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()

def load_scale(rep: dict, scenario: str):
    item = rep["scenarios"][scenario]
    paths = {k: ROOT / item[k]["path"] for k in ("predictor_inputs", "scale_fit_labels", "predictions")}
    for k, p in paths.items():
        assert sha(p) == item[k]["sha256"], (k, p)
    x = pd.read_csv(paths["predictor_inputs"], keep_default_na=False, float_precision="round_trip")
    labels = pd.read_csv(paths["scale_fit_labels"], keep_default_na=False, float_precision="round_trip")
    pred = pd.read_csv(paths["predictions"], keep_default_na=False, float_precision="round_trip")
    assert len(x) == len(pred) == 2000 and len(labels) == 200
    assert x.sample_id.is_unique and labels.sample_id.is_unique and pred.sample_id.is_unique
    assert ((labels.split == "local_calibration") & (labels.local_calibration_role == "scale_fit")).all()
    sx = x.loc[(x.split == "local_calibration") & (x.local_calibration_role == "scale_fit")].copy()
    assert len(sx) == 200
    f = sx.merge(labels, on=META, validate="one_to_one", sort=False).merge(pred, on=META, validate="one_to_one", sort=False)
    f = f.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    assert len(f) == 200 and f.sample_id.is_unique and f.isna().sum().sum() == 0
    assert np.isfinite(f[COORDS + FEATURES + ["y", "y_pred"]].to_numpy(float)).all()
    f["A_model"] = np.abs(f.y.to_numpy(float) - f.y_pred.to_numpy(float))
    assert np.isfinite(f.A_model).all()
    scientific = f[["sample_id"] + COORDS + FEATURES + ["A_model"]].copy()
    return scientific, {k: {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p), "rows": len(pd.read_csv(p))} for k, p in paths.items()}

def graph_from_xy(xy: np.ndarray):
    tri = Delaunay(xy)
    edges = sorted({tuple(sorted((int(t[i]) + 1, int(t[j]) + 1))) for t in tri.simplices for i, j in ((0,1),(1,2),(0,2))})
    assert all(1 <= u < v <= 200 for u, v in edges)
    adj = [set() for _ in range(200)]
    for u, v in edges: adj[u-1].add(v-1); adj[v-1].add(u-1)
    assert all(i not in adj[i] for i in range(200))
    rows, cols = zip(*([(u-1,v-1) for u,v in edges] + [(v-1,u-1) for u,v in edges]))
    comps = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(200,200)), directed=False)[0]
    deg = np.array([len(a) for a in adj])
    text = "u,v\n" + "".join(f"{u},{v}\n" for u,v in edges)
    return edges, {"node_count":200,"edge_count":len(edges),"self_edges":0,"symmetric":True,"connected_components":int(comps),"degree":{"min":int(deg.min()),"median":float(np.median(deg)),"mean":float(deg.mean()),"max":int(deg.max())},"adjacency_sha256":hashlib.sha256(text.encode()).hexdigest()}

def main():
    for name in ("gcmc_primary_results.csv","gcmc_sensitivity_results.csv","gcmc_direction_results.csv","fnn_curves.csv","stage10b_summary.json"):
        if (RECORD/name).exists(): raise FileExistsError(f"refusing to overwrite {name}")
    cfg = json.loads((RECORD/"config.json").read_text())
    assert cfg["generation_seed"] == 20270908
    manifest_path = FROZEN / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    rep = next(r for r in manifest["replicates"] if int(r["generation_seed"]) == 20270908)
    assert int(rep["replicate_index"]) == 1
    (RECORD/"inputs").mkdir(exist_ok=True)
    frames, artifacts = {}, {}
    for s in ("S2","S4"):
        frames[s], artifacts[s] = load_scale(rep, s)
        frames[s].to_csv(RECORD/f"inputs/{s}_scale_fit_gcmc.csv", index=False, lineterminator="\n", float_format="%.17g")
    paired_cols = ["sample_id"] + COORDS + FEATURES
    pd.testing.assert_frame_equal(frames["S2"][paired_cols], frames["S4"][paired_cols], check_dtype=False, check_exact=True)
    edges2, graph2 = graph_from_xy(frames["S2"][COORDS].to_numpy(float)); edges4, graph4 = graph_from_xy(frames["S4"][COORDS].to_numpy(float))
    assert edges2 == edges4 and graph2["adjacency_sha256"] == graph4["adjacency_sha256"]
    pd.DataFrame(edges2, columns=["u","v"]).to_csv(RECORD/"inputs/delaunay_edges.csv", index=False, lineterminator="\n")
    hashes = {}
    for s in ("S2","S4"):
        f=frames[s]
        hashes[s]={"row_order_sha256":stable_hash(f.sample_id.astype(str)+"\n"),"sample_id_sha256":stable_hash(sorted(f.sample_id.astype(str))+["\n"]),"coordinate_sha256":hashlib.sha256(f[COORDS].to_csv(index=False,lineterminator="\n",float_format="%.17g").encode()).hexdigest(),"analysis_table_sha256":sha(RECORD/f"inputs/{s}_scale_fit_gcmc.csv")}
    graph_payload={"method":"same SciPy Delaunay simplex-edge convention as simulation_pilot_20260908/gccm_runtime/run_gccm_scale_fit_v1.py","S2":graph2,"S4":graph4,"paired_graph_hash_identical":True,"table_hashes":hashes}
    (RECORD/"graph_diagnostics.json").write_text(json.dumps(graph_payload,indent=2)+"\n")
    inputs={"generation_seed":20270908,"replicate_index":1,"manifest":{"path":str(manifest_path.relative_to(ROOT)).replace("\\","/"),"sha256":sha(manifest_path)},"artifacts":artifacts,"derived_inputs":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted((RECORD/"inputs").glob("*.csv"))},"label_access":{"read":["local_calibration/scale_fit"],"not_read":["train","validation","tuning_calibration","final_calibration","test"]}}
    (RECORD/"input_manifest.json").write_text(json.dumps(inputs,indent=2)+"\n")
    env={**os.environ,"R_LIBS_USER":str(RLIB).replace("\\","/"),"LC_ALL":"C"}
    rscript=HERE/"run_stage10b_gcmc_one_seed_smoke_v1.R"
    commands=[]; started=time.perf_counter()
    for mode in ("full","repro"):
        cmd=[str(RSCRIPT),"--vanilla",str(rscript),str(RECORD),mode]
        commands.append(cmd); proc=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=7200)
        (RECORD/("run_log.txt" if mode=="full" else "repro_run_log.txt")).write_text("COMMAND: "+subprocess.list2cmdline(cmd)+"\n"+proc.stdout)
        (RECORD/("stderr.txt" if mode=="full" else "repro_stderr.txt")).write_text(proc.stderr)
        if proc.returncode: raise RuntimeError(f"R {mode} failed ({proc.returncode}): {proc.stderr[-2000:]}")
    a=pd.read_csv(RECORD/"gcmc_primary_results.csv").sort_values(["scenario","feature"]).reset_index(drop=True)
    b=pd.read_csv(RECORD/"gcmc_primary_repro_results.csv").sort_values(["scenario","feature"]).reset_index(drop=True)
    cols=["forward_cs","reverse_cs","forward_xmap","reverse_xmap","forward_sig","reverse_sig","forward_lower","reverse_lower","forward_upper","reverse_upper"]
    diffs={c:float(np.nanmax(np.abs(a[c].to_numpy(float)-b[c].to_numpy(float)))) for c in cols}
    repro={"fresh_process":True,"rows":len(a),"numeric_columns":cols,"max_absolute_difference":diffs,"tolerance":1e-12,"pass":max(diffs.values())<=1e-12}
    (RECORD/"reproducibility_check.json").write_text(json.dumps(repro,indent=2)+"\n")
    if not repro["pass"]: raise AssertionError("primary GCMC not deterministic")
    cmi=pd.read_csv(HERE/"stage9_feature_recovery_results.csv")
    cmi=cmi[(cmi.generation_seed==20270908)][["scenario","condition","feature","raw_cmi","p_value","bh_q"]]
    cmi.to_csv(RECORD/"stage9_seed_20270908_context.csv",index=False,lineterminator="\n",float_format="%.17g")
    sens=pd.read_csv(RECORD/"gcmc_sensitivity_results.csv")
    diagnostic=[]
    for keys,g in sens.groupby(["scenario","style","E","k"],sort=True):
        g=g.sort_values(["forward_cs","feature"],ascending=[False,True]).reset_index(drop=True)
        vals=dict(zip(g.feature,g.forward_cs)); top=g.iloc[0]
        diagnostic.append({"scenario":keys[0],"style":int(keys[1]),"E":int(keys[2]),"k":int(keys[3]),"top_feature":top.feature,"top_score":float(top.forward_cs),"x2_rank":int(g.index[g.feature.eq('x2')][0]+1),"x2_score":float(vals['x2']),"x2_minus_x_noise":float(vals['x2']-vals['x_noise']),"x2_minus_x1":float(vals['x2']-vals['x1']),"x2_minus_g_proxy":float(vals['x2']-vals['g_proxy'])})
    primary=a.sort_values(["scenario","forward_cs"],ascending=[True,False]).to_dict("records")
    summary={"stage":"10B","status":"ONE-SEED_SMOKE_DIAGNOSTIC","generation_seed":20270908,"primary":primary,"configuration_diagnostics":diagnostic,"reproducibility":repro,"elapsed_seconds":time.perf_counter()-started,"decision_classification":"PENDING_INTERPRETATION"}
    (RECORD/"stage10b_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print("STAGE10B_EXECUTION_PASS",flush=True)

if __name__ == "__main__": main()
