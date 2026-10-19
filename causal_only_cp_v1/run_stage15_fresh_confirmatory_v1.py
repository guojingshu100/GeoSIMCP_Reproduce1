"""Stage 15 fresh confirmatory simulation execution.

This runner is governed by the immutable Stage-15A preregistration.  It
generates serialized fresh replicates, freezes train-only RF predictions,
executes CMI/GCCM, preserves a test-label barrier, and evaluates the six
predeclared methods at all three alpha levels.  It is checkpoint-aware so a
recoverable process interruption does not require discarding completed
scenario artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
PREREG = ROOT / "experiments" / "confirmatory_simulation" / "20260918_stage15_confirmatory_preregistration_v1"
RECORD = ROOT / "experiments" / "confirmatory_simulation" / "20260918_stage15_fresh_confirmatory_v1"
GENERATOR_SOURCE = ROOT / "simulation_pilot_20260908" / "generator" / "generate_scenarios.py"
GENERATOR_CONFIG = ROOT / "simulation_pilot_20260908" / "generator" / "config_v1.json"
PREDICTOR_SOURCE = ROOT / "simulation_pilot_20260908" / "predictor" / "fit_predictors.py"
PREDICTOR_CONFIG = ROOT / "simulation_pilot_20260908" / "predictor" / "predictor_config_v1.json"
CMI_WORKER = HERE / "stage15_cmi_worker_v1.py"
CMI_PYTHON = ROOT / "causal_strength_diagnostic_v1" / ".venv_cmi_v1" / "Scripts" / "python.exe"
GCCM_WRAPPER = HERE / "gccm_stage12_attribute_wrapper_v1.R"
GCCM_CORE = ROOT / "geoconformal-main" / "GCCP_Seattle" / "02_gccm" / "GCCM_updated 2" / "R" / "GCCM" / "GCCM4Lattice.r"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")

FEATURES_ALL = ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"]
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]
META = ["sample_id", "scenario", "split", "local_calibration_role"]
SCENARIOS = ["S2", "S4"]
METHODS = ["Raw-CP", "GeoOnly-NormCP", "CausalOnly-GCCM-NormCP",
           "CausalOnly-CMI-NormCP", "GeoCMI-Balanced-NormCP", "Oracle-NormCP"]
ALPHAS = [.05, .10, .20]
K_VALUES = [20, 40, 60, 80]
FLOOR = 1e-12
TIE_TOL = 1e-12
CMI_MASTER_SEED = 20260910
GCCM_PARAMS = {"E": 3, "tau": 1, "b": 4, "seed": 42,
               "direction": "effect_xmap_cause", "library_sizes": [10, 18, 33, 60, 109, 198]}
EXPECTED_COUNTS = {"train": 1000, "scale_fit": 200, "tuning_calibration": 100,
                   "validation": 200, "final_calibration": 200, "test": 300}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    raw = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
    return hashlib.sha256(raw).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    raise TypeError(type(value).__name__)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_preregistration() -> dict:
    seal = json.loads((PREREG / "PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    for name, expected in seal["files"].items():
        if sha(PREREG / name) != expected:
            raise AssertionError(f"preregistration changed: {name}")
    payload = "\n".join(f"{name} {seal['files'][name]}" for name in sorted(seal["files"]))
    if hashlib.sha256(payload.encode()).hexdigest() != seal["composite_sha256"]:
        raise AssertionError("preregistration composite mismatch")
    seeds = json.loads((PREREG / "seed_manifest.json").read_text(encoding="utf-8"))
    expected = list(range(20320001, 20320051))
    actual = [int(row["generation_seed"]) for row in seeds["paired_seed_records"]]
    if actual != expected or seeds["seed_scan"]["collisions"]:
        raise AssertionError("preregistered seed gate failed")
    return {"seal": seal, "seeds": actual, "cmi999": [int(x) for x in seeds["cmi999_preregistered_first20"]]}


def scenario_dir(index: int, seed: int, scenario: str) -> Path:
    return RECORD / "frozen_inputs" / f"replicate_{index:02d}_seed_{seed}" / scenario


def split_role(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(frame["split"].eq("local_calibration"), frame["local_calibration_role"], frame["split"]), index=frame.index)


def generate_reproducibility_probe(generator, config: dict, seed: int) -> dict:
    hashes = []
    with tempfile.TemporaryDirectory(prefix="stage15_generator_probe_", dir=RECORD) as td:
        root = Path(td)
        for run in (1, 2):
            local = json.loads(json.dumps(config)); local["random_seeds"]["base_variables"] = seed
            scenarios = generator.generate_all_scenarios(local)
            out = root / str(run); out.mkdir()
            run_hashes = {}
            for scenario in SCENARIOS:
                observed, truth = scenarios[scenario]
                for kind, frame in (("observed", observed), ("truth", truth)):
                    path = out / f"{scenario}_{kind}.csv"; write_csv(frame, path); run_hashes[path.name] = sha(path)
            hashes.append(run_hashes)
    if hashes[0] != hashes[1]: raise AssertionError("generator reproducibility failure")
    return {"status": "PASS", "seed": seed, "two_run_hashes_identical": True, "hashes": hashes[0]}


def generate_scenario_artifacts(index: int, seed: int, scenario: str, observed: pd.DataFrame,
                                truth: pd.DataFrame, predictor, predictor_cfg: dict) -> tuple[dict, dict, dict]:
    out = scenario_dir(index, seed, scenario); out.mkdir(parents=True, exist_ok=True)
    completion = out / "COMPLETE.json"
    if completion.exists():
        saved = json.loads(completion.read_text(encoding="utf-8"))
        for name, expected in saved["hashes"].items():
            if sha(out / name) != expected: raise AssertionError(f"checkpoint hash mismatch: {out/name}")
        return saved["generator_audit"], saved["split_integrity"], saved["prediction_manifest"]

    observed = observed.copy(); truth = truth.copy()
    observed["split_role"] = split_role(observed)
    truth["split_role"] = split_role(truth)
    keys = META + ["split_role"]
    pd.testing.assert_frame_equal(observed[keys], truth[keys], check_dtype=False)
    input_frame = observed[META + ["split_role", *FEATURES_ALL]].copy()
    label_frame = observed[META + ["split_role", "y"]].merge(
        truth[META + ["split_role", "sigma_true"]], on=META + ["split_role"], validate="one_to_one")
    label_frame = label_frame.rename(columns={"y": "y_true", "sigma_true": "oracle_sigma"})
    development = label_frame[~label_frame.split.eq("test")].copy()
    test_sealed = label_frame[label_frame.split.eq("test")].copy()
    input_path = out / "predictor_inputs.csv"; dev_labels_path = out / "development_labels.csv"
    test_path = out / "test_sealed.csv"
    write_csv(input_frame, input_path); write_csv(development, dev_labels_path); write_csv(test_sealed, test_path)

    # Downstream predictor uses only values reloaded from established %.17g CSV serialization.
    x = pd.read_csv(input_path, keep_default_na=False); labels = pd.read_csv(dev_labels_path, keep_default_na=False)
    train = x[x.split.eq("train")].merge(labels[labels.split.eq("train")][META + ["y_true"]], on=META, validate="one_to_one", sort=False)
    if len(train) != 1000 or set(train.split) != {"train"}: raise AssertionError("train-only predictor gate")
    model = predictor.make_model(predictor_cfg)
    model.fit(train[FEATURES_ALL].to_numpy(float), train.y_true.to_numpy(float))
    predictions = x[META + ["split_role"]].copy()
    predictions["y_pred"] = predictor.predict_in_fixed_tree_order(model, x[FEATURES_ALL].to_numpy(float))
    prediction_path = out / "predictions.csv"; model_path = out / "model.joblib"
    write_csv(predictions, prediction_path); joblib.dump(model, model_path, compress=3, protocol=5)
    predictions = pd.read_csv(prediction_path, keep_default_na=False)
    dev = x.merge(predictions, on=META + ["split_role"], validate="one_to_one", sort=False).merge(
        labels, on=META + ["split_role"], validate="one_to_one", sort=False)
    if len(dev) != 1700 or dev.split.eq("test").any(): raise AssertionError("development label leakage")
    write_csv(dev, out / "development.csv")

    counts = x.split_role.value_counts().to_dict()
    if counts != EXPECTED_COUNTS: raise AssertionError(f"split mismatch: {counts}")
    if not x.sample_id.is_unique or not test_sealed.sample_id.is_unique: raise AssertionError("duplicate sample id")
    generator_audit = {"replicate_index": index, "generation_seed": seed, "scenario": scenario,
       "rows": len(x), "serialized_reload": True, "finite_predictors": bool(np.isfinite(x[FEATURES_ALL].to_numpy(float)).all()),
       "finite_y_development": bool(np.isfinite(dev.y_true.to_numpy(float)).all()),
       "finite_oracle_sigma": bool(np.isfinite(pd.concat([development.oracle_sigma,test_sealed.oracle_sigma]).to_numpy(float)).all()),
       "sigma_min": float(label_frame.oracle_sigma.min()), "sigma_max": float(label_frame.oracle_sigma.max()),
       "predictor_inputs_sha256": sha(input_path), "development_labels_sha256": sha(dev_labels_path),
       "test_sealed_sha256": sha(test_path)}
    split_audit = {"replicate_index": index, "generation_seed": seed, "scenario": scenario,
       **{f"n_{key}": int(counts[key]) for key in EXPECTED_COUNTS}, "unique_ids": x.sample_id.nunique(),
       "roles_disjoint": True, "paired_split_seed": 620907, "local_role_seed": 620908, "status": "PASS"}
    pred_manifest = {"replicate_index": index, "generation_seed": seed, "scenario": scenario,
       "model_sha256": sha(model_path), "predictions_sha256": sha(prediction_path), "prediction_rows": len(predictions),
       "predictor_input_sha256": sha(input_path), "training_rows": len(train), "predictor_random_seed": 420907,
       "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
       "prediction_path": str(prediction_path.relative_to(ROOT)).replace("\\", "/")}
    hashes = {p.name: sha(p) for p in [input_path, dev_labels_path, test_path, prediction_path, model_path, out/"development.csv"]}
    dump(completion, {"hashes": hashes, "generator_audit": generator_audit,
                      "split_integrity": split_audit, "prediction_manifest": pred_manifest})
    return generator_audit, split_audit, pred_manifest


def generate_replicate(index: int, seed: int, generator, gen_cfg: dict, predictor, pred_cfg: dict):
    local = json.loads(json.dumps(gen_cfg)); local["random_seeds"]["base_variables"] = seed
    scenarios = generator.generate_all_scenarios(local)
    # Exact paired covariates and split roles are required.
    common = ["sample_id", "split", "local_calibration_role", *FEATURES_ALL]
    pd.testing.assert_frame_equal(scenarios["S2"][0][common], scenarios["S4"][0][common], check_dtype=False)
    rows = [generate_scenario_artifacts(index, seed, scenario, *scenarios[scenario], predictor, pred_cfg)
            for scenario in SCENARIOS]
    return rows


def load_development(index: int, seed: int, scenario: str) -> pd.DataFrame:
    frame = pd.read_csv(scenario_dir(index, seed, scenario) / "development.csv", keep_default_na=False)
    if len(frame) != 1700 or frame.split.eq("test").any(): raise AssertionError("invalid development table")
    return frame


def role(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame[frame.split_role.eq(name)].copy()
    if len(out) != EXPECTED_COUNTS[name]: raise AssertionError(f"role count {name}")
    return out


def cmi_seed(index: int, scenario: str) -> int:
    return CMI_MASTER_SEED + index * 10 + (1 if scenario == "S2" else 3)


def make_cmi_input(index: int, seed: int, scenario: str) -> Path:
    out = scenario_dir(index, seed, scenario); path = out / "cmi_input.csv"
    if not path.exists():
        scale = role(load_development(index, seed, scenario), "scale_fit")
        frame = scale[["sample_id", *COORDS, *FEATURES]].copy()
        frame["A_model"] = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
        write_csv(frame, path)
    return path


def run_cmi(index: int, seed: int, scenario: str, surrogates: int, output: Path | None = None) -> tuple[pd.DataFrame, float]:
    source = make_cmi_input(index, seed, scenario)
    if output is None:
        output = scenario_dir(index, seed, scenario) / f"cmi{surrogates}_feature_results.csv"
    if output.exists() and output.with_suffix(".json").exists():
        return pd.read_csv(output), float(json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["runtime_seconds"])
    cmd = [str(CMI_PYTHON), str(CMI_WORKER), "--input", str(source), "--output", str(output),
           "--seed", str(cmi_seed(index, scenario)), "--surrogates", str(surrogates)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    output.with_suffix(".stdout.txt").write_text(proc.stdout, encoding="utf-8")
    output.with_suffix(".stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode: raise RuntimeError(f"CMI worker failed: {index}/{scenario}: {proc.stderr}")
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    return pd.read_csv(output), float(metadata["runtime_seconds"])


def weight_info(frame: pd.DataFrame) -> dict:
    mapping = {row.feature: float(row.weight) for row in frame.itertuples()}
    selected = [feature for feature in FEATURES if mapping[feature] > 0]
    status = str(frame.status.iloc[0])
    if status == "SIGNAL" and not np.isclose(sum(mapping.values()), 1, atol=1e-12): raise AssertionError("CMI weights")
    if status == "NO_CAUSAL_SIGNAL" and selected: raise AssertionError("CMI no-signal weights")
    return {"weights": mapping, "selected": selected, "status": status}


def prepare_gccm(index: int, seed: int, scenario: str, out: Path) -> None:
    if (out / "weights.csv").exists(): return
    out.mkdir(parents=True, exist_ok=True)
    scale = role(load_development(index, seed, scenario), "scale_fit").sort_values("sample_id").reset_index(drop=True)
    residual = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
    x = scale[FEATURES].to_numpy(float); z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
    xy = scale[COORDS].to_numpy(float); tri = Delaunay(xy)
    edges = sorted({tuple(sorted((int(t[i])+1, int(t[j])+1))) for t in tri.simplices for i,j in ((0,1),(1,2),(0,2))})
    inp = pd.DataFrame({"node_id": np.arange(1,201), "sample_id": scale.sample_id,
                        "s_x": xy[:,0], "s_y": xy[:,1], "abs_residual": residual})
    for j, feature in enumerate(FEATURES): inp["z_"+feature] = z[:,j]
    write_csv(inp, out/"restricted_input.csv")
    pd.DataFrame(edges,columns=["u","v"]).to_csv(out/"edges.csv",index=False,lineterminator="\n")
    shutil.copyfile(GCCM_CORE,out/"historical_core.r")
    pd.DataFrame([("E","3"),("tau","1"),("b","4"),("seed","42"),
                  ("direction","effect_xmap_cause"),("library_sizes","10;18;33;60;109;198")],
                 columns=["name","value"]).to_csv(out/"parameters.csv",index=False,lineterminator="\n")


def run_gccm(index: int, seed: int, scenario: str, out: Path | None = None) -> tuple[pd.DataFrame, float]:
    if out is None: out = RECORD/"gccm_runs"/f"replicate_{index:02d}_seed_{seed}"/scenario
    prepare_gccm(index, seed, scenario, out)
    meta_path = out/"execution_metadata.json"
    if (out/"weights.csv").exists() and meta_path.exists():
        return pd.read_csv(out/"weights.csv",keep_default_na=False), float(json.loads(meta_path.read_text(encoding="utf-8"))["runtime_seconds"])
    cmd=[str(RSCRIPT),"--vanilla",str(GCCM_WRAPPER),str(out/"restricted_input.csv"),str(out/"edges.csv"),
         str(out/"historical_core.r"),str(out/"parameters.csv"),str(out),"estimate"]
    start=time.perf_counter(); proc=subprocess.run(cmd,cwd=out,env={**os.environ,"LC_ALL":"C"},capture_output=True,
        text=True,encoding="utf-8",errors="replace",timeout=1800); elapsed=time.perf_counter()-start
    (out/"stdout.log").write_text(proc.stdout,encoding="utf-8"); (out/"stderr.log").write_text(proc.stderr,encoding="utf-8")
    if proc.returncode: raise RuntimeError(f"GCCM failed: {index}/{scenario}: {proc.stderr}")
    weights=pd.read_csv(out/"weights.csv",keep_default_na=False)
    if weights.feature.tolist()!=FEATURES: raise AssertionError("GCCM schema")
    dump(meta_path,{"command":cmd,"runtime_seconds":elapsed,"returncode":proc.returncode,
                    "hashes":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    return weights,elapsed


def gccm_info(frame: pd.DataFrame) -> dict:
    status=str(frame.status.iloc[0])
    if status=="NO_CONVERGED_FEATURES": return {"weights":{f:0.0 for f in FEATURES},"selected":[],"status":"NO_USABLE_GCCM_SIGNAL"}
    mapping={row.feature:float(row.weight) for row in frame.itertuples()}; selected=[f for f in FEATURES if mapping[f]>0]
    if not np.isclose(sum(mapping.values()),1,atol=1e-12): raise AssertionError("GCCM weights")
    return {"weights":mapping,"selected":selected,"status":"WEIGHTS_AVAILABLE"}


def standardizer(reference: pd.DataFrame, columns: list[str]):
    x=reference[columns].to_numpy(float); mean=x.mean(axis=0); sd=x.std(axis=0,ddof=1)
    if np.any(sd<=1e-12): raise AssertionError("zero localization SD")
    return mean,sd


def scale_values(frame: pd.DataFrame, columns: list[str], mean, sd): return (frame[columns].to_numpy(float)-mean)/sd


def neighbor_scale(rz,ra,ids,qz,weights,k):
    out=np.empty(len(qz))
    for i,row in enumerate(qz):
        dist=np.sqrt(np.sum(weights*(rz-row)**2,axis=1)); order=np.lexsort((ids,dist)); out[i]=max(FLOOR,float(ra[order[:k]].mean()))
    return out


def fit_local(scale: pd.DataFrame,tuning: pd.DataFrame,columns:list[str],weights:np.ndarray,method:str):
    mean,sd=standardizer(scale,columns); rz=scale_values(scale,columns,mean,sd); tz=scale_values(tuning,columns,mean,sd)
    ra=np.abs(scale.y_true.to_numpy(float)-scale.y_pred.to_numpy(float)); ta=np.abs(tuning.y_true.to_numpy(float)-tuning.y_pred.to_numpy(float))
    maes={}
    for k in K_VALUES: maes[k]=float(np.mean(np.abs(ta-neighbor_scale(rz,ra,scale.sample_id.to_numpy(str),tz,weights,k))))
    best=min(maes.values()); chosen=max(k for k,v in maes.items() if abs(v-best)<=TIE_TOL)
    rows=[{"method":method,"k":k,"tuning_scale_mae":maes[k],"selected":k==chosen} for k in K_VALUES]
    return chosen,{"type":"local","columns":columns,"mean":mean,"sd":sd,"rz":rz,"ra":ra,
                   "ids":scale.sample_id.to_numpy(str),"weights":weights,"tuning_mae":maes[chosen]},rows


def constant_model(scale:pd.DataFrame,tuning:pd.DataFrame,method:str):
    ra=np.abs(scale.y_true.to_numpy(float)-scale.y_pred.to_numpy(float)); ta=np.abs(tuning.y_true.to_numpy(float)-tuning.y_pred.to_numpy(float))
    constant=max(FLOOR,float(ra.mean())); mae=float(np.mean(np.abs(ta-constant)))
    return None,{"type":"constant","constant":constant,"tuning_mae":mae},[{"method":method,"k":None,"tuning_scale_mae":mae,"selected":True}]


def fit_geocmi(scale,tuning,cmi):
    if cmi["status"]=="NO_CAUSAL_SIGNAL":
        k,model,rows=fit_local(scale,tuning,COORDS,np.ones(2),"GeoCMI-Balanced-NormCP")
        model["geocmi_fallback"]=True; return k,model,rows
    selected=cmi["selected"]; cm,cs=standardizer(scale,selected); gm,gs=standardizer(scale,COORDS)
    rz=np.column_stack((scale_values(scale,selected,cm,cs),scale_values(scale,COORDS,gm,gs)))
    tz=np.column_stack((scale_values(tuning,selected,cm,cs),scale_values(tuning,COORDS,gm,gs)))
    weights=np.concatenate((.5*np.array([cmi["weights"][f] for f in selected]),np.array([.25,.25])))
    ra=np.abs(scale.y_true.to_numpy(float)-scale.y_pred.to_numpy(float)); ta=np.abs(tuning.y_true.to_numpy(float)-tuning.y_pred.to_numpy(float))
    maes={k:float(np.mean(np.abs(ta-neighbor_scale(rz,ra,scale.sample_id.to_numpy(str),tz,weights,k)))) for k in K_VALUES}
    best=min(maes.values()); chosen=max(k for k,v in maes.items() if abs(v-best)<=TIE_TOL)
    model={"type":"geocmi","selected":selected,"cm":cm,"cs":cs,"gm":gm,"gs":gs,"rz":rz,"ra":ra,
           "ids":scale.sample_id.to_numpy(str),"weights":weights,"tuning_mae":maes[chosen],"geocmi_fallback":False}
    rows=[{"method":"GeoCMI-Balanced-NormCP","k":k,"tuning_scale_mae":maes[k],"selected":k==chosen} for k in K_VALUES]
    return chosen,model,rows


def query_model(frame:pd.DataFrame,model:dict,k:int|None):
    if model["type"]=="constant": return np.full(len(frame),model["constant"])
    if model["type"]=="local": qz=scale_values(frame,model["columns"],model["mean"],model["sd"])
    else: qz=np.column_stack((scale_values(frame,model["selected"],model["cm"],model["cs"]),scale_values(frame,COORDS,model["gm"],model["gs"])))
    return neighbor_scale(model["rz"],model["ra"],model["ids"],qz,model["weights"],int(k))


def quantile(scores,alpha):
    scores=np.asarray(scores,float); rank=math.ceil((len(scores)+1)*(1-alpha))
    if len(scores)!=200 or rank>len(scores): raise AssertionError("conformal rank")
    return float(np.sort(scores,kind="mergesort")[rank-1]),rank


def freeze_analysis(index:int,seed:int,scenario:str,cmi:dict,gccm:dict):
    dev=load_development(index,seed,scenario); scale=role(dev,"scale_fit"); tuning=role(dev,"tuning_calibration"); final=role(dev,"final_calibration")
    krows=[]; models={}; selected_k={}
    selected_k["GeoOnly-NormCP"],models["GeoOnly-NormCP"],rows=fit_local(scale,tuning,COORDS,np.ones(2),"GeoOnly-NormCP"); krows+=rows
    if cmi["status"]=="NO_CAUSAL_SIGNAL": selected_k["CausalOnly-CMI-NormCP"],models["CausalOnly-CMI-NormCP"],rows=constant_model(scale,tuning,"CausalOnly-CMI-NormCP")
    else: selected_k["CausalOnly-CMI-NormCP"],models["CausalOnly-CMI-NormCP"],rows=fit_local(scale,tuning,cmi["selected"],np.array([cmi["weights"][f] for f in cmi["selected"]]),"CausalOnly-CMI-NormCP")
    krows+=rows
    if gccm["status"]=="NO_USABLE_GCCM_SIGNAL": selected_k["CausalOnly-GCCM-NormCP"],models["CausalOnly-GCCM-NormCP"],rows=constant_model(scale,tuning,"CausalOnly-GCCM-NormCP")
    else: selected_k["CausalOnly-GCCM-NormCP"],models["CausalOnly-GCCM-NormCP"],rows=fit_local(scale,tuning,gccm["selected"],np.array([gccm["weights"][f] for f in gccm["selected"]]),"CausalOnly-GCCM-NormCP")
    krows+=rows
    selected_k["GeoCMI-Balanced-NormCP"],models["GeoCMI-Balanced-NormCP"],rows=fit_geocmi(scale,tuning,cmi); krows+=rows
    final_a=np.abs(final.y_true.to_numpy(float)-final.y_pred.to_numpy(float)); sigma_final={"Raw-CP":np.ones(200),"Oracle-NormCP":final.oracle_sigma.to_numpy(float)}
    for method in ["GeoOnly-NormCP","CausalOnly-GCCM-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP"]: sigma_final[method]=query_model(final,models[method],selected_k[method])
    q={}; ranks={}
    for alpha in ALPHAS:
        q[str(alpha)]={}; ranks[str(alpha)]={}
        for method in METHODS:
            q[str(alpha)][method],ranks[str(alpha)][method]=quantile(final_a/sigma_final[method],alpha)
    # Test covariates and predictions contain no test labels.
    base=scenario_dir(index,seed,scenario); x=pd.read_csv(base/"predictor_inputs.csv",keep_default_na=False); pred=pd.read_csv(base/"predictions.csv",keep_default_na=False)
    test=x[x.split.eq("test")].merge(pred,on=META+["split_role"],validate="one_to_one",sort=False)
    sigma_test={"Raw-CP":np.ones(300)}
    for method in ["GeoOnly-NormCP","CausalOnly-GCCM-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP"]: sigma_test[method]=query_model(test,models[method],selected_k[method])
    digest=hashlib.sha256()
    digest.update(json.dumps({"k":selected_k,"q":q},sort_keys=True,default=json_default).encode())
    for method in sorted(sigma_test): digest.update(np.asarray(sigma_test[method],float).tobytes())
    return {"dev":dev,"scale":scale,"tuning":tuning,"final":final,"test_cov":test,"models":models,"selected_k":selected_k,
            "krows":krows,"q":q,"ranks":ranks,"sigma_test":sigma_test,"hash":digest.hexdigest()}


UNSEAL_FIELDS=["phase","replicate_index","generation_seed","scenario","cmi_status","cmi_features","cmi_weights_json",
 "gccm_status","gccm_features","gccm_weights_json","selected_k_json","q_json","scale_fit_hash","tuning_hash",
 "final_calibration_hash","prediction_hash","test_covariate_hash","barrier_row_sha256"]


def append_unseal(row:dict):
    base={k:row[k] for k in UNSEAL_FIELDS if k!="barrier_row_sha256"}; row["barrier_row_sha256"]=hashlib.sha256(json.dumps(base,sort_keys=True,separators=(",",":"),default=json_default).encode()).hexdigest()
    path=RECORD/"test_unseal_audit.csv"; exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=UNSEAL_FIELDS,lineterminator="\n");
        if not exists: writer.writeheader()
        writer.writerow(row); handle.flush(); os.fsync(handle.fileno())


def interval_metrics(y,pred,q,sigma,alpha):
    lower=pred-q*sigma; upper=pred+q*sigma
    if not(np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower<=upper)): raise AssertionError("interval")
    covered=(y>=lower)&(y<=upper); width=upper-lower
    score=width+(2/alpha)*(lower-y)*(y<lower)+(2/alpha)*(y-upper)*(y>upper)
    return {"coverage":float(covered.mean()),"covered_count":int(covered.sum()),"test_n":len(y),"mean_width":float(width.mean()),
            "median_width":float(np.median(width)),"mean_interval_score":float(score.mean()),"lower":lower,"upper":upper}


def evaluate_one(index,seed,scenario,cmi,gccm,frozen,phase="primary"):
    barrier={"phase":phase,"replicate_index":index,"generation_seed":seed,"scenario":scenario,"cmi_status":cmi["status"],
      "cmi_features":";".join(cmi["selected"]),"cmi_weights_json":json.dumps(cmi["weights"],sort_keys=True),
      "gccm_status":gccm["status"],"gccm_features":";".join(gccm["selected"]),"gccm_weights_json":json.dumps(gccm["weights"],sort_keys=True),
      "selected_k_json":json.dumps(frozen["selected_k"],sort_keys=True),"q_json":json.dumps(frozen["q"],sort_keys=True),
      "scale_fit_hash":frame_hash(frozen["scale"]),"tuning_hash":frame_hash(frozen["tuning"]),"final_calibration_hash":frame_hash(frozen["final"]),
      "prediction_hash":sha(scenario_dir(index,seed,scenario)/"predictions.csv"),"test_covariate_hash":frame_hash(frozen["test_cov"])}
    append_unseal(barrier)
    sealed=pd.read_csv(scenario_dir(index,seed,scenario)/"test_sealed.csv",keep_default_na=False)
    test=frozen["test_cov"].merge(sealed,on=META+["split_role"],validate="one_to_one",sort=False)
    if len(test)!=300: raise AssertionError("test label join")
    y=test.y_true.to_numpy(float); pred=test.y_pred.to_numpy(float); sigma=dict(frozen["sigma_test"]); sigma["Oracle-NormCP"]=test.oracle_sigma.to_numpy(float)
    result_rows=[]; scale_rows=[]; metrics_cache={}
    for alpha in ALPHAS:
      for method in METHODS:
        m=interval_metrics(y,pred,frozen["q"][str(alpha)][method],sigma[method],alpha); metrics_cache[(alpha,method)]=m
        info=cmi if method in ("CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP") else gccm if method=="CausalOnly-GCCM-NormCP" else {"selected":[],"weights":{},"status":"NA"}
        result_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,"alpha":alpha,"method":method,
          **{k:m[k] for k in ("coverage","covered_count","test_n","mean_width","median_width","mean_interval_score")},
          "selected_k":frozen["selected_k"].get(method),"q_norm":frozen["q"][str(alpha)][method],
          "tuning_scale_mae":frozen["models"].get(method,{}).get("tuning_mae"),"signal_status":info["status"],
          "selected_features":";".join(info["selected"]),"feature_weights":json.dumps(info["weights"],sort_keys=True)})
    # Frozen fallback equalities across all alpha values.
    for alpha in ALPHAS:
      raw=metrics_cache[(alpha,"Raw-CP")]; geo=metrics_cache[(alpha,"GeoOnly-NormCP")]
      if cmi["status"]=="NO_CAUSAL_SIGNAL":
        cm=metrics_cache[(alpha,"CausalOnly-CMI-NormCP")]; diff=max(np.max(np.abs(raw["lower"]-cm["lower"])),np.max(np.abs(raw["upper"]-cm["upper"])))
        if diff>1e-12: raise AssertionError("CMI fallback")
        gm=metrics_cache[(alpha,"GeoCMI-Balanced-NormCP")]; diff=max(np.max(np.abs(geo["lower"]-gm["lower"])),np.max(np.abs(geo["upper"]-gm["upper"])))
        if diff>1e-12: raise AssertionError("GeoCMI fallback")
      if gccm["status"]=="NO_USABLE_GCCM_SIGNAL":
        gm=metrics_cache[(alpha,"CausalOnly-GCCM-NormCP")]; diff=max(np.max(np.abs(raw["lower"]-gm["lower"])),np.max(np.abs(raw["upper"]-gm["upper"])))
        if diff>1e-12: raise AssertionError("GCCM fallback")
    if scenario=="S2":
      oracle=test.oracle_sigma.to_numpy(float)
      for method in ["GeoOnly-NormCP","CausalOnly-GCCM-NormCP","CausalOnly-CMI-NormCP","GeoCMI-Balanced-NormCP","Oracle-NormCP"]:
        values=sigma[method]; corr=None if np.ptp(values)<=1e-15 else float(np.corrcoef(values,oracle)[0,1])
        scale_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,"method":method,
          "test_oracle_correlation":corr,"test_oracle_mae":float(np.mean(np.abs(values-oracle))),
          "tuning_scale_mae":frozen["models"].get(method,{}).get("tuning_mae"),"cmi_signal":cmi["status"]=="SIGNAL"})
    krows=[]
    for row in frozen["krows"]: krows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,**row})
    return result_rows,scale_rows,krows


def predictor_determinism_probe(index,seed,scenario,predictor,pred_cfg):
    base=scenario_dir(index,seed,scenario); x=pd.read_csv(base/"predictor_inputs.csv",keep_default_na=False); dev=pd.read_csv(base/"development_labels.csv",keep_default_na=False)
    train=x[x.split.eq("train")].merge(dev[dev.split.eq("train")][META+["y_true"]],on=META,validate="one_to_one",sort=False)
    models=[]; preds=[]
    for _ in range(2):
        model=predictor.make_model(pred_cfg); model.fit(train[FEATURES_ALL].to_numpy(float),train.y_true.to_numpy(float)); models.append(model)
        preds.append(predictor.predict_in_fixed_tree_order(model,x[FEATURES_ALL].to_numpy(float)))
    if not np.array_equal(preds[0],preds[1]): raise AssertionError("predictor nondeterminism")
    structures=all(np.array_equal(a.tree_.feature,b.tree_.feature) and np.array_equal(a.tree_.threshold,b.tree_.threshold)
                   for a,b in zip(models[0].estimators_,models[1].estimators_))
    if not structures: raise AssertionError("forest structure nondeterminism")
    frozen=pd.read_csv(base/"predictions.csv").y_pred.to_numpy(float)
    return {"status":"PASS","max_abs_repeat_difference":float(np.max(np.abs(preds[0]-preds[1]))),
            "all_300_tree_structures_identical":structures,"max_abs_frozen_csv_difference":float(np.max(np.abs(preds[0]-frozen)))}


def main():
    started=time.time(); prereg=verify_preregistration(); seeds=prereg["seeds"]
    RECORD.mkdir(parents=True,exist_ok=True)
    if (RECORD/"confirmatory_summary.json").exists(): raise FileExistsError("completed Stage-15 record exists")
    generator=load_module("stage15_generator",GENERATOR_SOURCE); predictor=load_module("stage15_predictor",PREDICTOR_SOURCE)
    gen_cfg=json.loads(GENERATOR_CONFIG.read_text(encoding="utf-8")); pred_cfg=json.loads(PREDICTOR_CONFIG.read_text(encoding="utf-8"))
    if pred_cfg["model_configuration"]["random_state"]!=420907 or pred_cfg["authorized_features"]!=FEATURES_ALL: raise AssertionError("predictor protocol")
    execution_config={"preregistration_composite_sha256":prereg["seal"]["composite_sha256"],"seeds":seeds,"scenarios":SCENARIOS,
      "methods":METHODS,"alphas":ALPHAS,"generator_source_sha256":sha(GENERATOR_SOURCE),"generator_config_sha256":sha(GENERATOR_CONFIG),
      "predictor_source_sha256":sha(PREDICTOR_SOURCE),"predictor_config_sha256":sha(PREDICTOR_CONFIG),"cmi_worker_sha256":sha(CMI_WORKER),
      "gccm_wrapper_sha256":sha(GCCM_WRAPPER),"gccm_core_sha256":sha(GCCM_CORE),"implementation_version":"v1",
      "implementation_scope":"execution of immutable Stage-15A protocol; no methodological changes"}
    dump(RECORD/"config.json",execution_config)

    # First-seed generation and predictor preflight precede the remaining 49 seeds.
    generator_probe=generate_reproducibility_probe(generator,gen_cfg,seeds[0])
    first_rows=generate_replicate(1,seeds[0],generator,gen_cfg,predictor,pred_cfg)
    predictor_probe={s:predictor_determinism_probe(1,seeds[0],s,predictor,pred_cfg) for s in SCENARIOS}
    cmi_infos={}; gccm_infos={}; cmi_frames={}; gccm_frames={}; cmi_runtimes=[]; gccm_runtimes=[]
    cmi_preflight={}; gccm_preflight={}
    for scenario in SCENARIOS:
        first,runtime=run_cmi(1,seeds[0],scenario,99); cmi_runtimes.append(runtime); cmi_frames[(1,scenario)]=first; cmi_infos[(1,scenario)]=weight_info(first)
        with tempfile.TemporaryDirectory(prefix="stage15_cmi_preflight_",dir=RECORD) as td:
            second,_=run_cmi(1,seeds[0],scenario,99,Path(td)/"second.csv")
            cols=["feature","raw_cmi","p_value","bh_q","eligible","weight","status"]
            pd.testing.assert_frame_equal(first[cols],second[cols],check_exact=True)
            cmi_preflight[scenario]="identical"
        out=RECORD/"gccm_runs"/f"replicate_01_seed_{seeds[0]}"/scenario
        gw,gr=run_gccm(1,seeds[0],scenario,out); gccm_runtimes.append(gr); gccm_frames[(1,scenario)]=gw; gccm_infos[(1,scenario)]=gccm_info(gw)
        with tempfile.TemporaryDirectory(prefix="stage15_gccm_preflight_",dir=RECORD) as td:
            second_dir=Path(td)/scenario; gw2,_=run_gccm(1,seeds[0],scenario,second_dir)
            for name in ["restricted_input.csv","edges.csv","parameters.csv","embedding.csv","raw_curves.csv","mean_curves.csv","convergence.csv","weights.csv"]:
                if sha(out/name)!=sha(second_dir/name): raise AssertionError("GCCM nondeterminism")
            gccm_preflight[scenario]="identical"
    frozen_preflight={}
    for scenario in SCENARIOS:
        a=freeze_analysis(1,seeds[0],scenario,cmi_infos[(1,scenario)],gccm_infos[(1,scenario)])
        b=freeze_analysis(1,seeds[0],scenario,cmi_infos[(1,scenario)],gccm_infos[(1,scenario)])
        if a["hash"]!=b["hash"]: raise AssertionError("CP freeze nondeterminism")
        # Algebraic fallback implementation checks independent of observed selection status.
        scale,tuning,final=a["scale"],a["tuning"],a["final"]; _,constant,_=constant_model(scale,tuning,"probe")
        fa=np.abs(final.y_true-final.y_pred).to_numpy(float); qs,_=quantile(fa/constant["constant"],.1); qr,_=quantile(fa,.1)
        if abs(qs*constant["constant"]-qr)>1e-12: raise AssertionError("constant fallback algebra")
        frozen_preflight[scenario]=a["hash"]
    preflight={"status":"PASS","seed":seeds[0],"generator":generator_probe,"predictor":predictor_probe,
      "cmi99_clean_rerun":cmi_preflight,"gccm_clean_rerun":gccm_preflight,"frozen_analysis_hashes":frozen_preflight,
      "coordinate_leakage":"PASS: causal final distances restricted to x1,x2,g_proxy,x_noise",
      "fallback_implementation":"PASS","test_labels_loaded":False}
    dump(RECORD/"preflight_summary.json",preflight)
    print("PREFLIGHT PASS",flush=True)

    generator_rows=[x[0] for group in first_rows for x in [group]]
    split_rows=[x[1] for group in first_rows for x in [group]]
    prediction_rows=[x[2] for group in first_rows for x in [group]]
    # Generate remaining fresh pairs only after full first-pair preflight.
    for index,seed in enumerate(seeds[1:],2):
        rows=generate_replicate(index,seed,generator,gen_cfg,predictor,pred_cfg)
        generator_rows.extend(x[0] for x in rows); split_rows.extend(x[1] for x in rows); prediction_rows.extend(x[2] for x in rows)
        if index%5==0: print(f"GENERATION {index}/50 pairs",flush=True)

    # Frozen CMI99 and GCCM weights for remaining scenario-replicates.
    for index,seed in enumerate(seeds,1):
      for scenario in SCENARIOS:
        if (index,scenario) not in cmi_frames:
          cf,rt=run_cmi(index,seed,scenario,99); cmi_frames[(index,scenario)]=cf; cmi_infos[(index,scenario)]=weight_info(cf); cmi_runtimes.append(rt)
        if (index,scenario) not in gccm_frames:
          gf,rt=run_gccm(index,seed,scenario); gccm_frames[(index,scenario)]=gf; gccm_infos[(index,scenario)]=gccm_info(gf); gccm_runtimes.append(rt)
        print(f"ESTIMATION {index:02d}/50 {scenario} CMI={cmi_infos[(index,scenario)]['status']} GCCM={gccm_infos[(index,scenario)]['status']}",flush=True)

    cmi_rows=[]; gccm_rows=[]
    for index,seed in enumerate(seeds,1):
      for scenario in SCENARIOS:
        for row in cmi_frames[(index,scenario)].to_dict("records"): cmi_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,**row})
        for row in gccm_frames[(index,scenario)].to_dict("records"): gccm_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,**row})
    write_csv(pd.DataFrame(generator_rows),RECORD/"generator_audit.csv"); write_csv(pd.DataFrame(split_rows),RECORD/"split_integrity.csv")
    write_csv(pd.DataFrame(prediction_rows),RECORD/"prediction_manifest.csv"); write_csv(pd.DataFrame(cmi_rows),RECORD/"cmi99_feature_results.csv")
    weight_rows=[]
    for (index,scenario),info in cmi_infos.items():
      seed=seeds[index-1]
      for feature in FEATURES: weight_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,"feature":feature,
        "weight":info["weights"][feature],"selected":feature in info["selected"],"status":info["status"]})
    write_csv(pd.DataFrame(weight_rows),RECORD/"cmi99_weight_audit.csv"); write_csv(pd.DataFrame(gccm_rows),RECORD/"gccm_weight_audit.csv")

    all_results=[]; scale_diagnostics=[]; k_tuning=[]
    for index,seed in enumerate(seeds,1):
      for scenario in SCENARIOS:
        frozen=freeze_analysis(index,seed,scenario,cmi_infos[(index,scenario)],gccm_infos[(index,scenario)])
        rows,scales,krows=evaluate_one(index,seed,scenario,cmi_infos[(index,scenario)],gccm_infos[(index,scenario)],frozen)
        all_results.extend(rows); scale_diagnostics.extend(scales); k_tuning.extend(krows)
      if index%5==0: print(f"CP EVALUATION {index}/50 pairs",flush=True)
    results=pd.DataFrame(all_results); scales=pd.DataFrame(scale_diagnostics); ktune=pd.DataFrame(k_tuning)
    for alpha,tag in ((.10,"010"),(.05,"005"),(.20,"020")):
        write_csv(results[np.isclose(results.alpha,alpha)].drop(columns="alpha"),RECORD/f"replicate_results_alpha{tag}.csv")
    write_csv(scales,RECORD/"scale_diagnostics.csv"); write_csv(ktune,RECORD/"k_tuning_results.csv")

    # CMI-999 runtime benchmark uses only the first two preregistered pairs and does not inspect rankings.
    cmi999_frames={}; benchmark_times=[]
    for index,seed in enumerate(prereg["cmi999"][:2],1):
      for scenario in SCENARIOS:
        cf,rt=run_cmi(index,seed,scenario,999); cmi999_frames[(index,scenario)]=cf; benchmark_times.append(rt)
    projected=float(sum(benchmark_times)/2*20); chosen_pairs=20 if projected<=10800 else 10
    runtime_decision={"benchmark_pair_seeds":prereg["cmi999"][:2],"scenario_job_seconds":benchmark_times,
      "observed_seconds_for_two_pairs":sum(benchmark_times),"projected_seconds_for_20_pairs":projected,
      "threshold_seconds":10800,"selected_pair_count":chosen_pairs,"decision_basis":"runtime only; rankings not inspected"}
    dump(RECORD/"cmi999_runtime_benchmark.json",runtime_decision)
    for index,seed in enumerate(prereg["cmi999"][:chosen_pairs],1):
      for scenario in SCENARIOS:
        if (index,scenario) not in cmi999_frames: cmi999_frames[(index,scenario)],_=run_cmi(index,seed,scenario,999)
      print(f"CMI999 {index}/{chosen_pairs} pairs",flush=True)
    cmi999_rows=[]; cmi_compare=[]; cmi999_cp=[]
    primary010=results[np.isclose(results.alpha,.10)]
    for index,seed in enumerate(prereg["cmi999"][:chosen_pairs],1):
      for scenario in SCENARIOS:
        f999=cmi999_frames[(index,scenario)]; info999=weight_info(f999); f99=cmi_frames[(index,scenario)].set_index("feature")
        for row in f999.to_dict("records"):
          cmi999_rows.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,**row})
          old=f99.loc[row["feature"]]; cmi_compare.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,"feature":row["feature"],
            "raw_cmi_99":float(old.raw_cmi),"raw_cmi_999":row["raw_cmi"],"raw_abs_difference":abs(float(old.raw_cmi)-row["raw_cmi"]),
            "p99":float(old.p_value),"q99":float(old.bh_q),"selected99":bool(old.eligible),
            "p999":row["p_value"],"q999":row["bh_q"],"selected999":bool(row["eligible"])})
        # Freeze and evaluate only the CMI999 method; primary comparators are reused read-only.
        dev=load_development(index,seed,scenario); scale=role(dev,"scale_fit"); tuning=role(dev,"tuning_calibration"); final=role(dev,"final_calibration")
        if info999["status"]=="NO_CAUSAL_SIGNAL": ck,cm,_=constant_model(scale,tuning,"CausalOnly-CMI999-NormCP")
        else: ck,cm,_=fit_local(scale,tuning,info999["selected"],np.array([info999["weights"][f] for f in info999["selected"]]),"CausalOnly-CMI999-NormCP")
        fa=np.abs(final.y_true.to_numpy(float)-final.y_pred.to_numpy(float)); fs=query_model(final,cm,ck); q,_=quantile(fa/fs,.1)
        base=scenario_dir(index,seed,scenario); x=pd.read_csv(base/"predictor_inputs.csv"); predframe=pd.read_csv(base/"predictions.csv")
        tc=x[x.split.eq("test")].merge(predframe,on=META+["split_role"],validate="one_to_one",sort=False); ts=query_model(tc,cm,ck)
        # Separate secondary barrier before reloading sealed labels.
        append_unseal({"phase":"cmi999","replicate_index":index,"generation_seed":seed,"scenario":scenario,"cmi_status":info999["status"],
          "cmi_features":";".join(info999["selected"]),"cmi_weights_json":json.dumps(info999["weights"],sort_keys=True),
          "gccm_status":"NOT_APPLICABLE","gccm_features":"","gccm_weights_json":"{}","selected_k_json":json.dumps({"CausalOnly-CMI999-NormCP":ck}),
          "q_json":json.dumps({"0.1":{"CausalOnly-CMI999-NormCP":q}}),"scale_fit_hash":frame_hash(scale),"tuning_hash":frame_hash(tuning),
          "final_calibration_hash":frame_hash(final),"prediction_hash":sha(base/"predictions.csv"),"test_covariate_hash":frame_hash(tc)})
        sealed=pd.read_csv(base/"test_sealed.csv"); test=tc.merge(sealed,on=META+["split_role"],validate="one_to_one",sort=False)
        m=interval_metrics(test.y_true.to_numpy(float),test.y_pred.to_numpy(float),q,ts,.1)
        cmi999_cp.append({"replicate_index":index,"generation_seed":seed,"scenario":scenario,"method":"CausalOnly-CMI999-NormCP",
          **{key:m[key] for key in ("coverage","covered_count","test_n","mean_width","median_width","mean_interval_score")},
          "selected_k":ck,"q_norm":q,"signal_status":info999["status"],"selected_features":";".join(info999["selected"]),
          "feature_weights":json.dumps(info999["weights"],sort_keys=True),"tuning_scale_mae":cm["tuning_mae"]})
        for method in ["Raw-CP","GeoOnly-NormCP","CausalOnly-CMI-NormCP","Oracle-NormCP"]:
          row=primary010[(primary010.replicate_index==index)&(primary010.scenario==scenario)&(primary010.method==method)].iloc[0].to_dict()
          row.pop("alpha",None); cmi999_cp.append(row)
    write_csv(pd.DataFrame(cmi999_rows),RECORD/"cmi999_feature_results.csv"); write_csv(pd.DataFrame(cmi_compare),RECORD/"cmi999_comparison.csv")
    write_csv(pd.DataFrame(cmi999_cp),RECORD/"cmi999_cp_sensitivity.csv")
    dump(RECORD/"execution_checkpoint.json",{"status":"SCIENTIFIC_COMPUTATION_COMPLETE","elapsed_seconds":time.time()-started,
      "cmi99_total_runtime_seconds":sum(cmi_runtimes),"gccm_total_runtime_seconds":sum(gccm_runtimes),"cmi999_pair_count":chosen_pairs,
      "primary_scenario_replicates":100,"result_rows":len(results)})
    print("SCIENTIFIC COMPUTATION COMPLETE",flush=True)


if __name__=="__main__": main()
