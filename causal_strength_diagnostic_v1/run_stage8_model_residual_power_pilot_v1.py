"""Stage 8 paired A_model CMI pilot using the original generator and RF procedure in memory."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from tigramite.independence_tests.cmiknn import CMIknn

from src.cmi_adapter_v1 import direct_score, standardize_selected
from run_stage7_oracle_power_pilot_v1 import CONFIG_PATH, MASTER_SEED, META, TOL, generator_module, scale_fit_oracle

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PREDICTOR_CONFIG_PATH = ROOT / "simulation_pilot_20260908" / "predictor" / "predictor_config_v1.json"
STAGE7_CSV = OUT / "stage7_oracle_power_replicates.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predictor_module():
    path = ROOT / "simulation_pilot_20260908" / "predictor" / "fit_predictors.py"
    spec = importlib.util.spec_from_file_location("stage8_predictor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import predictor module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmi_test(seed: int, sig_samples: int) -> CMIknn:
    return CMIknn(knn=.10, shuffle_neighbors=5, significance="shuffle_test", transform=False, workers=1, sig_samples=sig_samples, sig_blocklength=1, seed=seed)


def fit_and_predict_scale(observed: pd.DataFrame, predictor_config: dict, predictor) -> tuple[pd.DataFrame, dict]:
    features = list(predictor_config["authorized_features"])
    if features != ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"]:
        raise AssertionError("unexpected original predictor feature list")
    train_mask = observed["split"] == "train"
    scale_mask = (observed["split"] == "local_calibration") & (observed["local_calibration_role"] == "scale_fit")
    if int(train_mask.sum()) != 1000 or int(scale_mask.sum()) != 200:
        raise AssertionError("predictor train/scale-fit row count mismatch")
    model = predictor.make_model(predictor_config)
    if model.get_params(deep=False) != predictor_config["model_configuration"]:
        raise AssertionError("predictor parameter mismatch")
    x_train = observed.loc[train_mask, features].to_numpy(float)
    y_train = observed.loc[train_mask, "y"].to_numpy(float)
    model.fit(x_train, y_train)
    scale = observed.loc[scale_mask, META + features + ["y"]].copy().sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    scale["y_pred"] = predictor.predict_in_fixed_tree_order(model, scale[features].to_numpy(float))
    if not np.isfinite(scale["y_pred"].to_numpy(float)).all():
        raise AssertionError("nonfinite scale-fit prediction")
    return scale, {"train_rows": int(train_mask.sum()), "scale_fit_rows": int(scale_mask.sum()), "random_state": predictor_config["model_configuration"]["random_state"], "n_estimators": predictor_config["model_configuration"]["n_estimators"], "feature_list": features}


def frozen_reproduction(config: dict, generator, predictor_config: dict, predictor) -> dict:
    scenarios = generator.generate_all_scenarios(config)
    summary = {}
    for scenario in ("S2", "S4"):
        observed, _ = scenarios[scenario]
        features = list(predictor_config["authorized_features"])
        model = predictor.make_model(predictor_config)
        mask = observed["split"] == "train"
        model.fit(observed.loc[mask, features].to_numpy(float), observed.loc[mask, "y"].to_numpy(float))
        reproduced = predictor.predict_in_fixed_tree_order(model, observed[features].to_numpy(float))
        frozen = pd.read_csv(ROOT / "simulation_pilot_20260908" / "predictor_outputs" / f"{scenario}_predictions.csv", keep_default_na=False)
        pd.testing.assert_frame_equal(observed[META], frozen[META], check_dtype=False)
        diff = float(np.max(np.abs(reproduced - frozen["y_pred"].to_numpy(float))))
        if diff > 1e-12:
            raise AssertionError(f"{scenario}: frozen predictor reproduction mismatch max={diff}")
        summary[scenario] = {"max_abs_prediction_difference": diff, "tolerance": 1e-12}
    return summary


def oracle_expected() -> dict[tuple[int, str, str], dict]:
    with STAGE7_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 80:
        raise AssertionError("Stage 7 must contain exactly 80 tests")
    result = {(int(row["replicate_index"]), row["scenario"], row["conditioning"]): {"raw_score": float(row["raw_score"]), "p_value": float(row["p_value"]), "reject": row["reject_0_05"] == "True", "generation_seed": int(row["generation_seed"])} for row in rows}
    if len(result) != 80:
        raise AssertionError("duplicate Stage 7 test identity")
    return result


def run_replicate(index: int, generation_seed: int, config: dict, generator, predictor_config: dict, predictor, expected: dict | None, sig_samples: int) -> list[dict]:
    local_config = copy.deepcopy(config); local_config["random_seeds"]["base_variables"] = generation_seed
    scenarios = generator.generate_all_scenarios(local_config)
    oracle, errors, model_scales, predictor_specs = {}, {}, {}, {}
    for scenario in ("S2", "S4"):
        oracle[scenario], errors[scenario] = scale_fit_oracle(*scenarios[scenario], scenario)
        model_scales[scenario], predictor_specs[scenario] = fit_and_predict_scale(scenarios[scenario][0], predictor_config, predictor)
        pd.testing.assert_frame_equal(oracle[scenario][META], model_scales[scenario][META], check_dtype=False)
    pd.testing.assert_frame_equal(oracle["S2"][["sample_id", "split", "local_calibration_role"]], oracle["S4"][["sample_id", "split", "local_calibration_role"]], check_dtype=False)
    rows = []
    offset = 0
    for scenario in ("S2", "S4"):
        frame = oracle[scenario].copy()
        frame["A_model"] = np.abs(model_scales[scenario]["y"].to_numpy(float) - model_scales[scenario]["y_pred"].to_numpy(float))
        if not np.isfinite(frame["A_model"].to_numpy(float)).all():
            raise AssertionError("nonfinite A_model")
        diagnostics = {"scale_fit_mae_y": float(np.mean(frame["A_model"])), "mean_abs_y_pred_minus_mu_true": float(np.mean(np.abs(model_scales[scenario]["y_pred"].to_numpy(float) - frame["mu_true"].to_numpy(float)))), "median_abs_y_pred_minus_mu_true": float(np.median(np.abs(model_scales[scenario]["y_pred"].to_numpy(float) - frame["mu_true"].to_numpy(float)))), "mean_A_model": float(frame["A_model"].mean()), "median_A_model": float(frame["A_model"].median()), "mean_A_oracle": float(frame["A_oracle"].mean()), "median_A_oracle": float(frame["A_oracle"].median()), "mean_abs_A_model_minus_A_oracle": float(np.mean(np.abs(frame["A_model"].to_numpy(float) - frame["A_oracle"].to_numpy(float))))}
        oracle_values, _ = standardize_selected({name: frame[name].to_numpy(float) for name in ("x2", "s_x", "s_y", "A_oracle")})
        model_values, _ = standardize_selected({name: frame[name].to_numpy(float) for name in ("x2", "s_x", "s_y", "A_model")})
        cases = (("M0", ["x2", "A_oracle"], ["x2", "A_model"], [0, 1]), ("Mspace", ["x2", "A_oracle", "s_x", "s_y"], ["x2", "A_model", "s_x", "s_y"], [0, 1, 2, 2]))
        for condition, oracle_names, model_names, labels in cases:
            xyz = np.asarray(labels, dtype=int)
            oracle_array = np.vstack([oracle_values[name] for name in oracle_names])
            model_array = np.vstack([model_values[name] for name in model_names])
            oracle_raw, _ = direct_score(oracle_array, xyz, MASTER_SEED)
            stage7 = expected[(index, scenario, condition)] if expected is not None else {"raw_score": None, "p_value": None, "reject": False}
            if expected is not None and abs(oracle_raw - stage7["raw_score"]) > 1e-12:
                raise AssertionError(f"Stage 7 oracle raw mismatch: {index}/{scenario}/{condition}")
            shuffle_seed = MASTER_SEED + index * 10 + offset
            offset += 1
            test = cmi_test(shuffle_seed, sig_samples)
            raw = float(test.get_dependence_measure(model_array.copy(), xyz.copy()))
            p_value = float(test.get_shuffle_significance(model_array.copy(), xyz.copy(), raw))
            reject = p_value <= .05
            if expected is None: pattern = "smoke_not_compared"
            elif reject and stage7["reject"]: pattern = "both"
            elif stage7["reject"]: pattern = "oracle_only"
            elif reject: pattern = "model_only"
            else: pattern = "neither"
            rows.append({"replicate_index": index, "generation_seed": generation_seed, "shuffle_seed": shuffle_seed, "scenario": scenario, "conditioning": condition, "n": 200, "k": 20, "raw_score": raw, "p_value": p_value, "reject_0_05": reject, "stage7_oracle_raw_score": oracle_raw, "stage7_oracle_p_value": stage7["p_value"], "model_minus_oracle_raw_score": raw - oracle_raw, "oracle_model_rejection_pattern": pattern, "oracle_identity_max_error": errors[scenario], **diagnostics, "predictor_random_state": predictor_specs[scenario]["random_state"], "predictor_train_rows": predictor_specs[scenario]["train_rows"]})
    return rows


def wilson(k: int) -> list[float]:
    ci = binomtest(k, 20).proportion_ci(confidence_level=.95, method="wilson")
    return [float(ci.low), float(ci.high)]


def stats(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max())}


def summary(rows: list[dict]) -> dict:
    out = {}
    for condition in ("M0", "Mspace"):
        item = {"primary": condition == "M0", "by_scenario": {}}
        paired = {}
        for scenario in ("S2", "S4"):
            group = [r for r in rows if r["scenario"] == scenario and r["conditioning"] == condition]
            if len(group) != 20: raise AssertionError("incomplete summary group")
            raw = np.asarray([r["raw_score"] for r in group]); rejects = sum(r["reject_0_05"] for r in group); oracle_rejects = sum(r["stage7_oracle_p_value"] <= .05 for r in group)
            patterns = {key: sum(r["oracle_model_rejection_pattern"] == key for r in group) for key in ("both", "oracle_only", "model_only", "neither")}
            predictor_keys = ["scale_fit_mae_y", "mean_abs_y_pred_minus_mu_true", "median_abs_y_pred_minus_mu_true", "mean_A_model", "median_A_model", "mean_A_oracle", "median_A_oracle", "mean_abs_A_model_minus_A_oracle"]
            item["by_scenario"][scenario] = {"A_model_rejection_count": rejects, "A_model_rejection_rate": rejects / 20, "A_model_wilson_95": wilson(rejects), "Stage7_A_oracle_rejection_count": oracle_rejects, "Stage7_A_oracle_rejection_rate": oracle_rejects / 20, "raw": stats(raw), "median_A_model_p_value": float(np.median([r["p_value"] for r in group])), "median_model_minus_oracle_raw_score": float(np.median([r["model_minus_oracle_raw_score"] for r in group])), "rejection_patterns": patterns, "predictor_error_diagnostics_median": {key: float(np.median([r[key] for r in group])) for key in predictor_keys}}
            paired[scenario] = {r["replicate_index"]: r for r in group}
        s2 = np.asarray([paired["S2"][i]["raw_score"] for i in range(1,21)]); s4 = np.asarray([paired["S4"][i]["raw_score"] for i in range(1,21)])
        item["median_paired_raw_difference_S2_minus_S4"] = float(np.median(s2-s4)); item["S2_raw_greater_count"] = int(np.count_nonzero(s2>s4)); item["S2_raw_greater_proportion"] = float(np.mean(s2>s4))
        out[condition] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")); predictor_config = json.loads(PREDICTOR_CONFIG_PATH.read_text(encoding="utf-8")); generator, predictor = generator_module(), predictor_module()
    frozen_paths = [ROOT / "simulation_pilot_20260908" / "predictor_outputs" / f"{s}_{kind}.csv" for s in ("S2", "S4") for kind in ("predictions", "authorized_residuals")]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in frozen_paths}
    reproduction = frozen_reproduction(config, generator, predictor_config, predictor)
    expected = oracle_expected()
    if args.smoke:
        seed = int(config["random_seeds"]["base_variables"]) + 30000
        smoke = run_replicate(0, seed, config, generator, predictor_config, predictor, None, 19)
        print(json.dumps({"smoke_status":"PASS","smoke_generation_seed":seed,"tests":len(smoke),"frozen_reproduction":reproduction,"sig_samples":19},ensure_ascii=False),flush=True); return
    started=time.perf_counter(); all_rows=[]
    for index in range(1,21):
        seed=expected[(index,"S2","M0")]["generation_seed"]; all_rows.extend(run_replicate(index,seed,config,generator,predictor_config,predictor,expected,99))
        if index%5==0: print(f"PROGRESS: {index}/20 replicate seeds complete",flush=True)
    after={str(p.relative_to(ROOT)):sha256(p) for p in frozen_paths}
    if before!=after: raise AssertionError("frozen predictor outputs changed")
    fields=list(all_rows[0])
    with (OUT/"stage8_model_residual_replicates.csv").open("w",encoding="utf-8",newline="") as h: w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(all_rows)
    meta={"stage":"Stage 8","execution_status":"PASS","interpreter":sys.executable,"packages":{x:importlib.metadata.version(x) for x in ("tigramite","numpy","scipy","pandas","scikit-learn")},"frozen_predictor_reproduction":reproduction,"frozen_output_hashes_before_after_match":before==after,"generator_seeds_source":"stage7_oracle_power_replicates.csv","configuration":{"target":"A_model=abs(y-y_pred)","n":200,"k":20,"knn_fraction":.10,"sig_samples":99,"shuffle_neighbors":5,"sig_blocklength":1,"alpha":.05,"predictor_random_state":predictor_config["model_configuration"]["random_state"],"predictor_features":predictor_config["authorized_features"]},"validation":{"Stage7_oracle_raw_match_tolerance":1e-12,"all_20_train_only_fits":"PASS"},"summary":summary(all_rows),"replicates":all_rows,"elapsed_time_seconds":time.perf_counter()-started}
    (OUT/"stage8_model_residual_summary.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");print("PRODUCTION PASS: 20/20 replicate seeds complete",flush=True)


if __name__=="__main__": main()
