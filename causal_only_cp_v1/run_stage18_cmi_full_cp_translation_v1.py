"""Execute the preregistered Stage-18 CMI-Full CP translation benchmark."""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RECORD = ROOT / "experiments/causal_only_cp/20260919_stage18_cmi_full_cp_translation_v1"
STAGE15_SOURCE = HERE / "run_stage15_fresh_confirmatory_v1.py"
CMI_WORKER = HERE / "stage18_cmi_worker_v1.py"
CMI_PYTHON = ROOT / "causal_strength_diagnostic_v1/.venv_cmi_v1/Scripts/python.exe"
GENERATOR_SOURCE = ROOT / "simulation_pilot_20260908/generator/generate_scenarios.py"
GENERATOR_CONFIG = ROOT / "simulation_pilot_20260908/generator/config_v1.json"
PREDICTOR_SOURCE = ROOT / "simulation_pilot_20260908/predictor/fit_predictors.py"
PREDICTOR_CONFIG = ROOT / "simulation_pilot_20260908/predictor/predictor_config_v1.json"
SCENARIOS = ["S2", "S4"]
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]
META = ["sample_id", "scenario", "split", "local_calibration_role"]
CMI_METHODS = ["CMI_MSPACE_999", "CMI_FULL_999"]
METHODS = ["Raw-CP", "GeoOnly-NormCP", "CausalOnly-CMI-Mspace999-NormCP", "CausalOnly-CMI-Full999-NormCP", "Oracle-NormCP"]
ALPHA = .10
BOOTSTRAP_SEED = 21918001


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n", encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def verify_seal() -> tuple[dict, list[int]]:
    seal = json.loads((RECORD / "PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    for name, expected in seal["files"].items():
        if sha(RECORD / name) != expected: raise AssertionError(f"preregistration changed: {name}")
    payload = "\n".join(f"{n} {seal['files'][n]}" for n in sorted(seal["files"]))
    if hashlib.sha256(payload.encode()).hexdigest() != seal["composite_sha256"]: raise AssertionError("composite mismatch")
    seeds = json.loads((RECORD / "seed_manifest.json").read_text(encoding="utf-8"))["seeds"]
    if seeds != list(range(20730001, 20730051)): raise AssertionError("seed mismatch")
    return seal, seeds


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def scenario_dir(index: int, seed: int, scenario: str) -> Path:
    return RECORD / "frozen_inputs" / f"replicate_{index:02d}_seed_{seed}" / scenario


def make_cmi_input(s15, index: int, seed: int, scenario: str) -> Path:
    path = scenario_dir(index, seed, scenario) / "cmi_stage18_input.csv"
    if not path.exists():
        scale = s15.role(s15.load_development(index, seed, scenario), "scale_fit")
        frame = scale[["sample_id", *COORDS, *FEATURES]].copy()
        frame["A_model"] = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
        write_csv(frame, path)
    return path


def run_worker(index: int, seed: int, scenario: str, input_path: Path, output: Path) -> dict:
    if output.exists() and output.with_suffix(".json").exists():
        return {"index": index, "scenario": scenario, "runtime_seconds": json.loads(output.with_suffix(".json").read_text())["runtime_seconds"], "checkpoint": True}
    command = [str(CMI_PYTHON), str(CMI_WORKER), "--input", str(input_path), "--output", str(output),
               "--replicate-index", str(index), "--scenario", scenario]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    output.with_suffix(".stdout.txt").write_text(proc.stdout, encoding="utf-8")
    output.with_suffix(".stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode: raise RuntimeError(f"CMI failed {index}/{scenario}: {proc.stderr[-3000:]}")
    return {"index": index, "scenario": scenario, "runtime_seconds": time.perf_counter()-started, "checkpoint": False}


def cmi_info(frame: pd.DataFrame, method: str) -> dict:
    d = frame[frame.method.eq(method)].copy()
    if len(d) != 4 or d.feature.tolist() != FEATURES: raise AssertionError("CMI schema")
    weights = dict(zip(d.feature, d.weight.astype(float)))
    selected = [f for f in FEATURES if weights[f] > 0]
    status = str(d.status.iloc[0])
    if status == "SIGNAL" and not np.isclose(sum(weights.values()), 1, atol=1e-12): raise AssertionError("weight sum")
    if status == "NO_CAUSAL_SIGNAL" and selected: raise AssertionError("no-signal weights")
    return {"weights": weights, "selected": selected, "status": status}


def freeze_analysis(s15, index: int, seed: int, scenario: str, infos: dict[str, dict]) -> dict:
    dev = s15.load_development(index, seed, scenario)
    scale, tuning, final = (s15.role(dev, x) for x in ("scale_fit", "tuning_calibration", "final_calibration"))
    models, selected_k, krows = {}, {}, []
    k, model, rows = s15.fit_local(scale, tuning, COORDS, np.ones(2), "GeoOnly-NormCP")
    selected_k["GeoOnly-NormCP"], models["GeoOnly-NormCP"] = k, model; krows += rows
    mapping = {"CMI_MSPACE_999": "CausalOnly-CMI-Mspace999-NormCP", "CMI_FULL_999": "CausalOnly-CMI-Full999-NormCP"}
    for cmi_method, cp_method in mapping.items():
        info = infos[cmi_method]
        if info["status"] == "NO_CAUSAL_SIGNAL": k, model, rows = s15.constant_model(scale, tuning, cp_method)
        else: k, model, rows = s15.fit_local(scale, tuning, info["selected"], np.array([info["weights"][f] for f in info["selected"]]), cp_method)
        selected_k[cp_method], models[cp_method] = k, model; krows += rows
    final_a = np.abs(final.y_true.to_numpy(float) - final.y_pred.to_numpy(float))
    sigma_final = {"Raw-CP": np.ones(200), "Oracle-NormCP": final.oracle_sigma.to_numpy(float)}
    for method in ("GeoOnly-NormCP", "CausalOnly-CMI-Mspace999-NormCP", "CausalOnly-CMI-Full999-NormCP"):
        sigma_final[method] = s15.query_model(final, models[method], selected_k[method])
    q, ranks = {}, {}
    for method in METHODS:
        q[method], ranks[method] = s15.quantile(final_a / sigma_final[method], ALPHA)
    base = scenario_dir(index, seed, scenario)
    x = pd.read_csv(base / "predictor_inputs.csv", keep_default_na=False)
    pred = pd.read_csv(base / "predictions.csv", keep_default_na=False)
    test_cov = x[x.split.eq("test")].merge(pred, on=META + ["split_role"], validate="one_to_one", sort=False)
    sigma_test = {"Raw-CP": np.ones(300)}
    for method in ("GeoOnly-NormCP", "CausalOnly-CMI-Mspace999-NormCP", "CausalOnly-CMI-Full999-NormCP"):
        sigma_test[method] = s15.query_model(test_cov, models[method], selected_k[method])
    digest = hashlib.sha256(json.dumps({"k": selected_k, "q": q}, sort_keys=True).encode())
    for method in sorted(sigma_test): digest.update(np.asarray(sigma_test[method]).tobytes())
    return {"scale": scale, "tuning": tuning, "final": final, "test_cov": test_cov, "models": models,
            "selected_k": selected_k, "krows": krows, "q": q, "ranks": ranks, "sigma_test": sigma_test, "hash": digest.hexdigest()}


UNSEAL_FIELDS = ["phase", "replicate_index", "generation_seed", "scenario", "mspace_status", "mspace_features", "mspace_weights_json",
                 "full_status", "full_features", "full_weights_json", "selected_k_json", "q_json", "scale_fit_hash",
                 "tuning_hash", "final_calibration_hash", "prediction_hash", "test_covariate_hash", "barrier_row_sha256"]


def append_unseal(s15, row: dict) -> None:
    base = {k: row[k] for k in UNSEAL_FIELDS if k != "barrier_row_sha256"}
    row["barrier_row_sha256"] = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = RECORD / "test_unseal_audit.csv"; exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNSEAL_FIELDS, lineterminator="\n")
        if not exists: writer.writeheader()
        writer.writerow(row); handle.flush(); os.fsync(handle.fileno())


def evaluate(s15, index: int, seed: int, scenario: str, infos: dict[str, dict], frozen: dict, phase: str = "primary"):
    m, f = infos["CMI_MSPACE_999"], infos["CMI_FULL_999"]
    base = scenario_dir(index, seed, scenario)
    append_unseal(s15, {"phase": phase, "replicate_index": index, "generation_seed": seed, "scenario": scenario,
        "mspace_status": m["status"], "mspace_features": ";".join(m["selected"]), "mspace_weights_json": json.dumps(m["weights"], sort_keys=True),
        "full_status": f["status"], "full_features": ";".join(f["selected"]), "full_weights_json": json.dumps(f["weights"], sort_keys=True),
        "selected_k_json": json.dumps(frozen["selected_k"], sort_keys=True), "q_json": json.dumps(frozen["q"], sort_keys=True),
        "scale_fit_hash": s15.frame_hash(frozen["scale"]), "tuning_hash": s15.frame_hash(frozen["tuning"]),
        "final_calibration_hash": s15.frame_hash(frozen["final"]), "prediction_hash": sha(base/"predictions.csv"),
        "test_covariate_hash": s15.frame_hash(frozen["test_cov"])})
    sealed = pd.read_csv(base / "test_sealed.csv", keep_default_na=False)
    test = frozen["test_cov"].merge(sealed, on=META + ["split_role"], validate="one_to_one", sort=False)
    if len(test) != 300: raise AssertionError("test join")
    y, pred = test.y_true.to_numpy(float), test.y_pred.to_numpy(float)
    sigma = dict(frozen["sigma_test"]); sigma["Oracle-NormCP"] = test.oracle_sigma.to_numpy(float)
    rows, scale_rows = [], []
    for method in METHODS:
        metrics = s15.interval_metrics(y, pred, frozen["q"][method], sigma[method], ALPHA)
        info = m if "Mspace" in method else f if "Full" in method else {"status": "NA", "selected": [], "weights": {}}
        rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, "method": method,
                     **{k: metrics[k] for k in ("coverage", "covered_count", "test_n", "mean_width", "median_width", "mean_interval_score")},
                     "selected_k": frozen["selected_k"].get(method), "q_norm": frozen["q"][method],
                     "tuning_scale_mae": frozen["models"].get(method, {}).get("tuning_mae"), "signal_status": info["status"],
                     "selected_features": ";".join(info["selected"]), "feature_weights": json.dumps(info["weights"], sort_keys=True)})
        if info["status"] == "NO_CAUSAL_SIGNAL":
            raw = s15.interval_metrics(y, pred, frozen["q"]["Raw-CP"], sigma["Raw-CP"], ALPHA)
            diff = max(np.max(np.abs(metrics["lower"]-raw["lower"])), np.max(np.abs(metrics["upper"]-raw["upper"])))
            if diff > 1e-12: raise AssertionError(f"fallback inequality {method}: {diff}")
    if scenario == "S2":
        oracle = test.oracle_sigma.to_numpy(float)
        strata = {}
        for cmi_method, info in infos.items():
            selected = set(info["selected"])
            strata[cmi_method] = "NO_SIGNAL" if not selected else "x2_only" if selected == {"x2"} else "x2_plus_nuisance" if "x2" in selected else "nuisance_without_x2"
        for method in ("GeoOnly-NormCP", "CausalOnly-CMI-Mspace999-NormCP", "CausalOnly-CMI-Full999-NormCP", "Oracle-NormCP"):
            values = sigma[method]
            corr = None if np.ptp(values) <= 1e-15 else float(np.corrcoef(values, oracle)[0,1])
            key = "CMI_MSPACE_999" if "Mspace" in method else "CMI_FULL_999" if "Full" in method else None
            scale_rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, "method": method,
                               "test_oracle_correlation": corr, "test_oracle_mae": float(np.mean(np.abs(values-oracle))),
                               "selection_stratum": strata.get(key, "NOT_APPLICABLE")})
    krows = [{"replicate_index": index, "generation_seed": seed, "scenario": scenario, **row} for row in frozen["krows"]]
    return rows, scale_rows, krows


def main() -> None:
    started = time.time(); seal, seeds = verify_seal()
    if (RECORD / "execution_checkpoint.json").exists(): raise FileExistsError("Stage-18 scientific checkpoint already exists")
    s15 = load_module("stage15_reused", STAGE15_SOURCE); s15.RECORD = RECORD
    generator = s15.load_module("stage18_generator", GENERATOR_SOURCE); predictor = s15.load_module("stage18_predictor", PREDICTOR_SOURCE)
    gen_cfg = json.loads(GENERATOR_CONFIG.read_text()); pred_cfg = json.loads(PREDICTOR_CONFIG.read_text())
    if pred_cfg["model_configuration"]["random_state"] != 420907 or pred_cfg["authorized_features"] != s15.FEATURES_ALL: raise AssertionError("predictor protocol")
    execution_config = {"preregistration_composite_sha256": seal["composite_sha256"], "implementation": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
                        "implementation_sha256": sha(Path(__file__)), "cmi_worker_sha256": sha(CMI_WORKER), "stage15_reused_sha256": sha(STAGE15_SOURCE)}
    dump(RECORD / "execution_config.json", execution_config)

    generator_probe = s15.generate_reproducibility_probe(generator, gen_cfg, seeds[0])
    first = s15.generate_replicate(1, seeds[0], generator, gen_cfg, predictor, pred_cfg)
    predictor_probe = {sc: s15.predictor_determinism_probe(1, seeds[0], sc, predictor, pred_cfg) for sc in SCENARIOS}
    infos, frames, runtime_rows = {}, {}, []
    # First-pair CMI and exact clean rerun.
    for scenario in SCENARIOS:
        inp = make_cmi_input(s15, 1, seeds[0], scenario); out = scenario_dir(1, seeds[0], scenario) / "cmi_stage18_results.csv"
        runtime_rows.append(run_worker(1, seeds[0], scenario, inp, out)); frame = pd.read_csv(out); frames[(1, scenario)] = frame
        infos[(1, scenario)] = {m: cmi_info(frame, m) for m in CMI_METHODS}
        with tempfile.TemporaryDirectory(prefix="stage18_cmi_rerun_", dir=RECORD) as td:
            out2 = Path(td) / "rerun.csv"; run_worker(1, seeds[0], scenario, inp, out2)
            cols = ["method", "feature", "raw_cmi", "p_value", "bh_q", "eligible", "weight", "status", "rank"]
            pd.testing.assert_frame_equal(frame[cols], pd.read_csv(out2)[cols], check_exact=True)
    frozen_preflight = {}
    for scenario in SCENARIOS:
        a = freeze_analysis(s15, 1, seeds[0], scenario, infos[(1,scenario)])
        b = freeze_analysis(s15, 1, seeds[0], scenario, infos[(1,scenario)])
        if a["hash"] != b["hash"]: raise AssertionError("freeze nondeterminism")
        frozen_preflight[scenario] = a["hash"]
        # Evaluate only after the barrier written by evaluate().
        evaluate(s15, 1, seeds[0], scenario, infos[(1,scenario)], a, phase="preflight")
        # Fallback algebra is audited even if the observed first pair has signal.
        _, constant, _ = s15.constant_model(a["scale"], a["tuning"], "fallback_probe")
        fa = np.abs(a["final"].y_true.to_numpy(float) - a["final"].y_pred.to_numpy(float))
        qn, _ = s15.quantile(fa / constant["constant"], ALPHA); qr, _ = s15.quantile(fa, ALPHA)
        if abs(qn * constant["constant"] - qr) > 1e-12: raise AssertionError("fallback algebra")
    preflight = {"status": "PASS", "seed": seeds[0], "generator": generator_probe, "predictor": predictor_probe,
                 "cmi_clean_rerun": "bit-identical both methods/scenarios", "frozen_analysis_hashes": frozen_preflight,
                 "coordinate_leakage": "PASS: causal distance columns are eligible subsets of x1,x2,g_proxy,x_noise", "test_seal": "PASS"}
    dump(RECORD / "preflight_summary.json", preflight); print("PREFLIGHT PASS", flush=True)
    generator_rows = [x[0] for x in first]; split_rows = [x[1] for x in first]; prediction_rows = [x[2] for x in first]
    for index, seed in enumerate(seeds[1:], 2):
        rows = s15.generate_replicate(index, seed, generator, gen_cfg, predictor, pred_cfg)
        generator_rows.extend(x[0] for x in rows); split_rows.extend(x[1] for x in rows); prediction_rows.extend(x[2] for x in rows)
        if index % 5 == 0: print(f"GENERATION {index}/50", flush=True)
    # Ensure every CMI input exists before parallel estimator execution.
    jobs = []
    for index, seed in enumerate(seeds, 1):
        for scenario in SCENARIOS:
            inp = make_cmi_input(s15, index, seed, scenario); out = scenario_dir(index, seed, scenario) / "cmi_stage18_results.csv"
            if (index, scenario) not in frames: jobs.append((index, seed, scenario, inp, out))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_worker, *job) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            runtime_rows.append(future.result())
            if done % 10 == 0: print(f"CMI JOBS {done}/{len(jobs)}", flush=True)
    for index, seed in enumerate(seeds, 1):
        for scenario in SCENARIOS:
            if (index, scenario) not in frames:
                frame = pd.read_csv(scenario_dir(index, seed, scenario) / "cmi_stage18_results.csv")
                frames[(index, scenario)] = frame; infos[(index, scenario)] = {m: cmi_info(frame, m) for m in CMI_METHODS}

    cmi_rows, weight_rows = [], []
    for index, seed in enumerate(seeds, 1):
        for scenario in SCENARIOS:
            frame = frames[(index, scenario)]
            for row in frame.to_dict("records"): cmi_rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, **row})
            for method in CMI_METHODS:
                info = infos[(index,scenario)][method]
                for feature in FEATURES: weight_rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                    "cmi_method": method, "feature": feature, "selected": feature in info["selected"], "weight": info["weights"][feature], "status": info["status"]})
    cmi_all = pd.DataFrame(cmi_rows)
    write_csv(cmi_all[cmi_all.method.eq("CMI_MSPACE_999")], RECORD/"cmi_mspace999_results.csv")
    write_csv(cmi_all[cmi_all.method.eq("CMI_FULL_999")], RECORD/"cmi_full999_results.csv")
    write_csv(pd.DataFrame(weight_rows), RECORD/"weight_audit.csv")
    write_csv(pd.DataFrame(generator_rows), RECORD/"generator_audit.csv"); write_csv(pd.DataFrame(split_rows), RECORD/"split_integrity.csv")
    write_csv(pd.DataFrame(prediction_rows), RECORD/"prediction_manifest.csv")

    result_rows, scale_rows, krows = [], [], []
    for index, seed in enumerate(seeds, 1):
        for scenario in SCENARIOS:
            frozen = freeze_analysis(s15, index, seed, scenario, infos[(index,scenario)])
            rr, sr, kr = evaluate(s15, index, seed, scenario, infos[(index,scenario)], frozen, phase="primary")
            result_rows += rr; scale_rows += sr; krows += kr
        if index % 5 == 0: print(f"CP EVALUATION {index}/50", flush=True)
    write_csv(pd.DataFrame(result_rows), RECORD/"replicate_results.csv")
    write_csv(pd.DataFrame(scale_rows), RECORD/"scale_diagnostics.csv")
    write_csv(pd.DataFrame(krows), RECORD/"k_tuning_results.csv")
    dump(RECORD/"execution_checkpoint.json", {"status": "SCIENTIFIC_COMPUTATION_COMPLETE", "elapsed_seconds": time.time()-started,
        "scenario_replicates": 100, "cmi_method_feature_rows": len(cmi_rows), "result_rows": len(result_rows),
        "runtime_jobs": runtime_rows, "test_unseal_rows": len(pd.read_csv(RECORD/"test_unseal_audit.csv")),
        "expected_test_unseal_rows": 102})
    with (RECORD/"run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"Scientific computation complete; elapsed_seconds={time.time()-started:.3f}\n")
    print("STAGE18 SCIENTIFIC COMPUTATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
