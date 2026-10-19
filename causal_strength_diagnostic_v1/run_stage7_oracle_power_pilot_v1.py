"""Fixed 20-seed paired oracle-only CMIknn power pilot; generator outputs stay in memory."""

from __future__ import annotations

import argparse
import copy
import csv
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

from src.cmi_adapter_v1 import standardize_selected

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "simulation_pilot_20260908" / "generator" / "config_v1.json"
META = ["sample_id", "scenario", "split", "local_calibration_role"]
MASTER_SEED = 20260910
TOL = 1e-12


def generator_module():
    path = ROOT / "simulation_pilot_20260908" / "generator" / "generate_scenarios.py"
    spec = importlib.util.spec_from_file_location("stage7_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generator module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fresh_test(shuffle_seed: int, sig_samples: int) -> CMIknn:
    return CMIknn(knn=.10, shuffle_neighbors=5, significance="shuffle_test", transform=False, workers=1, sig_samples=sig_samples, sig_blocklength=1, seed=shuffle_seed)


def scale_fit_oracle(observed: pd.DataFrame, truth: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, float]:
    mask_obs = (observed["split"] == "local_calibration") & (observed["local_calibration_role"] == "scale_fit")
    mask_truth = (truth["split"] == "local_calibration") & (truth["local_calibration_role"] == "scale_fit")
    left = observed.loc[mask_obs, META + ["s_x", "s_y", "x2", "y"]]
    right = truth.loc[mask_truth, META + ["mu_true", "sigma_true", "epsilon"]]
    if len(left) != 200 or len(right) != 200 or left.sample_id.duplicated().any() or right.sample_id.duplicated().any():
        raise AssertionError(f"{scenario}: scale-fit selection is not 200 unique rows")
    frame = left.merge(right, on=META, how="inner", validate="one_to_one").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if len(frame) != 200 or set(frame["scenario"]) != {scenario} or set(frame["split"]) != {"local_calibration"} or set(frame["local_calibration_role"]) != {"scale_fit"}:
        raise AssertionError(f"{scenario}: metadata join or restriction failed")
    numeric = frame[["s_x", "s_y", "x2", "y", "mu_true", "sigma_true", "epsilon"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise AssertionError(f"{scenario}: nonfinite selected input")
    frame["A_oracle"] = np.abs(frame["y"].to_numpy(float) - frame["mu_true"].to_numpy(float))
    expected = frame["sigma_true"].to_numpy(float) * np.abs(frame["epsilon"].to_numpy(float))
    error = float(np.max(np.abs(frame["A_oracle"].to_numpy(float) - expected)))
    if not np.allclose(frame["A_oracle"].to_numpy(float), expected, rtol=TOL, atol=TOL):
        raise AssertionError(f"{scenario}: A_oracle identity failed ({error})")
    if scenario == "S4" and not np.allclose(frame["A_oracle"].to_numpy(float), 1.3 * np.abs(frame["epsilon"].to_numpy(float)), rtol=TOL, atol=TOL):
        raise AssertionError("S4: constant-sigma oracle identity failed")
    return frame, error


def one_replicate(index: int, generation_seed: int, config: dict, generator, sig_samples: int = 99) -> list[dict]:
    local_config = copy.deepcopy(config)
    local_config["random_seeds"]["base_variables"] = generation_seed
    scenarios = generator.generate_all_scenarios(local_config)  # in-memory only; no write_outputs/run_pipeline call
    frames, errors = {}, {}
    for scenario in ("S2", "S4"):
        frames[scenario], errors[scenario] = scale_fit_oracle(*scenarios[scenario], scenario)
    pd.testing.assert_frame_equal(frames["S2"][["sample_id", "split", "local_calibration_role"]], frames["S4"][["sample_id", "split", "local_calibration_role"]], check_dtype=False)
    for column in ("s_x", "s_y", "x2", "epsilon"):
        pd.testing.assert_series_equal(frames["S2"][column], frames["S4"][column], check_names=False, check_dtype=False)
    rows = []
    seed_offset = 0
    for scenario in ("S2", "S4"):
        values, _ = standardize_selected({name: frames[scenario][name].to_numpy(float) for name in ("x2", "s_x", "s_y", "A_oracle")})
        cases = (("M0", np.vstack([values["x2"], values["A_oracle"]]), np.array([0, 1], dtype=int)), ("Mspace", np.vstack([values["x2"], values["A_oracle"], values["s_x"], values["s_y"]]), np.array([0, 1, 2, 2], dtype=int)))
        for condition, array, xyz in cases:
            shuffle_seed = MASTER_SEED + index * 10 + seed_offset
            seed_offset += 1
            test = fresh_test(shuffle_seed, sig_samples)
            raw = float(test.get_dependence_measure(array.copy(), xyz.copy()))
            p_value = float(test.get_shuffle_significance(array.copy(), xyz.copy(), raw))
            rows.append({"replicate_index": index, "generation_seed": generation_seed, "shuffle_seed": shuffle_seed, "scenario": scenario, "conditioning": condition, "n": 200, "k": 20, "raw_score": raw, "p_value": p_value, "reject_0_05": p_value <= .05, "oracle_identity_max_error": errors[scenario]})
    return rows


def wilson(count: int, n: int) -> list[float]:
    interval = binomtest(count, n).proportion_ci(confidence_level=.95, method="wilson")
    return [float(interval.low), float(interval.high)]


def raw_stats(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max())}


def summarize(rows: list[dict]) -> dict:
    result = {}
    for condition in ("M0", "Mspace"):
        s2 = [row for row in rows if row["scenario"] == "S2" and row["conditioning"] == condition]
        s4 = [row for row in rows if row["scenario"] == "S4" and row["conditioning"] == condition]
        s2_by_index, s4_by_index = {row["replicate_index"]: row for row in s2}, {row["replicate_index"]: row for row in s4}
        if set(s2_by_index) != set(range(1, 21)) or set(s4_by_index) != set(range(1, 21)):
            raise AssertionError(f"{condition}: incomplete paired replicate results")
        s2_scores = np.asarray([s2_by_index[i]["raw_score"] for i in range(1, 21)])
        s4_scores = np.asarray([s4_by_index[i]["raw_score"] for i in range(1, 21)])
        s2_reject = sum(row["reject_0_05"] for row in s2)
        s4_reject = sum(row["reject_0_05"] for row in s4)
        result[condition] = {"primary": condition == "M0", "S2_rejection_count": s2_reject, "S2_rejection_rate": s2_reject / 20, "S2_wilson_95": wilson(s2_reject, 20), "S4_rejection_count": s4_reject, "S4_rejection_rate": s4_reject / 20, "S4_wilson_95": wilson(s4_reject, 20), "S2_raw": raw_stats(s2_scores), "S4_raw": raw_stats(s4_scores), "median_paired_raw_difference_S2_minus_S4": float(np.median(s2_scores - s4_scores)), "S2_raw_greater_count": int(np.count_nonzero(s2_scores > s4_scores)), "S2_raw_greater_proportion": float(np.mean(s2_scores > s4_scores)), "median_S2_p_value": float(np.median([row["p_value"] for row in s2])), "median_S4_p_value": float(np.median([row["p_value"] for row in s4]))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")); original_seed = int(config["random_seeds"]["base_variables"])
    generator = generator_module()
    if args.smoke:
        smoke_seed = original_seed + 20000
        smoke_rows = one_replicate(0, smoke_seed, config, generator, sig_samples=19)
        if len(smoke_rows) != 4:
            raise AssertionError("smoke did not produce four tests")
        print(json.dumps({"smoke_status": "PASS", "smoke_generation_seed": smoke_seed, "tests": len(smoke_rows), "sig_samples": 19}, ensure_ascii=False), flush=True)
        return
    started = time.perf_counter()
    all_rows = []
    for index in range(1, 21):
        generation_seed = original_seed + 10000 + index
        all_rows.extend(one_replicate(index, generation_seed, config, generator))
        if index % 5 == 0:
            print(f"PROGRESS: {index}/20 replicate seeds complete", flush=True)
    fields = ["replicate_index", "generation_seed", "shuffle_seed", "scenario", "conditioning", "n", "k", "raw_score", "p_value", "reject_0_05", "oracle_identity_max_error"]
    with (OUT / "stage7_oracle_power_replicates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    metadata = {"stage": "Stage 7", "execution_status": "PASS", "interpreter": sys.executable, "packages": {name: importlib.metadata.version(name) for name in ("tigramite", "numpy", "scipy", "pandas")}, "generator": {"config_path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"), "generator_function": "generate_all_scenarios (in memory only)", "original_base_seed": original_seed, "replicate_seed_formula": "original_base_seed + 10000 + replicate_index", "replicate_seeds": [original_seed + 10000 + i for i in range(1, 21)], "smoke_seed": original_seed + 20000}, "configuration": {"target": "A_oracle=abs(y-mu_true)", "n": 200, "knn_fraction": .10, "actual_k": 20, "transform": False, "workers": 1, "significance": "shuffle_test", "sig_samples": 99, "shuffle_neighbors": 5, "sig_blocklength": 1, "alpha": .05, "shuffle_seed_formula": "20260910 + replicate_index*10 + test_offset"}, "validation": {"tolerance": TOL, "paired_S2_S4_ids_metadata_x2_coordinates_epsilon": "PASS", "all_replicates_completed": 20}, "summary": summarize(all_rows), "replicates": all_rows, "elapsed_time_seconds": time.perf_counter() - started}
    (OUT / "stage7_oracle_power_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PRODUCTION PASS: 20/20 replicate seeds complete", flush=True)


if __name__ == "__main__":
    main()
