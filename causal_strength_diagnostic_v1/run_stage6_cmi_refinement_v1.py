"""Stage 6: replace only three Stage 5 borderline shuffle nulls with 999 surrogates."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import false_discovery_control
from tigramite.independence_tests.cmiknn import CMIknn

from src.cmi_adapter_v1 import direct_score
from run_stage4_oracle_cmi_v1 import load_oracle_rows
from run_stage5_cmi_significance_v1 import META, model_frame, prepared_cases

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MASTER_SEED = 20260910
SEED = 3
REFINED = {("A_model", "S2", "M0"), ("A_oracle", "S2", "M0"), ("A_oracle", "S2", "Mspace")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_estimator() -> CMIknn:
    return CMIknn(knn=0.10, shuffle_neighbors=5, significance="shuffle_test", transform=False, workers=1, sig_samples=999, sig_blocklength=1, seed=SEED)


def refine(array: np.ndarray, xyz: np.ndarray) -> tuple[float, float, np.ndarray, int]:
    test = fresh_estimator()
    raw = float(test.get_dependence_measure(array.copy(), xyz.copy()))
    p_value, null = test.get_shuffle_significance(array.copy(), xyz.copy(), raw, return_null_dist=True)
    null = np.asarray(null, dtype=float)
    exceedances = int(np.count_nonzero(null >= raw))
    if float(p_value) != (exceedances + 1) / 1000:
        raise AssertionError("official empirical p-value does not match recorded null distribution")
    return raw, float(p_value), null, exceedances


def main() -> None:
    started = time.perf_counter()
    stage5_csv, stage5_json = OUT / "stage5_cmi_significance.csv", OUT / "stage5_run.json"
    with stage5_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8:
        raise AssertionError("Stage 5 result family must contain exactly eight tests")
    stage5_meta = json.loads(stage5_json.read_text(encoding="utf-8"))
    index = {(row["target"], row["scenario"], row["conditioning"]): row for row in rows}
    if set(index) != {(target, scenario, condition) for target in ("A_model", "A_oracle") for scenario in ("S2", "S4") for condition in ("M0", "Mspace")}:
        raise AssertionError("Stage 5 test identities differ from the fixed eight-test family")
    feature_access, target_access = [], []
    model = model_frame("S2", feature_access, target_access)
    oracle, oracle_hashes = load_oracle_rows("S2", feature_access)
    oracle = oracle[META + ["x2", "s_x", "s_y"]].assign(target=oracle["A_oracle"].to_numpy(float))
    frames = {"A_model": model, "A_oracle": oracle}
    refined_records, preprocessing = {}, {}
    for target, scenario, condition in sorted(REFINED):
        cases, params = prepared_cases(target, scenario, frames[target])
        preprocessing[f"{target}/{scenario}"] = params
        array, xyz = cases[condition]
        if array.shape[1] != 200:
            raise AssertionError(f"{target}/{scenario}/{condition}: n is not 200")
        frozen_raw = float(index[(target, scenario, condition)]["raw_score"])
        recomputed, _ = direct_score(array, xyz, MASTER_SEED)
        if abs(recomputed - frozen_raw) > 1e-12:
            raise AssertionError(f"{target}/{scenario}/{condition}: frozen raw-score mismatch")
        raw, p_value, null, exceedances = refine(array, xyz)
        if abs(raw - frozen_raw) > 1e-12:
            raise AssertionError(f"{target}/{scenario}/{condition}: refined raw-score mismatch")
        refined_records[(target, scenario, condition)] = {"n": 200, "raw_score": raw, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, .95, method="linear")), "exceedance_count": exceedances, "p_value": p_value}
    output_rows = []
    for row in rows:
        key = (row["target"], row["scenario"], row["conditioning"])
        item = dict(row)
        if key in refined_records:
            record = refined_records[key]
            item.update({"sig_samples": 999, "raw_score": record["raw_score"], "null_mean": record["null_mean"], "null_q95": record["null_q95"], "p_value": record["p_value"], "exceedance_count": record["exceedance_count"], "refined": True})
        else:
            item.update({"sig_samples": int(item["sig_samples"]), "raw_score": float(item["raw_score"]), "null_mean": float(item["null_mean"]), "null_q95": float(item["null_q95"]), "p_value": float(item["p_value"]), "exceedance_count": None, "refined": False})
        output_rows.append(item)
    p_values = np.asarray([row["p_value"] for row in output_rows], dtype=float)
    q_values = false_discovery_control(p_values, method="bh")
    for row, q in zip(output_rows, q_values):
        row["bh_q_value"] = float(q)
        row["significant_raw_0_05"] = bool(row["p_value"] < .05)
        row["significant_bh_0_05"] = bool(q < .05)
    fields = ["target", "scenario", "conditioning", "n", "k", "knn_fraction", "shuffle_neighbors", "sig_samples", "seed", "raw_score", "null_mean", "null_q95", "exceedance_count", "p_value", "bh_q_value", "significant_raw_0_05", "significant_bh_0_05", "monte_carlo_flag", "refined"]
    with (OUT / "stage6_cmi_refinement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)
    inputs = dict(stage5_meta["input_paths_and_sha256"])
    inputs.update({str(stage5_csv.relative_to(ROOT)).replace("\\", "/"): sha256(stage5_csv), str(stage5_json.relative_to(ROOT)).replace("\\", "/"): sha256(stage5_json)})
    metadata = {"stage": "Stage 6", "execution_status": "PASS", "interpreter": sys.executable, "packages": {name: importlib.metadata.version(name) for name in ("tigramite", "numpy", "scipy", "pandas", "scikit-learn")}, "input_paths_and_sha256": inputs, "reused_stage5_files": [str(stage5_csv.relative_to(ROOT)).replace("\\", "/"), str(stage5_json.relative_to(ROOT)).replace("\\", "/"), "run_stage5_cmi_significance_v1.py"], "configuration": {"n": 200, "knn_fraction": .10, "actual_k": 20, "transform": False, "workers": 1, "significance": "shuffle_test", "shuffle_neighbors": 5, "sig_blocklength": 1, "seed": SEED, "sig_samples_refined": 999, "p_value": "one-sided empirical (exceedances+1)/1000"}, "refined_tests": [{"target": k[0], "scenario": k[1], "conditioning": k[2], **v} for k, v in refined_records.items()], "unchanged_tests_source": "Stage 5 199-surrogate p-values; not rerun", "alignment_and_raw_checks": {"all_refined_n": 200, "complete_metadata_alignment": True, "frozen_raw_agreement_tolerance": 1e-12}, "preprocessing": preprocessing, "access_logs": {"feature": feature_access, "A_model_target": target_access}, "bh_adjustment": "scipy.stats.false_discovery_control(method='bh') over all eight tests", "updated_eight_test_results": output_rows, "elapsed_time_seconds": time.perf_counter() - started}
    (OUT / "stage6_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"refined": [{"target": key[0], "scenario": key[1], "conditioning": key[2], **value} for key, value in refined_records.items()], "all": output_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
