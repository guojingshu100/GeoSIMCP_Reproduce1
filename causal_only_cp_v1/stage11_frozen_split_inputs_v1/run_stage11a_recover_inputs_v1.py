"""Recover the exact Stage-8 realizations into split-safe Stage-11 inputs.

The generator is replayed only with the frozen Stage-8 seeds/configuration.
Every recovered predictor-input and scale-fit label CSV is compared after the
same %.17g CSV round trip used by Stage 8. No predictor is fitted and no CP is
run.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import platform
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIM = ROOT / "simulation_pilot_20260908"
FROZEN = ROOT / "causal_strength_diagnostic_v1" / "stage8_frozen_replicates_v1"
STAGE9 = ROOT / "causal_strength_diagnostic_v1"
OUT = HERE
META = ["sample_id", "scenario", "split", "local_calibration_role"]
FEATURES = ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"]
SCENARIOS = ["S2", "S4"]
SPLITS = ["train", "scale_fit", "tuning_calibration", "validation", "final_calibration", "test"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def csv_roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    buf = StringIO()
    frame.to_csv(buf, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")
    return pd.read_csv(StringIO(buf.getvalue()), keep_default_na=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")


def equality_metrics(expected: pd.DataFrame, actual: pd.DataFrame, numeric: list[str]) -> dict:
    if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
        return {"row_count": int(len(actual)), "expected_row_count": int(len(expected)), "columns_equal": False,
                "max_abs_diff": None, "mean_abs_diff": None, "unequal_values": None, "status": "FAIL"}
    unequal = 0
    max_diff = 0.0
    sum_diff = 0.0
    for col in expected.columns:
        if col in numeric:
            a = pd.to_numeric(expected[col], errors="raise").to_numpy(float)
            b = pd.to_numeric(actual[col], errors="raise").to_numpy(float)
            diff = np.abs(a - b)
            unequal += int(np.count_nonzero(diff != 0.0))
            max_diff = max(max_diff, float(diff.max(initial=0.0)))
            sum_diff += float(diff.sum())
        else:
            a = expected[col].astype(str).to_numpy()
            b = actual[col].astype(str).to_numpy()
            unequal += int(np.count_nonzero(a != b))
    denom = max(1, len(expected) * max(1, len(expected.columns)))
    return {"row_count": int(len(actual)), "expected_row_count": int(len(expected)),
            "columns_equal": True, "max_abs_diff": max_diff, "mean_abs_diff": sum_diff / denom,
            "unequal_values": unequal, "status": "PASS" if unequal == 0 else "FAIL"}


def split_role(row: pd.Series) -> str:
    if row["split"] == "local_calibration":
        return str(row["local_calibration_role"])
    return str(row["split"])


def load_manifest() -> dict:
    return json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))


def recover() -> dict:
    if any((OUT / name).exists() for name in ("recovery_manifest.json", "split_inventory.csv")):
        raise FileExistsError("refusing to overwrite Stage-11A recovery outputs")
    config_path = SIM / "generator" / "config_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen_manifest = load_manifest()
    generator = module("stage11a_generator", SIM / "generator" / "generate_scenarios.py")
    OUT.mkdir(parents=True, exist_ok=True)

    inventory, equality, joins, integrity, oracle, records = [], [], [], [], [], []
    expected_counts = {"train": 1000, "scale_fit": 200, "tuning_calibration": 100,
                       "validation": 200, "final_calibration": 200, "test": 300}

    for rep in frozen_manifest["replicates"]:
        index, seed = int(rep["replicate_index"]), int(rep["generation_seed"])
        local = copy.deepcopy(config)
        local["random_seeds"]["base_variables"] = seed
        generated = generator.generate_all_scenarios(local)
        rep_root = OUT / f"replicate_{index:02d}_seed_{seed}"
        for scenario in SCENARIOS:
            observed, truth = generated[scenario]
            item = rep["scenarios"][scenario]
            frozen_input_path = ROOT / item["predictor_inputs"]["path"]
            frozen_scale_path = ROOT / item["scale_fit_labels"]["path"]
            frozen_pred_path = ROOT / item["predictions"]["path"]
            frozen_input = pd.read_csv(frozen_input_path, keep_default_na=False)
            frozen_scale = pd.read_csv(frozen_scale_path, keep_default_na=False)
            predictions = pd.read_csv(frozen_pred_path, keep_default_na=False)

            generated_input = csv_roundtrip(observed[META + FEATURES])
            generated_scale = csv_roundtrip(observed.loc[
                (observed["split"] == "local_calibration") &
                (observed["local_calibration_role"] == "scale_fit"), META + ["y"]])
            numeric_input = FEATURES
            equality.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                             "artifact": "predictor_inputs", **equality_metrics(frozen_input, generated_input, numeric_input),
                             "expected_path": str(frozen_input_path.relative_to(ROOT))})
            equality.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                             "artifact": "scale_fit_labels", **equality_metrics(frozen_scale, generated_scale, ["y"]),
                             "expected_path": str(frozen_scale_path.relative_to(ROOT))})
            if equality[-1]["status"] != "PASS" or equality[-2]["status"] != "PASS":
                raise RuntimeError(f"bit-for-bit recovery mismatch at replicate {index} {scenario}")

            joined = observed.merge(truth[META + ["sigma_true"]], on=META, how="left", validate="one_to_one", sort=False)
            joined = joined.merge(predictions, on=META, how="left", validate="one_to_one", sort=False)
            if joined["y_pred"].isna().any() or not joined["sample_id"].is_unique:
                raise RuntimeError(f"prediction join failure at replicate {index} {scenario}")
            joined["replicate_id"] = index
            joined["seed"] = seed
            joined["split_role"] = joined.apply(split_role, axis=1)
            joined = joined[["sample_id", "scenario", "replicate_id", "seed", "split", "local_calibration_role",
                             "split_role", "s_x", "s_y", "x1", "x2", "g_proxy", "x_noise", "y", "y_pred", "sigma_true"]]
            joined = joined.rename(columns={"y": "y_true", "sigma_true": "oracle_sigma"})
            if not np.isfinite(joined[["y_true", "y_pred", "oracle_sigma"]].to_numpy(float)).all():
                raise RuntimeError(f"nonfinite joined values at replicate {index} {scenario}")

            development = joined[joined["split_role"] != "test"].copy()
            test = joined[joined["split_role"] == "test"].copy()
            for role, expected in expected_counts.items():
                n = int((joined["split_role"] == role).sum())
                if n != expected:
                    raise RuntimeError(f"count mismatch at replicate {index} {scenario} {role}: {n}")
            role_sets = {role: set(joined.loc[joined["split_role"] == role, "sample_id"]) for role in expected_counts}
            overlap = {f"{a}__{b}": len(role_sets[a] & role_sets[b]) for i, a in enumerate(role_sets) for b in list(role_sets)[i + 1:]}
            local_set = role_sets["scale_fit"] | role_sets["tuning_calibration"]
            parent_local = set(joined.loc[joined["split"] == "local_calibration", "sample_id"])
            if max(overlap.values(), default=0) != 0 or local_set != parent_local:
                raise RuntimeError(f"split disjointness/union failure at replicate {index} {scenario}")
            integrity.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                              "counts": json.dumps({k: len(v) for k, v in role_sets.items()}, sort_keys=True),
                              "max_pairwise_overlap": max(overlap.values(), default=0),
                              "local_union_matches_parent": local_set == parent_local, "status": "PASS"})
            oracle_dev = development["oracle_sigma"].to_numpy(float)
            oracle.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                           "development_rows": len(development), "oracle_sigma_min": float(oracle_dev.min()),
                           "oracle_sigma_median": float(np.median(oracle_dev)), "oracle_sigma_max": float(oracle_dev.max()),
                           "oracle_sigma_unique_development": int(np.unique(oracle_dev).size),
                           "expected_behavior": "variable" if scenario == "S2" else "constant",
                           "status": "PASS" if ((scenario == "S2" and np.unique(oracle_dev).size > 1) or
                                                  (scenario == "S4" and np.all(oracle_dev == 1.3)) ) else "FAIL"})

            dev_path = rep_root / scenario / "development.csv"
            test_path = rep_root / scenario / "test_sealed.csv"
            write_csv(development, dev_path)
            write_csv(test, test_path)
            joins.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                          "prediction_rows": len(joined), "missing_predictions": int(joined["y_pred"].isna().sum()),
                          "duplicate_sample_ids": int(joined["sample_id"].duplicated().sum()),
                          "join_key": "sample_id,scenario,split,local_calibration_role", "status": "PASS",
                          "development_path": str(dev_path.relative_to(ROOT)), "test_sealed_path": str(test_path.relative_to(ROOT))})
            for kind, path, frame in (("development", dev_path, development), ("test_sealed", test_path, test)):
                records.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                                "artifact": kind, "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                                "rows": len(frame), "columns": json.dumps(list(frame.columns), separators=(",", ":")),
                                "source": "exact deterministic generator replay + frozen prediction join"})
                for role, n in frame["split_role"].value_counts().sort_index().items():
                    inventory.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                                      "artifact": kind, "split_role": role, "rows": int(n),
                                      "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                                      "source": "exact deterministic generator replay + frozen prediction join"})

        paired_columns = ["sample_id", "split", "local_calibration_role"] + FEATURES
        s2 = generated["S2"][0][paired_columns].reset_index(drop=True)
        s4 = generated["S4"][0][paired_columns].reset_index(drop=True)
        if not s2.equals(s4):
            raise RuntimeError(f"paired S2/S4 generated structure mismatch at replicate {index}")

    # Frozen Stage-9 weights are copied/audited, never recomputed.
    weights = pd.read_csv(STAGE9 / "stage9_provisional_weights.csv")
    weights = weights[(weights["condition"] == "Mspace") & (weights["transform"] == "bh_gated_raw_positive")].copy()
    weights["stage11_status"] = weights["status"].replace({"NO_SIGNAL": "NO_CAUSAL_SIGNAL"})
    weights.to_csv(OUT / "weight_input_audit.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(inventory).to_csv(OUT / "split_inventory.csv", index=False)
    pd.DataFrame(equality).to_csv(OUT / "frozen_input_equality_audit.csv", index=False)
    pd.DataFrame(joins).to_csv(OUT / "prediction_join_audit.csv", index=False)
    pd.DataFrame(integrity).to_csv(OUT / "split_integrity.csv", index=False)
    pd.DataFrame(oracle).to_csv(OUT / "oracle_sigma_audit.csv", index=False)
    (OUT / "recovery_manifest.json").write_text(json.dumps({"artifacts": records}, indent=2) + "\n", encoding="utf-8")
    return {"records": records, "equality": equality, "joins": joins, "integrity": integrity, "oracle": oracle,
            "seeds": [int(r["generation_seed"]) for r in frozen_manifest["replicates"]],
            "frozen_manifest_sha256": sha256(FROZEN / "manifest.json"), "generator_config_sha256": sha256(config_path),
            "generator_source_sha256": sha256(SIM / "generator" / "generate_scenarios.py"),
            "status": "PASS"}


if __name__ == "__main__":
    result = recover()
    print(json.dumps({"status": result["status"], "replicates": 20, "paired_scenarios": 40,
                      "equality_rows": len(result["equality"]), "prediction_join_rows": len(result["joins"]),
                      "oracle_rows": len(result["oracle"])}, indent=2))
