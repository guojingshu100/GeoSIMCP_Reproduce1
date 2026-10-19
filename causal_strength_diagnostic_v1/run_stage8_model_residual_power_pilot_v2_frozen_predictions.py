"""Stage 8 model-residual CMI pilot using serialized frozen replicate prediction artifacts; no RF refitting."""
from __future__ import annotations

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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = HERE / "stage8_frozen_replicates_v1"
OUT = HERE
META = ["sample_id", "scenario", "split", "local_calibration_role"]
FEATURES = ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"]
MASTER_SEED = 20260910


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(result)
    return result


def stage7_expected() -> dict:
    with (HERE / "stage7_oracle_power_replicates.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 80:
        raise AssertionError("Stage 7 must contain exactly 80 tests")
    result = {(int(r["replicate_index"]), r["scenario"], r["conditioning"]): {"raw_score": float(r["raw_score"]), "p_value": float(r["p_value"]), "reject": r["reject_0_05"] == "True", "generation_seed": int(r["generation_seed"])} for r in rows}
    if len(result) != 80:
        raise AssertionError("duplicate Stage 7 test identity")
    return result


def load_scale(rep: dict, scenario: str) -> tuple[pd.DataFrame, dict]:
    item = rep["scenarios"][scenario]
    paths = {name: ROOT / item[name]["path"] for name in ("predictor_inputs", "scale_fit_labels", "predictions")}
    for name, path in paths.items():
        if sha256(path) != item[name]["sha256"]:
            raise AssertionError(f"frozen artifact hash mismatch: {path}")
    x = pd.read_csv(paths["predictor_inputs"], keep_default_na=False)
    labels = pd.read_csv(paths["scale_fit_labels"], keep_default_na=False)
    pred = pd.read_csv(paths["predictions"], keep_default_na=False)
    if list(x.columns) != META + FEATURES or list(labels.columns) != META + ["y"] or list(pred.columns) != META + ["y_pred"]:
        raise AssertionError("unexpected frozen artifact schema")
    if len(x) != 2000 or len(labels) != 200 or len(pred) != 2000:
        raise AssertionError("unexpected frozen artifact row count")
    if not x["sample_id"].is_unique or not labels["sample_id"].is_unique or not pred["sample_id"].is_unique:
        raise AssertionError("duplicate frozen sample_id")
    if not ((labels["split"] == "local_calibration") & (labels["local_calibration_role"] == "scale_fit")).all():
        raise AssertionError("unauthorized label row in scale-fit labels")
    scale_x = x.loc[(x["split"] == "local_calibration") & (x["local_calibration_role"] == "scale_fit")].copy()
    if len(scale_x) != 200:
        raise AssertionError("unexpected scale-fit predictor-input count")
    frame = scale_x.merge(labels, on=META, how="left", validate="one_to_one", sort=False).merge(pred, on=META, how="left", validate="one_to_one", sort=False)
    if len(frame) != 200 or frame[["y", "y_pred"]].isna().any().any():
        raise AssertionError("missing scale-fit label or frozen prediction")
    if not np.isfinite(frame[["y", "y_pred"]].to_numpy(float)).all():
        raise AssertionError("nonfinite scale-fit label or frozen prediction")
    frame = frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    frame["A_model"] = np.abs(frame["y"].to_numpy(float) - frame["y_pred"].to_numpy(float))
    if not np.isfinite(frame["A_model"].to_numpy(float)).all():
        raise AssertionError("nonfinite A_model")
    table = frame[META + FEATURES + ["y", "y_pred", "A_model"]]
    table_hash = hashlib.sha256(table.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")).hexdigest()
    preflight = {"authorized_scale_fit_rows": 200, "matched_frozen_predictions": 200, "missing_sample_ids": [], "duplicate_sample_ids": 0, "finite_y_count": 200, "finite_y_pred_count": 200, "finite_A_model_count": 200, "A_model_min": float(frame["A_model"].min()), "A_model_median": float(frame["A_model"].median()), "A_model_mean": float(frame["A_model"].mean()), "A_model_max": float(frame["A_model"].max()), "analysis_table_sha256": table_hash, "prediction_artifact": item["predictions"]}
    return frame, preflight


def standardize(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = list(values)
    matrix = np.column_stack([np.asarray(values[n], dtype=float) for n in names])
    if not np.isfinite(matrix).all():
        raise AssertionError("nonfinite CMI input")
    scale = matrix.std(axis=0, ddof=0)
    if np.any(scale == 0):
        raise AssertionError("constant CMI input")
    result = (matrix - matrix.mean(axis=0)) / scale
    return {name: result[:, i] for i, name in enumerate(names)}


def cmi_test(seed: int):
    from tigramite.independence_tests.cmiknn import CMIknn
    return CMIknn(knn=.10, shuffle_neighbors=5, significance="shuffle_test", transform=False, workers=1, sig_samples=99, sig_blocklength=1, seed=seed)


def wilson(k: int) -> list[float]:
    ci = binomtest(k, 20).proportion_ci(confidence_level=.95, method="wilson")
    return [float(ci.low), float(ci.high)]


def stats(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max())}


def summarize(rows: list[dict]) -> dict:
    output = {}
    for condition in ("M0", "Mspace"):
        by_scenario, paired = {}, {}
        for scenario in ("S2", "S4"):
            group = [r for r in rows if r["scenario"] == scenario and r["conditioning"] == condition]
            if len(group) != 20:
                raise AssertionError("incomplete summary group")
            raw = np.asarray([r["raw_score"] for r in group]); p = np.asarray([r["p_value"] for r in group]); reject = sum(r["reject_0_05"] for r in group)
            by_scenario[scenario] = {"A_model_rejection_count": reject, "A_model_rejection_rate": reject / 20, "A_model_wilson_95": wilson(reject), "raw": stats(raw), "p_value": stats(p), "failed_replicates": 0}
            paired[scenario] = {r["replicate_index"]: r for r in group}
        s2 = np.asarray([paired["S2"][i]["raw_score"] for i in range(1, 21)]); s4 = np.asarray([paired["S4"][i]["raw_score"] for i in range(1, 21)]); diff = s2 - s4
        output[condition] = {"primary": condition == "M0", "by_scenario": by_scenario, "S2_raw_greater_count": int(np.count_nonzero(s2 > s4)), "S2_raw_greater_proportion": float(np.mean(s2 > s4)), "paired_raw_difference_S2_minus_S4": stats(diff)}
    return output


def main() -> None:
    if (OUT / "stage8_model_residual_replicates.csv").exists() or (OUT / "stage8_model_residual_summary.json").exists():
        raise FileExistsError("Refusing to overwrite canonical Stage-8 outputs")
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    expected = stage7_expected()
    if len(manifest["replicates"]) != 20:
        raise AssertionError("expected 20 frozen replicate artifacts")
    frames, preflight = {}, {}
    for rep in manifest["replicates"]:
        index, seed = int(rep["replicate_index"]), int(rep["generation_seed"])
        if expected[(index, "S2", "M0")]["generation_seed"] != seed:
            raise AssertionError("frozen replicate seed does not match Stage-7 design")
        for scenario in ("S2", "S4"):
            frame, check = load_scale(rep, scenario)
            frames[(index, scenario)] = frame; preflight[f"{index:02d}_{scenario}"] = check
        paired_columns = ["sample_id", "split", "local_calibration_role"] + FEATURES
        pd.testing.assert_frame_equal(frames[(index, "S2")][paired_columns], frames[(index, "S4")][paired_columns], check_dtype=False)
    started, rows = time.perf_counter(), []
    for index in range(1, 21):
        offset = 0
        for scenario in ("S2", "S4"):
            frame = frames[(index, scenario)]
            values = standardize({name: frame[name].to_numpy(float) for name in ("x2", "s_x", "s_y", "A_model")})
            for condition, names, xyz in (("M0", ["x2", "A_model"], [0, 1]), ("Mspace", ["x2", "A_model", "s_x", "s_y"], [0, 1, 2, 2])):
                array = np.vstack([values[name] for name in names]); labels = np.asarray(xyz, dtype=int)
                shuffle_seed = MASTER_SEED + index * 10 + offset; offset += 1
                test = cmi_test(shuffle_seed)
                raw = float(test.get_dependence_measure(array.copy(), labels.copy()))
                p_value = float(test.get_shuffle_significance(array.copy(), labels.copy(), raw))
                stage7 = expected[(index, scenario, condition)]
                rows.append({"replicate_index": index, "generation_seed": stage7["generation_seed"], "shuffle_seed": shuffle_seed, "scenario": scenario, "conditioning": condition, "n": 200, "k": 20, "raw_score": raw, "p_value": p_value, "reject_0_05": p_value <= .05, "stage7_oracle_raw_score": stage7["raw_score"], "stage7_oracle_p_value": stage7["p_value"], "model_minus_oracle_raw_score": raw - stage7["raw_score"], "frozen_prediction_sha256": preflight[f"{index:02d}_{scenario}"]["prediction_artifact"]["sha256"], "analysis_table_sha256": preflight[f"{index:02d}_{scenario}"]["analysis_table_sha256"]})
        if index % 5 == 0:
            print(f"PROGRESS: {index}/20 replicate seeds complete", flush=True)
    if len(rows) != 80:
        raise AssertionError("expected 80 CMI tests")
    fields = list(rows[0])
    with (OUT / "stage8_model_residual_replicates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    result = {"stage": "Stage 8", "execution_status": "PASS", "source_script": str(Path(__file__).relative_to(ROOT)), "interpreter": sys.executable, "packages": {p: importlib.metadata.version(p) for p in ("tigramite", "numpy", "scipy", "pandas", "scikit-learn")}, "frozen_replicate_manifest": {"path": str((FROZEN / "manifest.json").relative_to(ROOT)), "sha256": sha256(FROZEN / "manifest.json")}, "base_frozen_prediction_artifacts": manifest["base_artifacts"], "configuration": {"target": "A_model=abs(y-y_pred_frozen)", "scenarios": ["S2", "S4"], "paired_replicates": 20, "n": 200, "k": 20, "knn_fraction": .10, "sig_samples": 99, "shuffle_neighbors": 5, "sig_blocklength": 1, "alpha": .05, "conditioning": {"M0": ["x2", "A_model"], "Mspace": ["x2", "A_model", "s_x", "s_y"]}, "shuffle_seed_formula": "20260910 + replicate_index*10 + test_offset", "predictor_refit_at_CMI_time": False}, "label_access_contract": {"labels_read": ["local_calibration/scale_fit only"], "prohibited_label_reads": ["validation", "final_calibration", "test"], "status": "PASS"}, "preflight": preflight, "summary": summarize(rows), "replicates": rows, "elapsed_time_seconds": time.perf_counter() - started}
    (OUT / "stage8_model_residual_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("PRODUCTION PASS: 20/20 replicate seeds complete", flush=True)


if __name__ == "__main__":
    main()
