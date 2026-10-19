"""Freeze serialized, label-restricted Stage-8 inputs and RF predictions for 20 prescribed seeds."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIM = ROOT / "simulation_pilot_20260908"
OUT = HERE / "stage8_frozen_replicates_v1"
META = ["sample_id", "scenario", "split", "local_calibration_role"]
FEATURES = ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(result)
    return result


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")


def read_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != META + ["y"]:
        raise AssertionError(f"unexpected authorized label schema: {path}")
    if not np.isfinite(frame["y"].to_numpy(float)).all():
        raise AssertionError("nonfinite authorized label")
    if not (frame["split"] == "local_calibration").all() and path.name.endswith("scale_fit_labels.csv"):
        raise AssertionError("scale-fit labels include non-local-calibration row")
    return frame


def seeds() -> list[tuple[int, int]]:
    with (HERE / "stage7_oracle_power_replicates.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for index in range(1, 21):
        matches = {int(row["generation_seed"]) for row in rows if int(row["replicate_index"]) == index}
        if len(matches) != 1:
            raise AssertionError(f"unexpected Stage-7 seed mapping for {index}")
        output.append((index, matches.pop()))
    return output


def base_artifact_check() -> dict:
    output = {}
    oracle_columns = {"y", "mu_true", "sigma_true", "epsilon", "g_true", "eta"}
    for scenario in ("S2", "S4"):
        path = SIM / "predictor_outputs" / f"{scenario}_predictions.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        if len(frame) != 2000 or not frame["sample_id"].is_unique or "y_pred" not in frame:
            raise AssertionError(f"invalid historical frozen prediction artifact: {scenario}")
        if oracle_columns.intersection(frame.columns):
            raise AssertionError(f"historical prediction contains target/oracle column: {scenario}")
        output[scenario] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(frame), "columns": list(frame.columns)}
    return output


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen replicate directory: {OUT}")
    config = json.loads((SIM / "generator" / "config_v1.json").read_text(encoding="utf-8"))
    predictor_config = json.loads((SIM / "predictor" / "predictor_config_v1.json").read_text(encoding="utf-8"))
    if list(predictor_config["authorized_features"]) != FEATURES or predictor_config["preprocessing"]["predictor_scaler"] is not None:
        raise AssertionError("unexpected frozen predictor feature protocol")
    generator = module("stage8_freeze_generator", SIM / "generator" / "generate_scenarios.py")
    predictor = module("stage8_freeze_predictor", SIM / "predictor" / "fit_predictors.py")
    OUT.mkdir(parents=True)
    manifest = {"stage": "Stage 8 frozen replicate preparation", "version": "v1", "base_artifacts": base_artifact_check(), "generator_config_sha256": sha256(SIM / "generator" / "config_v1.json"), "predictor_config_sha256": sha256(SIM / "predictor" / "predictor_config_v1.json"), "predictor_source_sha256": sha256(SIM / "predictor" / "fit_predictors.py"), "feature_columns": FEATURES, "predictor_configuration": predictor_config["model_configuration"], "replicates": []}
    for index, seed in seeds():
        local = copy.deepcopy(config)
        local["random_seeds"]["base_variables"] = seed
        scenarios = generator.generate_all_scenarios(local)
        rep = OUT / f"replicate_{index:02d}_seed_{seed}"
        rep.mkdir()
        entry = {"replicate_index": index, "generation_seed": seed, "scenarios": {}}
        for scenario in ("S2", "S4"):
            observed, _ = scenarios[scenario]
            predictor_input = observed[META + FEATURES].copy()
            train = observed.loc[observed["split"] == "train", META + ["y"]].copy()
            scale = observed.loc[(observed["split"] == "local_calibration") & (observed["local_calibration_role"] == "scale_fit"), META + ["y"]].copy()
            if len(predictor_input) != 2000 or len(train) != 1000 or len(scale) != 200:
                raise AssertionError("unexpected frozen replicate rows")
            if not (train["split"] == "train").all() or not ((scale["split"] == "local_calibration") & (scale["local_calibration_role"] == "scale_fit")).all():
                raise AssertionError("authorized label contract violation")
            input_path, train_path, scale_path = (rep / f"{scenario}_{suffix}" for suffix in ("predictor_inputs.csv", "train_labels.csv", "scale_fit_labels.csv"))
            write_csv(predictor_input, input_path); write_csv(train, train_path); write_csv(scale, scale_path)
            x = pd.read_csv(input_path, keep_default_na=False)
            y_train = pd.read_csv(train_path, keep_default_na=False)
            pd.testing.assert_frame_equal(x.loc[x["split"] == "train", META].reset_index(drop=True), y_train[META].reset_index(drop=True), check_dtype=False)
            model = predictor.make_model(predictor_config)
            model.fit(x.loc[x["split"] == "train", FEATURES].to_numpy(float), y_train["y"].to_numpy(float))
            pred = x[META].copy()
            pred["y_pred"] = predictor.predict_in_fixed_tree_order(model, x[FEATURES].to_numpy(float))
            if not np.isfinite(pred["y_pred"].to_numpy(float)).all():
                raise AssertionError("nonfinite frozen prediction")
            prediction_path = rep / f"{scenario}_predictions.csv"
            write_csv(pred, prediction_path)
            final_scale = read_labels(scale_path).merge(pred, on=META, how="left", validate="one_to_one", sort=False)
            if len(final_scale) != 200 or final_scale["y_pred"].isna().any() or not np.isfinite(final_scale[["y", "y_pred"]].to_numpy(float)).all():
                raise AssertionError("scale-fit prediction alignment failure")
            final_scale["A_model"] = np.abs(final_scale["y"] - final_scale["y_pred"])
            entry["scenarios"][scenario] = {"predictor_inputs": {"path": str(input_path.relative_to(ROOT)), "sha256": sha256(input_path), "rows": 2000}, "train_labels": {"path": str(train_path.relative_to(ROOT)), "sha256": sha256(train_path), "rows": 1000}, "scale_fit_labels": {"path": str(scale_path.relative_to(ROOT)), "sha256": sha256(scale_path), "rows": 200}, "predictions": {"path": str(prediction_path.relative_to(ROOT)), "sha256": sha256(prediction_path), "rows": 2000}, "preflight": {"matched_predictions": 200, "missing_sample_ids": [], "duplicate_sample_ids": 0, "finite_y": 200, "finite_y_pred": 200, "finite_A_model": 200, "A_model_min": float(final_scale["A_model"].min()), "A_model_median": float(final_scale["A_model"].median()), "A_model_mean": float(final_scale["A_model"].mean()), "A_model_max": float(final_scale["A_model"].max())}}
        manifest["replicates"].append(entry)
        if index % 5 == 0:
            print(f"PREPARATION PROGRESS: {index}/20 frozen replicate seeds complete", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("PREPARATION PASS: 20/20 frozen replicate seeds complete", flush=True)


if __name__ == "__main__":
    main()
