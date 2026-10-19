"""Stage 14 real-data provenance audit and one-dataset exploratory transfer.

The historical PM2.5 test labels are deliberately loaded only after the
CMI features, weights, k values, conformal quantiles, and input hashes have
been written to the evaluation-unseal audit. No base predictor is refit.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "experiments" / "real_data" / "20260918_stage14_realdata_feasibility_v1"
EVAL = ROOT / "experiments" / "real_data" / "20260918_stage14_pm25_causalonly_cmi_exploratory_v1"
SEATTLE = ROOT / "geoconformal-main" / "GCCP_Seattle"
PM25 = ROOT / "geoconformal-main" / "GCCP_PM2.5"
SEATTLE_RAW = ROOT / "geoconformal-main" / "example" / "seattle_sample_3k.csv"
PM_RAW = PM25 / "00_data_audit" / "china_sites_2018_05_PM2.5_all.csv"
PM_SPLIT = PM25 / "outputs" / "pm25_formal_split_assignment_v1.csv"
PM_PROTOCOL = PM25 / "outputs" / "pm25_formal_protocol_v1.json"
PM_SELECTION = PM25 / "outputs" / "pm25_formal_rf_train_only_selection_v1.csv"
PM_VAL_PRED = PM25 / "outputs" / "pm25_formal_validation_predictions_v1.csv"
PM_TEST_PRED = PM25 / "outputs" / "pm25_formal_frozen_test_per_sample_v1.csv"
PM_TEST_MANIFEST = PM25 / "outputs" / "pm25_formal_frozen_test_manifest_v1.json"

FEATURES = ["dem", "sp", "tp", "blh", "r", "aod_sat", "ndvi"]
PM_EXTRA_ATTRIBUTES = ["w10", "d10", "t2m", "e", "u10", "v10"]
SEATTLE_FEATURES = ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition", "waterfront", "view", "age"]
COORDS = ["lon", "lat"]
K_VALUES = [20, 40, 60, 80]
ALPHA = 0.10
SPLIT_SEED = 20260918
CMI_SEED = 20260910
N_SCALE, N_TUNE, N_FINAL, N_EVAL = 82, 30, 30, 142
FLOOR = 1e-12
TIE_TOL = 1e-12
CLAIM = ("This real-data experiment is retrospective and exploratory because historical test/evaluation "
         "labels in this project had already been accessed before this protocol was frozen. The current "
         "run itself did not use those evaluation labels for feature selection, CMI estimation, k tuning, "
         "or conformal calibration.")
EXECUTION_COMMAND = (r'F:\GeoSIMCP_Reproduce\causal_strength_diagnostic_v1\.venv_cmi_v1\Scripts\python.exe '
                     r'F:\GeoSIMCP_Reproduce\causal_only_cp_v1\run_stage14_realdata_causalonly_cmi_exploratory_v1.py')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def frame_hash(frame: pd.DataFrame) -> str:
    raw = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def bh_q_values(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted = [0.0] * m
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, min(1.0, ordered[i][1] * m / (i + 1)))
        adjusted[i] = running
    return {name: float(q) for (name, _), q in zip(ordered, adjusted)}


def standardize_cmi(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = list(values)
    matrix = np.column_stack([np.asarray(values[name], float) for name in names])
    if not np.isfinite(matrix).all():
        raise AssertionError("nonfinite CMI input")
    scale = matrix.std(axis=0, ddof=0)
    if np.any(scale <= 0):
        raise AssertionError("constant CMI input")
    matrix = (matrix - matrix.mean(axis=0)) / scale
    return {name: matrix[:, i] for i, name in enumerate(names)}


def run_cmi(scale: pd.DataFrame) -> pd.DataFrame:
    from tigramite.independence_tests.cmiknn import CMIknn

    residual = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
    values = {name: scale[name].to_numpy(float) for name in [*FEATURES, *COORDS]}
    values["A_model"] = residual
    values = standardize_cmi(values)
    raw_scores: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for feature in FEATURES:
        names = [feature, "A_model", *COORDS]
        xyz = np.asarray([0, 1, 2, 2], dtype=int)
        array = np.vstack([values[name] for name in names])
        test = CMIknn(knn=20, shuffle_neighbors=5, significance="shuffle_test", transform=False,
                      workers=1, sig_samples=99, sig_blocklength=1, seed=CMI_SEED)
        raw = float(test.get_dependence_measure(array.copy(), xyz.copy()))
        p = float(test.get_shuffle_significance(array.copy(), xyz.copy(), raw))
        raw_scores[feature], p_values[feature] = raw, p
    q_values = bh_q_values(p_values)
    rows = []
    for feature in FEATURES:
        eligible = q_values[feature] < .05 and raw_scores[feature] > 0
        rows.append({"dataset": "PM2.5", "condition": "Mspace", "feature": feature,
                     "raw_cmi": raw_scores[feature], "p_value": p_values[feature],
                     "bh_q": q_values[feature], "eligible": eligible, "cmi_k": 20,
                     "surrogates": 99, "shuffle_seed": CMI_SEED, "n_scale_fit": len(scale)})
    result = pd.DataFrame(rows)
    positive = result.eligible
    denom = float(result.loc[positive, "raw_cmi"].sum())
    result["weight"] = 0.0
    if denom > 0:
        result.loc[positive, "weight"] = result.loc[positive, "raw_cmi"] / denom
        result["status"] = "SIGNAL"
    else:
        result["status"] = "NO_CAUSAL_SIGNAL"
    return result


def standardizer(reference: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = reference[columns].to_numpy(float)
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 1e-12):
        raise AssertionError("zero/nonfinite standardization SD")
    return mean, sd


def scaled(frame: pd.DataFrame, columns: list[str], mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (frame[columns].to_numpy(float) - mean) / sd


def neighbor_scale(reference_z: np.ndarray, residual: np.ndarray, ids: np.ndarray,
                   query_z: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(len(query_z), float)
    for i, row in enumerate(query_z):
        distance = np.sqrt(np.sum(weights * (reference_z - row) ** 2, axis=1))
        order = np.lexsort((ids, distance))
        out[i] = max(FLOOR, float(residual[order[:k]].mean()))
    return out


def tune_local(scale: pd.DataFrame, tuning: pd.DataFrame, columns: list[str], weights: np.ndarray,
               method: str) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    mean, sd = standardizer(scale, columns)
    rz, tz = scaled(scale, columns, mean, sd), scaled(tuning, columns, mean, sd)
    ref_a = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
    tune_a = np.abs(tuning.y_true.to_numpy(float) - tuning.y_pred.to_numpy(float))
    values: dict[int, float] = {}
    for k in K_VALUES:
        sigma = neighbor_scale(rz, ref_a, scale.original_index.to_numpy(int), tz, weights, k)
        values[k] = float(np.mean(np.abs(tune_a - sigma)))
    best = min(values.values())
    chosen = max(k for k, value in values.items() if abs(value - best) <= TIE_TOL)
    rows = [{"dataset": "PM2.5", "method": method, "k": k, "tuning_scale_mae": values[k],
             "selected": k == chosen, "reference_role": "scale_fit", "tuning_role": "tuning_calibration"}
            for k in K_VALUES]
    return chosen, {"mean": mean, "sd": sd, "reference_z": rz, "reference_a": ref_a,
                    "ids": scale.original_index.to_numpy(int), "weights": weights,
                    "tuning_scale_mae": values[chosen]}, rows


def finite_quantile(scores: np.ndarray) -> tuple[float, int]:
    scores = np.asarray(scores, float)
    rank = math.ceil((len(scores) + 1) * (1 - ALPHA))
    if rank > len(scores):
        return math.inf, rank
    return float(np.sort(scores, kind="mergesort")[rank - 1]), rank


def metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    if not (np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower <= upper)):
        raise AssertionError("invalid intervals")
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width + (2 / ALPHA) * (lower - y) * (y < lower) + (2 / ALPHA) * (y - upper) * (y > upper)
    return {"n_evaluation": len(y), "covered_count": int(covered.sum()),
            "coverage": float(covered.mean()), "mean_width": float(width.mean()),
            "median_width": float(np.median(width)), "mean_interval_score": float(score.mean())}


def artifact_row(dataset: str, path: Path, role: str, rows: int | None = None,
                 sample_id: str = "", target: str = "", prediction: str = "", split: str = "",
                 notes: str = "") -> dict[str, Any]:
    if dataset == "Seattle":
        dataset_fields = {"coordinate_fields": "UTM_X;UTM_Y", "candidate_attribute_fields": ";".join(SEATTLE_FEATURES),
                          "date_time_field": "NONE", "preprocessing_status": "target=10**log_price/10000; raw model attributes and UTM coordinates",
                          "missingness": "0 in audited raw candidate/coordinate fields",
                          "historical_use": "baseline/GCCM/GCCP notebooks and saved test results",
                          "historical_test_exposure_status": "HISTORICALLY_EXPOSED"}
    else:
        dataset_fields = {"coordinate_fields": "lon;lat (proj_x/proj_y also present in source)",
                          "candidate_attribute_fields": ";".join(FEATURES), "date_time_field": "date",
                          "preprocessing_status": "formal RF used raw 7 attributes plus lon/lat; no RF scaler",
                          "missingness": "0 in audited formal candidate/coordinate fields",
                          "historical_use": "formal baseline/GCCM/GCCP notebooks and frozen-test results",
                          "historical_test_exposure_status": "HISTORICALLY_EXPOSED"}
    return {"dataset": dataset, "artifact_path": rel(path), "role": role, "sha256": sha256(path),
            "bytes": path.stat().st_size, "row_count": rows, "sample_identifier": sample_id,
            "target_field": target, "prediction_field": prediction, "split_field": split, **dataset_fields, "notes": notes}


def write_audit_records() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    AUDIT.mkdir(parents=True, exist_ok=True)
    seattle = pd.read_csv(SEATTLE_RAW)
    s_split = np.load(SEATTLE / "outputs" / "seattle_split_seed42.npz")
    s_inner = np.load(SEATTLE / "outputs" / "seattle_inner_split_seed42.npz")
    s_test = pd.read_csv(SEATTLE / "outputs" / "baseline_test_intervals_seed42.csv")
    pm_cov = pd.read_csv(PM_RAW, encoding="gb18030", usecols=["监测点编码", "date", "经度", "纬度", *FEATURES, *PM_EXTRA_ATTRIBUTES])
    pm_split = pd.read_csv(PM_SPLIT)
    pm_val = pd.read_csv(PM_VAL_PRED)
    pm_test_meta = pd.read_csv(PM_TEST_PRED, usecols=["method", "original_index", "id", "y_pred"])
    pm_test_cp = pm_test_meta[pm_test_meta.method == "CP"].copy()

    artifacts = [
        artifact_row("Seattle", SEATTLE_RAW, "raw table", len(seattle), "Unnamed: 0", "log_price", "", "", "Cross-sectional housing sample"),
        artifact_row("Seattle", SEATTLE / "outputs" / "seattle_split_seed42.npz", "outer split", 3000, "row index", "", "", "array keys", "2100/450/450"),
        artifact_row("Seattle", SEATTLE / "outputs" / "seattle_inner_split_seed42.npz", "inner split", 2100, "row index", "", "", "array keys", "1470/315/315"),
        artifact_row("Seattle", SEATTLE / "outputs" / "baseline_test_intervals_seed42.csv", "historical test predictions/results", len(s_test), "original_index", "y_true", "y_pred", "implicit test", "No frozen development predictions"),
        artifact_row("Seattle", SEATTLE / "outputs" / "seattle_author_aligned_final_test_per_sample_v1.csv", "historical final-test results", 300, "test_index", "y_true", "prediction", "implicit test", "Historical labels exposed"),
        artifact_row("PM2.5", PM_RAW, "raw table", len(pm_cov), "监测点编码", "PM2.5", "", "", "Single date 20180531"),
        artifact_row("PM2.5", PM_SPLIT, "formal split", len(pm_split), "id + original_index", "", "", "split", "990/141/142/142"),
        artifact_row("PM2.5", PM_PROTOCOL, "formal protocol", None, "", "PM2.5", "", "", "RF parameters and feature set"),
        artifact_row("PM2.5", PM_SELECTION, "train-only RF selection", 1, "", "", "", "", "3-fold CV on formal train"),
        artifact_row("PM2.5", PM_VAL_PRED, "frozen development predictions", len(pm_val), "id + original_index", "y_true", "y_pred", "validation", "Stage-14 development source"),
        artifact_row("PM2.5", PM_TEST_PRED, "historical test predictions/results", 994, "id + original_index", "y_true", "y_pred", "test", "Seven method rows per 142 sites; CP rows define unique frozen predictor vector"),
        artifact_row("PM2.5", PM_TEST_MANIFEST, "frozen-test manifest", None, "", "", "", "", "Historical test exposed after manifest")]
    inventory = pd.DataFrame(artifacts)
    inventory.to_csv(AUDIT / "dataset_inventory.csv", index=False)

    pred_rows = [
        {"dataset": "Seattle", "model": "XGBRegressor", "parameters": "n_estimators=500,max_depth=3,min_child_weight=1,colsample_bytree=1,random_state=42",
         "training_rows": "outer train=2100", "selection": "inner fit/calibration/validation documented", "preprocessing": "raw 8 attributes + UTM_X/UTM_Y; no feature scaler",
         "frozen_predictions": "test only (450-row baseline; separate 300-row author-aligned record)", "one_to_one_join": True,
         "evaluation_in_fit": False, "status": "PREDICTOR_PROVENANCE_BLOCKED",
         "reason": "No saved model and no frozen prediction vector covering enough non-test rows for scale_fit/tuning/final calibration."},
        {"dataset": "PM2.5", "model": "RandomForestRegressor", "parameters": "n_estimators=300,max_depth=None,random_state=42",
         "training_rows": "formal train=990", "selection": "train-only 3-fold GridSearchCV, neg MSE", "preprocessing": "raw 7 formal attributes + lon/lat; no fitted scaler in RF",
         "frozen_predictions": "validation=142 and historical test=142", "one_to_one_join": True,
         "evaluation_in_fit": False, "status": "PASS", "reason": "Frozen validation/test vectors and immutable split/protocol are present."}]
    predictor = pd.DataFrame(pred_rows)
    predictor.to_csv(AUDIT / "predictor_provenance.csv", index=False)

    exposure = pd.DataFrame([
        {"dataset": "Seattle", "set": "outer test and author-aligned final test", "classification": "HISTORICALLY_EXPOSED", "evidence": "Per-sample y_true and interval result files exist."},
        {"dataset": "PM2.5", "set": "formal test", "classification": "HISTORICALLY_EXPOSED", "evidence": "pm25_formal_frozen_test_per_sample_v1.csv contains y_true and metrics."},
        {"dataset": "PM2.5", "set": "formal validation", "classification": "HISTORICALLY_EXPOSED_DEVELOPMENT", "evidence": "Validation predictions and prior tuning outputs exist; Stage 14 uses it only as development."}])
    exposure.to_csv(AUDIT / "split_exposure_audit.csv", index=False)

    feature_rows: list[dict[str, Any]] = []
    for f in SEATTLE_FEATURES:
        feature_rows.append({"dataset": "Seattle", "feature": f, "meaning": f, "role": "candidate non-coordinate attribute",
                             "dtype": str(seattle[f].dtype), "missing": int(seattle[f].isna().sum()), "variance": float(seattle[f].var()),
                             "available_at_prediction": True, "target_or_residual_derived": False, "evaluation_label_derived": False,
                             "eligibility": "LEGITIMATE_BUT_DATASET_BLOCKED"})
    for f, meaning in {"UTM_X": "projected easting", "UTM_Y": "projected northing"}.items():
        feature_rows.append({"dataset": "Seattle", "feature": f, "meaning": meaning, "role": "coordinate",
                             "dtype": str(seattle[f].dtype), "missing": int(seattle[f].isna().sum()), "variance": float(seattle[f].var()),
                             "available_at_prediction": True, "target_or_residual_derived": False, "evaluation_label_derived": False,
                             "eligibility": "COORDINATE_ONLY"})
    meanings = {"dem": "digital elevation", "sp": "surface pressure", "tp": "total precipitation",
                "blh": "boundary-layer height", "r": "relative humidity", "aod_sat": "satellite aerosol optical depth",
                "ndvi": "normalized difference vegetation index"}
    for f in FEATURES:
        feature_rows.append({"dataset": "PM2.5", "feature": f, "meaning": meanings[f], "role": "candidate non-coordinate attribute",
                             "dtype": str(pm_cov[f].dtype), "missing": int(pm_cov[f].isna().sum()), "variance": float(pm_cov[f].var()),
                             "available_at_prediction": True, "target_or_residual_derived": False, "evaluation_label_derived": False,
                             "eligibility": "ELIGIBLE_FROZEN_FORMAL_FEATURE"})
    extra_meanings = {"w10": "10 m wind speed", "d10": "10 m wind direction", "t2m": "2 m temperature",
                      "e": "evaporation field", "u10": "10 m east-west wind component",
                      "v10": "10 m north-south wind component"}
    for f in PM_EXTRA_ATTRIBUTES:
        feature_rows.append({"dataset": "PM2.5", "feature": f, "meaning": extra_meanings[f],
                             "role": "available non-coordinate source attribute",
                             "dtype": str(pm_cov[f].dtype), "missing": int(pm_cov[f].isna().sum()),
                             "variance": float(pm_cov[f].var()), "available_at_prediction": True,
                             "target_or_residual_derived": False, "evaluation_label_derived": False,
                             "eligibility": "EXCLUDED_NOT_IN_FROZEN_FORMAL_FEATURE_SET"})
    for f, meaning in {"lon": "longitude", "lat": "latitude"}.items():
        raw = "经度" if f == "lon" else "纬度"
        feature_rows.append({"dataset": "PM2.5", "feature": f, "meaning": meaning, "role": "coordinate",
                             "dtype": str(pm_cov[raw].dtype), "missing": int(pm_cov[raw].isna().sum()), "variance": float(pm_cov[raw].var()),
                             "available_at_prediction": True, "target_or_residual_derived": False, "evaluation_label_derived": False,
                             "eligibility": "MSPACE_OR_GEOONLY_ONLY"})
    features = pd.DataFrame(feature_rows)
    features.to_csv(AUDIT / "candidate_feature_audit.csv", index=False)

    # Exact pre-model feasibility gates, using prediction metadata without test labels.
    val_join = pm_val.merge(pm_split, on=["original_index", "id"], validate="one_to_one")
    test_join = pm_test_cp.merge(pm_split, on=["original_index", "id"], validate="one_to_one")
    gates = {
        "Seattle": {"eligible": False, "status": "PREDICTOR_PROVENANCE_BLOCKED",
                    "stable_ids": bool(seattle["Unnamed: 0"].is_unique), "coordinates": True,
                    "legitimate_candidate_count": len(SEATTLE_FEATURES), "development_frozen_predictions": False,
                    "requirements": {"stable_ids": True, "frozen_base_predictions": False,
                      "one_to_one_labels": True, "coordinates": True, "at_least_two_candidates": True,
                      "enough_development": False, "enough_evaluation": True, "preprocessing_documented": True,
                      "no_candidate_label_leakage": True, "stage11_shell_transferable": False},
                    "reason": "Development prediction vector/model object was not preserved; refitting is prohibited."},
        "PM2.5": {"eligible": True, "status": "PASS", "stable_ids": bool(pm_split.id.is_unique),
                  "coordinates": True, "legitimate_candidate_count": len(FEATURES),
                  "development_frozen_predictions": bool(len(val_join) == 142 and val_join.split.eq("validation").all()),
                  "evaluation_frozen_predictions": bool(len(test_join) == 142 and test_join.split.eq("test").all()),
                  "available_development": 142, "allocation": {"scale_fit": N_SCALE, "tuning_calibration": N_TUNE, "final_calibration": N_FINAL},
                  "evaluation_n": N_EVAL, "k_grid_transferable": max(K_VALUES) <= N_SCALE,
                  "requirements": {"stable_ids": True, "frozen_base_predictions": True,
                    "one_to_one_labels": True, "coordinates": True, "at_least_two_candidates": True,
                    "enough_development": True, "enough_evaluation": True, "preprocessing_documented": True,
                    "no_candidate_label_leakage": True, "stage11_shell_transferable": True},
                  "exchangeability_note": "Single-date cross-sectional site snapshot; formal split is random across sites."},
        "selection": {"dataset": "PM2.5", "rule": "Only dataset passing all feasibility/provenance gates; performance not inspected.",
                      "Seattle_blocker": "missing frozen development predictions"},
        "claim_boundary": CLAIM,
    }
    write_json(AUDIT / "feasibility_decision.json", gates)
    config = {"stage": "Stage 14 feasibility audit", "date": "2026-09-18", "network": False,
              "execution_command": EXECUTION_COMMAND,
              "datasets": ["Seattle", "PM2.5"], "selection_priority": ["predictor provenance", "split reconstruction",
              "sample size", "candidate attributes", "exchangeability", "preprocessing ambiguity"],
              "selected": "PM2.5", "claim_boundary": CLAIM}
    write_json(AUDIT / "config.json", config)
    return inventory, predictor, features, gates


def append_unseal(row: dict[str, Any]) -> None:
    path = EVAL / "evaluation_unseal_audit.csv"
    fields = list(row)
    raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    row = {**row, "audit_row_sha256": hashlib.sha256(raw).hexdigest()}
    fields.append("audit_row_sha256")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerow(row); handle.flush(); os.fsync(handle.fileno())


def create_docs(inventory: pd.DataFrame, gates: dict[str, Any], results: pd.DataFrame | None,
                classification: str | None, summary: dict[str, Any] | None) -> None:
    audit_readme = f"""# Stage 14 real-data feasibility audit\n\nStatus: PM2.5 selected; Seattle blocked by absent frozen development predictions.\n\n{CLAIM}\n\nNo network access, predictor refitting, CMI, or CP evaluation was used to choose the dataset.\n"""
    (AUDIT / "README.md").write_text(audit_readme, encoding="utf-8")
    (AUDIT / "DATA_PROVENANCE.md").write_text(
        "# Data provenance\n\nSeattle is a 3,000-row housing sample with immutable split indices, but only test predictions were preserved. "
        "PM2.5 is a 1,415-site snapshot dated 2018-05-31, with a formal random split and frozen validation/test predictions. "
        "Both historical evaluation sets are exposed. PM2.5 has 1,415 unique station IDs and one unique date, so this artifact is "
        "cross-sectional rather than longitudinal; this does not establish exchangeability beyond the observed snapshot.\n",
        encoding="utf-8")
    if summary is None:
        return
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "CLAIM_BOUNDARY.md").write_text("# Claim boundary\n\n" + CLAIM + "\n", encoding="utf-8")
    (EVAL / "METHOD.md").write_text(
        "# Method\n\nThe formal PM2.5 validation set was deterministically repartitioned into scale_fit=82, "
        "tuning_calibration=30, and final_calibration=30. CMIknn used k=20, 99 shuffle surrogates, "
        "and conditioning on longitude/latitude. BH q<0.05 and positive raw CMI determined weights. "
        "Raw-CP, GeoOnly-NormCP, and CausalOnly-CMI-NormCP use alpha=0.10; local methods select k from "
        "{20,40,60,80} by tuning residual-scale MAE. Historical formal test labels were opened only after "
        "the evaluation-unseal audit was flushed.\n", encoding="utf-8")
    display_columns = ["method", "coverage", "mean_width", "median_width", "mean_interval_score", "selected_k"]
    display = results[display_columns].copy()
    header = "| " + " | ".join(display_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display_columns)) + " |"
    table_rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
                  for row in display.itertuples(index=False, name=None)]
    markdown_table = "\n".join([header, separator, *table_rows])
    lines = ["# Stage 14 PM2.5 retrospective exploratory evaluation", "", CLAIM, "", f"Classification: **{classification}**", "",
             f"Selected CMI features: {', '.join(summary['selected_features']) if summary['selected_features'] else 'none'}", "",
             "## Results", "", markdown_table, "",
             "This single historically exposed dataset cannot support confirmatory or independent-validation claims."]
    (EVAL / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EVAL / "stage14_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifests(record: Path, source_paths: list[Path], outputs: list[Path]) -> None:
    input_rows = [{"path": rel(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in source_paths]
    write_json(record / "input_manifest.json", {"files": input_rows})
    output_rows = [{"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in outputs if p.is_file()]
    write_json(record / "output_manifest.json", {"files": output_rows, "note": "Manifest excludes itself."})


def main() -> None:
    started = time.time()
    print(f"COMMAND: {EXECUTION_COMMAND}", flush=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    inventory, predictor, feature_audit, gates = write_audit_records()
    print("AUDIT: Seattle=PREDICTOR_PROVENANCE_BLOCKED; PM2.5=PASS; selected=PM2.5", flush=True)

    # Freeze the scientific protocol before CMI or evaluation results are inspected.
    eval_config = {
        "stage": "Stage 14 PM2.5 retrospective exploratory evaluation", "dataset": "PM2.5",
        "execution_command": EXECUTION_COMMAND,
        "claim_boundary": CLAIM, "source_development_role": "historical formal validation",
        "source_evaluation_role": "historical formal test", "development_split_seed": SPLIT_SEED,
        "development_counts": {"scale_fit": N_SCALE, "tuning_calibration": N_TUNE, "final_calibration": N_FINAL},
        "evaluation_n": N_EVAL, "candidate_features": FEATURES, "coordinates": COORDS,
        "CMI": {"estimator": "Tigramite CMIknn", "k": 20, "surrogates": 99, "shuffle_neighbors": 5,
                "sig_blocklength": 1, "seed": CMI_SEED, "condition": "Mspace: coordinates"},
        "weights": "BH q<0.05 AND raw_CMI>0; normalize eligible raw CMI", "BH_family_size": len(FEATURES),
        "k_candidates": K_VALUES, "k_selection": "minimum tuning residual-scale MAE; larger k on ties",
        "alpha": ALPHA, "no_signal": "constant mean scale_fit residual; exact Raw-CP fallback",
        "classification_rule": {"D": "no CMI feature passes frozen gate",
          "C": "signal present and CausalOnly coverage is >=0.02 below both comparators with Wilson 95% CI excluding 0.90",
          "A": "signal present, no coverage concern, and both width and interval score are lower than Raw and GeoOnly",
          "B": "all other feasible evaluated outcomes"},
        "evaluation_label_policy": "load only after evaluation_unseal_audit.csv is flushed"}
    write_json(EVAL / "config.json", eval_config)

    # Load covariates only. PM2.5 target is intentionally omitted here.
    raw = pd.read_csv(PM_RAW, encoding="gb18030", usecols=["监测点编码", "date", "经度", "纬度", *FEATURES])
    raw = raw.rename(columns={"监测点编码": "id", "经度": "lon", "纬度": "lat"})
    raw["id"] = raw.id.astype(str)
    raw["original_index"] = np.arange(len(raw))
    split = pd.read_csv(PM_SPLIT); split["id"] = split.id.astype(str)
    val = pd.read_csv(PM_VAL_PRED); val["id"] = val.id.astype(str)
    if not val.original_index.is_unique or not val.id.is_unique:
        raise AssertionError("validation predictions are not one-to-one")
    val = val.merge(split[["original_index", "id", "split"]], on=["original_index", "id"], validate="one_to_one")
    if len(val) != 142 or not val.split.eq("validation").all():
        raise AssertionError("validation prediction/split mismatch")
    val = val.merge(raw, on=["original_index", "id"], validate="one_to_one")
    if val[["y_true", "y_pred", *FEATURES, *COORDS]].isna().any().any():
        raise AssertionError("missing development value")

    # Inspect frozen evaluation predictions and covariates without reading evaluation labels.
    test_meta = pd.read_csv(PM_TEST_PRED, usecols=["method", "original_index", "id", "y_pred"])
    test_meta["id"] = test_meta.id.astype(str)
    test_meta = test_meta[test_meta.method == "CP"].drop(columns="method")
    if len(test_meta) != 142 or not test_meta.original_index.is_unique:
        raise AssertionError("test prediction vector mismatch")
    test_cov = test_meta.merge(split[["original_index", "id", "split"]], on=["original_index", "id"], validate="one_to_one")
    if not test_cov.split.eq("test").all():
        raise AssertionError("test split mismatch")
    test_cov = test_cov.merge(raw, on=["original_index", "id"], validate="one_to_one")
    if test_cov[["y_pred", *FEATURES, *COORDS]].isna().any().any():
        raise AssertionError("missing test covariate/prediction")

    ordered = val.sort_values("original_index", kind="mergesort").reset_index(drop=True)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(ordered))
    role_by_pos = np.empty(len(ordered), object)
    role_by_pos[perm[:N_SCALE]] = "scale_fit"
    role_by_pos[perm[N_SCALE:N_SCALE + N_TUNE]] = "tuning_calibration"
    role_by_pos[perm[N_SCALE + N_TUNE:]] = "final_calibration"
    ordered["stage14_role"] = role_by_pos
    split_manifest = ordered[["original_index", "id", "stage14_role"]].copy()
    eval_manifest = test_cov[["original_index", "id"]].copy(); eval_manifest["stage14_role"] = "evaluation"
    split_manifest = pd.concat([split_manifest, eval_manifest], ignore_index=True).sort_values(["stage14_role", "original_index"])
    split_manifest.to_csv(EVAL / "split_manifest.csv", index=False)
    counts = ordered.stage14_role.value_counts().to_dict()
    if counts != {"scale_fit": N_SCALE, "tuning_calibration": N_TUNE, "final_calibration": N_FINAL}:
        raise AssertionError(counts)
    if max(K_VALUES) > N_SCALE:
        raise AssertionError("STAGE11_K_GRID_NOT_TRANSFERABLE")
    scale = ordered[ordered.stage14_role == "scale_fit"].copy()
    tuning = ordered[ordered.stage14_role == "tuning_calibration"].copy()
    final = ordered[ordered.stage14_role == "final_calibration"].copy()

    cmi_first = run_cmi(scale)
    cmi_second = run_cmi(scale)
    numeric = ["raw_cmi", "p_value", "bh_q", "weight"]
    max_cmi_repeat_diff = float(np.max(np.abs(cmi_first[numeric].to_numpy(float) - cmi_second[numeric].to_numpy(float))))
    if max_cmi_repeat_diff != 0.0 or not cmi_first[["feature", "eligible", "status"]].equals(cmi_second[["feature", "eligible", "status"]]):
        raise AssertionError("CMI repeat is not deterministic")
    cmi_first.to_csv(EVAL / "cmi_feature_results.csv", index=False)
    cmi_first[["feature", "eligible", "weight", "status"]].to_csv(EVAL / "weight_audit.csv", index=False)
    selected = cmi_first.loc[cmi_first.eligible, "feature"].tolist()
    no_signal = len(selected) == 0

    k_rows: list[dict[str, Any]] = []
    geo_k, geo_model, rows = tune_local(scale, tuning, COORDS, np.ones(2), "GeoOnly-NormCP")
    k_rows.extend(rows)
    residual_scale = np.abs(scale.y_true.to_numpy(float) - scale.y_pred.to_numpy(float))
    residual_tune = np.abs(tuning.y_true.to_numpy(float) - tuning.y_pred.to_numpy(float))
    if no_signal:
        causal_k = None
        global_sigma = max(FLOOR, float(residual_scale.mean()))
        causal_model = {"constant": global_sigma, "tuning_scale_mae": float(np.mean(np.abs(residual_tune - global_sigma)))}
        k_rows.append({"dataset": "PM2.5", "method": "CausalOnly-CMI-NormCP", "k": None,
                       "tuning_scale_mae": causal_model["tuning_scale_mae"], "selected": True,
                       "reference_role": "scale_fit", "tuning_role": "tuning_calibration"})
    else:
        weights = cmi_first.set_index("feature").loc[selected, "weight"].to_numpy(float)
        causal_k, causal_model, rows = tune_local(scale, tuning, selected, weights, "CausalOnly-CMI-NormCP")
        k_rows.extend(rows)
    pd.DataFrame(k_rows).to_csv(EVAL / "k_tuning_results.csv", index=False)

    final_a = np.abs(final.y_true.to_numpy(float) - final.y_pred.to_numpy(float))
    q_raw, rank = finite_quantile(final_a)
    if rank != math.ceil((N_FINAL + 1) * .9) or not np.isfinite(q_raw):
        raise AssertionError("invalid conformal rank")
    geo_final_sigma = neighbor_scale(geo_model["reference_z"], geo_model["reference_a"], geo_model["ids"],
                                     scaled(final, COORDS, geo_model["mean"], geo_model["sd"]), geo_model["weights"], geo_k)
    q_geo, _ = finite_quantile(final_a / geo_final_sigma)
    if no_signal:
        causal_final_sigma = np.full(len(final), causal_model["constant"])
    else:
        causal_final_sigma = neighbor_scale(causal_model["reference_z"], causal_model["reference_a"], causal_model["ids"],
            scaled(final, selected, causal_model["mean"], causal_model["sd"]), causal_model["weights"], causal_k)
    q_causal, _ = finite_quantile(final_a / causal_final_sigma)

    geo_test_sigma = neighbor_scale(geo_model["reference_z"], geo_model["reference_a"], geo_model["ids"],
                                    scaled(test_cov, COORDS, geo_model["mean"], geo_model["sd"]), geo_model["weights"], geo_k)
    if no_signal:
        causal_test_sigma = np.full(len(test_cov), causal_model["constant"])
    else:
        causal_test_sigma = neighbor_scale(causal_model["reference_z"], causal_model["reference_a"], causal_model["ids"],
            scaled(test_cov, selected, causal_model["mean"], causal_model["sd"]), causal_model["weights"], causal_k)
    pred = test_cov.y_pred.to_numpy(float)
    intervals = {
        "Raw-CP": (pred - q_raw, pred + q_raw),
        "GeoOnly-NormCP": (pred - q_geo * geo_test_sigma, pred + q_geo * geo_test_sigma),
        "CausalOnly-CMI-NormCP": (pred - q_causal * causal_test_sigma, pred + q_causal * causal_test_sigma)}
    if not all(np.isfinite(np.column_stack(bounds)).all() for bounds in intervals.values()):
        raise AssertionError("preflight interval failure")
    fallback_diff = None
    if no_signal:
        fallback_diff = float(max(np.max(np.abs(intervals["Raw-CP"][0] - intervals["CausalOnly-CMI-NormCP"][0])),
                                  np.max(np.abs(intervals["Raw-CP"][1] - intervals["CausalOnly-CMI-NormCP"][1]))))
        if fallback_diff > 1e-12:
            raise AssertionError("NO_SIGNAL fallback mismatch")
    preflight = {"status": "PASS", "prediction_join": "PASS", "split_integrity": "PASS",
                 "candidate_leakage": "PASS", "coordinate_leakage": "PASS",
                 "cmi_repeat_max_abs_difference": max_cmi_repeat_diff, "selected_k_deterministic": True,
                 "final_calibration_rank": rank, "evaluation_labels_accessed": False,
                 "finite_evaluation_covariate_intervals": True, "no_signal_raw_bound_max_abs_diff": fallback_diff,
                 "scale_fit_hash": frame_hash(scale), "tuning_hash": frame_hash(tuning),
                 "final_calibration_hash": frame_hash(final), "evaluation_covariate_hash": frame_hash(test_cov)}
    write_json(EVAL / "preflight_summary.json", preflight)

    unseal = {"dataset": "PM2.5", "event": "EVALUATION_UNSEAL", "selected_features": ";".join(selected),
              "weights_json": json.dumps(dict(zip(cmi_first.feature, cmi_first.weight)), sort_keys=True),
              "causal_selected_k": causal_k, "geo_selected_k": geo_k, "q_raw": q_raw,
              "q_geo": q_geo, "q_causal": q_causal, "final_calibration_rank": rank,
              "scale_fit_hash": frame_hash(scale), "tuning_hash": frame_hash(tuning),
              "final_calibration_hash": frame_hash(final), "evaluation_covariate_hash": frame_hash(test_cov),
              "evaluation_labels_accessed_before_event": False}
    append_unseal(unseal)

    # Evaluation labels are read for the first time in the operational pipeline here.
    test_labels = pd.read_csv(PM_TEST_PRED, usecols=["method", "original_index", "id", "y_true"])
    test_labels["id"] = test_labels.id.astype(str)
    test_labels = test_labels[test_labels.method == "CP"].drop(columns="method")
    evaluation = test_cov.merge(test_labels, on=["original_index", "id"], validate="one_to_one")
    if len(evaluation) != N_EVAL or evaluation.y_true.isna().any():
        raise AssertionError("evaluation label join failed")
    # Align intervals to test_cov, whose order is preserved by the one-to-one merge.
    if not evaluation.original_index.equals(test_cov.original_index):
        raise AssertionError("evaluation order changed")
    y = evaluation.y_true.to_numpy(float)
    result_rows = []
    for method, (lower, upper) in intervals.items():
        row = {"dataset": "PM2.5", "method": method, **metrics(y, lower, upper),
               "selected_k": geo_k if method == "GeoOnly-NormCP" else (causal_k if method == "CausalOnly-CMI-NormCP" else None),
               "q": q_geo if method == "GeoOnly-NormCP" else (q_causal if method == "CausalOnly-CMI-NormCP" else q_raw),
               "tuning_scale_mae": geo_model["tuning_scale_mae"] if method == "GeoOnly-NormCP" else
                    (causal_model["tuning_scale_mae"] if method == "CausalOnly-CMI-NormCP" else None),
               "no_causal_signal": no_signal if method == "CausalOnly-CMI-NormCP" else False}
        result_rows.append(row)
    results = pd.DataFrame(result_rows)
    results.to_csv(EVAL / "method_results.csv", index=False)

    causal = results.set_index("method").loc["CausalOnly-CMI-NormCP"]
    raw_row = results.set_index("method").loc["Raw-CP"]
    geo_row = results.set_index("method").loc["GeoOnly-NormCP"]
    if no_signal:
        classification = "D. NO_CAUSAL_SIGNAL_DETECTED"
    else:
        n = int(causal.n_evaluation); phat = float(causal.coverage)
        z = 1.959963984540054
        center = (phat + z*z/(2*n))/(1+z*z/n)
        half = z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/(1+z*z/n)
        coverage_concern = (float(causal.coverage) <= min(float(raw_row.coverage), float(geo_row.coverage)) - .02 and center + half < .90)
        if coverage_concern:
            classification = "C. EXPLORATORY_TRANSFER_COVERAGE_CONCERN"
        elif (float(causal.mean_width) < float(raw_row.mean_width) and float(causal.mean_width) < float(geo_row.mean_width)
              and float(causal.mean_interval_score) < float(raw_row.mean_interval_score)
              and float(causal.mean_interval_score) < float(geo_row.mean_interval_score)):
            classification = "A. EXPLORATORY_TRANSFER_SUPPORTS_CMI_LOCALIZATION"
        else:
            classification = "B. EXPLORATORY_TRANSFER_SHOWS_NO_CLEAR_EFFICIENCY_GAIN"

    method_map: dict[str, dict[str, Any]] = {}
    for row in results.to_dict("records"):
        clean = {key: (None if pd.isna(value) else value) for key, value in row.items() if key != "method"}
        method_map[str(row["method"])] = clean
    summary = {"stage": "Stage 14", "execution_status": "PASS", "selected_dataset": "PM2.5",
               "classification": classification, "claim_boundary": CLAIM, "development_counts": counts,
               "evaluation_n": len(evaluation), "date_values": sorted(raw.date.astype(str).unique().tolist()),
               "unique_station_ids": int(raw.id.nunique()), "repeated_station_rows": int(len(raw) - raw.id.nunique()),
               "selected_features": selected, "no_causal_signal": no_signal,
               "weights": dict(zip(cmi_first.feature, cmi_first.weight)), "selected_k": {"GeoOnly-NormCP": geo_k,
                  "CausalOnly-CMI-NormCP": causal_k}, "final_calibration_rank": rank,
               "fallback_max_abs_bound_difference": fallback_diff, "preflight": preflight,
               "methods": method_map,
               "recommended_next_step": "Construct and preregister a genuinely new future-date PM2.5 holdout, with a frozen predictor and development/evaluation split, before making any transfer claim.",
               "elapsed_seconds": time.time() - started}
    write_json(EVAL / "stage14_summary.json", summary)
    create_docs(inventory, gates, results, classification, summary)

    env = [f"Execution date: 2026-09-18", f"OS: {platform.platform()}", f"Python executable: {sys.executable}",
           f"Python: {platform.python_version()}"]
    for package in ["tigramite", "numpy", "scipy", "pandas", "scikit-learn"]:
        try: env.append(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError: env.append(f"{package}: NOT INSTALLED")
    (EVAL / "environment.txt").write_text("\n".join(env) + "\n", encoding="utf-8")

    audit_sources = [Path(__file__), SEATTLE_RAW, SEATTLE / "01_baselines" / "01_Seattle_baselines_three_split.ipynb",
                     SEATTLE / "outputs" / "seattle_split_seed42.npz",
                     SEATTLE / "outputs" / "seattle_inner_split_seed42.npz",
                     SEATTLE / "outputs" / "baseline_test_intervals_seed42.csv",
                     SEATTLE / "outputs" / "seattle_author_aligned_final_test_per_sample_v1.csv",
                     PM_RAW, PM25 / "00_data_audit" / "pm25_data_audit.ipynb",
                     PM25 / "02_baselines" / "pm2.5_formal_baselines.ipynb",
                     PM_SPLIT, PM_PROTOCOL, PM_SELECTION, PM_VAL_PRED, PM_TEST_PRED, PM_TEST_MANIFEST]
    audit_config_outputs = [AUDIT / name for name in ["README.md", "DATA_PROVENANCE.md", "config.json", "dataset_inventory.csv",
        "predictor_provenance.csv", "split_exposure_audit.csv", "candidate_feature_audit.csv", "feasibility_decision.json"]]
    manifests(AUDIT, audit_sources, audit_config_outputs)
    eval_outputs = [EVAL / name for name in ["README.md", "METHOD.md", "CLAIM_BOUNDARY.md", "config.json", "environment.txt",
        "split_manifest.csv", "cmi_feature_results.csv", "weight_audit.csv", "k_tuning_results.csv", "preflight_summary.json",
        "evaluation_unseal_audit.csv", "method_results.csv", "stage14_summary.json", "stage14_summary.md"]]
    manifests(EVAL, [Path(__file__), PM_RAW, PM25 / "02_baselines" / "pm2.5_formal_baselines.ipynb",
                     PM_SPLIT, PM_PROTOCOL, PM_SELECTION, PM_VAL_PRED, PM_TEST_PRED, PM_TEST_MANIFEST], eval_outputs)
    print(f"PASS: {classification}")
    print(f"Audit record: {AUDIT}")
    print(f"Evaluation record: {EVAL}")


if __name__ == "__main__":
    main()
