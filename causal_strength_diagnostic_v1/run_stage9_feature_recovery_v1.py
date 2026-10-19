"""Stage 9 frozen-artifact feature-level residual-scale CMI diagnostic; no predictor refitting."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
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
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDINATES = ["s_x", "s_y"]
MASTER_SEED = 20260910


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bh_q_values(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda pair: (pair[1], pair[0]))
    m, adjusted = len(ordered), [0.0] * len(ordered)
    running = 1.0
    for index in range(len(ordered) - 1, -1, -1):
        candidate = min(1.0, ordered[index][1] * m / (index + 1))
        running = min(running, candidate)
        adjusted[index] = running
    return {feature: float(value) for (feature, _), value in zip(ordered, adjusted)}


def standardize(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = list(values)
    matrix = np.column_stack([np.asarray(values[name], dtype=float) for name in names])
    if not np.isfinite(matrix).all():
        raise AssertionError("nonfinite CMI input")
    scale = matrix.std(axis=0, ddof=0)
    if np.any(scale == 0.0):
        raise AssertionError("constant CMI input")
    normalized = (matrix - matrix.mean(axis=0)) / scale
    return {name: normalized[:, index] for index, name in enumerate(names)}


def cmi_test(seed: int):
    from tigramite.independence_tests.cmiknn import CMIknn
    return CMIknn(knn=.10, shuffle_neighbors=5, significance="shuffle_test", transform=False, workers=1, sig_samples=99, sig_blocklength=1, seed=seed)


def load_scale(rep: dict, scenario: str) -> tuple[pd.DataFrame, dict]:
    item = rep["scenarios"][scenario]
    paths = {name: ROOT / item[name]["path"] for name in ("predictor_inputs", "scale_fit_labels", "predictions")}
    for name, path in paths.items():
        if sha256(path) != item[name]["sha256"]:
            raise AssertionError(f"frozen artifact hash mismatch: {path}")
    x = pd.read_csv(paths["predictor_inputs"], keep_default_na=False)
    labels = pd.read_csv(paths["scale_fit_labels"], keep_default_na=False)
    predictions = pd.read_csv(paths["predictions"], keep_default_na=False)
    if list(x.columns) != META + ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"] or list(labels.columns) != META + ["y"] or list(predictions.columns) != META + ["y_pred"]:
        raise AssertionError("unexpected frozen artifact schema")
    if len(x) != 2000 or len(labels) != 200 or len(predictions) != 2000:
        raise AssertionError("unexpected frozen artifact row count")
    if not x.sample_id.is_unique or not labels.sample_id.is_unique or not predictions.sample_id.is_unique:
        raise AssertionError("duplicate sample ID")
    if not ((labels["split"] == "local_calibration") & (labels["local_calibration_role"] == "scale_fit")).all():
        raise AssertionError("unauthorized target-label access")
    scale_x = x.loc[(x["split"] == "local_calibration") & (x["local_calibration_role"] == "scale_fit")].copy()
    frame = scale_x.merge(labels, on=META, how="left", validate="one_to_one", sort=False).merge(predictions, on=META, how="left", validate="one_to_one", sort=False)
    if len(frame) != 200 or frame[["y", "y_pred"]].isna().any().any() or not np.isfinite(frame[["y", "y_pred"]].to_numpy(float)).all():
        raise AssertionError("scale-fit alignment or finite-value failure")
    frame = frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    frame["A_model"] = np.abs(frame["y"].to_numpy(float) - frame["y_pred"].to_numpy(float))
    if not np.isfinite(frame["A_model"].to_numpy(float)).all():
        raise AssertionError("nonfinite A_model")
    table_hash = hashlib.sha256(frame[META + ["s_x", "s_y"] + FEATURES + ["y", "y_pred", "A_model"]].to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")).hexdigest()
    return frame, {"authorized_scale_fit_rows": 200, "matched_frozen_predictions": 200, "missing_sample_ids": [], "duplicate_sample_ids": 0, "finite_y_count": 200, "finite_y_pred_count": 200, "finite_A_model_count": 200, "analysis_table_sha256": table_hash}


def stats(values: pd.Series | np.ndarray) -> dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "std": float(array.std(ddof=1)), "min": float(array.min()), "max": float(array.max())}


def transform_weights(group: pd.DataFrame, transform: str) -> list[dict]:
    raw = group["raw_cmi"].to_numpy(float)
    if transform == "raw_positive":
        u = np.maximum(raw, 0.0)
    elif transform == "bh_gated_raw_positive":
        u = np.where(group["bh_q"].to_numpy(float) < .05, np.maximum(raw, 0.0), 0.0)
    else:
        raise ValueError(transform)
    status = "OK" if float(u.sum()) > 0.0 else "NO_SIGNAL"
    weights = u / u.sum() if status == "OK" else np.zeros(len(u), dtype=float)
    if set(group["feature"]) != set(FEATURES):
        raise AssertionError("candidate feature mismatch")
    return [{"replicate_index": int(row.replicate_index), "generation_seed": int(row.generation_seed), "scenario": row.scenario, "condition": row.condition, "transform": transform, "feature": row.feature, "raw_cmi": float(row.raw_cmi), "bh_q": float(row.bh_q), "weight": float(weight), "status": status} for row, weight in zip(group.itertuples(index=False), weights)]


def feature_summary(results: pd.DataFrame, weights: pd.DataFrame, scenario: str, condition: str) -> dict:
    group = results[(results.scenario == scenario) & (results.condition == condition)]
    piv = group.pivot(index="replicate_index", columns="feature", values="raw_cmi")[FEATURES]
    tops = piv.eq(piv.max(axis=1), axis=0)
    out = {"feature_raw_cmi": {f: stats(piv[f]) for f in FEATURES}, "top_rank_counts": {f: int(tops[f].sum()) for f in FEATURES}, "bh_q_lt_0_05_counts": {f: int((group[group.feature == f].bh_q < .05).sum()) for f in FEATURES}, "top_feature_by_replicate": {str(i): [f for f in FEATURES if bool(tops.loc[i, f])] for i in piv.index}}
    if scenario == "S2":
        margins = piv["x2"] - piv[["x1", "g_proxy", "x_noise"]].max(axis=1)
        out["x2_pairwise_raw_greater_counts"] = {f: int((piv["x2"] > piv[f]).sum()) for f in ("x1", "g_proxy", "x_noise")}
        out["x2_minus_strongest_non_x2_margin"] = stats(margins)
    else:
        out["replicates_any_bh_q_lt_0_05"] = int(group.groupby("replicate_index").apply(lambda x: bool((x.bh_q < .05).any()), include_groups=False).sum())
    transforms = {}
    for transform in ("raw_positive", "bh_gated_raw_positive"):
        subset = weights[(weights["scenario"] == scenario) & (weights["condition"] == condition) & (weights["transform"] == transform)]
        wp = subset.pivot(index="replicate_index", columns="feature", values="weight")[FEATURES]
        status = subset.groupby("replicate_index").status.first()
        transforms[transform] = {"no_signal_replicates": int((status == "NO_SIGNAL").sum()), "x2_median_weight": float(np.median(wp["x2"])), "x2_largest_weight_replicates": int(wp["x2"].eq(wp.max(axis=1)).sum()), "spurious_nonzero_weight_replicates": int((wp.sum(axis=1) > 0).sum()) if scenario == "S4" else None, "maximum_weight": stats(wp.max(axis=1))}
    out["provisional_weight_diagnostics"] = transforms
    return out


def main() -> None:
    if any((OUT / name).exists() for name in ("stage9_feature_recovery_results.csv", "stage9_feature_recovery_summary.json", "stage9_provisional_weights.csv")):
        raise FileExistsError("Refusing to overwrite Stage-9 canonical outputs")
    if set(FEATURES).intersection(COORDINATES):
        raise AssertionError("coordinates are prohibited candidate features")
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["replicates"]) != 20:
        raise AssertionError("expected 20 frozen replicates")
    frames, preflight = {}, {}
    for rep in manifest["replicates"]:
        index = int(rep["replicate_index"])
        for scenario in ("S2", "S4"):
            frames[(index, scenario)], preflight[f"{index:02d}_{scenario}"] = load_scale(rep, scenario)
        paired = ["sample_id", "split", "local_calibration_role", "s_x", "s_y", *FEATURES]
        pd.testing.assert_frame_equal(frames[(index, "S2")][paired], frames[(index, "S4")][paired], check_dtype=False)
    started, rows = time.perf_counter(), []
    for rep in manifest["replicates"]:
        index, seed = int(rep["replicate_index"]), int(rep["generation_seed"])
        for scenario, scenario_offset in (("S2", 0), ("S4", 2)):
            frame = frames[(index, scenario)]
            values = standardize({name: frame[name].to_numpy(float) for name in [*FEATURES, *COORDINATES, "A_model"]})
            for condition, conditioning, condition_offset in (("M0", [], 0), ("Mspace", COORDINATES, 1)):
                # Reuse the original Stage-8 seed for this scenario/condition across four feature tests.
                shuffle_seed = MASTER_SEED + index * 10 + scenario_offset + condition_offset
                p_values, raw_scores = {}, {}
                for feature in FEATURES:
                    names, xyz = [feature, "A_model", *conditioning], [0, 1] + [2] * len(conditioning)
                    array = np.vstack([values[name] for name in names]); labels = np.asarray(xyz, dtype=int)
                    test = cmi_test(shuffle_seed)
                    raw = float(test.get_dependence_measure(array.copy(), labels.copy()))
                    p_values[feature] = float(test.get_shuffle_significance(array.copy(), labels.copy(), raw)); raw_scores[feature] = raw
                q_values = bh_q_values(p_values)
                for feature in FEATURES:
                    rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario, "condition": condition, "feature": feature, "raw_cmi": raw_scores[feature], "p_value": p_values[feature], "bh_q": q_values[feature], "bh_q_lt_0_05": q_values[feature] < .05, "shuffle_seed": shuffle_seed, "n": 200, "k": 20})
        if index % 5 == 0:
            print(f"PROGRESS: {index}/20 paired replicate seeds complete", flush=True)
    results = pd.DataFrame(rows)
    if len(results) != 320:
        raise AssertionError("expected 320 feature-level CMI tests")
    weights = pd.DataFrame([entry for _, group in results.groupby(["replicate_index", "scenario", "condition"], sort=True) for transform in ("raw_positive", "bh_gated_raw_positive") for entry in transform_weights(group.sort_values("feature"), transform)])
    if set(weights.feature) != set(FEATURES) or set(weights.feature).intersection(COORDINATES):
        raise AssertionError("coordinate leaked into provisional weights")
    summary = {condition: {scenario: feature_summary(results, weights, scenario, condition) for scenario in ("S2", "S4")} for condition in ("Mspace", "M0")}
    primary = summary["Mspace"]
    s2, s4 = primary["S2"], primary["S4"]
    top_s4 = s4["top_rank_counts"]
    gate = {"x2_top_at_least_15_of_20": {"observed": s2["top_rank_counts"]["x2"], "pass": s2["top_rank_counts"]["x2"] >= 15}, "x2_gt_x_noise_at_least_18_of_20": {"observed": s2["x2_pairwise_raw_greater_counts"]["x_noise"], "pass": s2["x2_pairwise_raw_greater_counts"]["x_noise"] >= 18}, "x2_gt_x1_at_least_15_of_20": {"observed": s2["x2_pairwise_raw_greater_counts"]["x1"], "pass": s2["x2_pairwise_raw_greater_counts"]["x1"] >= 15}, "x2_gt_g_proxy_at_least_15_of_20": {"observed": s2["x2_pairwise_raw_greater_counts"]["g_proxy"], "pass": s2["x2_pairwise_raw_greater_counts"]["g_proxy"] >= 15}, "x2_bh_q_lt_0_05_at_least_10_of_20": {"observed": s2["bh_q_lt_0_05_counts"]["x2"], "pass": s2["bh_q_lt_0_05_counts"]["x2"] >= 10}, "S4_each_feature_bh_q_lt_0_05_at_most_2_of_20": {"observed": s4["bh_q_lt_0_05_counts"], "pass": max(s4["bh_q_lt_0_05_counts"].values()) <= 2}, "S4_no_concentrated_top_feature": {"operational_definition": "no one feature is top-ranked in at least 15 of 20 S4 replicates", "observed_top_counts": top_s4, "pass": max(top_s4.values()) < 15}}
    gate["overall_primary_gate_pass"] = all(item["pass"] for item in gate.values())
    results.to_csv(OUT / "stage9_feature_recovery_results.csv", index=False, lineterminator="\n", float_format="%.17g")
    weights.to_csv(OUT / "stage9_provisional_weights.csv", index=False, lineterminator="\n", float_format="%.17g")
    payload = {"stage": "Stage 9", "execution_status": "PASS", "source_script": str(Path(__file__).relative_to(ROOT)), "interpreter": sys.executable, "packages": {p: importlib.metadata.version(p) for p in ("tigramite", "numpy", "scipy", "pandas", "scikit-learn")}, "frozen_replicate_manifest": {"path": str((FROZEN / "manifest.json").relative_to(ROOT)), "sha256": sha256(FROZEN / "manifest.json")}, "configuration": {"target": "A_model=abs(y-y_pred_frozen)", "candidate_features": FEATURES, "coordinates_not_candidates": COORDINATES, "primary_condition": "Mspace: I(X_j; A_model | s_x,s_y)", "secondary_condition": "M0: I(X_j; A_model)", "CMIknn": {"n": 200, "knn_fraction": .10, "k": 20, "sig_samples": 99, "shuffle_neighbors": 5, "sig_blocklength": 1, "alpha": .05}, "shuffle_seed_rule": "Original Stage-8 scenario/condition seed reused for all four feature tests within that replicate", "bh": "Benjamini-Hochberg across four candidate features within scenario x replicate x condition; q<0.05"}, "label_access_contract": {"allowed": "local_calibration/scale_fit only", "prohibited": ["validation", "final_calibration", "test"], "status": "PASS"}, "preflight": preflight, "summary": summary, "primary_feature_recovery_gate": gate, "elapsed_time_seconds": time.perf_counter() - started}
    (OUT / "stage9_feature_recovery_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("PRODUCTION PASS: 20/20 paired replicate seeds complete", flush=True)


if __name__ == "__main__":
    main()
