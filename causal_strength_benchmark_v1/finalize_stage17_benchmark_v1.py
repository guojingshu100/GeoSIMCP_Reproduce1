"""Validate, enrich, and summarize the completed Stage-17 estimator benchmark."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RECORD = ROOT / "experiments/causal_strength_benchmark/20260918_stage17_causal_estimator_validation_v1"
METHODS = ["CMI_MSPACE", "CMI_FULL", "GPC", "SCT", "SCPCM"]
MECHANISMS = ["C0_INDEPENDENT_NULL", "C1_RESIDUAL_SCALE_CAUSAL", "C2_PURE_CONFOUNDING", "C3_REVERSE_DIRECTION", "C4_SPATIAL_LAG_CAUSAL", "C5_NONLINEAR_DIRECT"]
FEATURES = ["x1", "x2", "g_proxy", "x_noise", "Z"]
DRIVER = {
    "C0_INDEPENDENT_NULL": None,
    "C1_RESIDUAL_SCALE_CAUSAL": "x2",
    "C2_PURE_CONFOUNDING": "Z",
    "C3_REVERSE_DIRECTION": None,
    "C4_SPATIAL_LAG_CAUSAL": "x2",
    "C5_NONLINEAR_DIRECT": "x2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bh(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(ordered)
    running = 1.0
    for i in range(len(ordered) - 1, -1, -1):
        running = min(running, min(1.0, ordered[i][1] * len(ordered) / (i + 1)))
        adjusted[i] = running
    return {name: float(value) for (name, _), value in zip(ordered, adjusted)}


def enrich_directional(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    out_parts = []
    for _, group in frame.groupby(["mechanism", "replicate_index"], sort=True):
        group = group.copy()
        if method == "GPC":
            group["forward_q"] = np.nan
            group["reverse_q"] = np.nan
            group["forward_selected"] = (group.forward_score > 0) & (group.forward_score > group.reverse_score)
            group["reverse_selected"] = (group.reverse_score > 0) & (group.reverse_score > group.forward_score)
        else:
            fq = bh(dict(zip(group.feature, group.forward_p)))
            rq = bh(dict(zip(group.feature, group.reverse_p)))
            group["forward_q"] = group.feature.map(fq)
            group["reverse_q"] = group.feature.map(rq)
            group["forward_selected"] = (group.forward_q < .05) & (group.forward_score > 0)
            group["reverse_selected"] = (group.reverse_q < .05) & (group.reverse_score > 0)
        order = sorted(zip(group.feature, group.forward_score), key=lambda x: (-x[1], x[0]))
        ranks = {feature: rank for rank, (feature, _) in enumerate(order, start=1)}
        group["rank"] = group.feature.map(ranks)
        group["no_signal"] = ~group.forward_selected.any()
        out_parts.append(group)
    return pd.concat(out_parts, ignore_index=True)


def proportion_ci(count: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = count / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return center-half, center+half


def combined_long() -> pd.DataFrame:
    pieces = []
    for filename in ("cmi_mspace_results.csv", "cmi_full_results.csv"):
        d = pd.read_csv(RECORD / filename)
        d = d.rename(columns={"raw_score": "forward_score", "p_value": "forward_p", "bh_q": "forward_q", "selected": "forward_selected"})
        d["reverse_score"] = np.nan; d["reverse_p"] = np.nan; d["reverse_q"] = np.nan; d["reverse_selected"] = False
        pieces.append(d[["dataset_id", "mechanism", "replicate_index", "generation_seed", "method", "feature", "forward_score", "reverse_score", "forward_p", "reverse_p", "forward_q", "reverse_q", "forward_selected", "reverse_selected", "rank", "no_signal"]])
    for method, filename in (("GPC", "gpc_results.csv"), ("SCT", "sct_results.csv"), ("SCPCM", "scpcm_results.csv")):
        raw = pd.read_csv(RECORD / filename)
        d = enrich_directional(raw, method)
        d.to_csv(RECORD / filename, index=False, lineterminator="\n", float_format="%.17g")
        pieces.append(d[["dataset_id", "mechanism", "replicate_index", "generation_seed", "method", "feature", "forward_score", "reverse_score", "forward_p", "reverse_p", "forward_q", "reverse_q", "forward_selected", "reverse_selected", "rank", "no_signal"]])
    all_results = pd.concat(pieces, ignore_index=True)
    all_results["forward_selected"] = all_results.forward_selected.astype(bool)
    all_results["reverse_selected"] = all_results.reverse_selected.astype(bool)
    all_results["no_signal"] = all_results.no_signal.astype(bool)
    return all_results


def main() -> None:
    checkpoint = json.loads((RECORD / "execution_checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint["status"] != "SCIENTIFIC_EXECUTION_COMPLETE":
        raise AssertionError("execution checkpoint incomplete")
    seal = json.loads((RECORD / "PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    for name, expected in seal["files"].items():
        if sha256(RECORD / name) != expected:
            raise AssertionError(f"preregistration mutation: {name}")
    all_results = combined_long()
    expected_rows = 5 * 6 * 50 * 5
    if len(all_results) != expected_rows or all_results.duplicated(["method", "dataset_id", "feature"]).any():
        raise AssertionError("canonical result completeness failure")

    recovery_rows = []
    for method in METHODS:
        for mechanism in MECHANISMS:
            d = all_results[(all_results.method == method) & (all_results.mechanism == mechanism)]
            datasets = d.dataset_id.nunique()
            driver = DRIVER[mechanism]
            x2 = d[d.feature == "x2"]
            target = d[d.feature == driver] if driver else None
            nuisance = d[~d.feature.isin([driver] if driver else [])]
            selected_by_dataset = nuisance.groupby("dataset_id").forward_selected.any()
            row = {
                "method": method, "mechanism": mechanism, "replicates": datasets,
                "truth_forward_driver": driver or "NONE",
                "x2_mean_rank": float(x2["rank"].mean()),
                "x2_top1_count": int((x2["rank"] == 1).sum()),
                "x2_top1_frequency": float((x2["rank"] == 1).mean()),
                "x2_selected_count": int(x2.forward_selected.sum()),
                "x2_selected_frequency": float(x2.forward_selected.mean()),
                "no_signal_count": int(d.groupby("dataset_id").no_signal.first().sum()),
                "no_signal_frequency": float(d.groupby("dataset_id").no_signal.first().mean()),
                "any_nondriver_selected_count": int(selected_by_dataset.sum()),
                "any_nondriver_selected_frequency": float(selected_by_dataset.mean()),
                "g_proxy_selected_count": int(d.loc[d.feature.eq("g_proxy"), "forward_selected"].sum()),
                "g_proxy_selected_frequency": float(d.loc[d.feature.eq("g_proxy"), "forward_selected"].mean()),
            }
            if target is not None:
                row.update({
                    "true_driver_top1_count": int((target["rank"] == 1).sum()),
                    "true_driver_top1_frequency": float((target["rank"] == 1).mean()),
                    "true_driver_selected_count": int(target.forward_selected.sum()),
                    "true_driver_selected_frequency": float(target.forward_selected.mean()),
                })
            else:
                row.update({"true_driver_top1_count": np.nan, "true_driver_top1_frequency": np.nan, "true_driver_selected_count": np.nan, "true_driver_selected_frequency": np.nan})
            recovery_rows.append(row)
    recovery = pd.DataFrame(recovery_rows)
    recovery.to_csv(RECORD / "method_recovery_summary.csv", index=False, lineterminator="\n", float_format="%.17g")

    null = recovery[recovery.mechanism.eq("C0_INDEPENDENT_NULL")].copy()
    null[["method", "replicates", "no_signal_count", "no_signal_frequency", "any_nondriver_selected_count", "any_nondriver_selected_frequency", "g_proxy_selected_count", "g_proxy_selected_frequency"]].to_csv(RECORD / "null_control_summary.csv", index=False, lineterminator="\n", float_format="%.17g")

    conf_rows = []
    for method in METHODS:
        d = all_results[(all_results.method == method) & all_results.mechanism.eq("C2_PURE_CONFOUNDING")]
        for feature, label in (("x2", "confounded_noncausal_x2"), ("Z", "observed_direct_driver_Z"), ("g_proxy", "noncausal_spatial_proxy")):
            f = d[d.feature.eq(feature)]
            conf_rows.append({"method": method, "feature": feature, "role": label, "replicates": 50,
                              "selected_count": int(f.forward_selected.sum()), "selected_frequency": float(f.forward_selected.mean()),
                              "top1_count": int((f["rank"] == 1).sum()), "top1_frequency": float((f["rank"] == 1).mean()),
                              "mean_rank": float(f["rank"].mean())})
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(RECORD / "confounding_summary.csv", index=False, lineterminator="\n", float_format="%.17g")

    direction_rows = []
    truth_specs = [
        ("C1_RESIDUAL_SCALE_CAUSAL", "x2", "forward"),
        ("C2_PURE_CONFOUNDING", "Z", "forward"),
        ("C3_REVERSE_DIRECTION", "x2", "reverse"),
        ("C4_SPATIAL_LAG_CAUSAL", "x2", "forward"),
        ("C5_NONLINEAR_DIRECT", "x2", "forward"),
    ]
    for method in ("GPC", "SCT", "SCPCM"):
        for mechanism, feature, correct in truth_specs:
            d = all_results[(all_results.method == method) & all_results.mechanism.eq(mechanism) & all_results.feature.eq(feature)]
            if correct == "forward":
                pref = d.forward_score > d.reverse_score; opposite = d.reverse_score > d.forward_score
                selected = d.forward_selected; opposite_selected = d.reverse_selected
            else:
                pref = d.reverse_score > d.forward_score; opposite = d.forward_score > d.reverse_score
                selected = d.reverse_selected; opposite_selected = d.forward_selected
            direction_rows.append({
                "method": method, "mechanism": mechanism, "feature": feature, "truth_direction": correct,
                "replicates": len(d), "correct_preference_count": int(pref.sum()), "correct_preference_frequency": float(pref.mean()),
                "opposite_preference_count": int(opposite.sum()), "opposite_preference_frequency": float(opposite.mean()),
                "correct_selected_count": int(selected.sum()), "correct_selected_frequency": float(selected.mean()),
                "opposite_selected_count": int(opposite_selected.sum()), "opposite_selected_frequency": float(opposite_selected.mean()),
            })
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(RECORD / "directionality_summary.csv", index=False, lineterminator="\n", float_format="%.17g")

    # Supporting full long table makes all selection/ranking transformations auditable.
    all_results.to_csv(RECORD / "all_method_results_long.csv", index=False, lineterminator="\n", float_format="%.17g")
    summary = {
        "stage": "17", "execution_status": "PASS", "conformal_prediction_run": False,
        "preregistration_composite_sha256": seal["composite_sha256"],
        "seed_block": [20420001, 20420050], "replicates_per_mechanism": 50,
        "completed_dataset_count_per_method": {method: int(all_results[all_results.method.eq(method)].dataset_id.nunique()) for method in METHODS},
        "score_claim_boundary": {
            "CMI_MSPACE": "symmetric coordinate-conditioned dependence; not directional causality",
            "CMI_FULL": "symmetric dependence conditioned on coordinates and observed alternative candidates; not directional causality",
            "GPC": "directional pattern-strength output; descriptive forward preference has no calibrated full-library p-value",
            "SCT": "directional statistic with bootstrap p-value and five-feature BH gate",
            "SCPCM": "secondary partial-cross-map directional mean/significance; documentation maturity limitation retained",
        },
        "tables": {
            "recovery": "method_recovery_summary.csv", "null": "null_control_summary.csv",
            "confounding": "confounding_summary.csv", "directionality": "directionality_summary.csv",
        },
        "gccm_frozen_reference": {
            "source": "Stage-15 fresh confirmatory record; not rerun on Stage-17 controls",
            "S2_x2_positive": "1/50", "S2_x2_top": "0/50", "S2_g_proxy_top": "32/50", "S2_fallback": "18/50",
            "interpretation": "historical negative reference only; denominators/mechanisms are separate",
        },
    }
    (RECORD / "stage17_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    execution_inputs = {
      "sealed_preregistration_input_manifest": {"path": "input_manifest.json", "sha256": sha256(RECORD / "input_manifest.json")},
      "implementation": [
        {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
        for path in (HERE / "preregister_stage17_v1.py", HERE / "run_stage17_benchmark_v1.py", HERE / "stage17_cmi_worker_v1.py", HERE / "run_stage17_spedm_v1.R", HERE / "finalize_stage17_benchmark_v1.py", RECORD / "IMPLEMENTATION_FREEZE.md")
      ],
      "generated_controlled_input": {"path": "controlled_inputs.csv", "sha256": sha256(RECORD / "controlled_inputs.csv"), "rows": 60000},
    }
    (RECORD / "execution_input_manifest.json").write_text(json.dumps(execution_inputs, indent=2) + "\n", encoding="utf-8")
    (RECORD / "runtime_environment.txt").write_text(
        f"Finalizer Python: {sys.version}\nFinalizer platform: {platform.platform()}\n"
        "CMI Python: causal_strength_diagnostic_v1/.venv_cmi_v1/Scripts/python.exe\n"
        "Tigramite: 5.2.10.1\nNumPy: 2.5.3\nPandas: 3.0.5\nSciPy: 1.18.1\n"
        "R: 4.6.1\nspEDM: 1.12\nsf: 1.1.3\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
