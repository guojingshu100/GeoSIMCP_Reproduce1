"""Stage 11 CausalOnly-CMI normalized split conformal prediction.

This runner fails closed before any CP calculation unless immutable outcome
artifacts exist for tuning_calibration, final_calibration, and test. It never
regenerates the Stage-8 simulations or refits their predictors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = ROOT / "causal_strength_diagnostic_v1" / "stage8_frozen_replicates_v1"
STAGE9_RESULTS = ROOT / "causal_strength_diagnostic_v1" / "stage9_feature_recovery_results.csv"
STAGE9_WEIGHTS = ROOT / "causal_strength_diagnostic_v1" / "stage9_provisional_weights.csv"
CONFIG = HERE / "stage11_config.json"
META = ["sample_id", "scenario", "split", "local_calibration_role"]
FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]
SCENARIOS = ["S2", "S4"]
REQUIRED_DOWNSTREAM = {
    "tuning_calibration": "{scenario}_tuning_calibration_labels.csv",
    "final_calibration": "{scenario}_final_calibration_labels.csv",
    "test": "{scenario}_test_labels.csv",
    "oracle_truth": "{scenario}_truth.csv",
}


class ProtocolBlocker(RuntimeError):
    """A fail-closed scientific-input or leakage-contract violation."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv_checked(path: Path, columns: list[str], rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != columns:
        raise ProtocolBlocker(f"unexpected schema for {path}: {list(frame.columns)}")
    if len(frame) != rows or not frame["sample_id"].is_unique:
        raise ProtocolBlocker(f"row/ID integrity failure for {path}")
    return frame


def derive_weights(results: pd.DataFrame) -> pd.DataFrame:
    selected = results[(results["condition"] == "Mspace")].copy()
    if set(selected["feature"]) != set(FEATURES):
        raise ProtocolBlocker("Stage-9 feature set is not the frozen four-feature set")
    selected["eligible"] = (selected["bh_q"] < 0.05) & (selected["raw_cmi"] > 0.0)
    selected["u"] = np.where(selected["eligible"], selected["raw_cmi"], 0.0)
    totals = selected.groupby(["replicate_index", "scenario"])["u"].transform("sum")
    selected["weight"] = 0.0
    positive = totals > 0.0
    selected.loc[positive, "weight"] = selected.loc[positive, "u"] / totals.loc[positive]
    # Preserve Stage-9's recorded vocabulary for the provenance cross-check.
    # Stage 11 translates NO_SIGNAL to the method-facing NO_CAUSAL_SIGNAL state.
    selected["status"] = np.where(positive, "OK", "NO_SIGNAL")
    return selected


def artifact_preflight() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest_path = FROZEN / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("replicates", [])) != config["replicates"]:
        raise ProtocolBlocker("frozen manifest does not contain 20 replicates")

    stage9 = pd.read_csv(STAGE9_RESULTS)
    weights = derive_weights(stage9)
    recorded = pd.read_csv(STAGE9_WEIGHTS)
    recorded = recorded[(recorded["condition"] == "Mspace") &
                        (recorded["transform"] == "bh_gated_raw_positive")]
    joined = weights.merge(
        recorded[["replicate_index", "scenario", "feature", "weight", "status"]],
        on=["replicate_index", "scenario", "feature"], suffixes=("_derived", "_recorded"),
        validate="one_to_one",
    )
    if len(joined) != 160 or not np.allclose(joined["weight_derived"], joined["weight_recorded"], atol=1e-15, rtol=0):
        raise ProtocolBlocker("derived frozen weights do not reproduce Stage-9 provisional weights")
    if not (joined["status_derived"] == joined["status_recorded"]).all():
        raise ProtocolBlocker("derived Stage-9 weight statuses disagree with the recorded artifact")

    missing: list[dict] = []
    artifact_rows: list[dict] = []
    paired_checks: list[dict] = []
    expected_counts = config["split_roles"]
    for rep in manifest["replicates"]:
        index, seed = int(rep["replicate_index"]), int(rep["generation_seed"])
        scenario_frames: dict[str, pd.DataFrame] = {}
        for scenario in SCENARIOS:
            item = rep["scenarios"][scenario]
            paths = {name: ROOT / item[name]["path"] for name in ("predictor_inputs", "scale_fit_labels", "predictions")}
            for kind, path in paths.items():
                actual = sha256(path)
                if actual != item[kind]["sha256"]:
                    raise ProtocolBlocker(f"hash mismatch: {path}")
                artifact_rows.append({"replicate_index": index, "generation_seed": seed, "scenario": scenario,
                                      "kind": kind, "path": str(path.relative_to(ROOT)), "sha256": actual,
                                      "rows": int(item[kind]["rows"])})
            x = read_csv_checked(paths["predictor_inputs"], META + COORDS + FEATURES, 2000)
            p = read_csv_checked(paths["predictions"], META + ["y_pred"], 2000)
            scale = read_csv_checked(paths["scale_fit_labels"], META + ["y"], 200)
            counts = x["split"].value_counts().to_dict()
            roles = x.loc[x["split"] == "local_calibration", "local_calibration_role"].value_counts().to_dict()
            for role in ("train", "local_calibration", "validation", "final_calibration", "test"):
                if int(counts.get(role, -1)) != int(expected_counts[role]):
                    raise ProtocolBlocker(f"split count mismatch in {path.parent.name}/{scenario}: {role}")
            for role in ("scale_fit", "tuning_calibration"):
                if int(roles.get(role, -1)) != int(expected_counts[role]):
                    raise ProtocolBlocker(f"local role count mismatch in {path.parent.name}/{scenario}: {role}")
            if set(x["sample_id"]) != set(p["sample_id"]):
                raise ProtocolBlocker(f"prediction ID mismatch in {path.parent.name}/{scenario}")
            expected_scale = set(x.loc[(x["split"] == "local_calibration") &
                                       (x["local_calibration_role"] == "scale_fit"), "sample_id"])
            if expected_scale != set(scale["sample_id"]):
                raise ProtocolBlocker(f"scale_fit label ID mismatch in {path.parent.name}/{scenario}")
            scenario_frames[scenario] = x.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
            for role, pattern in REQUIRED_DOWNSTREAM.items():
                candidate = path.parent / pattern.format(scenario=scenario)
                if not candidate.is_file():
                    missing.append({"replicate_index": index, "generation_seed": seed,
                                    "scenario": scenario, "required_role": role,
                                    "expected_path": str(candidate.relative_to(ROOT))})
        columns = ["sample_id", "split", "local_calibration_role"] + COORDS + FEATURES
        pair_equal = scenario_frames["S2"][columns].equals(scenario_frames["S4"][columns])
        paired_checks.append({"replicate_index": index, "generation_seed": seed,
                              "paired_feature_and_split_identity": pair_equal})
        if not pair_equal:
            raise ProtocolBlocker(f"paired S2/S4 input mismatch for replicate {index}")

    overlap_checks = []
    for rep in manifest["replicates"]:
        index = int(rep["replicate_index"])
        path = ROOT / rep["scenarios"]["S2"]["predictor_inputs"]["path"]
        x = pd.read_csv(path, usecols=META, keep_default_na=False)
        id_sets = {
            "train": set(x.loc[x["split"] == "train", "sample_id"]),
            "scale_fit": set(x.loc[(x["split"] == "local_calibration") & (x["local_calibration_role"] == "scale_fit"), "sample_id"]),
            "tuning_calibration": set(x.loc[(x["split"] == "local_calibration") & (x["local_calibration_role"] == "tuning_calibration"), "sample_id"]),
            "validation": set(x.loc[x["split"] == "validation", "sample_id"]),
            "final_calibration": set(x.loc[x["split"] == "final_calibration", "sample_id"]),
            "test": set(x.loc[x["split"] == "test", "sample_id"]),
        }
        overlaps = {f"{a}__{b}": len(id_sets[a] & id_sets[b]) for i, a in enumerate(id_sets) for b in list(id_sets)[i + 1:]}
        overlap_checks.append({"replicate_index": index, "all_disjoint": max(overlaps.values(), default=0) == 0,
                               "pairwise_overlap_counts": overlaps})

    status_counts = (weights.groupby(["scenario", "replicate_index"])["status"].first()
                     .groupby(level=0).value_counts().unstack(fill_value=0).to_dict(orient="index"))
    selected = weights[weights["weight"] > 0]
    feature_counts = selected.groupby(["scenario", "feature"])["replicate_index"].nunique().to_dict()
    report = {
        "status": "BLOCKED_MISSING_FROZEN_DOWNSTREAM_LABELS" if missing else "PASS",
        "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256(manifest_path)},
        "stage9_results": {"path": str(STAGE9_RESULTS.relative_to(ROOT)), "sha256": sha256(STAGE9_RESULTS)},
        "stage9_weights": {"path": str(STAGE9_WEIGHTS.relative_to(ROOT)), "sha256": sha256(STAGE9_WEIGHTS)},
        "split_integrity": "PASS",
        "paired_integrity": "PASS",
        "weight_reproduction": "PASS",
        "weight_status_counts": status_counts,
        "selected_feature_replicate_counts": {f"{s}/{f}": int(v) for (s, f), v in feature_counts.items()},
        "missing_required_artifacts_count": len(missing),
        "missing_required_artifacts": missing,
        "artifact_inventory": artifact_rows,
        "paired_checks": paired_checks,
        "role_disjointness": overlap_checks,
        "forbidden_recovery": "Regenerating outcomes from generator seed/config is prohibited by Stage-11 boundaries.",
    }
    return report


def conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ProtocolBlocker("nonfinite conformal score")
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    if rank > len(scores):
        return math.inf, rank
    return float(np.sort(scores, kind="mergesort")[rank - 1]), rank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args()
    report = artifact_preflight()
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(text, encoding="utf-8")
    print(text, end="")
    if report["status"] != "PASS":
        print("STAGE11 STOP: downstream frozen labels/truth are absent; simulation regeneration is forbidden.", file=sys.stderr)
        return 2
    raise NotImplementedError("Full CP execution is gated until immutable downstream outcome artifacts pass preflight.")


if __name__ == "__main__":
    raise SystemExit(main())
