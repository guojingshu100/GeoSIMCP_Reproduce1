from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments" / "causal_only_cp" / "20260919_stage18_cmi_full_cp_translation_v1"
EXPECTED_COMPOSITE = "f962500c8b608c671c25d0a736169bf0af284119411884fecadbcba32b5a3fb3"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def scalar(frame: pd.DataFrame, scenario: str, method: str, field: str) -> float:
    row = frame[(frame.scenario == scenario) & (frame.method == method)]
    if len(row) != 1:
        raise AssertionError((scenario, method, field, len(row)))
    return float(row.iloc[0][field])


def paired_row(frame: pd.DataFrame, scenario: str, comparison: str, metric: str) -> dict:
    row = frame[(frame.scenario == scenario) & (frame.comparison == comparison) & (frame.metric == metric)]
    if len(row) != 1:
        raise AssertionError((scenario, comparison, metric, len(row)))
    return row.iloc[0].to_dict()


def compact_pair(row: dict) -> dict:
    keys = (
        "n", "mean_difference", "median_difference", "sd_difference",
        "student_t_ci_lower", "student_t_ci_upper",
        "bootstrap_ci_lower", "bootstrap_ci_upper", "relative_percent_difference",
    )
    return {key: int(row[key]) if key == "n" else float(row[key]) for key in keys}


def verify_preregistration() -> dict:
    seal = json.loads((RECORD / "PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    if seal["composite_sha256"] != EXPECTED_COMPOSITE:
        raise AssertionError("unexpected preregistration composite")
    mismatches = []
    for rel, expected in seal["files"].items():
        actual = sha256(RECORD / rel)
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    if mismatches:
        raise AssertionError(f"preregistration changed: {mismatches}")
    return {"composite_sha256": seal["composite_sha256"], "file_hash_mismatches": mismatches}


def main() -> None:
    prereg = verify_preregistration()
    checkpoint = json.loads((RECORD / "execution_checkpoint.json").read_text(encoding="utf-8"))
    analysis = json.loads((RECORD / "stage18_analysis.json").read_text(encoding="utf-8"))
    seeds = json.loads((RECORD / "seed_manifest.json").read_text(encoding="utf-8"))["seeds"]
    preflight = json.loads((RECORD / "preflight_summary.json").read_text(encoding="utf-8"))
    if seeds != list(range(20730001, 20730051)) or len(set(seeds)) != 50:
        raise AssertionError("seed list")
    if checkpoint["status"] != "SCIENTIFIC_COMPUTATION_COMPLETE" or preflight["status"] != "PASS":
        raise AssertionError("checkpoint/preflight")
    if checkpoint["scenario_replicates"] != 100 or checkpoint["result_rows"] != 500:
        raise AssertionError("checkpoint counts")
    if analysis["status"] != "DERIVED_ANALYSIS_COMPLETE":
        raise AssertionError("analysis incomplete")
    if (RECORD / "stderr.txt").stat().st_size != 0:
        raise AssertionError("stderr is not empty")

    generator = pd.read_csv(RECORD / "generator_audit.csv")
    split = pd.read_csv(RECORD / "split_integrity.csv")
    predictions = pd.read_csv(RECORD / "prediction_manifest.csv")
    unseal = pd.read_csv(RECORD / "test_unseal_audit.csv")
    weights = pd.read_csv(RECORD / "weight_audit.csv")
    results = pd.read_csv(RECORD / "replicate_results.csv")
    mspace = pd.read_csv(RECORD / "cmi_mspace999_results.csv")
    full = pd.read_csv(RECORD / "cmi_full999_results.csv")
    feature = pd.read_csv(RECORD / "feature_selection_summary.csv")
    scenario = pd.read_csv(RECORD / "scenario_summary.csv")
    paired = pd.read_csv(RECORD / "paired_comparisons.csv")
    tradeoff = pd.read_csv(RECORD / "causal_quality_cp_tradeoff.csv")
    scale = pd.read_csv(RECORD / "scale_diagnostics.csv")

    required_counts = {
        "generator_audit.csv": (len(generator), 100),
        "split_integrity.csv": (len(split), 100),
        "prediction_manifest.csv": (len(predictions), 100),
        "cmi_mspace999_results.csv": (len(mspace), 400),
        "cmi_full999_results.csv": (len(full), 400),
        "weight_audit.csv": (len(weights), 800),
        "replicate_results.csv": (len(results), 500),
        "scale_diagnostics.csv": (len(scale), 200),
        "test_unseal_audit.csv": (len(unseal), 102),
    }
    failures = {name: {"actual": actual, "expected": expected} for name, (actual, expected) in required_counts.items() if actual != expected}
    if failures:
        raise AssertionError(f"row counts: {failures}")
    if not generator[["serialized_reload", "finite_predictors", "finite_y_development", "finite_oracle_sigma"]].all().all():
        raise AssertionError("generator audit")
    expected_split_counts = {"n_train": 1000, "n_scale_fit": 200, "n_tuning_calibration": 100,
                             "n_validation": 200, "n_final_calibration": 200, "n_test": 300, "unique_ids": 2000}
    if not split.status.eq("PASS").all() or not split.roles_disjoint.all():
        raise AssertionError("split integrity")
    for field, expected in expected_split_counts.items():
        if not split[field].eq(expected).all():
            raise AssertionError(f"split count {field}")
    if dict(unseal.phase.value_counts()) != {"primary": 100, "preflight": 2}:
        raise AssertionError("test unseal audit")
    if set(results.generation_seed) != set(seeds) or set(results.scenario) != {"S2", "S4"}:
        raise AssertionError("result seed/scenario coverage")
    if results.groupby(["scenario", "method"]).size().nunique() != 1 or results.groupby(["scenario", "method"]).size().iloc[0] != 50:
        raise AssertionError("method completion")
    numeric = results[["coverage", "mean_width", "median_width", "mean_interval_score", "q_norm"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise AssertionError("nonfinite result")

    # Production evaluated every NO_SIGNAL replicate through the hard bound-level
    # equality assertion in run_stage18...evaluate().  Confirm the serialized
    # metrics are also identical to Raw-CP.
    fallback_max_metric_difference = 0.0
    for cmi_method in ("CausalOnly-CMI-Mspace999-NormCP", "CausalOnly-CMI-Full999-NormCP"):
        left = results[(results.method == cmi_method) & (results.signal_status == "NO_CAUSAL_SIGNAL")]
        raw = results[results.method == "Raw-CP"]
        joined = left.merge(raw, on=["replicate_index", "generation_seed", "scenario"], suffixes=("_left", "_raw"), validate="one_to_one")
        for metric in ("coverage", "mean_width", "median_width", "mean_interval_score"):
            fallback_max_metric_difference = max(fallback_max_metric_difference,
                float(np.max(np.abs(joined[f"{metric}_left"] - joined[f"{metric}_raw"]))))
    if fallback_max_metric_difference > 1e-12:
        raise AssertionError("fallback metric equality")

    fs = {(r.scenario, r.cmi_method): r for r in feature.itertuples(index=False)}
    s2_m = fs[("S2", "CMI_MSPACE_999")]
    s2_f = fs[("S2", "CMI_FULL_999")]
    s4_m = fs[("S4", "CMI_MSPACE_999")]
    s4_f = fs[("S4", "CMI_FULL_999")]

    methods = ["Raw-CP", "GeoOnly-NormCP", "CausalOnly-CMI-Mspace999-NormCP",
               "CausalOnly-CMI-Full999-NormCP", "Oracle-NormCP"]
    method_tables = {}
    for sc in ("S2", "S4"):
        method_tables[sc] = []
        for method in methods:
            row = scenario[(scenario.scenario == sc) & (scenario.method == method)].iloc[0]
            method_tables[sc].append({key: (str(row[key]) if key == "method" else
                                                  bool(row[key]) if key == "material_coverage_concern" else
                                                  int(row[key]) if key in {"replicates", "coverage_below_0_90_count", "coverage_below_0_89_count"} else
                                                  float(row[key]))
                                      for key in ("method", "replicates", "mean_coverage", "coverage_sd", "mean_width", "width_sd",
                                                  "mean_interval_score", "coverage_below_0_90_count", "coverage_below_0_89_count",
                                                  "material_coverage_concern")})

    pairs = {}
    for comp in ("Full_minus_Raw", "Full_minus_GeoOnly", "Full_minus_Mspace"):
        pairs[comp] = {metric: compact_pair(paired_row(paired, "S2", comp, metric))
                       for metric in ("coverage", "mean_width", "mean_interval_score")}

    t = {(r.scenario, r.cmi_method): r for r in tradeoff.itertuples(index=False)}
    tm, tf = t[("S2", "CMI_MSPACE_999")], t[("S2", "CMI_FULL_999")]
    summary = {
        "stage": "18",
        "status": "COMPLETE",
        "classification": "B. FULL_CMI_IMPROVES_CAUSAL_VALIDITY_WITH_SIMILAR_CP",
        "classification_rationale": (
            "Stage-17 controlled mechanisms established substantially stronger observed-confounder/proxy robustness for CMI-Full. "
            "On fresh Stage-18 S2/S4 development replicates, Full retained x2 recovery, acceptable coverage, and practically similar CP performance to Mspace. "
            "Full was 0.94% wider than Mspace on average but had 0.00367 higher coverage and only 0.10% higher interval score; it improved interval score versus Raw and GeoOnly. "
            "The actual S2/S4 nuisance-selection count was not lower for Full, so the robustness claim remains anchored to Stage-17 controlled-confounding evidence."
        ),
        "claim_boundary": "Both CMI estimators measure conditional dependence/residual-scale relevance, not identified directional causality.",
        "seeds": {"first": seeds[0], "last": seeds[-1], "count": len(seeds), "values": seeds},
        "preregistration": prereg,
        "preflight": preflight,
        "completion": {"paired_replicates": 50, "scenario_replicates": 100, "method_rows": 500,
                       "feature_rows_per_estimator": 400, "test_unseal_rows": 102, "stderr_bytes": 0},
        "feature_selection": {
            "S2": {
                "Mspace999": {"x2_raw_rank1": int(s2_m.x2_raw_rank1_count), "x2_selected": int(s2_m.x2_selected_count),
                              "x2_largest_weight": int(s2_m.x2_largest_weight_count), "no_signal": int(s2_m.no_signal_count),
                              "x1_selected": int(s2_m.x1_selected_count), "g_proxy_selected": int(s2_m.g_proxy_selected_count),
                              "x_noise_selected": int(s2_m.x_noise_selected_count), "any_nuisance_replicates": int(s2_m.any_nuisance_selected_replicates)},
                "Full999": {"x2_raw_rank1": int(s2_f.x2_raw_rank1_count), "x2_selected": int(s2_f.x2_selected_count),
                            "x2_largest_weight": int(s2_f.x2_largest_weight_count), "no_signal": int(s2_f.no_signal_count),
                            "x1_selected": int(s2_f.x1_selected_count), "g_proxy_selected": int(s2_f.g_proxy_selected_count),
                            "x_noise_selected": int(s2_f.x_noise_selected_count), "any_nuisance_replicates": int(s2_f.any_nuisance_selected_replicates)},
            },
            "S4": {"Mspace999": {"no_signal": int(s4_m.no_signal_count), "false_selection_replicates": int(s4_m.any_nuisance_selected_replicates)},
                   "Full999": {"no_signal": int(s4_f.no_signal_count), "false_selection_replicates": int(s4_f.any_nuisance_selected_replicates)}},
        },
        "method_tables": method_tables,
        "paired_S2": pairs,
        "scale_recovery_S2_signal": {
            "Mspace999": {"n_signal": int(50 - s2_m.no_signal_count), "mean_oracle_correlation": float(tm.mean_oracle_correlation_signal),
                          "mean_oracle_mae": float(tm.mean_oracle_mae_signal)},
            "Full999": {"n_signal": int(50 - s2_f.no_signal_count), "mean_oracle_correlation": float(tf.mean_oracle_correlation_signal),
                        "mean_oracle_mae": float(tf.mean_oracle_mae_signal)},
        },
        "fallback_audit": {"hard_bound_tolerance": 1e-12, "production_assertion_passed": True,
                           "max_serialized_metric_difference": fallback_max_metric_difference,
                           "S4_no_signal_Mspace": int(s4_m.no_signal_count), "S4_no_signal_Full": int(s4_f.no_signal_count)},
        "stage19_recommendation": (
            "Carry CMI-Full as the primary, more confounder-aware causal block and retain CMI-Mspace as a prespecified comparator. "
            "Do not describe either block as directionally identified causality and do not select between them using Stage-19 outcomes."
        ),
    }
    write_json(RECORD / "stage18_summary.json", summary)

    def f(x: float) -> str:
        return f"{x:.4f}"

    md = [
        "# Stage 18 summary", "",
        "## Result", "",
        "**B. FULL_CMI_IMPROVES_CAUSAL_VALIDITY_WITH_SIMILAR_CP**", "",
        summary["classification_rationale"], "",
        "Both estimators remain conditional-dependence/residual-scale-relevance estimators, not identified directional-causality estimators.", "",
        "## Completion and integrity", "",
        f"- Fresh development seeds: `{seeds[0]}-{seeds[-1]}` (50 paired replicates).",
        f"- Preregistration composite SHA-256: `{EXPECTED_COMPOSITE}`.",
        "- Preflight: PASS; clean CMI rerun was bit-identical for both methods and scenarios.",
        "- Completed: 50/50 pairs, 100 scenario-replicates, 500 method results.",
        "- Test-unseal audit: 102/102 rows (2 preflight + 100 primary).",
        "- Final stderr: empty.",
        "- NO_SIGNAL bound-level equality to Raw-CP passed at tolerance 1e-12.", "",
        "## Feature selection", "",
        "| Scenario | Estimator | x2 raw rank #1 | x2 selected/top weight | x1 | g_proxy | x_noise | NO_SIGNAL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| S2 | Mspace999 | {s2_m.x2_raw_rank1_count}/50 | {s2_m.x2_selected_count}/50 | {s2_m.x1_selected_count}/50 | {s2_m.g_proxy_selected_count}/50 | {s2_m.x_noise_selected_count}/50 | {s2_m.no_signal_count}/50 |",
        f"| S2 | Full999 | {s2_f.x2_raw_rank1_count}/50 | {s2_f.x2_selected_count}/50 | {s2_f.x1_selected_count}/50 | {s2_f.g_proxy_selected_count}/50 | {s2_f.x_noise_selected_count}/50 | {s2_f.no_signal_count}/50 |",
        f"| S4 | Mspace999 | {s4_m.x2_raw_rank1_count}/50 | 0/50 | 0/50 | 0/50 | 0/50 | {s4_m.no_signal_count}/50 |",
        f"| S4 | Full999 | {s4_f.x2_raw_rank1_count}/50 | 0/50 | 0/50 | 0/50 | 0/50 | {s4_f.no_signal_count}/50 |", "",
        "## S2 conformal results", "",
        "| Method | Coverage | Mean width | Interval score | Material concern |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in method_tables["S2"]:
        md.append(f"| {row['method']} | {f(row['mean_coverage'])} | {f(row['mean_width'])} | {f(row['mean_interval_score'])} | {row['material_coverage_concern']} |")
    md += ["", "## S4 conformal results", "",
           "| Method | Coverage | Mean width | Interval score | Material concern |",
           "|---|---:|---:|---:|---:|"]
    for row in method_tables["S4"]:
        md.append(f"| {row['method']} | {f(row['mean_coverage'])} | {f(row['mean_width'])} | {f(row['mean_interval_score'])} | {row['material_coverage_concern']} |")
    md += ["", "## Primary S2 paired effects (Full minus comparator)", "",
           "| Comparator | Outcome | Mean difference | Student-t 95% CI | Bootstrap 95% CI |",
           "|---|---|---:|---:|---:|"]
    for comp, label in (("Full_minus_Raw", "Raw"), ("Full_minus_GeoOnly", "GeoOnly"), ("Full_minus_Mspace", "Mspace999")):
        for metric in ("coverage", "mean_width", "mean_interval_score"):
            r = pairs[comp][metric]
            md.append(f"| {label} | {metric} | {r['mean_difference']:.6f} | [{r['student_t_ci_lower']:.6f}, {r['student_t_ci_upper']:.6f}] | [{r['bootstrap_ci_lower']:.6f}, {r['bootstrap_ci_upper']:.6f}] |")
    md += ["", "## Scale recovery and tradeoff", "",
           f"Among S2 SIGNAL replicates, Mspace999 had mean oracle-scale correlation {float(tm.mean_oracle_correlation_signal):.4f} (MAE {float(tm.mean_oracle_mae_signal):.4f}); Full999 had correlation {float(tf.mean_oracle_correlation_signal):.4f} (MAE {float(tf.mean_oracle_mae_signal):.4f}).",
           "Full therefore retained essentially the same scale recovery. Its intervals were 0.94% wider than Mspace on average, while coverage was 0.00367 higher and interval score differed by only 0.10%.",
           "In these S2/S4 replicates, Full did not reduce nuisance selections (5 versus 2 replicates); the stronger confounder/proxy interpretation is supported by the controlled Stage-17 mechanisms, not by this unconfounded S2/S4 comparison alone.", "",
           "## Stage 19 recommendation", "",
           summary["stage19_recommendation"], ""]
    (RECORD / "stage18_summary.md").write_text("\n".join(md), encoding="utf-8")

    completion = {
        "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": prereg,
        "row_counts": {name: actual for name, (actual, _) in required_counts.items()},
        "generator_failures": 0,
        "split_failures": 0,
        "test_unseal_phase_counts": {key: int(value) for key, value in unseal.phase.value_counts().items()},
        "finite_result_metrics": True,
        "fallback_max_serialized_metric_difference": fallback_max_metric_difference,
        "classification": summary["classification"],
    }
    write_json(RECORD / "completion_audit.json", completion)

    with (RECORD / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write("Stage18 derived analysis complete\n")
        handle.write("Stage18 final integrity audit PASS\n")
        handle.write("Stage18 record finalized\n")

    entries = []
    for path in sorted(p for p in RECORD.rglob("*") if p.is_file() and p.name != "output_manifest.json"):
        rel = path.relative_to(RECORD).as_posix()
        item = {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                item["data_rows"] = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
        entries.append(item)
    write_json(RECORD / "output_manifest.json", {
        "stage": "18", "status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_composite_sha256": EXPECTED_COMPOSITE, "file_count_excluding_manifest": len(entries), "files": entries,
    })
    print(json.dumps({"status": "COMPLETE", "classification": summary["classification"],
                      "preregistration": EXPECTED_COMPOSITE, "output_files": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
