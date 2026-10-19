"""Create the immutable Stage-15 preregistration before scientific generation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "confirmatory_simulation" / "20260918_stage15_confirmatory_preregistration_v1"
SEEDS = list(range(20320001, 20320051))
CMI999 = SEEDS[:20]
SOURCE_PATHS = [
    ROOT / "simulation_pilot_20260908" / "generator" / "config_v1.json",
    ROOT / "simulation_pilot_20260908" / "generator" / "generate_scenarios.py",
    ROOT / "simulation_pilot_20260908" / "predictor" / "predictor_config_v1.json",
    ROOT / "simulation_pilot_20260908" / "predictor" / "fit_predictors.py",
    ROOT / "causal_strength_diagnostic_v1" / "run_stage9_feature_recovery_v1.py",
    ROOT / "causal_only_cp_v1" / "run_stage11_causalonly_cmi_normcp_v2.py",
    ROOT / "causal_only_cp_v1" / "run_stage12_causalonly_gccm_ablation_v1.py",
    ROOT / "causal_only_cp_v1" / "gccm_stage12_attribute_wrapper_v1.R",
    ROOT / "causal_only_cp_v1" / "run_stage13_geocmi_balanced_normcp_v1.py",
    ROOT / "geoconformal-main" / "GCCP_Seattle" / "02_gccm" / "GCCM_updated 2" / "R" / "GCCM" / "GCCM4Lattice.r",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def used_seed_scan() -> dict:
    values: set[int] = set()
    files = 0
    for path in (ROOT / "experiments").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
            continue
        if not any(token in path.name.lower() for token in
                   ("manifest", "config", "summary", "audit", "result", "log", "readme", "method", "protocol")):
            continue
        files += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        values.update(int(x) for x in re.findall(r"(?<!\d)(20\d{6})(?!\d)", text))
    collision = sorted(set(SEEDS) & values)
    if collision:
        raise AssertionError(f"confirmatory seed collision: {collision}")
    return {"experiment_record_files_scanned": files, "distinct_eight_digit_values_found": len(values),
            "candidate_block": [SEEDS[0], SEEDS[-1]], "collisions": collision,
            "known_stage8_13_block_present": all(x in values for x in range(20270908, 20270928)),
            "status": "PASS"}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to modify preregistration: {OUT}")
    for path in SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)
    scan = used_seed_scan()
    OUT.mkdir(parents=True)
    local = datetime.now().astimezone()
    utc = datetime.now(timezone.utc)
    common = {
        "stage": "Stage 15 fresh confirmatory simulation",
        "preregistered_local_timestamp": local.isoformat(),
        "preregistered_utc_timestamp": utc.isoformat(),
        "fresh_paired_replicates": 50,
        "scenarios": ["S2", "S4"],
        "seeds": SEEDS,
        "claim": "fresh confirmatory simulation replicates for an already-frozen method family",
    }
    config = {
        **common,
        "generator": {"version": "gccp_sim_s1_s4_v1", "n": 2000,
                      "base_variable_seed": "confirmatory seed", "outer_split_seed": 620907,
                      "local_calibration_role_seed": 620908, "serialization": "CSV %.17g then reload"},
        "splits": {"train": 1000, "scale_fit": 200, "tuning_calibration": 100,
                   "validation": 200, "final_calibration": 200, "test": 300},
        "predictor": {"features": ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"],
                      "class": "RandomForestRegressor", "n_estimators": 300, "min_samples_leaf": 5,
                      "max_features": 1.0, "random_state": 420907, "train_only": True,
                      "prediction_aggregation": "fixed_estimators_order_mean"},
        "cmi99": {"condition": "Mspace", "features": ["x1", "x2", "g_proxy", "x_noise"],
                  "target": "abs(y-y_pred_frozen) on scale_fit", "knn": 20, "surrogates": 99,
                  "shuffle_neighbors": 5, "sig_blocklength": 1,
                  "seed_policy": "20260910 + replicate_index*10 + scenario_offset + 1; S2 offset=0, S4 offset=2",
                  "BH_family": 4, "eligibility": "BH q<0.05 and raw CMI>0",
                  "weights": "eligible raw CMI / sum eligible raw CMI", "fallback": "NO_CAUSAL_SIGNAL"},
        "gccm": {"E": 3, "tau": 1, "b": 4, "seed": 42, "libraries": [10, 18, 33, 60, 109, 198],
                 "direction": "effect_xmap_cause", "candidate_features": ["x1", "x2", "g_proxy", "x_noise"],
                 "convergence": {"gain_gt": 0, "spearman_gte": .60,
                    "terminal_pearson_two_sided_p_lt": .05, "fisher_z_95_ci_lower_gt": 0},
                 "strength": "terminal mean rho iff all gates pass; otherwise zero",
                 "fallback": "NO_USABLE_GCCM_SIGNAL"},
        "local_scale": {"reference": "scale_fit", "estimator": "unweighted mean absolute residual of k nearest",
                        "k_candidates": [20, 40, 60, 80], "selection": "tuning_calibration MAE; larger k on exact tie",
                        "standardization": "scale_fit only", "floor": 1e-12, "deterministic_ties": "sample_id"},
        "alphas": {"primary": .10, "secondary": [.05, .20],
                   "secondary_rule": "reuse predictor, weights, k, and test scale; recalibrate quantile only"},
        "bootstrap": {"resamples": 10000, "seed": 20260918, "role": "secondary robustness"},
        "test_policy": "separate sealed labels; audit barrier before label load",
        "validation_role": "unused by all Stage-15 methods",
    }
    methods = {
        "methods": ["Raw-CP", "GeoOnly-NormCP", "CausalOnly-GCCM-NormCP",
                    "CausalOnly-CMI-NormCP", "GeoCMI-Balanced-NormCP", "Oracle-NormCP"],
        "primary": "CausalOnly-CMI-NormCP",
        "primary_comparators": ["Raw-CP", "GeoOnly-NormCP"],
        "causal_distance": "sum w_j*(delta standardized selected feature_j)^2; no coordinates",
        "geo_distance": "0.5*delta_z_sx^2 + 0.5*delta_z_sy^2",
        "geocmi": "0.5*d_causal^2 + 0.5*d_geo^2; no mixing-weight tuning",
        "fallbacks": {"CausalOnly-CMI-NormCP": "Raw-CP exact", "CausalOnly-GCCM-NormCP": "Raw-CP exact",
                      "GeoCMI-Balanced-NormCP": "GeoOnly-NormCP exact"},
        "oracle": "generator sigma_true; never used for practical tuning",
        "prohibited": ["new methods", "new thresholds", "new k values", "new kernels", "new lambdas",
                       "new transforms", "new predictors", "test-based tuning"],
    }
    seed_manifest = {**common, "seed_scan": scan, "paired_seed_records": [
        {"replicate_index": i, "generation_seed": seed} for i, seed in enumerate(SEEDS, 1)],
        "cmi999_preregistered_first20": CMI999,
        "cmi999_runtime_benchmark_first2": CMI999[:2],
        "cmi999_runtime_gate": "projected first-20 runtime <=10800 seconds => 20 pairs; otherwise first 10 pairs",
    }
    analysis = """# Stage 15 analysis plan

## Primary hypotheses

At alpha=0.10, CausalOnly-CMI-NormCP is expected to avoid material coverage loss and reduce mean width and interval score relative to both Raw-CP and GeoOnly-NormCP.

Material undercoverage is mean coverage <0.89 or more than 0.01 below Raw-CP. Paired differences are method minus comparator. Every paired contrast reports mean, median, SD, Student-t 95% CI, and a fixed-seed 10,000-resample paired bootstrap 95% CI.

## Frozen classification

- D: the primary CMI method triggers material undercoverage.
- A: no material undercoverage; width and interval-score means are lower versus both primary comparators; for each comparator at least one efficiency metric has a Student-t CI wholly below zero; and each comparator has at least a 1% relative width reduction or 2% relative interval-score reduction.
- B: no material undercoverage and the expected efficiency direction generally replicates (at least three of four mean width/score contrasts are negative), but A is not met.
- C: no material undercoverage and fewer than three of the four efficiency contrasts are negative.

Classification uses coverage and effect sizes, not p-values alone.

## Secondary analyses

Alpha 0.05 and 0.20 reuse fitted scales and k. Their descriptive concern threshold is nominal minus 0.01 or more than 0.01 below Raw at the same alpha. CMI-999 uses the preregistered subset and runtime gate. Stability is first 25 versus second 25. S2 is stratified by frozen CMI99 SIGNAL/NO_SIGNAL status. GCCM and GeoCMI are unchanged negative/incremental-geography ablations.
"""
    prereg = """# Stage 15 preregistration

This record was sealed before generating any Stage-15 scientific observation. Fifty paired fresh seeds are used for unchanged S2/S4 mechanisms. The complete six-method family, CMI and GCCM protocols, conformal shell, alpha levels, hypotheses, inference, CMI-999 runtime gate, stability split, and classification are fixed in the accompanying JSON and analysis plan.

No result-driven tuning or method substitution is permitted. Any code repair must preserve this protocol and be documented only in the separate execution record.
"""
    readme = """# Stage 15 immutable preregistration

Fresh confirmatory simulation for the frozen GCCP causal-only method family. This directory must not be edited after its seal is created. Scientific outputs belong in the separate `20260918_stage15_fresh_confirmatory_v1` record.
"""
    env = [f"Local timestamp: {local.isoformat()}", f"UTC timestamp: {utc.isoformat()}",
           f"OS: {platform.platform()}", f"Python executable: {sys.executable}", f"Python: {platform.python_version()}"]
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "tigramite", "joblib"]:
        try: env.append(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError: env.append(f"{package}: NOT INSTALLED")
    rscript = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")
    r_version = subprocess.run([str(rscript), "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    env.append("R: " + (r_version.stdout.strip() or r_version.stderr.strip()))
    inputs = {"seed_scan": scan, "source_files": [
        {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in SOURCE_PATHS]}

    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "PREREGISTRATION.md").write_text(prereg, encoding="utf-8")
    dump(OUT / "config.json", config)
    dump(OUT / "method_family.json", methods)
    dump(OUT / "seed_manifest.json", seed_manifest)
    (OUT / "analysis_plan.md").write_text(analysis, encoding="utf-8")
    (OUT / "environment.txt").write_text("\n".join(env) + "\n", encoding="utf-8")
    dump(OUT / "input_manifest.json", inputs)

    named = ["README.md", "PREREGISTRATION.md", "config.json", "method_family.json", "seed_manifest.json",
             "analysis_plan.md", "environment.txt", "input_manifest.json"]
    hashes = {name: sha(OUT / name) for name in named}
    composite = hashlib.sha256("\n".join(f"{name} {hashes[name]}" for name in sorted(hashes)).encode()).hexdigest()
    dump(OUT / "PREREGISTRATION_SEAL.json", {"files": hashes, "composite_sha256": composite,
                                              "sealed_utc": utc.isoformat(), "immutable": True})
    print(json.dumps({"status": "SEALED", "path": str(OUT), "composite_sha256": composite,
                      "seeds": [SEEDS[0], SEEDS[-1]], "seed_count": len(SEEDS)}, indent=2))


if __name__ == "__main__":
    main()
