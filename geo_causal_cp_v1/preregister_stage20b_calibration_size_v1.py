"""Create and cryptographically seal Stage-20B before scientific generation."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/geo_causal_cp/20260920_stage20b_calibration_size_feasibility_v1"
PREREG = RECORD / "preregistration"
SEEDS = list(range(20950001, 20950051))
OFFICIAL = ROOT / "external/geoconformal_stage19a_audit_26419ef"
COMMIT = "26419ef06d7e3711e6618f97f4f1ed02af16bcb7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def scan_seeds() -> tuple[list[str], list[int]]:
    files: list[str] = []
    used: set[int] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or RECORD in path.parents:
            continue
        name = path.name.lower()
        if path.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
            continue
        if "manifest" not in name and name != "config.json":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        files.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        used.update(int(x) for x in re.findall(r"(?<!\d)(20\d{6})(?!\d)", text))
    return sorted(files), sorted(used)


def main() -> None:
    if RECORD.exists():
        raise FileExistsError(RECORD)
    PREREG.mkdir(parents=True)
    files, used = scan_seeds()
    overlap = sorted(set(SEEDS) & set(used))
    if overlap:
        raise RuntimeError(f"seed collision: {overlap}")

    utc = datetime.now(timezone.utc)
    local = datetime.now().astimezone()
    joint_h = [float(x) for x in np.linspace(0.1, 5.0, 20)]
    lambdas = [float(x) for x in np.arange(0, 1.05, 0.05)]
    split_counts = {
        "train": 1000,
        "CMI_scale_fit": 200,
        "CP_calibration_pool": 800,
        "validation": 200,
        "test": 300,
    }
    methods = ["Raw-CP", "GeoCP", "GeoSIMCP-EUC", "GeoSIMCP-MND", "GeoCausalCP-Full"]

    dump(PREREG / "seed_manifest.json", {
        "stage": "20B", "created_before_generation": True, "seeds": SEEDS,
        "first": SEEDS[0], "last": SEEDS[-1], "paired_S2_S4": True,
        "selection_rule": "unused deterministic contiguous 50-seed block after project manifest/config scan",
        "manifest_config_files_scanned": len(files), "collision_count": 0,
        "utc_timestamp": utc.isoformat(), "local_timestamp": local.isoformat(),
    })
    dump(PREREG / "config.json", {
        "stage": "20B", "role": "DEVELOPMENT_CALIBRATION_SIZE_FEASIBILITY",
        "scientific_question": "vary only CP-calibration support while holding CMI scale-fit and the Stage-19B official engine fixed",
        "official_commit": COMMIT, "paired_replicates": 50, "scenarios": ["S2", "S4"],
        "alpha": 0.10, "split_counts": split_counts, "total_n": sum(split_counts.values()),
        "calibration_sizes": [200, 400, 800],
        "nested_rule": "sort the frozen 800-row CP-calibration pool by sample_id; C200=first 200, C400=first 400, C800=all 800",
        "serialization": "%.17g CSV then reload before predictor fitting/downstream use",
        "methods": methods,
        "tracks": {
            "A": {"include_test_atom": False, "role": "historical manuscript parity"},
            "B": {"include_test_atom": True, "role": "primary finite-sample-valid feasibility"},
        },
        "bootstrap": {"resamples": 10000, "seed": 22020001},
        "coverage_concern": "mean coverage<0.89 OR >0.01 below the relevant comparator",
    })
    dump(PREREG / "method_specification.json", {
        "generator": "exact frozen S2/S4 equations and fixed split RNG semantics, with n expanded only to supply an 800-row calibration pool",
        "predictor": "exact Stage-15/19B train-only frozen RF protocol",
        "CMI_Full999": {
            "estimand": "I(X_j;A_model|s_x,s_y,all other candidate attributes)",
            "data": "fixed 200-row CMI_scale_fit only; computed once per scenario-replicate",
            "candidates": ["x1", "x2", "g_proxy", "x_noise"],
            "CMIknn_k": 20, "surrogates": 999, "BH_family": 4,
            "eligibility": "BH q<0.05 and raw CMI>0",
            "weights": "eligible raw CMI divided by eligible sum",
            "invariance": "SIGNAL status, eligibility and weights must be byte/numerically identical for n_cal=200,400,800",
        },
        "Raw_CP": "ordinary finite-sample split conformal using the selected nested calibration subset",
        "GeoCP": "raw-coordinate Euclidean Gaussian weights and official weighted step quantile",
        "GeoSIMCP_EUC": "calibration-fitted ddof=0 z-score EUC distance over x1,x2,g_proxy,x_noise",
        "GeoSIMCP_MND": "official maximum normalized absolute feature difference with small-range safeguard",
        "GeoCausalCP_Full": "Stage-19B minimal-change official joint engine; only EUC feature distance is replaced by weighted CMI-Full causal distance",
        "no_signal": "exact GeoCP fallback under the same n_cal, track and selected GeoCP bandwidth",
        "kernel": "Gaussian exp(-0.5*(distance/(bandwidth+1e-8))^2)",
        "coordinates": "raw and unstandardized in the geographic component",
    })
    dump(PREREG / "track_specification.json", {
        "A": {
            "include_test_atom": False,
            "quantile_level": "ceil((1-alpha)*(n_cal+1))/n_cal, capped at 1",
            "infinite_atom": False,
        },
        "B": {
            "include_test_atom": True, "quantile_level": "1-alpha", "self_weight": 1.0,
            "infinite_policy": "retain +infinity without truncation",
            "exact_mass_condition": "finite iff S/(S+1)>=0.90 up to official step-quantile tolerance; at alpha=.10 threshold is S>=9",
        },
        "Raw_CP": "rank ceil((n_cal+1)*(1-alpha)), recomputed at every n_cal",
    })
    dump(PREREG / "tuning_specification.json", {
        "GeoCP_bandwidth": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
        "joint_bandwidth": joint_h,
        "joint_lambda": lambdas,
        "iteration_order": "bandwidth ascending outer loop; lambda ascending inner loop",
        "labels": "fixed 200-row validation split only",
        "feasibility": "validation coverage>=0.90",
        "objective": "minimum mean interval score among feasible candidates",
        "tie": "first candidate in frozen grid order",
        "no_feasible": {
            "status": "NO_FEASIBLE_HYPERPARAMETER",
            "ordinary_test_metrics": "not evaluated",
            "diagnostic": "highest-validation-coverage candidate; ties by smaller interval score then grid order; never selected",
        },
    })
    dump(PREREG / "analysis_plan.json", {
        "primary_track": "B", "primary_scenario": "S2",
        "primary_contrast": "GeoCausalCP-Full minus GeoSIMCP-EUC separately at n_cal=200,400,800",
        "strata": ["all", "SIGNAL", "clean_x2_only", "x2_plus_nuisance", "NO_SIGNAL"],
        "mass_outputs": ["mean S", "median S", "p05 S", "fraction S<9", "fraction S>=9", "infinite fraction"],
        "paired_inference": "Student-t and fixed-seed 10,000-resample paired bootstrap; never pool calibration sizes",
        "aggregate_infinity": "aggregate mean width and interval score remain +infinity whenever any interval in the summary is infinite",
        "finite_only": "diagnostic only",
        "substantial_mass_improvement": ">=50% relative reduction in Track-B infinite fraction from n_cal=200 to 800",
        "resolved_infinity": "Track-B infinite fraction <=0.01 at n_cal=800 in the relevant S2 SIGNAL comparison",
        "competitive_efficiency": "no coverage concern and neither width nor interval score is >1% worse than EUC; improvement may be in either efficiency outcome",
        "decision_order": [
            "F if coverage concern",
            "E if infinity reduction is <50% and n800 remains >1%",
            "B if reduction is >=50% but n800 remains >1%",
            "C if infinity is resolved but SIGNAL GeoCausal is not competitive",
            "D if infinity is resolved and SIGNAL is competitive but aggregate failure is explained chiefly by unchanged NO_SIGNAL fallback",
            "A if infinity is resolved/substantially reduced and SIGNAL GeoCausal remains competitive or better without coverage concern",
        ],
        "next_step_gate": {
            "A": "official minimal-change route may proceed to dedicated confirmation",
            "D": "hold engine fixed; separately vary only CMI-scale-fit sample size on new development seeds",
            "C": "investigate official weighted-quantile shell interaction; do not increase CMI power first",
            "B_or_E": "do not promote; reassess feasibility",
        },
    })
    (PREREG / "PREREGISTRATION.md").write_text(
        "# Stage 20B preregistration\n\n"
        "This development feasibility study was sealed before generating outcomes. It varies only the nested CP-calibration size (200, 400, 800) drawn from one frozen 800-row pool. CMI-Full999 is computed once from a separate fixed 200-row scale-fit sample. The S2/S4 generator, train-only RF predictor, CMI gate and weights, feature/geographic metrics, Gaussian kernel, grids, validation objective, weighted step quantile, test atom, no-signal fallback, and alpha remain identical to Stage 19B. Track B is primary; Track A preserves historical parity.\n",
        encoding="utf-8",
    )
    (PREREG / "analysis_plan.md").write_text(
        "# Analysis plan\n\n"
        "Report each calibration size separately. Coverage precedes efficiency. Retain every Track-B infinity, and report finite-only quantities only as diagnostics. The primary mass analysis verifies the exact S/(S+1) condition and never substitutes ESS. Stratify S2 by fixed CMI status, with clean-x2 and x2+nuisance subgroups. CMI power is outside the intervention and must remain invariant across sizes.\n",
        encoding="utf-8",
    )

    sources = [
        OFFICIAL / "geoconformal/geocp/utils.py",
        OFFICIAL / "geoconformal/geocp/weights.py",
        OFFICIAL / "geoconformal/GeoConformalSpatialRegression.py",
        OFFICIAL / "geoconformal/GeoSIMConformalSpatialRegression.py",
        ROOT / "geo_causal_cp_v1/run_stage19b_direct_parity_v1.py",
        ROOT / "geo_causal_cp_v1/stage19b_cmi_full_worker_v1.py",
        ROOT / "experiments/geo_causal_cp/20260919_stage20a_stage19b_failure_audit_v1/stage20a_summary.json",
        ROOT / "simulation_pilot_20260908/generator/generate_scenarios.py",
        ROOT / "simulation_pilot_20260908/generator/config_v1.json",
        ROOT / "simulation_pilot_20260908/predictor/fit_predictors.py",
        ROOT / "simulation_pilot_20260908/predictor/predictor_config_v1.json",
    ]
    dump(PREREG / "input_manifest.json", {
        "created_before_generation": True,
        "official_checkout": str(OFFICIAL.relative_to(ROOT)).replace("\\", "/"),
        "official_commit": COMMIT,
        "seed_scan_files": files, "seed_scan_numeric_values": used,
        "sources": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p)} for p in sources],
    })
    (PREREG / "environment.txt").write_text(
        f"Created UTC: {utc.isoformat()}\nCreated local: {local.isoformat()}\nPython: {sys.version}\nExecutable: {sys.executable}\nPlatform: {platform.platform()}\n",
        encoding="utf-8",
    )
    sealed = [
        "PREREGISTRATION.md", "config.json", "seed_manifest.json", "method_specification.json",
        "track_specification.json", "tuning_specification.json", "analysis_plan.json", "analysis_plan.md",
        "environment.txt", "input_manifest.json",
    ]
    file_hashes = {name: sha(PREREG / name) for name in sealed}
    payload = "\n".join(f"{name} {file_hashes[name]}" for name in sorted(file_hashes))
    dump(PREREG / "PREREGISTRATION_SEAL.json", {
        "created_utc": utc.isoformat(), "files": file_hashes,
        "composite_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    })
    print(json.dumps({
        "record": str(RECORD), "seed_range": [SEEDS[0], SEEDS[-1]],
        "scanned_files": len(files), "composite_sha256": json.loads((PREREG / "PREREGISTRATION_SEAL.json").read_text())["composite_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
