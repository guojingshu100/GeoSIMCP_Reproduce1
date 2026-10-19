"""Create and cryptographically seal Stage-19B before scientific generation."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/geo_causal_cp/20260919_stage19b_direct_geosimcp_parity_v1"
PREREG = RECORD / "preregistration"
SEEDS = list(range(20840001, 20840051))
OFFICIAL = ROOT / "external/geoconformal_stage19a_audit_26419ef"
COMMIT = "26419ef06d7e3711e6618f97f4f1ed02af16bcb7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def scan_seeds() -> tuple[list[str], list[int]]:
    files, used = [], set()
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
    utc, local = datetime.now(timezone.utc), datetime.now().astimezone()
    grids = {
        "GeoCP": {"bandwidth": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0], "lambda": None},
        "joint_methods": {
            "bandwidth": [float(x) for x in __import__("numpy").linspace(0.1, 5.0, 20)],
            "lambda": [float(x) for x in __import__("numpy").arange(0, 1.05, 0.05)],
            "iteration_order": "bandwidth ascending outer loop; lambda ascending inner loop",
        },
    }
    dump(PREREG / "seed_manifest.json", {
        "stage": "19B", "created_before_generation": True, "seeds": SEEDS,
        "first": SEEDS[0], "last": SEEDS[-1], "paired_S2_S4": True,
        "selection_rule": "unused deterministic contiguous 50-seed block after manifest/config scan",
        "manifest_config_files_scanned": len(files), "collision_count": 0,
        "utc_timestamp": utc.isoformat(), "local_timestamp": local.isoformat(),
    })
    dump(PREREG / "config.json", {
        "stage": "19B", "role": "FRESH_DIRECT_PARITY_EXPERIMENT", "alpha": 0.10,
        "official_commit": COMMIT, "paired_replicates": 50, "scenarios": ["S2", "S4"],
        "split_counts": {"train": 1000, "cmi_scale_fit": 200, "unused_tuning_calibration": 100,
                         "validation": 200, "cp_calibration": 200, "test": 300},
        "serialization": "%.17g CSV then reload before predictor fitting/downstream use",
        "methods": ["Raw-CP", "GeoCP", "GeoSIMCP-EUC", "GeoSIMCP-MND", "GeoCausalCP-Full", "CausalOnly-Full"],
        "principal_family": ["Raw-CP", "GeoCP", "GeoSIMCP-EUC", "GeoSIMCP-MND", "GeoCausalCP-Full"],
        "primary_contrast": "GeoCausalCP-Full minus GeoSIMCP-EUC",
        "tracks": {"A": {"include_test_atom": False, "role": "historical manuscript parity"},
                   "B": {"include_test_atom": True, "role": "primary current finite-sample-valid parity"}},
        "bootstrap": {"resamples": 10000, "seed": 21919001},
        "coverage_concern": "mean coverage<0.89 OR >0.01 below relevant comparator",
    })
    dump(PREREG / "method_specification.json", {
        "predictor": "exact Stage-15 train-only frozen RF protocol",
        "CMI_Full999": {"estimand": "I(X_j;A_model|s_x,s_y,all other candidate attributes)",
            "candidates": ["x1", "x2", "g_proxy", "x_noise"], "CMIknn_k": 20,
            "surrogates": 999, "BH_family": 4, "eligibility": "BH q<0.05 and raw CMI>0",
            "weights": "eligible raw CMI divided by eligible sum", "data": "CMI-scale-fit only"},
        "GeoCP": "raw-coordinate Euclidean Gaussian weights and official weighted step quantile",
        "GeoSIMCP_EUC": "calibration-fitted ddof=0 z-score EUC feature distance over four declared attributes",
        "GeoSIMCP_MND": "official maximum normalized absolute feature difference with range safeguard",
        "GeoCausalCP_Full": "official joint GeoSIMCP engine with ordinary feature distance replaced only by weighted CMI-Full causal distance",
        "causal_distance": "sum omega_j*(z_ij-z_*j)^2; z transform fit on CP calibration; coordinates excluded",
        "no_signal": "exact GeoCP fallback under same track and selected GeoCP bandwidth",
        "constant_selected_feature": "drop CP-calibration columns with population SD<=1e-8 and renormalize frozen eligible CMI weights; if none remain use GeoCP fallback",
        "CausalOnly_Full": "supporting Stage-18 normalized-CP ablation only; not part of official-engine contrast",
    })
    dump(PREREG / "track_specification.json", {
        "A": {"include_test_atom": False, "quantile_level": "ceil((1-alpha)*(n_cal+1))/n_cal, capped at 1", "infinite_atom": False},
        "B": {"include_test_atom": True, "quantile_level": "1-alpha", "self_weight": 1.0,
              "infinite_policy": "retain +infinity without truncation; report counts/fractions, finite-only width, and Kish n_eff"},
        "Raw_CP": "finite-sample order statistic ceil((n_cal+1)*(1-alpha)); common across tracks",
    })
    dump(PREREG / "tuning_specification.json", {
        "grids": grids, "labels": "validation only", "feasibility": "coverage>=0.90",
        "objective": "minimum mean interval score among feasible candidates", "tie": "first grid-order candidate",
        "no_feasible": {"status": "NO_FEASIBLE_HYPERPARAMETER", "ordinary_test_metrics": "not evaluated",
            "diagnostic": "report highest-validation-coverage candidate; tie by smaller interval score then grid order; never call selected"},
        "track_B_infinite_validation": "infinite widths/scores remain infinity and cannot minimize over a finite feasible candidate",
        "paired_inference": "only pairs with selected hyperparameters; width/score inference further requires finite replicate metrics; always report n",
        "classification": {"coverage_first": True, "limited_if_valid_pairs_below": 40,
            "practical_efficiency_margin_relative": 0.01,
            "A": "coverage acceptable and >=1% improvement in width or interval score with no >=1% deterioration in the other",
            "B": "coverage acceptable and both efficiency differences within 1% or uncertainty spans no difference without >=1% harm",
            "C": "coverage acceptable but no >=1% efficiency gain and at least one >=1% efficiency deterioration",
            "D": "material coverage concern", "E": "fewer than 40 paired selected-hyperparameter replicates"},
    })
    prereg_text = """# Stage 19B preregistration\n\nThis record was sealed before any new outcomes were generated. The primary contrast replaces only the ordinary EUC feature-similarity block in the audited official GeoSIMCP weighted-conformal architecture with CMI-Full999 residual-scale-relevance-weighted similarity. Track A preserves historical no-test-atom behavior; Track B is the primary current finite-sample-valid analysis and retains all infinite intervals. Validation alone selects hyperparameters. Test labels remain sealed until every method setting is frozen.\n"""
    (PREREG / "PREREGISTRATION.md").write_text(prereg_text, encoding="utf-8")
    (PREREG / "analysis_plan.md").write_text("""# Analysis plan\n\nReport Track A and B separately. Coverage precedes efficiency. Primary paired inference is GeoCausalCP-Full minus GeoSIMCP-EUC; GeoCP is the principal secondary comparator. Use Student-t and fixed-seed 10,000-resample paired bootstrap intervals. Never cap Track-B infinite intervals. CMI feature recovery and S4 fallback are diagnostic and do not alter methods.\n""", encoding="utf-8")
    sources = [
        OFFICIAL / "geoconformal/geocp/utils.py", OFFICIAL / "geoconformal/geocp/weights.py",
        OFFICIAL / "geoconformal/GeoConformalSpatialRegression.py",
        OFFICIAL / "geoconformal/GeoSIMConformalSpatialRegression.py",
        ROOT / "causal_only_cp_v1/run_stage15_fresh_confirmatory_v1.py",
        ROOT / "causal_only_cp_v1/stage18_cmi_worker_v1.py",
        ROOT / "simulation_pilot_20260908/generator/generate_scenarios.py",
        ROOT / "simulation_pilot_20260908/generator/config_v1.json",
        ROOT / "simulation_pilot_20260908/predictor/fit_predictors.py",
        ROOT / "simulation_pilot_20260908/predictor/predictor_config_v1.json",
    ]
    dump(PREREG / "input_manifest.json", {"created_before_generation": True,
        "official_checkout": str(OFFICIAL.relative_to(ROOT)).replace("\\", "/"), "official_commit": COMMIT,
        "seed_scan_files": files, "seed_scan_numeric_values": used,
        "sources": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p)} for p in sources]})
    (PREREG / "environment.txt").write_text(
        f"Created UTC: {utc.isoformat()}\nCreated local: {local.isoformat()}\nPython: {sys.version}\nExecutable: {sys.executable}\nPlatform: {platform.platform()}\n",
        encoding="utf-8")
    sealed = ["PREREGISTRATION.md", "config.json", "seed_manifest.json", "method_specification.json",
              "track_specification.json", "tuning_specification.json", "analysis_plan.md", "environment.txt", "input_manifest.json"]
    file_hashes = {name: sha(PREREG / name) for name in sealed}
    payload = "\n".join(f"{name} {file_hashes[name]}" for name in sorted(file_hashes))
    dump(PREREG / "PREREGISTRATION_SEAL.json", {"created_utc": utc.isoformat(), "files": file_hashes,
        "composite_sha256": hashlib.sha256(payload.encode()).hexdigest()})
    print(json.dumps({"record": str(RECORD), "seeds": [SEEDS[0], SEEDS[-1]],
                      "scanned": len(files), "composite": json.loads((PREREG/"PREREGISTRATION_SEAL.json").read_text())["composite_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
