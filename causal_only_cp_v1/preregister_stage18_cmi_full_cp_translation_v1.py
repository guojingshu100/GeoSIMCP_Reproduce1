"""Create and seal Stage-18 before any new S2/S4 outcome generation."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/causal_only_cp/20260919_stage18_cmi_full_cp_translation_v1"
SEEDS = list(range(20730001, 20730051))
SOURCES = [
    ROOT / "simulation_pilot_20260908/generator/generate_scenarios.py",
    ROOT / "simulation_pilot_20260908/generator/config_v1.json",
    ROOT / "simulation_pilot_20260908/predictor/fit_predictors.py",
    ROOT / "simulation_pilot_20260908/predictor/predictor_config_v1.json",
    ROOT / "causal_only_cp_v1/run_stage15_fresh_confirmatory_v1.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def scan() -> tuple[list[str], list[int]]:
    files, used = [], set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or RECORD in path.parents:
            continue
        name = path.name.lower()
        if path.suffix.lower() not in {".json", ".csv", ".md", ".txt"} or not ("manifest" in name or name == "config.json"):
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
    RECORD.mkdir(parents=True)
    files, used = scan()
    overlap = sorted(set(SEEDS) & set(used))
    if overlap:
        raise RuntimeError(f"seed collision: {overlap}")
    now_utc, now_local = datetime.now(timezone.utc), datetime.now().astimezone()
    seed_manifest = {
        "stage": "18", "created_before_generation": True,
        "selection_rule": "first chosen unused contiguous 50-seed development block after scanning project manifest/config artifacts",
        "seeds": SEEDS, "first": SEEDS[0], "last": SEEDS[-1], "paired_S2_S4": True,
        "excluded_stage15": list(range(20320001, 20320051)),
        "excluded_stage17": list(range(20420001, 20420051)),
        "manifest_config_files_scanned": len(files), "collision_count": len(overlap),
        "utc_timestamp": now_utc.isoformat(), "local_timestamp": now_local.isoformat(),
    }
    dump(RECORD / "seed_manifest.json", seed_manifest)
    dump(RECORD / "seed_scan_audit.json", {"files_scanned": files, "numeric_seed_values": used, "candidate_overlap": overlap})
    config = {
        "stage": "18", "role": "DEVELOPMENT", "created_before_generation": True,
        "scenarios": ["S2", "S4"], "paired_replicates": 50,
        "split_counts": {"train": 1000, "scale_fit": 200, "tuning_calibration": 100, "validation": 200, "final_calibration": 200, "test": 300},
        "serialization": "%.17g CSV then reload before predictor fit/downstream use",
        "predictor": "exact Stage-15 train-only frozen RF implementation/configuration",
        "candidates": ["x1", "x2", "g_proxy", "x_noise"],
        "cmi": {
            "methods": {
                "CMI_MSPACE_999": "I(X_j;A_model|s_x,s_y)",
                "CMI_FULL_999": "I(X_j;A_model|s_x,s_y,all other candidate attributes)",
            },
            "estimator": "Tigramite CMIknn", "k": 20, "surrogates": 999,
            "shuffle_neighbors": 5, "sig_blocklength": 1, "transform": False, "workers": 1,
            "target": "A_model=abs(y-y_pred_frozen) on scale_fit only",
            "selection": "BH q<0.05 and raw CMI>0 across four candidates",
            "weight": "raw CMI / sum eligible raw CMI", "fallback": "NO_CAUSAL_SIGNAL",
            "shuffle_seed_rule": "21900000 + 1000*replicate_index + scenario_offset(0 S2,100 S4) + 10*method_index + feature_index; method/feature indices zero-based",
        },
        "methods": ["Raw-CP", "GeoOnly-NormCP", "CausalOnly-CMI-Mspace999-NormCP", "CausalOnly-CMI-Full999-NormCP", "Oracle-NormCP"],
        "local_scale": {"reference": "scale_fit", "k_candidates": [20, 40, 60, 80], "criterion": "tuning-calibration residual-scale MAE", "tie": "larger k", "estimator": "mean absolute residual among deterministic k nearest neighbors"},
        "causal_distance": "sum eligible w_j*(scale-fit standardized feature difference)^2; coordinates prohibited",
        "geo_distance": "standardized s_x,s_y only, equal coordinate weights",
        "alpha": 0.10, "final_calibration_n": 200, "finite_sample_rank_formula": "ceil((n_cal+1)*(1-alpha))",
        "material_coverage_concern": "mean coverage<0.89 OR more than 0.01 below Raw-CP",
        "paired_ci": {"student_t": 0.95, "bootstrap": 0.95, "bootstrap_resamples": 10000, "bootstrap_seed": 21918001},
        "decision_classes": ["FULL_CMI_IMPROVES_CAUSAL_VALIDITY_AND_CP", "FULL_CMI_IMPROVES_CAUSAL_VALIDITY_WITH_SIMILAR_CP", "FULL_CMI_IMPROVES_CAUSAL_VALIDITY_BUT_LOSES_TOO_MUCH_CP_POWER", "FULL_CMI_HAS_COVERAGE_CONCERN", "FULL_CMI_DOES_NOT_TRANSFER_TO_S2_S4"],
        "prohibitions": ["no Stage-15/17 mutation", "no directional causality claim", "no geography integration", "no GCCM/GCMC/GPC/SCT/SCPCM/GeoCMI", "no result-driven tuning"],
    }
    dump(RECORD / "config.json", config)
    prereg = """# Stage 18 preregistration\n\nThis development benchmark was sealed before generating the new S2/S4 outcomes. It compares CMI-Mspace and CMI-Full at identical CMIknn k=20 and 999-surrogate resolution inside the same causal-only normalized split-conformal shell. The only intended causal-method difference is the conditioning set.\n\nThe primary questions are coverage, efficiency versus Raw/GeoOnly, direct Full-versus-Mspace efficiency, oracle-scale recovery, and nuisance/proxy selection. Test labels remain sealed until weights, k, and q are frozen. CMI is described only as conditional-dependence/residual-scale-relevance evidence. No Geo-Causal architecture is developed here.\n"""
    (RECORD / "PREREGISTRATION.md").write_text(prereg, encoding="utf-8")
    input_manifest = {"created_before_generation": True, "sources": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p)} for p in SOURCES], "stage15_record_read_only": "experiments/confirmatory_simulation/20260918_stage15_fresh_confirmatory_v1", "stage17_record_read_only": "experiments/causal_strength_benchmark/20260918_stage17_causal_estimator_validation_v1"}
    dump(RECORD / "input_manifest.json", input_manifest)
    (RECORD / "environment.txt").write_text(f"Created UTC: {now_utc.isoformat()}\nCreated local: {now_local.isoformat()}\nPython: {sys.version}\nExecutable: {sys.executable}\nPlatform: {platform.platform()}\nCMI runtime: causal_strength_diagnostic_v1/.venv_cmi_v1/Scripts/python.exe\n", encoding="utf-8")
    (RECORD / "run_log.txt").write_text("Stage-18 preregistration created before generation.\n", encoding="utf-8")
    (RECORD / "stderr.txt").write_text("", encoding="utf-8")
    sealed_names = ["PREREGISTRATION.md", "config.json", "seed_manifest.json", "input_manifest.json", "environment.txt", "seed_scan_audit.json"]
    seal = {"created_utc": now_utc.isoformat(), "files": {n: sha(RECORD / n) for n in sealed_names}}
    payload = "\n".join(f"{n} {seal['files'][n]}" for n in sorted(seal["files"]))
    seal["composite_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    dump(RECORD / "PREREGISTRATION_SEAL.json", seal)
    print(json.dumps({"record": str(RECORD), "seeds": [SEEDS[0], SEEDS[-1]], "files_scanned": len(files), "composite_sha256": seal["composite_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
