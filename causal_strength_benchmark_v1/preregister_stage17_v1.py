"""Create and seal the Stage-17 causal-estimator benchmark preregistration."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/causal_strength_benchmark/20260918_stage17_causal_estimator_validation_v1"
SEEDS = list(range(20420001, 20420051))
GRAPH = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b_gcmc_one_seed_smoke_v1/inputs/delaunay_edges.csv"
COORDS = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b_gcmc_one_seed_smoke_v1/inputs/S2_scale_fit_gcmc.csv"
SPEDM_ARCHIVE = ROOT / "external/spEDM_1.12_cran/spEDM_1.12.tar.gz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def scan_used_seeds() -> tuple[list[str], list[int]]:
    files: list[str] = []
    values: set[int] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or RECORD in path.parents:
            continue
        lower = path.name.lower()
        if not ("manifest" in lower and path.suffix.lower() in {".json", ".csv", ".md", ".txt"}):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        files.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        values.update(int(x) for x in re.findall(r"(?<!\d)(20\d{6})(?!\d)", text))
    return sorted(files), sorted(values)


def main() -> None:
    if RECORD.exists():
        raise FileExistsError(f"refusing to overwrite existing record: {RECORD}")
    RECORD.mkdir(parents=True)
    scanned, used = scan_used_seeds()
    overlap = sorted(set(SEEDS) & set(used))
    if overlap:
        raise RuntimeError(f"candidate Stage-17 seeds already occur in project manifests: {overlap}")

    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now().astimezone()
    seed_manifest = {
        "stage": "17",
        "created_before_scientific_generation": True,
        "selection_rule": "first unused contiguous 50-seed block beginning at 20420001 after scanning project manifest artifacts",
        "seeds": SEEDS,
        "first": SEEDS[0],
        "last": SEEDS[-1],
        "replicates_per_mechanism": 50,
        "paired_across_mechanisms": True,
        "stage15_excluded": list(range(20320001, 20320051)),
        "manifest_files_scanned": len(scanned),
        "candidate_overlap": overlap,
        "utc_timestamp": now_utc.isoformat(),
        "local_timestamp": now_local.isoformat(),
    }
    dump(RECORD / "seed_manifest.json", seed_manifest)
    dump(RECORD / "seed_scan_audit.json", {"files_scanned": scanned, "numeric_seeds_observed": used, "candidate_overlap": overlap})

    config = {
        "stage": "17",
        "status_at_creation": "PREREGISTERED_BEFORE_RESULTS",
        "n": 200,
        "replicates_per_mechanism": 50,
        "mechanisms": ["C0_INDEPENDENT_NULL", "C1_RESIDUAL_SCALE_CAUSAL", "C2_PURE_CONFOUNDING", "C3_REVERSE_DIRECTION", "C4_SPATIAL_LAG_CAUSAL", "C5_NONLINEAR_DIRECT"],
        "candidate_attributes": ["x1", "x2", "g_proxy", "x_noise", "Z"],
        "coordinates": ["s_x", "s_y"],
        "target": "A",
        "fixed_support": {
            "coordinates_path": str(COORDS.relative_to(ROOT)).replace("\\", "/"),
            "coordinates_sha256": sha256(COORDS),
            "graph_path": str(GRAPH.relative_to(ROOT)).replace("\\", "/"),
            "graph_sha256": sha256(GRAPH),
            "nodes": 200,
            "edges": 583,
            "spatial_covariance": "K_ij=exp(-EuclideanDistance_ij/rho)",
            "rho_rule": "2 * median frozen Delaunay edge length",
            "median_edge_length": 0.07956259239781337,
            "rho": 0.15912518479562674,
            "cholesky_jitter": 1e-10,
        },
        "mechanism_parameters": {
            "common": {
                "spatial_fields": "sample-standardized Gaussian fields (ddof=1)",
                "g_proxy": "standardize(0.75*x2 + 0.50*g_spatial + 0.25*standardize(s_x+s_y)); never enters A equation",
                "x1": "iid N(0,1) nuisance",
                "x_noise": "iid N(0,1) nuisance",
                "Z_outside_C2": "independent standardized spatial Gaussian nuisance",
            },
            "C0_INDEPENDENT_NULL": "x2 spatial; A=abs(epsilon), epsilon iid N(0,1); no candidate causes A",
            "C1_RESIDUAL_SCALE_CAUSAL": "x2 spatial; sigma=exp(0.6*x2); A=sigma*abs(epsilon)",
            "C2_PURE_CONFOUNDING": "Z spatial; x2=standardize(0.8*Z+0.6*u), u iid N(0,1); sigma=exp(0.6*Z); A=sigma*abs(epsilon); Z causes x2 and A; x2 has no direct arrow to A",
            "C3_REVERSE_DIRECTION": "A=exp(0.5*a_driver), a_driver spatial; x2=standardize(a_driver+0.5*eta); A/a_driver causes x2; x2 has no arrow to A",
            "C4_SPATIAL_LAG_CAUSAL": "x2 spatial; L=standardize(row-mean Delaunay lag of x2); sigma=exp(0.6*L); A=sigma*abs(epsilon)",
            "C5_NONLINEAR_DIRECT": "x2 spatial; A=x2^2+0.5*abs(epsilon); direct non-monotone x2-to-A effect",
        },
        "methods": {
            "CMI_MSPACE": {"estimator": "Tigramite CMIknn", "k": 20, "surrogates": 999, "conditioning": ["s_x", "s_y"], "selection": "BH q<0.05 and raw CMI>0 across five candidates", "directional": False},
            "CMI_FULL": {"estimator": "Tigramite CMIknn", "k": 20, "surrogates": 999, "conditioning": "s_x,s_y plus every other candidate", "selection": "BH q<0.05 and raw CMI>0 across five candidates", "directional": False},
            "GPC": {"implementation": "spEDM 1.12 gpc", "E": 3, "k": 5, "tau": 1, "style": 1, "detrend": True, "bidirectional": True, "threads": 1, "nb": "explicit frozen Delaunay", "score": "sum of forward positive+negative+dark pattern strengths", "reverse_score": "sum of reverse positive+negative+dark pattern strengths", "selection": "descriptive direction-preference only: forward>reverse and forward>0; no calibrated p-value", "directional": True},
            "SCT": {"implementation": "spEDM 1.12 sc.test", "k": 15, "block": 3, "boot": 399, "seed_rule": "method base plus mechanism/replicate/candidate offsets", "detrend": True, "normalize": False, "threads": 1, "nb": "explicit frozen Delaunay", "score": "sc[1] forward X->A; sc[3] reverse A->X", "selection": "BH-adjust forward bootstrap p<0.05 and forward statistic>0", "directional": True},
            "SCPCM": {"implementation": "spEDM 1.12 scpcm", "E": 3, "k": 5, "tau": 1, "style": 1, "detrend": True, "bidirectional": True, "threads": 1, "nb": "explicit frozen Delaunay", "conditioning": "all other candidates", "score": "pxmap y_xmap_x_mean (forward X->A); pxmap x_xmap_y_mean (reverse)", "selection": "BH q<0.05 using pxmap directional sig and mean>0", "secondary": True, "directional": True},
            "GCCM_REFERENCE": {"rerun": False, "role": "frozen Stage-15 reference only"},
        },
        "ranking": "descending primary forward score; deterministic candidate-name tie break",
        "truth_interpretation": {
            "driver_scenarios": ["C1_RESIDUAL_SCALE_CAUSAL", "C4_SPATIAL_LAG_CAUSAL", "C5_NONLINEAR_DIRECT"],
            "null_scenario": "C0_INDEPENDENT_NULL",
            "confounded_noncausal_x2": "C2_PURE_CONFOUNDING",
            "reverse_noncausal_x2": "C3_REVERSE_DIRECTION",
            "C2_direct_driver": "Z",
        },
        "prohibitions": ["no conformal prediction", "no GCMC rerun", "no Stage-15 seeds", "no result-driven parameter tuning", "no forced direction metric for symmetric CMI", "no modification of Stage-15 evidence"],
    }
    dump(RECORD / "config.json", config)

    mechanisms = """# Frozen Stage-17 controlled mechanisms\n\nAll six mechanisms use the same 200 fixed coordinates and 583-edge Delaunay graph. Fifty seeds are paired across mechanisms. Spatial Gaussian fields have exponential covariance with rho equal to twice the median frozen edge length and are sample-standardized. `g_proxy` is a deliberately strong geography-aligned proxy and never appears in the target structural equation.\n\n- **C0 independent null:** spatial `x2`; `A=|epsilon|`, independently.\n- **C1 target residual scale:** `sigma=exp(0.6*x2)` and `A=sigma|epsilon|`.\n- **C2 pure confounding:** observed spatial `Z` causes both `x2` and `sigma`; `A=sigma|epsilon|`; there is no direct `x2 -> A`.\n- **C3 reverse direction:** spatial `a_driver` determines positive `A`, and `x2` is generated downstream from `a_driver`; there is no `x2 -> A`.\n- **C4 spatial-lag causality:** the standardized first-order graph lag of `x2` determines `sigma`, and `A=sigma|epsilon|`.\n- **C5 nonlinear direct causality:** `A=x2^2+0.5|epsilon|`, a non-monotone direct effect.\n\nThe exact equations and coefficients are frozen in `config.json`.\n"""
    (RECORD / "mechanism_definitions.md").write_text(mechanisms, encoding="utf-8")
    prereg = """# Stage 17 preregistration\n\nThis benchmark was written and hashed before any Stage-17 scientific outcomes were generated. It validates feature-strength estimators only; no conformal prediction is permitted. The 50 development seeds are new and exclude Stage-15 confirmatory seeds.\n\nThe benchmark compares CMI-Mspace, CMI-Full, GPC, the spatial causality test, and SCPCM on six frozen mechanisms. CMI is interpreted as symmetric conditional dependence, never identified directionality. GPC, SCT, and SCPCM directions follow audited spEDM 1.12 implementation call order. Parameters and selection/ranking rules are fixed in `config.json`. Existing GCCM is a read-only reference and GCMC is not rerun.\n\nPrimary questions are residual-scale driver recovery in C1, null control in C0, false `x2` attribution under observed confounding in C2, reverse-direction rejection in C3, spatial-lag recovery in C4, and nonlinear recovery in C5. At most two estimators may be recommended for the next Geo-Causal CP development stage.\n"""
    (RECORD / "PREREGISTRATION.md").write_text(prereg, encoding="utf-8")
    provenance = """# Method provenance frozen before outcomes\n\n- CMI: Tigramite `CMIknn`, k=20, 999 restricted-neighbor shuffle surrogates, five-feature BH family. Mspace conditions on coordinates; Full additionally conditions on all other candidates.\n- GPC: installed `spEDM` 1.12 `gpc()`. Forward `X -> A` is the `y_xmap_x` direction; the score is the sum of positive, negative, and dark forward pattern strengths. Full-library GPC exposes no calibrated selection p-value, so its selection rate is explicitly a descriptive forward-over-reverse rule.\n- SCT: installed `spEDM` 1.12 `sc.test()`. The C++ return order is forward statistic, forward bootstrap p, reverse statistic, reverse bootstrap p.\n- SCPCM: installed `spEDM` 1.12 `scpcm()`. The partial output is `pxmap`; forward `X -> A` is `y_xmap_x_mean/sig`. `xmap` is retained as an unpartialed diagnostic. SCPCM is secondary because documentation is less mature.\n- spEDM upstream: `stscl/spEDM`, release v1.12, known release commit `43c5c136ae909c5de83535438d5d5953001429c5`; local CRAN source archive SHA-256 `70dad0f8d41a4a31d0d4cb3b74273e2034dd87227d85abcaa8eb4c927e4cc153`. No byte identity to the GitHub commit is claimed.\n"""
    (RECORD / "METHOD_PROVENANCE.md").write_text(provenance, encoding="utf-8")

    input_manifest = {
        "created_before_results": True,
        "fixed_inputs": [
            {"path": str(GRAPH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(GRAPH)},
            {"path": str(COORDS.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(COORDS), "columns_used": ["sample_id", "s_x", "s_y"]},
            {"path": str(SPEDM_ARCHIVE.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(SPEDM_ARCHIVE)},
        ],
        "scientific_data_reused": "fixed coordinates and graph only; no Stage-15 outcomes or labels",
    }
    dump(RECORD / "input_manifest.json", input_manifest)
    (RECORD / "environment.txt").write_text(
        f"Created UTC: {now_utc.isoformat()}\nCreated local: {now_local.isoformat()}\nPython: {sys.version}\nExecutable: {sys.executable}\nPlatform: {platform.platform()}\nR: C:/Program Files/R/R-4.6.1\nspEDM: 1.12\nR library: {ROOT / 'environment/gcmc_spedm_1.12_r4.6.1/library'}\n",
        encoding="utf-8",
    )
    (RECORD / "run_log.txt").write_text("Stage-17 preregistration created before scientific generation.\n", encoding="utf-8")
    (RECORD / "stderr.txt").write_text("", encoding="utf-8")

    prereg_files = ["PREREGISTRATION.md", "config.json", "seed_manifest.json", "METHOD_PROVENANCE.md", "mechanism_definitions.md", "input_manifest.json", "environment.txt", "seed_scan_audit.json"]
    seal = {"created_utc": now_utc.isoformat(), "files": {name: sha256(RECORD / name) for name in prereg_files}}
    seal_bytes = json.dumps(seal, sort_keys=True, separators=(",", ":")).encode("utf-8")
    seal["composite_sha256"] = hashlib.sha256(seal_bytes).hexdigest()
    dump(RECORD / "PREREGISTRATION_SEAL.json", seal)
    print(json.dumps({"record": str(RECORD), "seed_block": [SEEDS[0], SEEDS[-1]], "manifest_files_scanned": len(scanned), "composite_sha256": seal["composite_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
