"""Stage 10B.2: source audit plus frozen one-seed k=25 GCMC sensitivity."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORD = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b2_gcmc_output_semantics_audit_v1"
STAGE10B = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b_gcmc_one_seed_smoke_v1"
SOURCE_TAR = ROOT / "external/spEDM_1.12_cran/spEDM_1.12.tar.gz"
RLIB = ROOT / "environment/gcmc_spedm_1.12_r4.6.1/library"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")
R_RUNNER = HERE / "run_stage10b2_gcmc_output_semantics_audit_v1.R"
EXPECTED_TAR_SHA = "70dad0f8d41a4a31d0d4cb3b74273e2034dd87227d85abcaa8eb4c927e4cc153"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def source_member(tf: tarfile.TarFile, name: str) -> str:
    stream = tf.extractfile(name)
    if stream is None:
        raise FileNotFoundError(name)
    return stream.read().decode("utf-8")


def exact_random_overlap_curve(n: int, k: int) -> list[dict]:
    """Expected IC under independent uniform neighbor rankings, respecting self exclusion."""
    rows = []
    common_universe = n - 2
    for r in range(1, k + 1):
        hit_probability = 0.0
        # Direct Y-neighbor set excludes focal i; mapped Y list excludes X-neighbor nx.
        # Only N-2 identities are common. The two unique excluded identities are averaged exactly.
        for direct_has_nx, p_direct in ((0, 1 - k / (n - 1)), (1, k / (n - 1))):
            direct_common = k - direct_has_nx
            for mapped_has_i, p_mapped in ((0, 1 - r / (n - 1)), (1, r / (n - 1))):
                mapped_common = r - mapped_has_i
                no_overlap = (
                    math.comb(common_universe - direct_common, mapped_common)
                    / math.comb(common_universe, mapped_common)
                )
                hit_probability += p_direct * p_mapped * (1 - no_overlap)
        rows.append({
            "N": n,
            "k": k,
            "r": r,
            "expected_ic_ratio": hit_probability,
            "cpp_control_value": r / k,
        })
    return rows


def implemented_auc(rows: list[dict]) -> float:
    # CppCMCTest passes direction=">"; CppDeLongPlacements negates values, so theta is
    # P(original control > original case) + 0.5 P(tie), not trapezoidal curve area.
    cases = [row["expected_ic_ratio"] for row in rows]
    controls = [row["cpp_control_value"] for row in rows]
    total = 0.0
    for case in cases:
        for control in controls:
            total += float(control > case) + 0.5 * float(math.isclose(control, case, abs_tol=1e-15))
    return total / (len(cases) * len(controls))


def main() -> None:
    required_absent = [
        "gcmc_xmap_vs_cs_primary.csv",
        "gcmc_vignette_k25_results.csv",
        "stage10b_original_field_trace.json",
    ]
    for name in required_absent:
        if (RECORD / name).exists():
            raise FileExistsError(f"refusing to overwrite completed output: {name}")
    if sha256(SOURCE_TAR) != EXPECTED_TAR_SHA:
        raise AssertionError("spEDM source archive hash mismatch")

    stage10b_manifest = json.loads((STAGE10B / "input_manifest.json").read_text(encoding="utf-8"))
    expected_derived = stage10b_manifest["derived_inputs"]
    for path_text, expected_hash in expected_derived.items():
        if sha256(ROOT / path_text) != expected_hash:
            raise AssertionError(f"Stage-10B frozen input changed: {path_text}")

    stage10b_r = ROOT / "causal_strength_diagnostic_v1/run_stage10b_gcmc_one_seed_smoke_v1.R"
    stage10b_py = ROOT / "causal_strength_diagnostic_v1/run_stage10b_gcmc_one_seed_smoke_v1.py"
    r_lines = stage10b_r.read_text(encoding="utf-8").splitlines()
    py_lines = stage10b_py.read_text(encoding="utf-8").splitlines()
    extraction_line = next((i + 1, line) for i, line in enumerate(r_lines) if "forward_cs=z$cs$y_xmap_x_mean" in line)
    ranking_line = next((i + 1, line) for i, line in enumerate(py_lines) if "sort_values([\"forward_cs\"" in line)
    trace = {
        "original_r_script": {"path": rel(stage10b_r), "sha256": sha256(stage10b_r)},
        "original_python_script": {"path": rel(stage10b_py), "sha256": sha256(stage10b_py)},
        "gcmc_call": "z <- spEDM::gcmc(...) in run_one",
        "csv_forward_cs": "z$cs$y_xmap_x_mean",
        "csv_forward_xmap": "z$xmap$y_xmap_x_mean",
        "csv_forward_significance": "z$xmap$y_xmap_x_sig",
        "ranking_field": "forward_cs",
        "r_extraction_line": {"number": extraction_line[0], "text": extraction_line[1].strip()},
        "python_ranking_line": {"number": ranking_line[0], "text": ranking_line[1].strip()},
        "observed_full_library_identity": "forward_cs equals forward_xmap in all eight original primary rows",
    }
    with (STAGE10B / "gcmc_primary_results.csv").open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    trace["full_library_max_abs_cs_minus_xmap"] = max(
        abs(float(row["forward_cs"]) - float(row["forward_xmap"])) for row in old_rows
    )
    (RECORD / "stage10b_original_field_trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    members = [
        "spEDM/R/gcmc.R",
        "spEDM/R/internal_utility.R",
        "spEDM/R/formatoutput.R",
        "spEDM/src/CrossMappingCardinality.h",
        "spEDM/src/CrossMappingCardinality.cpp",
        "spEDM/src/IntersectionalCardinality.h",
        "spEDM/src/IntersectionalCardinality.cpp",
        "spEDM/src/CppStats.cpp",
        "spEDM/src/DeLongPlacements.cpp",
        "spEDM/src/LatticeExp.cpp",
        "spEDM/man/gcmc.Rd",
        "spEDM/vignettes/main5_gcmc.Rmd",
    ]
    source_hashes = {}
    with tarfile.open(SOURCE_TAR, "r:gz") as tf:
        source = {name: source_member(tf, name) for name in members}
        source_hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in source.items()}
    assertions = [
        ("R call order x_xmap_y", "RcppGCMC4Lattice(cause,effect" in source["spEDM/R/gcmc.R"]),
        ("R call order y_xmap_x", "RcppGCMC4Lattice(effect,cause" in source["spEDM/R/gcmc.R"]),
        ("xmap binding", "x_xmap_y$xmap" in source["spEDM/R/internal_utility.R"]),
        ("cs binding", "x_xmap_y$cs" in source["spEDM/R/internal_utility.R"]),
        ("print uses first list item", "resdf = x[[1]]" in source["spEDM/R/formatoutput.R"]),
        ("plot removes first list item", "xmap = x[-1]" in source["spEDM/R/formatoutput.R"]),
        ("CMCTest controls", "static_cast<double>(i + 1) / num_samples" in source["spEDM/src/CppStats.cpp"]),
        ("CMCTest null 0.5", "theta - 0.5" in source["spEDM/src/CppStats.cpp"]),
        ("vignette reports xmap", "tempdf = g$xmap" in source["spEDM/vignettes/main5_gcmc.Rmd"]),
        ("vignette empirical k", "ceiling(sqrt(11 * nrow(popd_sf)))" in source["spEDM/vignettes/main5_gcmc.Rmd"]),
    ]
    if not all(passed for _, passed in assertions):
        raise AssertionError([label for label, passed in assertions if not passed])

    baseline_rows = exact_random_overlap_curve(200, 9) + exact_random_overlap_curve(200, 25)
    for k in (9, 25):
        subset = [row for row in baseline_rows if row["k"] == k]
        auc = implemented_auc(subset)
        for row in subset:
            row["implemented_random_overlap_auc"] = auc
    with (RECORD / "random_overlap_baseline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)

    direction = {
        "cause_argument": "X_j",
        "effect_argument": "A_model",
        "r_call_order": {
            "x_xmap_y": "RcppGCMC4Lattice(cause, effect, ...)",
            "y_xmap_x": "RcppGCMC4Lattice(effect, cause, ...)"
        },
        "xmap": {
            "X_j_to_A_model": "y_xmap_x_mean",
            "A_model_to_X_j": "x_xmap_y_mean",
            "evidence": "formatoutput labels y_xmap_x_mean as varname[1] -> varname[2]"
        },
        "cs": {
            "X_j_to_A_model": "y_xmap_x_mean",
            "A_model_to_X_j": "x_xmap_y_mean",
            "evidence": "the same directional call results are separately bound from each returned $cs"
        }
    }
    (RECORD / "direction_mapping.json").write_text(json.dumps(direction, indent=2) + "\n", encoding="utf-8")

    semantics = {
        "xmap": {
            "key": "neighbors",
            "quantity": "At the largest valid library size, DeLong-placement ROC AUC theta comparing the averaged IC-curve values ('cases') with deterministic controls r/k for r=1..k.",
            "raw_intersection_cardinality": False,
            "auc_derived": True,
            "delong": "Mean is DeLong theta; accompanying sig is a two-sided z-test of theta=0.5 and lower/upper are DeLong-variance confidence limits.",
            "null_reference": "theta=0.5, using deterministic controls r/k; this is not an empirically calibrated spatial null."
        },
        "cs": {
            "key": "libsizes",
            "quantity": "For each library size, mean across circular moving-library windows of CppCMCTest(IC_curve)[0], the same DeLong-placement AUC theta.",
            "raw_intersection_cardinality": False,
            "auc_derived": True,
            "delong": "Each component AUC is DeLong theta, but only the window mean is returned; no p-value or confidence interval is attached.",
            "null_reference": None
        },
        "full_library_relation": "When libsizes=N, only one library window exists, so xmap mean and cs mean are exactly equal; xmap additionally carries significance and confidence limits.",
        "official_behavior": {
            "documentation": "gcmc.Rd calls xmap 'cross mapping results' and cs 'causal strength'.",
            "print": "print.cmc_res uses x[[1]], which is xmap, and labels directions as cause->effect.",
            "plot": "plot.cmc_res removes x[-1] (xmap), then plot.ccm_res uses the remaining first item, cs, with y-axis 'Causal Score'.",
            "vignette": "The official GCMC vignette prints g (therefore xmap) and explicitly renames g$xmap mean/sig fields to cs/sig for reported causal-strength matrices.",
            "classification": "OUTPUT_SEMANTICS_AMBIGUITY"
        },
        "source_member_sha256": source_hashes,
        "source_assertions": [{"check": label, "pass": passed} for label, passed in assertions],
        "analytic_baseline": {
            "assumptions": "Independent uniform neighbor rankings with self excluded, N=200, direct Y set size k, mapped prefixes r=1..k; spatial embedding dependence is deliberately absent.",
            "k9_auc": implemented_auc([r for r in baseline_rows if r["k"] == 9]),
            "k25_auc": implemented_auc([r for r in baseline_rows if r["k"] == 25]),
            "conclusion": "The implemented statistic has a k- and N-dependent random-overlap baseline; 0.5 is not generally the independent-overlap expectation."
        }
    }
    (RECORD / "gcmc_output_semantics.json").write_text(json.dumps(semantics, indent=2) + "\n", encoding="utf-8")

    input_manifest = {
        "generation_seed": 20270908,
        "stage10b_record": rel(STAGE10B),
        "frozen_inputs": [
            {"path": path_text, "sha256": expected_hash}
            for path_text, expected_hash in sorted(expected_derived.items())
        ],
        "stage10b_outputs_read": [
            {"path": rel(STAGE10B / name), "sha256": sha256(STAGE10B / name)}
            for name in (
                "gcmc_primary_results.csv", "gcmc_sensitivity_results.csv",
                "gcmc_direction_results.csv", "stage10b_summary.json", "stage10b_summary.md"
            )
        ],
        "spEDM_source_archive": {"path": rel(SOURCE_TAR), "sha256": sha256(SOURCE_TAR)},
        "source_members": source_hashes,
        "new_source_code": [
            {"path": rel(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            {"path": rel(R_RUNNER), "sha256": sha256(R_RUNNER)},
        ],
        "label_access": {
            "read": ["existing derived 200-row Stage-10B scientific tables"],
            "not_read": ["train", "validation", "tuning_calibration", "final_calibration", "test"]
        }
    }
    (RECORD / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n", encoding="utf-8")

    command = [str(RSCRIPT), "--vanilla", str(R_RUNNER), str(RECORD), str(STAGE10B), str(RLIB)]
    env = dict(os.environ)
    env["R_LIBS_USER"] = str(RLIB).replace("\\", "/")
    env["LC_ALL"] = "C"
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=7200)
    (RECORD / "run_log.txt").write_text(
        "COMMAND: " + subprocess.list2cmdline(command) + "\n" + process.stdout,
        encoding="utf-8"
    )
    (RECORD / "stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"Stage 10B.2 R execution failed ({process.returncode}): {process.stderr[-3000:]}")
    print("STAGE10B2_EXECUTION_PASS")


if __name__ == "__main__":
    main()
