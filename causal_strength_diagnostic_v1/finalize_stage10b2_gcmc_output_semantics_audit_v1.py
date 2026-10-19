"""Finalize Stage 10B.2 documentation and immutable manifests without rerunning GCMC."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b2_gcmc_output_semantics_audit_v1"
STAGE10B = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b_gcmc_one_seed_smoke_v1"
RUNNER_PY = ROOT / "causal_strength_diagnostic_v1/run_stage10b2_gcmc_output_semantics_audit_v1.py"
RUNNER_R = ROOT / "causal_strength_diagnostic_v1/run_stage10b2_gcmc_output_semantics_audit_v1.R"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")
RLIB = ROOT / "environment/gcmc_spedm_1.12_r4.6.1/library"
TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def diagnostics(frame: pd.DataFrame, output_family: str) -> list[dict]:
    if output_family == "xmap":
        forward, reverse, sig = "xmap_y_xmap_x_mean", "xmap_x_xmap_y_mean", "xmap_y_xmap_x_sig"
    else:
        forward, reverse, sig = "cs_y_xmap_x_mean", "cs_x_xmap_y_mean", None
    result = []
    for (scenario, style, k), group in frame.groupby(["scenario", "style", "k"], sort=True):
        scores = dict(zip(group.feature, group[forward].astype(float)))
        reverse_scores = dict(zip(group.feature, group[reverse].astype(float)))
        maximum = max(scores.values())
        top = sorted(feature for feature, value in scores.items() if abs(value - maximum) <= TOL)
        x2 = scores["x2"]
        result.append({
            "scenario": scenario,
            "E": 3,
            "k": int(k),
            "style": int(style),
            "output_family": output_family,
            "forward_field": forward,
            "reverse_field": reverse,
            "top_features": top,
            "top_score": maximum,
            "x2_tie_aware_rank": 1 + sum(value > x2 + TOL for value in scores.values()),
            "x2_score": x2,
            "x2_minus_x_noise": x2 - scores["x_noise"],
            "x2_minus_x1": x2 - scores["x1"],
            "x2_minus_g_proxy": x2 - scores["g_proxy"],
            "x2_forward_minus_reverse": x2 - reverse_scores["x2"],
            "x2_significance": None if sig is None else float(group.loc[group.feature == "x2", sig].iloc[0]),
            "maximum_s4_score": maximum if scenario == "S4" else None,
        })
    return result


def table(frame: pd.DataFrame, family: str) -> str:
    if family == "xmap":
        forward, reverse, sig = "xmap_y_xmap_x_mean", "xmap_x_xmap_y_mean", "xmap_y_xmap_x_sig"
    else:
        forward, reverse, sig = "cs_y_xmap_x_mean", "cs_x_xmap_y_mean", None
    lines = []
    for row in frame.sort_values([forward, "feature"], ascending=[False, True]).itertuples():
        significance = "-" if sig is None else f"{getattr(row, sig):.3g}"
        fwd = getattr(row, forward)
        rev = getattr(row, reverse)
        lines.append(f"| {row.feature} | {fwd:.6f} | {rev:.6f} | {fwd-rev:+.6f} | {significance} |")
    return "\n".join([
        "| Feature | Forward | Reverse | Forward - reverse | Significance |",
        "|---|---:|---:|---:|---:|",
        *lines,
    ])


def main() -> None:
    primary = pd.read_csv(RECORD / "gcmc_xmap_vs_cs_primary.csv")
    k25 = pd.read_csv(RECORD / "gcmc_vignette_k25_results.csv")
    combined = pd.concat([primary, k25], ignore_index=True)
    semantics = json.loads((RECORD / "gcmc_output_semantics.json").read_text(encoding="utf-8"))
    trace = json.loads((RECORD / "stage10b_original_field_trace.json").read_text(encoding="utf-8"))
    direction = json.loads((RECORD / "direction_mapping.json").read_text(encoding="utf-8"))
    old = pd.read_csv(STAGE10B / "gcmc_primary_results.csv").sort_values(["scenario", "feature"]).reset_index(drop=True)
    new = primary.sort_values(["scenario", "feature"]).reset_index(drop=True)
    comparisons = {
        "forward_cs": np.abs(old.forward_cs - new.cs_y_xmap_x_mean),
        "reverse_cs": np.abs(old.reverse_cs - new.cs_x_xmap_y_mean),
        "forward_xmap": np.abs(old.forward_xmap - new.xmap_y_xmap_x_mean),
        "reverse_xmap": np.abs(old.reverse_xmap - new.xmap_x_xmap_y_mean),
        "forward_sig": np.abs(old.forward_sig - new.xmap_y_xmap_x_sig),
    }
    stage10b_reextraction = {name: float(values.max()) for name, values in comparisons.items()}
    if max(stage10b_reextraction.values()) > TOL:
        raise AssertionError("Stage-10B primary re-extraction mismatch")

    all_diagnostics = diagnostics(combined, "xmap") + diagnostics(combined, "cs")
    for d in all_diagnostics:
        d["top_score"] = float(d["top_score"])
        for key in ("x2_score", "x2_minus_x_noise", "x2_minus_x1", "x2_minus_g_proxy", "x2_forward_minus_reverse"):
            d[key] = float(d[key])

    source_baseline = semantics["analytic_baseline"]
    full_identity = {
        "max_abs_forward_xmap_minus_cs": float(np.max(np.abs(combined.xmap_y_xmap_x_mean - combined.cs_y_xmap_x_mean))),
        "max_abs_reverse_xmap_minus_cs": float(np.max(np.abs(combined.xmap_x_xmap_y_mean - combined.cs_x_xmap_y_mean))),
        "reason": "libsizes=N=200 yields one full-library window, so cs mean equals xmap theta",
    }
    if max(full_identity["max_abs_forward_xmap_minus_cs"], full_identity["max_abs_reverse_xmap_minus_cs"]) > TOL:
        raise AssertionError("full-library xmap/cs identity failed")

    official_example = (RECORD / "official_example_status.txt").read_text(encoding="utf-8").strip().splitlines()
    summary = {
        "stage": "10B.2",
        "status": "COMPLETE",
        "generation_seed": 20270908,
        "original_stage10b_field_trace": trace,
        "semantics": semantics,
        "direction_mapping": direction,
        "stage10b_reextraction_max_absolute_differences": stage10b_reextraction,
        "full_library_xmap_cs_identity": full_identity,
        "configuration_diagnostics": all_diagnostics,
        "official_vignette_k": {"formula": "ceiling(sqrt(E*N))", "E": 3, "N": 200, "k": 25},
        "official_example": official_example,
        "officially_justified_reporting_for_this_audit": {
            "quantity": "xmap",
            "reason": "the official v1.12 GCMC vignette reports g$xmap mean as cs and takes significance from g$xmap",
            "qualification": "the package documentation/plot naming remains inconsistent, so this does not resolve the general slot ambiguity",
            "s2_x2_recovered_k9_style1": False,
            "s2_x2_recovered_k25_style1": False,
            "s2_x2_recovered_k25_style0": False,
        },
        "baseline": {
            "implemented_independent_random_overlap_auc_k9": source_baseline["k9_auc"],
            "implemented_independent_random_overlap_auc_k25": source_baseline["k25_auc"],
            "meaning": "source-derived analytic expectation under independent uniform neighbor rankings; not a graph/spatial null calibration",
            "conclusion": "The score baseline changes strongly with k; the Stage-10B 0.89-0.94 range cannot be interpreted against a universal 0.5 random-overlap baseline.",
        },
        "final_classification": "D. spEDM_1.12_OUTPUT_SEMANTICS_REMAIN_AMBIGUOUS",
        "twenty_replicate_benchmark_justified": False,
        "next_recommended_stage": {
            "name": "STAGE 10B.3 - GCMC NULL-REFERENCE AND MECHANISM-COMPATIBILITY DIAGNOSTIC",
            "objective": "Predeclare and evaluate graph/k/N-aware null behavior and controlled same-location versus spatial-lag mechanisms before any 20-replicate feature-recovery benchmark.",
            "required_first_step": "Obtain a definitive semantic resolution for xmap versus cs from authoritative package/paper clarification, or explicitly freeze xmap as the vignette-reporting convention while preserving the ambiguity record.",
        },
        "warnings": [
            "No R runtime warnings or stderr were recorded.",
            "The official example confirms print() uses xmap and plot() uses cs; this is labeled OUTPUT_SEMANTICS_AMBIGUITY, not a software bug.",
            "The DeLong p-value tests theta=0.5, but the implemented independent-overlap expectation depends on N and k.",
        ],
    }
    (RECORD / "stage10b2_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    semantics_md = f"""# spEDM 1.12 GCMC source-semantics audit

## Computation path

`gcmc.R` constructs cause/effect embeddings and calls `RcppGCMC4Lattice`. `CrossMappingCardinality.cpp` obtains nearest-neighbor rankings for both embeddings and calls `IntersectionalCardinalitySingle`. For each prediction point and each prefix `r=1..k`, that routine computes the fraction of the k cause-manifold neighbors whose first r mapped effect-manifold neighbors intersect the focal point's full k effect-manifold-neighbor set. Averaging over prediction points yields the IC curve.

`CppCMCTest(IC_curve, ">")` does not return the raw IC curve or a trapezoidal geometric area. It creates deterministic controls `(1/k, 2/k, ..., 1)`, applies DeLong placements, and returns ROC AUC theta. Because direction `">"` negates both samples before placement, theta is `P(control > IC) + 0.5 P(tie)`. Its p-value is a two-sided z-test of theta=0.5; confidence limits use the DeLong placement variance.

## `result$xmap`

- Derived from the largest valid library size only.
- `neighbors` is k.
- Each `*_mean` is the DeLong-placement AUC theta for that direction.
- `*_sig` and `*_lower`/`*_upper` are the theta=0.5 p-value and confidence limits.
- It is AUC-derived, not raw intersection-cardinality output.

## `result$cs`

- Indexed by library size.
- For each circular moving-library window, computes the same `CppCMCTest(IC_curve)[0]` theta, then averages theta across windows at that library size.
- It is AUC-derived, not raw intersection-cardinality output.
- It carries no significance or confidence limits.
- At full library size N, there is one window, so `cs` mean equals `xmap` mean exactly.

## Baseline implication

Under independent uniform neighbor rankings, self exclusion, N=200, a direct set of k neighbors and mapped prefixes r=1..k, the exact expected IC hit curve can be derived hypergeometrically. Passing that expected curve through the implemented DeLong-placement orientation yields:

- k=9: AUC `{source_baseline['k9_auc']:.9f}`
- k=25: AUC `{source_baseline['k25_auc']:.9f}`

These are analytic independent-ranking expectations, not calibrated spatial-graph nulls. They demonstrate that 0.5 is not generally the random-overlap expectation and that the score scale is intrinsically k/N dependent. The observed k=9 scores are above, but close to, an already-high construction baseline; the baseline does not by itself explain feature ordering.

## Official interface behavior

- `gcmc.Rd`: calls `xmap` "cross mapping results" and `cs` "causal strength."
- `print.cmc_res`: uses `x[[1]]`, which is `xmap`.
- Official vignette: prints the object and explicitly renames `g$xmap` mean/significance fields to `cs`/`sig` for reported causal-strength matrices.
- `plot.cmc_res`: removes the first list item (`xmap`) and plots the remaining first item (`cs`) as "Causal Score."
- Package-bundled Columbus runtime check reproduced this split: print data equaled `xmap`; plot data equaled `cs`.

This is classified as **OUTPUT_SEMANTICS_AMBIGUITY**, not as a demonstrated software bug. At full library size the means coincide, masking the ambiguity numerically in Stage 10B.
"""
    (RECORD / "source_semantics_audit.md").write_text(semantics_md, encoding="utf-8")

    s2_k9 = primary[primary.scenario == "S2"]
    s4_k9 = primary[primary.scenario == "S4"]
    s2_k25_s1 = k25[(k25.scenario == "S2") & (k25["style"] == 1)]
    s4_k25_s1 = k25[(k25.scenario == "S4") & (k25["style"] == 1)]
    s2_k25_s0 = k25[(k25.scenario == "S2") & (k25["style"] == 0)]
    s4_k25_s0 = k25[(k25.scenario == "S4") & (k25["style"] == 0)]
    summary_md = f"""# Stage 10B.2 GCMC output-semantics audit

Final classification: **D. spEDM_1.12_OUTPUT_SEMANTICS_REMAIN_AMBIGUOUS**.

## Original Stage-10B trace

Stage 10B ranked `z$cs$y_xmap_x_mean` through the CSV field `forward_cs`. Its reported significance was `z$xmap$y_xmap_x_sig`. Re-extraction reproduced every original primary score/significance exactly (maximum absolute difference 0).

Because the only library size was the full N=200, `xmap` and `cs` means were identical in both directions for every original and k=25 call. The prior feature ranking was therefore numerically unchanged by using the vignette's `xmap` reporting convention.

## Direction

For cause X and effect A_model, both output families use `y_xmap_x_mean` for X -> A_model and `x_xmap_y_mean` for A_model -> X. This follows the R call order and the separate binding of each call's `xmap` and `cs` results.

## Original k=9, style=1 - xmap

### S2

{table(s2_k9, 'xmap')}

### S4

{table(s4_k9, 'xmap')}

The `cs` tables have identical forward/reverse means and no significance fields.

## OFFICIAL_VIGNETTE_EMPIRICAL_K_SENSITIVITY: k=25, style=1 - xmap

`k = ceiling(sqrt(3*200)) = 25`. This is a specific vignette sensitivity, not a universal GCMC rule.

### S2

{table(s2_k25_s1, 'xmap')}

Ranking: g_proxy > x_noise > x1 > x2. x2 was not recovered.

### S4

{table(s4_k25_s1, 'xmap')}

Ranking: g_proxy > x2 > x_noise > x1. The S4 maximum (0.7328) exceeded the S2 maximum (0.7216).

## k=25, style=0 same-location sensitivity - xmap

### S2

{table(s2_k25_s0, 'xmap')}

### S4

{table(s4_k25_s0, 'xmap')}

Both are led by g_proxy; x2 ranks fourth in S2 and third in S4. Again, `cs` has identical full-library means.

## Decision

The official vignette-reporting convention is `xmap`, because it explicitly converts `g$xmap` to `cs`/`sig`. However, package documentation names the separate `$cs` slot "causal strength," `print()` shows xmap, and `plot()` plots cs. The general interface semantics therefore remain ambiguous even though the full-library values coincide here.

x2 is not recovered under k=9 or the vignette-derived k=25, under either style. S4 remains structured and is repeatedly dominated by g_proxy. A 20-replicate benchmark is **not justified**.

Next recommended stage: **STAGE 10B.3 - GCMC NULL-REFERENCE AND MECHANISM-COMPATIBILITY DIAGNOSTIC**, after obtaining authoritative xmap/cs clarification or explicitly freezing the vignette xmap convention. It should preregister graph/k/N-aware null calibration and controlled same-location versus spatial-lag mechanisms; it must not tune parameters against x2.
"""
    (RECORD / "stage10b2_summary.md").write_text(summary_md, encoding="utf-8")

    method_md = """# Method

This audit used the verified local spEDM 1.12 CRAN source archive and installed runtime without modifying either. Source members were read directly from the immutable archive and hashed in `input_manifest.json` and `gcmc_output_semantics.json`.

Scientific execution reused only the Stage-10B derived 200-row S2/S4 scale-fit tables and explicit Delaunay edge list. No scenario generation, predictor fitting, or access to labels from other splits occurred.

The original E=3/k=9/style=1 configuration was rerun only to extract every `xmap` and `cs` field side-by-side and verify exact agreement with Stage 10B. The only new project-data calls were preregistered E=3/k=25 with style 1 and style 0, following the official vignette empirical `ceiling(sqrt(E*N))` sensitivity. No other parameter value was tested.

The analytic random-overlap calculation reproduces the implemented combinatorial structure under independent uniform rankings; it is not a permutation test or empirical spatial null. The package-bundled Columbus example was used only to verify print/plot behavior.
"""
    (RECORD / "METHOD.md").write_text(method_md, encoding="utf-8")

    readme = """# Stage 10B.2: GCMC output-semantics audit

This immutable record traces spEDM 1.12 `gcmc()` from R input through C++ IC/AUC computation, reconciles user-facing print/plot/vignette behavior, re-extracts both output families for the Stage-10B primary configuration, and runs the single preregistered official-vignette k=25 sensitivity.

Final classification: `D. spEDM_1.12_OUTPUT_SEMANTICS_REMAIN_AMBIGUOUS`.

Exact execution command:

```powershell
D:\\conda\\python.exe causal_strength_diagnostic_v1\\run_stage10b2_gcmc_output_semantics_audit_v1.py
```

No remaining replicate, conformal prediction, final weight construction, data regeneration, predictor fitting, network access, package installation, or spEDM modification occurred.
"""
    (RECORD / "README.md").write_text(readme, encoding="utf-8")

    r_expr = (
        f'.libPaths(c("{str(RLIB).replace(os.sep, "/")}",.libPaths())); '
        'pkgs <- c("spEDM","sf","terra","sdsfun","dplyr","ggplot2","Rcpp","RcppThread","RcppArmadillo"); '
        'cat(R.version.string,"\\n"); for(p in pkgs) cat(p,as.character(packageVersion(p)),"\\n"); sessionInfo()'
    )
    env = dict(os.environ)
    env["R_LIBS_USER"] = str(RLIB).replace("\\", "/")
    process = subprocess.run([str(RSCRIPT), "--vanilla", "-e", r_expr], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    (RECORD / "environment.txt").write_text(
        "R executable: C:\\Program Files\\R\\R-4.6.1\\bin\\Rscript.exe\n"
        f"Project-local library: {RLIB}\n" + process.stdout,
        encoding="utf-8"
    )

    output_names = [
        "README.md", "METHOD.md", "source_semantics_audit.md", "config.json", "input_manifest.json",
        "run_log.txt", "stderr.txt", "r_warnings.txt", "environment.txt",
        "stage10b_original_field_trace.json", "gcmc_xmap_vs_cs_primary.csv",
        "gcmc_vignette_k25_results.csv", "gcmc_output_semantics.json", "direction_mapping.json",
        "random_overlap_baseline.csv", "official_example_status.txt", "official_example_print.txt",
        "official_example_xmap.csv", "official_example_cs.csv", "official_example_plot_data.csv",
        "stage10b2_summary.json", "stage10b2_summary.md",
    ]
    manifest = {
        "record": rel(RECORD),
        "canonical_outputs": [
            {"path": name, "sha256": sha256(RECORD / name), "bytes": (RECORD / name).stat().st_size}
            for name in output_names
        ],
        "source_code": [
            {"path": rel(path), "sha256": sha256(path)}
            for path in (RUNNER_PY, RUNNER_R, Path(__file__).resolve())
        ],
    }
    (RECORD / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("STAGE10B2_FINALIZATION_PASS")


if __name__ == "__main__":
    main()
