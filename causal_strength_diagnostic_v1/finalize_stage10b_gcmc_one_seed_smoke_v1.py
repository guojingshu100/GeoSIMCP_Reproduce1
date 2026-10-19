"""Finalize the immutable Stage-10B record without rerunning GCMC or FNN."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b_gcmc_one_seed_smoke_v1"
RUNNER_PY = ROOT / "causal_strength_diagnostic_v1/run_stage10b_gcmc_one_seed_smoke_v1.py"
RUNNER_R = ROOT / "causal_strength_diagnostic_v1/run_stage10b_gcmc_one_seed_smoke_v1.R"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")
RLIB = ROOT / "environment/gcmc_spedm_1.12_r4.6.1/library"
TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def finite_float(x):
    value = float(x)
    if not np.isfinite(value):
        raise ValueError(f"non-finite value: {x}")
    return value


def config_summary(group: pd.DataFrame) -> dict:
    g = group.copy()
    scores = dict(zip(g.feature, g.forward_cs.astype(float)))
    maximum = max(scores.values())
    top = sorted([f for f, value in scores.items() if abs(value - maximum) <= TOL])
    x2 = scores["x2"]
    return {
        "scenario": str(g.scenario.iloc[0]),
        "style": int(g["style"].iloc[0]),
        "E": int(g.E.iloc[0]),
        "k": int(g.k.iloc[0]),
        "top_features": top,
        "top_score": finite_float(maximum),
        "x2_tie_aware_rank": 1 + sum(value > x2 + TOL for value in scores.values()),
        "x2_is_top": "x2" in top,
        "x2_score": finite_float(x2),
        "x2_minus_x_noise": finite_float(x2 - scores["x_noise"]),
        "x2_minus_x1": finite_float(x2 - scores["x1"]),
        "x2_minus_g_proxy": finite_float(x2 - scores["g_proxy"]),
    }


def score_table(group: pd.DataFrame) -> str:
    rows = []
    for row in group.sort_values(["forward_cs", "feature"], ascending=[False, True]).itertuples():
        rows.append(
            f"| {row.feature} | {row.forward_cs:.9f} | {row.reverse_cs:.9f} | "
            f"{row.forward_minus_reverse:+.9f} | {row.forward_sig:.3g} |"
        )
    return "\n".join([
        "| Feature | Forward CS | Reverse CS | Forward - reverse | xmap significance |",
        "|---|---:|---:|---:|---:|",
        *rows,
    ])


def main() -> None:
    sensitivity = pd.read_csv(RECORD / "gcmc_sensitivity_results.csv")
    primary = pd.read_csv(RECORD / "gcmc_primary_results.csv")
    fnn = pd.read_csv(RECORD / "fnn_curves.csv")
    cmi = pd.read_csv(RECORD / "stage9_seed_20270908_context.csv")
    graph = json.loads((RECORD / "graph_diagnostics.json").read_text(encoding="utf-8"))
    inputs = json.loads((RECORD / "input_manifest.json").read_text(encoding="utf-8"))
    repro = json.loads((RECORD / "reproducibility_check.json").read_text(encoding="utf-8"))
    config = json.loads((RECORD / "config.json").read_text(encoding="utf-8"))

    diagnostics = [
        config_summary(group)
        for _, group in sensitivity.groupby(["scenario", "style", "E", "k"], sort=True)
    ]
    primary_diag = [d for d in diagnostics if d["style"] == 1 and d["E"] == 3 and d["k"] == 9]
    style0_primary_diag = [d for d in diagnostics if d["style"] == 0 and d["E"] == 3 and d["k"] == 9]
    s2_diag = [d for d in diagnostics if d["scenario"] == "S2"]
    s4_diag = [d for d in diagnostics if d["scenario"] == "S4"]

    fnn_summary = []
    for (scenario, variable), group in fnn.groupby(["scenario", "variable"], sort=True):
        e3 = group.loc[group.E == 3, "fnn"]
        fnn_summary.append({
            "scenario": scenario,
            "variable": variable,
            "returned_E": [int(v) for v in group.E],
            "requested_E_max": int(group.requested_E_max.iloc[0]),
            "truncated": bool(group.truncated.iloc[0]),
            "fnn_at_E3": finite_float(e3.iloc[0]),
            "minimum_fnn": finite_float(group.fnn.min()),
        })

    cmi_context = []
    for (scenario, condition), group in cmi.groupby(["scenario", "condition"], sort=True):
        ordered = group.sort_values(["raw_cmi", "feature"], ascending=[False, True])
        cmi_context.append({
            "scenario": scenario,
            "condition": condition,
            "ranking": [str(v) for v in ordered.feature],
            "features": [
                {"feature": r.feature, "raw_cmi": finite_float(r.raw_cmi),
                 "p_value": finite_float(r.p_value), "bh_q": finite_float(r.bh_q)}
                for r in ordered.itertuples()
            ],
        })

    summary = {
        "stage": "10B",
        "label": "ONE-SEED SMOKE DIAGNOSTIC",
        "generation_seed": 20270908,
        "scientific_scope": "GCMC feature recovery on frozen scale_fit rows only; not power, FPR, validated recovery, or final weights",
        "direction_mapping": {
            "forward": "cause X_j -> effect A_model = result$cs$y_xmap_x_mean",
            "reverse": "effect A_model -> cause X_j = result$cs$x_xmap_y_mean",
            "basis": "spEDM 1.12 gcmc.R call order plus formatoutput.R varname mapping",
        },
        "primary_configuration": config["primary"],
        "primary_diagnostics": primary_diag,
        "style0_primary_diagnostics": style0_primary_diag,
        "configuration_diagnostics": diagnostics,
        "s2_x2_top_count": int(sum(d["x2_is_top"] for d in s2_diag)),
        "s2_configuration_count": len(s2_diag),
        "s4_top_feature_sets": [d["top_features"] for d in s4_diag],
        "fnn": {
            "diagnostic_only": True,
            "requested_E": list(range(1, 11)),
            "returned_E": list(range(1, 10)),
            "E10_truncated_for_all_variables": True,
            "E3_obviously_pathological": False,
            "details": fnn_summary,
        },
        "reproducibility": repro,
        "stage9_cmi_context": cmi_context,
        "warnings": [
            "spEDM::fnn truncated requested E=1:10 to E=1:9 for every scenario-variable curve because only nine dimensions were embeddable on this graph.",
            "No R warnings, non-finite outputs, missing rows, or runtime errors were recorded in the successful run.",
            "GCMC xmap significance values were small for all primary S4 features and are not evidence of null control in this one-seed diagnostic.",
        ],
        "decision_classification": "NOT_READY_METHOD_OR_IMPLEMENTATION_CONCERN",
        "decision_reasons": [
            "Leak-safe execution, explicit-neighbor integration, direction extraction, and clean-process determinism all passed.",
            "x2 was not top-ranked in S2 under any of the 12 preregistered style/E/k configurations.",
            "Under primary style=1/E=3/k=9, x2 ranked fourth in S2 while x_noise ranked first.",
            "S4 produced high, structured forward scores; its primary maximum was a three-way tie among g_proxy, x2, and x_noise.",
            "style=0 did not recover x2 and broadly increased same-location scores rather than isolating the known scale driver.",
        ],
        "protocol_issues_before_stage10c": [
            "Predeclare whether raw cardinality CS and xmap significance are scientifically interpretable as feature-recovery diagnostics under a full-library, single-scale-fit graph; they must not yet be converted to weights.",
            "Keep style=1 and style=0 as separately labeled variants; do not select the better-looking style post hoc.",
            "Resolve the repeated g_proxy/S4 structure and establish a defensible null-calibration strategy before a 20-replicate benchmark.",
            "Predeclare tie handling and acknowledge the graph-specific FNN limit of E<=9.",
            "Assess whether spatial cross mapping is appropriate for a same-location residual-scale mechanism before scaling the experiment.",
        ],
    }
    (RECORD / "stage10b_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    primary_s2 = primary[primary.scenario == "S2"]
    primary_s4 = primary[primary.scenario == "S4"]
    style0 = sensitivity[(sensitivity["style"] == 0) & (sensitivity.E == 3) & (sensitivity.k == 9)]
    style0_s2 = style0[style0.scenario == "S2"]
    style0_s4 = style0[style0.scenario == "S4"]
    grid_lines = []
    for d in diagnostics:
        top = ", ".join(d["top_features"])
        grid_lines.append(
            f"| {d['scenario']} | {d['style']} | {d['E']} | {d['k']} | {top} | "
            f"{d['x2_tie_aware_rank']} | {d['x2_score']:.9f} | {d['x2_minus_x_noise']:+.9f} | "
            f"{d['x2_minus_x1']:+.9f} | {d['x2_minus_g_proxy']:+.9f} |"
        )
    fnn_e3 = fnn[fnn.E == 3]
    fnn_lines = [f"| {r.scenario} | {r.variable} | {r.fnn:.3f} | 1-9 (10 truncated) |" for r in fnn_e3.itertuples()]

    summary_md = f"""# Stage 10B: GCMC one-seed smoke diagnostic

Status: **NOT_READY_METHOD_OR_IMPLEMENTATION_CONCERN**.

This is a **ONE-SEED SMOKE DIAGNOSTIC**, not a power estimate, false-positive-rate estimate, validated causal-recovery claim, final method comparison, or weight specification.

## Integrity and protocol

- Generation seed: `20270908`; scenarios: S2 and S4.
- Each scientific table contains exactly 200 unique, finite `local_calibration / scale_fit` rows.
- `A_model = abs(y - y_pred_frozen)` was constructed only after restricting to authorized scale-fit IDs.
- No train, validation, tuning-calibration, final-calibration, or test labels were read for feature scoring.
- Candidate causes: `x1`, `x2`, `g_proxy`, `x_noise`; coordinates define geometry only.
- Explicit undirected Delaunay graph: 200 nodes, 583 edges, one component, no self-edges, degree min/median/mean/max = 3/6/5.83/10.
- S2/S4 row, sample-ID, coordinate, and adjacency hashes match exactly.
- Direction mapping: `X_j -> A_model` is `y_xmap_x_mean`; `A_model -> X_j` is `x_xmap_y_mean`, verified from installed spEDM 1.12 call order and output formatter.
- Frozen primary: E=3, k=9, tau=1, style=1, L2, no detrending, bidirectional, one thread, explicit Delaunay `nb`, full 200-row library.

## Primary style=1 results

### S2

{score_table(primary_s2)}

x2 ranked fourth. x_noise ranked first. Therefore the known S2 scale driver was not recovered by the primary configuration.

### S4

{score_table(primary_s4)}

The maximum forward score was an exact three-way tie among `g_proxy`, `x2`, and `x_noise` (0.913580247). This is not weak or unstructured null behavior.

## Same-location sensitivity: style=0, E=3, k=9

### S2

{score_table(style0_s2)}

g_proxy ranked first; x1, x2, and x_noise tied for second. Including the focal state did not isolate x2.

### S4

{score_table(style0_s4)}

g_proxy ranked first; x1 and x2 tied for second.

## FNN descriptive diagnostic

FNN was descriptive only and did not select E. At E=3, all curves were zero except x_noise=0.02 in both scenarios, so E=3 was not obviously pathological by this check. spEDM returned E=1:9 and truncated requested E=10 for all ten scenario-variable curves.

| Scenario | Variable | FNN at E=3 | Returned E |
|---|---|---:|---|
{chr(10).join(fnn_lines)}

## Complete preregistered E/k/style diagnostic

Ranks below are tie-aware (competition ranking).

| Scenario | Style | E | k | Top feature(s) | x2 rank | x2 CS | x2-noise | x2-x1 | x2-g_proxy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(grid_lines)}

x2 was top-ranked in 0/12 S2 configurations. Except for primary style=1/E=3/k=9, which selected x_noise, every S2 grid configuration selected g_proxy; style=0/E=3/k=9 had a three-way second-place tie containing x2. The finding is not rescued by the limited preregistered sensitivity grid.

## Reproducibility

The primary configuration was repeated in a separate fresh R process. Across eight scenario-feature calls and all recorded CS/xmap/significance/interval fields, the maximum absolute difference was exactly 0 (tolerance 1e-12).

## Stage-9 CMI context

No CMI was rerun. In stored seed-20270908 Mspace results, S2 ranked x2 first (`raw_cmi=0.0355692`, `p=0.01`, `BH q=0.04`), while S4 ranked x1 first with no BH-significant feature. This is descriptive context only and does not establish superiority from one seed.

## Decision

**NOT_READY_METHOD_OR_IMPLEMENTATION_CONCERN**. Integration and determinism passed, but feature recovery and null behavior did not. Before any Stage-10C run, freeze the interpretation/null-calibration of cardinality CS and xmap significance, retain style variants as separately labeled, predeclare tie handling and the E<=9 FNN limit, and determine whether the same-location residual-scale mechanism is compatible with this spatial cross-mapping representation.
"""
    (RECORD / "stage10b_summary.md").write_text(summary_md, encoding="utf-8")

    method_md = """# Method

Stage 10B calls the installed `spEDM` 1.12 implementations directly: `spEDM::gcmc()` for causal-strength/cross-mapping output and `spEDM::fnn()` for descriptive embedding diagnostics. It does not reimplement either algorithm.

The frozen predictor inputs identify authorized scale-fit rows. Labels are joined only for those 200 IDs, frozen predictions are joined one-to-one by stable metadata, and `A_model=abs(y-y_pred_frozen)` is then constructed. The exported R scientific tables omit `y`, `y_pred`, split labels, and all non-scale-fit observations.

The explicit neighbor list uses the same SciPy Delaunay simplex-edge construction as the existing simulation GCCM pipeline. Rows are stably sorted by `sample_id`; Delaunay simplex edges are deduplicated and made undirected; R receives a 1-based `nb` list. Coordinates are removed as ordinary sf columns after POINT geometry construction.

For a supplied cause X and effect Y, installed `gcmc.R` computes `x_xmap_y` from `(X,Y)` and `y_xmap_x` from `(Y,X)`. Installed `formatoutput.R` labels `y_xmap_x_mean` as `X -> Y`; this is the forward feature-to-residual score used here. The reverse score is `x_xmap_y_mean`.

The primary and complete sensitivity grid were written to `config.json` before any GCMC result was read. FNN did not select E, and no extra E/k/style setting was added after results.
"""
    (RECORD / "METHOD.md").write_text(method_md, encoding="utf-8")

    r_expr = (
        f'.libPaths(c("{str(RLIB).replace(os.sep, "/")}", .libPaths())); '
        'pkgs <- c("spEDM","sf","terra","sdsfun","dplyr","ggplot2","Rcpp","RcppThread","RcppArmadillo"); '
        'cat(R.version.string,"\\n",sep=""); cat(R.home(),"\\n",sep=""); '
        'for(p in pkgs) cat(p,as.character(packageVersion(p)),"\\n"); sessionInfo()'
    )
    env = dict(os.environ)
    env["R_LIBS_USER"] = str(RLIB).replace("\\", "/")
    proc = subprocess.run([str(RSCRIPT), "--vanilla", "-e", r_expr], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    environment_text = (
        "R executable: C:\\Program Files\\R\\R-4.6.1\\bin\\Rscript.exe\n"
        f"Project-local R library: {RLIB}\n"
        "Scientific runtime: threads=1; parallel.level=low\n\n" + proc.stdout
    )
    (RECORD / "environment.txt").write_text(environment_text, encoding="utf-8")

    inputs["source_code"] = {
        rel(RUNNER_PY): sha256(RUNNER_PY),
        rel(RUNNER_R): sha256(RUNNER_R),
        rel(RECORD / "config.json"): sha256(RECORD / "config.json"),
    }
    inputs["runtime"] = {
        "R": "4.6.1",
        "spEDM": "1.12",
        "local_library": rel(RLIB),
        "spEDM_runtime_source_sha256": "70dad0f8d41a4a31d0d4cb3b74273e2034dd87227d85abcaa8eb4c927e4cc153",
    }
    (RECORD / "input_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Stage 10B GCMC one-seed smoke v1

This immutable record contains the first scientific GCMC feature-recovery smoke diagnostic for seed 20270908 on S2/S4 frozen scale-fit data.

Scientific conclusion: **NOT_READY_METHOD_OR_IMPLEMENTATION_CONCERN**. The runtime integration and deterministic reproduction passed, but x2 was not top-ranked under any preregistered configuration and S4 did not appear weak/unstructured.

Exact executed command:

```powershell
D:\\conda\\python.exe causal_strength_diagnostic_v1\\run_stage10b_gcmc_one_seed_smoke_v1.py
```

The Python orchestrator invoked the R implementation twice (full run and clean-process primary reproduction) with R 4.6.1, project-local spEDM 1.12, `threads=1`, and `LC_ALL=C`.

Key files:

- `config.json`: preregistered primary and sensitivity protocol, created before result inspection.
- `METHOD.md`: data, graph, API, and direction logic.
- `input_manifest.json`: frozen artifact/source hashes and label-access contract.
- `gcmc_direction_results.csv`: complete preregistered GCMC CS/xmap fields for both directions.
- `gcmc_primary_results.csv`: frozen primary subset.
- `gcmc_sensitivity_results.csv`: full style/E/k grid.
- `fnn_curves.csv`: descriptive FNN results.
- `reproducibility_check.json`: clean-process comparison.
- `stage10b_summary.md` and `.json`: interpretation and decision gate.

One recoverable serialization issue occurred before the successful run: spEDM returned E=1:9 when E=1:10 was requested. The output writer was changed to record returned E labels and an explicit truncation flag; no scientific parameter or result was changed. The successful run had empty R warning and stderr files.
"""
    (RECORD / "README.md").write_text(readme, encoding="utf-8")

    (RECORD / "finalization_log.txt").write_text(
        "The first documentation-finalizer invocation failed before completing the record because "
        "an unused pandas pivot attempted to reshape repeated FNN dimensions. The line was removed; "
        "this was a documentation-only issue and did not rerun or alter any GCMC/FNN calculation.\n",
        encoding="utf-8",
    )

    output_names = [
        "README.md", "METHOD.md", "config.json", "input_manifest.json", "environment.txt",
        "run_log.txt", "stderr.txt", "repro_run_log.txt", "repro_stderr.txt", "r_warnings.txt",
        "gcmc_primary_results.csv", "gcmc_sensitivity_results.csv", "gcmc_direction_results.csv",
        "gcmc_primary_repro_results.csv", "gcmc_direction_repro_results.csv", "fnn_curves.csv",
        "graph_diagnostics.json", "reproducibility_check.json", "stage9_seed_20270908_context.csv",
        "stage10b_summary.json", "stage10b_summary.md", "finalization_log.txt",
    ]
    manifest = {
        "created_by": rel(Path(__file__).resolve()),
        "canonical_outputs": [
            {"path": name, "sha256": sha256(RECORD / name), "bytes": (RECORD / name).stat().st_size}
            for name in output_names
        ],
        "source_code": [
            {"path": rel(p), "sha256": sha256(p)} for p in (RUNNER_PY, RUNNER_R, Path(__file__).resolve())
        ],
        "updated_project_files": [
            {
                "path": "docs/methodology/METHOD_PROVENANCE.md",
                "sha256": sha256(ROOT / "docs/methodology/METHOD_PROVENANCE.md"),
            }
        ],
    }
    (RECORD / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("STAGE10B_FINALIZATION_PASS")


if __name__ == "__main__":
    main()
