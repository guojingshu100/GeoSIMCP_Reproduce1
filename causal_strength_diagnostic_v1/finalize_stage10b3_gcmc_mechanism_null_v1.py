"""Finalize Stage 10B.3 summaries and manifests without rerunning GCMC."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b3_gcmc_mechanism_null_v1"
STAGE9 = ROOT / "experiments/causal_strength_diagnostic/20260917_stage9_cmi_feature_recovery_v1"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")
RLIB = ROOT / "environment/gcmc_spedm_1.12_r4.6.1/library"
RUNNER_PY = ROOT / "causal_strength_diagnostic_v1/run_stage10b3_gcmc_mechanism_null_v1.py"
RUNNER_R = ROOT / "causal_strength_diagnostic_v1/run_stage10b3_gcmc_mechanism_null_v1.R"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def stats(series: pd.Series, prefix: str) -> dict:
    x = series.astype(float)
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_sd": float(x.std(ddof=1)),
        f"{prefix}_q025": float(x.quantile(0.025)),
        f"{prefix}_q05": float(x.quantile(0.05)),
        f"{prefix}_q50": float(x.quantile(0.50)),
        f"{prefix}_q95": float(x.quantile(0.95)),
        f"{prefix}_q975": float(x.quantile(0.975)),
    }


def main() -> None:
    g = pd.read_csv(RECORD / "gcmc_controlled_results.csv")
    truth = pd.read_csv(RECORD / "controlled_truth_diagnostics.csv")
    observed = pd.read_csv(RECORD / "observed_spatial_diagnostics.csv")
    permutations = pd.read_csv(RECORD / "observed_naive_permutation_results.csv")
    config = json.loads((RECORD / "config.json").read_text(encoding="utf-8"))
    if len(g) != 500 or len(truth) != 250 or len(permutations) != 800:
        raise AssertionError("unexpected Stage-10B.3 row count")
    if g.groupby(["mechanism", "style"]).size().ne(50).any():
        raise AssertionError("incomplete controlled replicate group")
    if np.max(np.abs(g.forward_xmap - g.forward_cs)) > 1e-12 or np.max(np.abs(g.reverse_xmap - g.reverse_cs)) > 1e-12:
        raise AssertionError("full-library xmap/cs mismatch")

    null_rows = []
    for (mechanism, style), group in g[g.mechanism.isin(["M0_IID_NULL", "M1_SPATIALLY_STRUCTURED_NULL"])].groupby(["mechanism", "style"], sort=True):
        row = {"mechanism": mechanism, "style": int(style), "replicates": len(group)}
        row.update(stats(group.forward_xmap, "forward"))
        row.update({
            "builtin_sig_lt_0_05_count": int((group.forward_sig < 0.05).sum()),
            "builtin_sig_lt_0_05_rate": float((group.forward_sig < 0.05).mean()),
            "mean_minus_theta_0_5": float(group.forward_xmap.mean() - 0.5),
            "mean_minus_analytic_0_2832": float(group.forward_xmap.mean() - 0.2832),
        })
        null_rows.append(row)
    null_summary = pd.DataFrame(null_rows)
    null_summary.to_csv(RECORD / "gcmc_null_summary.csv", index=False, lineterminator="\n", float_format="%.17g")

    mechanism_rows = []
    for (mechanism, style), group in g[g.mechanism.isin([
        "M2_SAME_LOCATION_RESIDUAL_SCALE", "M3_SPATIAL_LAG_CAUSAL", "M4_DIRECT_SAME_LOCATION_MEAN"
    ])].groupby(["mechanism", "style"], sort=True):
        null = g[(g.mechanism == "M1_SPATIALLY_STRUCTURED_NULL") & (g["style"] == style)]
        q05, q95 = float(null.forward_xmap.quantile(0.05)), float(null.forward_xmap.quantile(0.95))
        paired = group[["generation_seed", "forward_xmap"]].merge(
            null[["generation_seed", "forward_xmap"]], on="generation_seed", suffixes=("_signal", "_M1")
        )
        row = {"mechanism": mechanism, "style": int(style), "replicates": len(group)}
        row.update(stats(group.forward_xmap, "forward"))
        row.update(stats(group.reverse_xmap, "reverse"))
        row.update(stats(group.forward_minus_reverse, "forward_minus_reverse"))
        row.update({
            "forward_p_mean": float(group.forward_sig.mean()),
            "forward_p_median": float(group.forward_sig.median()),
            "builtin_sig_lt_0_05_count": int((group.forward_sig < 0.05).sum()),
            "builtin_sig_lt_0_05_rate": float((group.forward_sig < 0.05).mean()),
            "M1_q05": q05,
            "M1_q95": q95,
            "fraction_below_M1_q05": float((group.forward_xmap < q05).mean()),
            "fraction_exceeding_M1_q95": float((group.forward_xmap > q95).mean()),
            "paired_score_below_M1_count": int((paired.forward_xmap_signal < paired.forward_xmap_M1).sum()),
            "paired_mean_score_minus_M1": float((paired.forward_xmap_signal - paired.forward_xmap_M1).mean()),
        })
        mechanism_rows.append(row)
    mechanism_summary = pd.DataFrame(mechanism_rows)
    mechanism_summary.to_csv(RECORD / "gcmc_mechanism_summary.csv", index=False, lineterminator="\n", float_format="%.17g")

    permutation_rows = []
    for (scenario, feature), group in permutations.groupby(["scenario", "feature"], sort=True):
        observed_score = float(group.loc[group.kind == "observed", "forward_xmap"].iloc[0])
        perm = group.loc[group.kind == "permuted", "forward_xmap"].astype(float)
        row = {"scenario": scenario, "feature": feature, "observed_forward_xmap": observed_score, "permutations": len(perm)}
        row.update(stats(perm, "permutation"))
        row.update({
            "plus_one_lower_tail": float((1 + (perm <= observed_score).sum()) / (len(perm) + 1)),
            "plus_one_upper_tail": float((1 + (perm >= observed_score).sum()) / (len(perm) + 1)),
            "observed_minus_permutation_mean": float(observed_score - perm.mean()),
            "label": "NAIVE_LABEL_PERMUTATION_SANITY_CHECK",
            "valid_spatial_null": False,
        })
        permutation_rows.append(row)
    permutation_summary = pd.DataFrame(permutation_rows)
    permutation_summary.to_csv(RECORD / "observed_naive_permutation_summary.csv", index=False,
                               lineterminator="\n", float_format="%.17g")

    truth_summary = []
    truth_columns = [
        "pearson_X_A", "spearman_X_A", "moran_X", "moran_A", "moran_sigma",
        "pearson_X_sigma", "pearson_spatial_lag_X_A"
    ]
    for mechanism, group in truth.groupby("mechanism", sort=True):
        truth_summary.append({
            "mechanism": mechanism,
            "replicates": len(group),
            **{f"{column}_mean": None if group[column].isna().all() else float(group[column].mean()) for column in truth_columns}
        })

    def row_for(frame: pd.DataFrame, mechanism: str, style: int) -> dict:
        return frame[(frame.mechanism == mechanism) & (frame["style"] == style)].iloc[0].to_dict()

    summary = {
        "stage": "10B.3",
        "status": "COMPLETE",
        "output_convention": {
            "primary": "result$xmap$y_xmap_x_mean and result$xmap$y_xmap_x_sig",
            "reverse": "result$xmap$x_xmap_y_mean",
            "cs_retained": True,
            "stage10b2_ambiguity_preserved": True,
        },
        "successful_replicates": {mechanism: 50 for mechanism in sorted(g.mechanism.unique())},
        "truth_diagnostics": truth_summary,
        "null_summary": null_summary.to_dict("records"),
        "mechanism_summary": mechanism_summary.to_dict("records"),
        "observed_spatial_diagnostics": observed.to_dict("records"),
        "naive_permutation_summary": permutation_summary.to_dict("records"),
        "key_findings": {
            "M0_style1_builtin_false_positive_rate": row_for(null_summary, "M0_IID_NULL", 1)["builtin_sig_lt_0_05_rate"],
            "M1_style1_builtin_false_positive_rate": row_for(null_summary, "M1_SPATIALLY_STRUCTURED_NULL", 1)["builtin_sig_lt_0_05_rate"],
            "M0_style0_builtin_false_positive_rate": row_for(null_summary, "M0_IID_NULL", 0)["builtin_sig_lt_0_05_rate"],
            "M1_style0_builtin_false_positive_rate": row_for(null_summary, "M1_SPATIALLY_STRUCTURED_NULL", 0)["builtin_sig_lt_0_05_rate"],
            "M2_style1_fraction_below_M1_q05": row_for(mechanism_summary, "M2_SAME_LOCATION_RESIDUAL_SCALE", 1)["fraction_below_M1_q05"],
            "M3_style1_fraction_below_M1_q05": row_for(mechanism_summary, "M3_SPATIAL_LAG_CAUSAL", 1)["fraction_below_M1_q05"],
            "M2_style1_fraction_exceeding_M1_q95": row_for(mechanism_summary, "M2_SAME_LOCATION_RESIDUAL_SCALE", 1)["fraction_exceeding_M1_q95"],
            "M3_style1_fraction_exceeding_M1_q95": row_for(mechanism_summary, "M3_SPATIAL_LAG_CAUSAL", 1)["fraction_exceeding_M1_q95"],
        },
        "classification": "C. GENERAL_SIGNAL_DETECTION_BUT_NULL_CALIBRATION_FAILURE",
        "classification_evidence": [
            "M3 was separated from M1 in all 50 paired replicates under both styles, so the integration detects the aligned spatial-lag mechanism.",
            "M2 was also separated from M1 in all 50 paired replicates under both styles, so same-location residual-scale compatibility is not the primary failure.",
            "True signals lowered the implemented xmap AUC; descending raw-score rankings have the wrong empirical orientation for these controls.",
            "M3 forward-minus-reverse means were -0.0091 for style1 and +0.0281 for style0, so the strong null separation was not accompanied by stable directional discrimination.",
            "Built-in significance was inflated for M1/style1 and severely inflated for both style0 nulls.",
            "The analytic 0.2832 independent-ranking value did not describe actual IID variables embedded on the shared graph.",
        ],
        "gccp_implication": "3. GCMC should not be used to generate primary residual-scale causal weights for the current GCCP formulation.",
        "cmi_context": "Stage 9 already demonstrated x2 recovery across 20 residual-scale replicates with strong S4 control; mechanism compatibility is currently better supported for CMI for this GCCP task, without claiming universal superiority.",
        "next_recommended_experiment": {
            "name": "STAGE 10B.4 - GRAPH-PRESERVING GCMC NULL CALIBRATION AND SCORE-ORIENTATION AUDIT",
            "objective": "Predeclare feature/target graph-preserving null surrogates, calibrate both tails of xmap conditional on spatial structure, and validate a null-standardized diagnostic on M0-M4 before any Stage-10C feature benchmark.",
            "prohibitions": ["no GCMC weights", "no conformal prediction", "no parameter tuning using x2"]
        },
        "warnings": [
            "No R warnings, stderr, failed controlled replicates, or failed permutation calls were recorded.",
            "The naive label permutation destroys A_model spatial structure and is not a valid final spatial null.",
            "M1 thresholds and below/above-tail rates are descriptive diagnostics, not validated inference or power estimates."
        ],
    }
    (RECORD / "stage10b3_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    m0s1 = row_for(null_summary, "M0_IID_NULL", 1)
    m1s1 = row_for(null_summary, "M1_SPATIALLY_STRUCTURED_NULL", 1)
    m0s0 = row_for(null_summary, "M0_IID_NULL", 0)
    m1s0 = row_for(null_summary, "M1_SPATIALLY_STRUCTURED_NULL", 0)
    m2s1 = row_for(mechanism_summary, "M2_SAME_LOCATION_RESIDUAL_SCALE", 1)
    m3s1 = row_for(mechanism_summary, "M3_SPATIAL_LAG_CAUSAL", 1)
    m4s1 = row_for(mechanism_summary, "M4_DIRECT_SAME_LOCATION_MEAN", 1)
    m2s0 = row_for(mechanism_summary, "M2_SAME_LOCATION_RESIDUAL_SCALE", 0)
    m3s0 = row_for(mechanism_summary, "M3_SPATIAL_LAG_CAUSAL", 0)
    m4s0 = row_for(mechanism_summary, "M4_DIRECT_SAME_LOCATION_MEAN", 0)

    md = f"""# Stage 10B.3 GCMC mechanism/null diagnostic

Final classification: **C. GENERAL_SIGNAL_DETECTION_BUT_NULL_CALIBRATION_FAILURE**.

## Frozen protocol

- Exact Stage-10B 200-node Delaunay graph, SHA-256 `7155d18b4f4263d373eca061707165630c9977f3e67b4786d9e6e5556e8d2159`.
- 50 paired seeds per mechanism, M0-M4, fixed before output inspection.
- Spatial covariance `exp(-distance/rho)` with `rho=0.15912518479562674`, twice the median Delaunay edge length.
- GCMC E=3, k=25, tau=1, styles 1 primary and 0 secondary, explicit graph, one thread.
- Primary output is vignette-convention `xmap$y_xmap_x_mean`; `cs` was retained and remained exactly equal at full library size.

## Mechanism verification

Mean truth diagnostics confirm the intended controls: M0 Moran(X/A) approximately -0.004/-0.003; M1 approximately 0.551/0.289 with near-zero X-A correlation; M2 corr(X,sigma)=0.924 and corr(X,A)=0.538; M3 corr(lag(X),A)=0.895; M4 corr(X,A)=0.893.

## Null distributions

| Null | Style | Mean | Median | SD | 2.5% | 5% | 95% | 97.5% | built-in p<0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 IID | 1 | {m0s1['forward_mean']:.6f} | {m0s1['forward_median']:.6f} | {m0s1['forward_sd']:.6f} | {m0s1['forward_q025']:.6f} | {m0s1['forward_q05']:.6f} | {m0s1['forward_q95']:.6f} | {m0s1['forward_q975']:.6f} | {int(m0s1['builtin_sig_lt_0_05_count'])}/50 |
| M1 spatial | 1 | {m1s1['forward_mean']:.6f} | {m1s1['forward_median']:.6f} | {m1s1['forward_sd']:.6f} | {m1s1['forward_q025']:.6f} | {m1s1['forward_q05']:.6f} | {m1s1['forward_q95']:.6f} | {m1s1['forward_q975']:.6f} | {int(m1s1['builtin_sig_lt_0_05_count'])}/50 |
| M0 IID | 0 | {m0s0['forward_mean']:.6f} | {m0s0['forward_median']:.6f} | {m0s0['forward_sd']:.6f} | {m0s0['forward_q025']:.6f} | {m0s0['forward_q05']:.6f} | {m0s0['forward_q95']:.6f} | {m0s0['forward_q975']:.6f} | {int(m0s0['builtin_sig_lt_0_05_count'])}/50 |
| M1 spatial | 0 | {m1s0['forward_mean']:.6f} | {m1s0['forward_median']:.6f} | {m1s0['forward_sd']:.6f} | {m1s0['forward_q025']:.6f} | {m1s0['forward_q05']:.6f} | {m1s0['forward_q95']:.6f} | {m1s0['forward_q975']:.6f} | {int(m1s0['builtin_sig_lt_0_05_count'])}/50 |

The theta=0.5 test is roughly conservative for M0/style1 in this 50-replicate diagnostic (2%), inflated for M1/style1 (18%), and severely inflated for style0 (92% M0; 76% M1). The analytic independent-ranking value 0.2832 is far below the actual M0/M1 means: sharing the graph embedding makes actual neighbor rankings non-independent.

## Controlled signals

| Mechanism | Style | Forward mean | Reverse mean | Mean F-R | built-in p<0.05 | Below M1 5% | Above M1 95% |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2 residual scale | 1 | {m2s1['forward_mean']:.6f} | {m2s1['reverse_mean']:.6f} | {m2s1['forward_minus_reverse_mean']:.6f} | {int(m2s1['builtin_sig_lt_0_05_count'])}/50 | {m2s1['fraction_below_M1_q05']:.0%} | {m2s1['fraction_exceeding_M1_q95']:.0%} |
| M3 spatial lag | 1 | {m3s1['forward_mean']:.6f} | {m3s1['reverse_mean']:.6f} | {m3s1['forward_minus_reverse_mean']:.6f} | {int(m3s1['builtin_sig_lt_0_05_count'])}/50 | {m3s1['fraction_below_M1_q05']:.0%} | {m3s1['fraction_exceeding_M1_q95']:.0%} |
| M4 direct mean | 1 | {m4s1['forward_mean']:.6f} | {m4s1['reverse_mean']:.6f} | {m4s1['forward_minus_reverse_mean']:.6f} | {int(m4s1['builtin_sig_lt_0_05_count'])}/50 | {m4s1['fraction_below_M1_q05']:.0%} | {m4s1['fraction_exceeding_M1_q95']:.0%} |
| M2 residual scale | 0 | {m2s0['forward_mean']:.6f} | {m2s0['reverse_mean']:.6f} | {m2s0['forward_minus_reverse_mean']:.6f} | {int(m2s0['builtin_sig_lt_0_05_count'])}/50 | {m2s0['fraction_below_M1_q05']:.0%} | {m2s0['fraction_exceeding_M1_q95']:.0%} |
| M3 spatial lag | 0 | {m3s0['forward_mean']:.6f} | {m3s0['reverse_mean']:.6f} | {m3s0['forward_minus_reverse_mean']:.6f} | {int(m3s0['builtin_sig_lt_0_05_count'])}/50 | {m3s0['fraction_below_M1_q05']:.0%} | {m3s0['fraction_exceeding_M1_q95']:.0%} |
| M4 direct mean | 0 | {m4s0['forward_mean']:.6f} | {m4s0['reverse_mean']:.6f} | {m4s0['forward_minus_reverse_mean']:.6f} | {int(m4s0['builtin_sig_lt_0_05_count'])}/50 | {m4s0['fraction_below_M1_q05']:.0%} | {m4s0['fraction_exceeding_M1_q95']:.0%} |

All M2/M3/M4 scores fell below the M1 5th percentile, and all 50 paired scores were lower than their paired M1 score under both styles. None exceeded the requested M1 95th-percentile threshold. True alignment therefore moves this implemented AUC downward; raw descending feature ranking is not calibrated as causal strength. M3's mean forward-minus-reverse contrast was only -0.0091 for style1 and +0.0281 for style0, so signal separation was strong but causal directionality was not stable.

## Observed g_proxy diagnosis

`g_proxy` Moran's I is 0.8289 in both S2/S4, versus -0.0137 for x2, 0.0386 for x1, and 0.0079 for x_noise. Yet g_proxy's direct correlations with A_model are only 0.0587 (S2) and 0.0626 (S4). Its observed k=25 scores are near the naive-permutation means rather than in an informative tail. This supports a graph-aligned spatial-structure baseline explanation for persistent high g_proxy scores; it does not prove confounding or causality.

In contrast, S2 x2 has corr(x2,A_model)=0.4039 and observed xmap 0.576, below all 99 naive permutations (plus-one lower-tail 0.01). The permutation destroys A_model spatial structure and is only a sanity check, not a valid final spatial null.

## Decision and GCCP implication

GCMC detects M3 and M2, so neither implementation failure nor same-location incompatibility is primary. The result is **C. GENERAL_SIGNAL_DETECTION_BUT_NULL_CALIBRATION_FAILURE**.

GCMC should not generate primary residual-scale causal weights for the current GCCP formulation. Stage-9 CMI currently has better demonstrated compatibility for this specific residual-scale task, without implying universal superiority.

Next: **STAGE 10B.4 - GRAPH-PRESERVING GCMC NULL CALIBRATION AND SCORE-ORIENTATION AUDIT**. Predeclare spatially preserving null surrogates and validate both-tail, null-standardized interpretation on M0-M4 before any Stage-10C benchmark.
"""
    (RECORD / "stage10b3_summary.md").write_text(md, encoding="utf-8")

    method = """# Method

Stage 10B.3 fixes the Stage-10B Delaunay graph and coordinates and changes only the controlled causal mechanism. Fifty paired master seeds generate M0-M4. M1-M4 use the same predeclared exponential Gaussian covariance and paired spatial X realization per seed; M0 is IID. The generator never reads S2/S4 outcomes or A_model.

Moran's I and graph lags use the same row-standardized binary adjacency. GCMC calls installed `spEDM::gcmc()` directly with E=3, k=25, tau=1, explicit `nb`, and styles 1/0. The R loop is identical across mechanisms and does not select parameters from truth labels.

Observed S2/S4 derived Stage-10B tables are accessed only after the controlled benchmark completes. The 99-permutation check fixes each feature/graph and shuffles A_model; because this destroys spatial structure it is explicitly not treated as a final spatial null.
"""
    (RECORD / "METHOD.md").write_text(method, encoding="utf-8")
    readme = """# Stage 10B.3: GCMC mechanism/null diagnostic

This immutable record tests IID null, spatial null, same-location residual scale, spatial-lag causality, and direct same-location signal on the exact frozen Stage-10B graph.

Final classification: `C. GENERAL_SIGNAL_DETECTION_BUT_NULL_CALIBRATION_FAILURE`.

Exact execution command:

```powershell
D:\\conda\\python.exe causal_strength_diagnostic_v1\\run_stage10b3_gcmc_mechanism_null_v1.py
```

No Stage 10C, conformal prediction, causal-weight construction, upstream-data regeneration, predictor fitting, package modification, or network access occurred.
"""
    (RECORD / "README.md").write_text(readme, encoding="utf-8")

    r_expr = (
        f'.libPaths(c("{str(RLIB).replace(os.sep, "/")}",.libPaths())); '
        'pkgs<-c("spEDM","sf","terra","sdsfun","dplyr","ggplot2","Rcpp","RcppThread","RcppArmadillo"); '
        'cat(R.version.string,"\\n"); for(p in pkgs) cat(p,as.character(packageVersion(p)),"\\n"); sessionInfo()'
    )
    env = dict(os.environ)
    env["R_LIBS_USER"] = str(RLIB).replace("\\", "/")
    process = subprocess.run([str(RSCRIPT), "--vanilla", "-e", r_expr], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    (RECORD / "environment.txt").write_text(
        "R executable: C:\\Program Files\\R\\R-4.6.1\\bin\\Rscript.exe\n"
        f"Project-local library: {RLIB}\nPython executable: D:\\conda\\python.exe\n" + process.stdout,
        encoding="utf-8"
    )

    output_names = [
        "README.md", "METHOD.md", "config.json", "input_manifest.json", "environment.txt",
        "run_log.txt", "stderr.txt", "controlled_run_log.txt", "controlled_stderr.txt",
        "permutation_run_log.txt", "permutation_stderr.txt", "controlled_r_warnings.txt",
        "permutation_r_warnings.txt", "controlled_mechanisms.csv", "controlled_truth_diagnostics.csv",
        "gcmc_controlled_results.csv", "gcmc_null_summary.csv", "gcmc_mechanism_summary.csv",
        "observed_spatial_diagnostics.csv", "observed_naive_permutation_results.csv",
        "observed_naive_permutation_summary.csv", "stage10b3_summary.json", "stage10b3_summary.md",
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
    print("STAGE10B3_FINALIZATION_PASS")


if __name__ == "__main__":
    main()
