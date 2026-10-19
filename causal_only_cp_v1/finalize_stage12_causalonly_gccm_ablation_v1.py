"""Finalize Stage 12 after all scientific CSVs were written.

This is a reporting-only recovery from a strict-JSON NaN serialization error;
it does not rerun GCCM or alter any estimator result.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "experiments" / "causal_only_cp" / "20260918_stage12_causalonly_gccm_ablation_v1"
RUNNER = ROOT / "causal_only_cp_v1" / "run_stage12_causalonly_gccm_ablation_v1.py"
WRAPPER = ROOT / "causal_only_cp_v1" / "gccm_stage12_attribute_wrapper_v1.R"
RECOVERED = ROOT / "causal_only_cp_v1" / "stage11_frozen_split_inputs_v1"
STAGE11 = ROOT / "experiments" / "causal_only_cp" / "20260918_stage11_causalonly_cmi_normcp_v2"
STAGE11A = ROOT / "experiments" / "causal_only_cp" / "20260918_stage11a_frozen_input_recovery_v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def finite(value):
    if isinstance(value, dict): return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list): return [finite(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def main():
    target = RECORD / "stage12_summary.json"
    if target.exists(): raise FileExistsError(target)
    scenarios = pd.read_csv(RECORD / "scenario_summary.csv")
    pairs = pd.read_csv(RECORD / "paired_comparisons.csv")
    results = pd.read_csv(RECORD / "replicate_results.csv")
    weights = pd.read_csv(RECORD / "gccm_weight_audit.csv")
    ktable = pd.read_csv(RECORD / "k_tuning_results.csv")
    preflight = json.loads((RECORD / "preflight_summary.json").read_text(encoding="utf-8"))
    if len(results) != 200 or len(weights) != 160 or len(pairs) != 18 or preflight["status"] != "PASS":
        raise AssertionError("completed-output integrity gate failed")

    wp = weights.pivot(index=["replicate_index", "scenario"], columns="feature", values="weight")
    wp = wp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    s2 = wp.xs("S2", level="scenario")
    s4 = wp.xs("S4", level="scenario")
    gccm = results[results.method == "CausalOnly-GCCM-NormCP"].copy()
    fallback = gccm.set_index(["replicate_index", "scenario"]).no_usable_gccm_signal.astype(bool)
    top_s2 = s2.idxmax(axis=1); top_s4 = s4.idxmax(axis=1)
    s2_nonzero = s2.max(axis=1) > 0; s4_nonzero = s4.max(axis=1) > 0

    # Re-evaluate only the frozen constant fallback identity assertion.  GCCM
    # itself is not rerun; weights and q rules are already fixed and sealed.
    stage12 = load_module("stage12_runner_finalize", RUNNER)
    loader = load_module("stage11a_loader_finalize", RECOVERED / "loader.py")
    fallback_diffs = []
    for (rep, scenario), is_fallback in fallback.items():
        if not is_fallback: continue
        seed = 20270907 + int(rep)
        info, _ = stage12.read_weight_info(RECORD / "gccm_runs" / f"replicate_{int(rep):02d}_seed_{seed}" / scenario)
        _, _, diff, _, _ = stage12.run_cp(loader, int(rep), seed, scenario, info, write_barrier=False)
        fallback_diffs.append({"replicate_index": int(rep), "scenario": scenario,
                               "max_abs_bound_diff_vs_raw": float(diff)})
    max_fallback = max(x["max_abs_bound_diff_vs_raw"] for x in fallback_diffs)
    if max_fallback > 1e-12: raise AssertionError("constant fallback is not Raw-CP equivalent")

    s2_res = results[results.scenario == "S2"].pivot(index="replicate_index", columns="method",
                                                      values=["coverage", "mean_width", "mean_interval_score"])
    x2 = s2.x2.reindex(s2_res.index).to_numpy(float)
    width_delta = (s2_res.mean_width["CausalOnly-GCCM-NormCP"] - s2_res.mean_width["Raw-CP"]).to_numpy(float)
    score_delta = (s2_res.mean_interval_score["CausalOnly-GCCM-NormCP"] - s2_res.mean_interval_score["Raw-CP"]).to_numpy(float)
    relationship = {
        "x2_weight_vs_width_delta_pearson": float(np.corrcoef(x2, width_delta)[0, 1]),
        "x2_weight_vs_width_delta_spearman": float(stats.spearmanr(x2, width_delta).statistic),
        "x2_weight_vs_interval_score_delta_pearson": float(np.corrcoef(x2, score_delta)[0, 1]),
        "x2_weight_vs_interval_score_delta_spearman": float(stats.spearmanr(x2, score_delta).statistic),
    }
    frequencies = {
        "S2_x2_positive": int((s2.x2 > 0).sum()),
        "S2_x2_largest": int(((s2.x2 > 0) & s2.x2.eq(s2.max(axis=1))).sum()),
        "S2_g_proxy_largest": int(((s2.g_proxy > 0) & s2.g_proxy.eq(s2.max(axis=1))).sum()),
        "S2_fallback": int((~s2_nonzero).sum()),
        "S4_fallback": int((~s4_nonzero).sum()),
        "S4_false_selection_replicates": int(s4_nonzero.sum()),
        "S4_dominant_counts": {k: int(v) for k, v in top_s4[s4_nonzero].value_counts().items()},
    }
    s2_g = scenarios[(scenarios.scenario == "S2") & (scenarios.method == "CausalOnly-GCCM-NormCP")].iloc[0]
    s2_raw = scenarios[(scenarios.scenario == "S2") & (scenarios.method == "Raw-CP")].iloc[0]
    scientific_class = ("GCCM_CAUSAL_ONLY_COVERAGE_FAILURE" if bool(s2_g.material_undercoverage) else
                        "GCCM_CAUSAL_ONLY_ALSO_IMPROVES_CP" if
                        (s2_g.mean_width < s2_raw.mean_width or s2_g.mean_interval_score < s2_raw.mean_interval_score)
                        else "GCCM_SIGNAL_DOES_NOT_TRANSLATE_TO_CP_GAIN")
    summary = finite({
        "status": "PASS", "reporting_recovery": "strict-JSON NaN was replaced by null; scientific CSVs unchanged",
        "readiness_class": "READY_RECOMPUTE_EXACT_ORIGINAL_GCCM_PROTOCOL", "preflight": preflight,
        "gccm_protocol": {"E": 3, "tau": 1, "b": 4, "gccm_seed": 42,
                          "library_sizes": [10, 18, 33, 60, 109, 198], "direction": "effect_xmap_cause"},
        "feature_frequencies": frequencies, "x2_weight_relationships": relationship,
        "fallback_audit": {"replicates": fallback_diffs, "maximum_bound_difference": max_fallback,
                           "tolerance": 1e-12, "status": "PASS"},
        "selected_k_distributions": {s: {str(int(k)): int(v) for k, v in
            gccm[gccm.scenario == s].selected_k.dropna().astype(int).value_counts().sort_index().items()}
            for s in ("S2", "S4")},
        "scenario_summary": scenarios.to_dict("records"), "paired_comparisons": pairs.to_dict("records"),
        "scientific_class": scientific_class, "test_unseal_rows": len(pd.read_csv(RECORD / "test_unseal_audit.csv")),
        "warnings": ["Original finalizer rejected NaN relative-effect fields under strict JSON; recovered without changing results."],
        "interpretation": "GCCM did not improve S2 width or interval score versus Raw-CP and was materially worse than CMI; S4 selections were frequent."
    })
    target.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    env_lines = [f"Python executable: {sys.executable}", f"Python: {platform.python_version()}",
                 f"Platform: {platform.platform()}"]
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn"):
        env_lines.append(f"{pkg}: {importlib.metadata.version(pkg)}")
    env_lines += ["R executable: C:/Program Files/R/R-4.6.1/bin/Rscript.exe", "R: 4.6.1",
                  "GCCM core: historical GCCM4Lattice.r; SHA-256 " + stage12.CORE_HASH]
    (RECORD / "environment.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    audit = f"""# GCCM provenance audit

Readiness: **READY_RECOMPUTE_EXACT_ORIGINAL_GCCM_PROTOCOL**.

The existing `simulation_pilot_20260908/gccm_outputs` contains one S2/S4
pilot realization (generator seed 20260907), not the Stage-11 seeds
20270908--20270927. Its target is nevertheless compatible: frozen
`abs_residual = abs(y-y_pred)` on 200 `scale_fit` rows.

Stage 12 therefore reused the historical core byte-for-byte (SHA-256
`{stage12.CORE_HASH}`), direction `effect_xmap_cause`, E=3, tau=1, b=4,
library grid [10,18,33,60,109,198], deterministic cyclic libraries, seed 42,
and all four original convergence gates. Per-feature calculations are
unchanged. The direct ablation restricts normalization to x1, x2, g_proxy,
and x_noise as preregistered; s_x/s_y construct Delaunay geometry only.

Historical rule: terminal rho is usable only when gain>0, Spearman>=0.60,
terminal Pearson p<0.05, and the Fisher-z 95% lower bound is >0. Usable
strengths are normalized by their sum. If none are usable, Stage 12 invokes
the preregistered constant-scale Raw-CP-equivalent fallback.
"""
    (RECORD / "gccm_provenance_audit.md").write_text(audit, encoding="utf-8")

    method = """# Method

Stage 12 is a direct causal-strength-estimator ablation. It uses the original
frozen GCCM point-graph estimator on model-residual magnitude from scale_fit,
but permits only x1, x2, g_proxy, and x_noise in the final causal distance.
Coordinates are used solely to build the Delaunay graph. Downstream feature
standardization, k candidates (20/40/60/80), tuning-MAE selection, final
calibration (n=200, rank 181 at alpha=.10), sealed test evaluation, interval
score, and fallback are identical to Stage 11.
"""
    (RECORD / "METHOD.md").write_text(method, encoding="utf-8")
    readme = f"""# Stage 12 — CausalOnly-GCCM direct ablation

Status: **PASS**  
Readiness audit: **READY_RECOMPUTE_EXACT_ORIGINAL_GCCM_PROTOCOL**  
Scientific class: **{scientific_class}**

Exact command:

`python causal_only_cp_v1\\run_stage12_causalonly_gccm_ablation_v1.py`

The initial execution completed all estimator/CP CSV outputs but strict JSON
serialization rejected NaN in optional relative-effect fields. Reporting was
completed with:

`python causal_only_cp_v1\\finalize_stage12_causalonly_gccm_ablation_v1.py`

No GCCM estimate was rerun by the finalizer. See `stage12_summary.md` for the
scientific interpretation and `stderr.txt` for the preserved reporting error.
"""
    (RECORD / "README.md").write_text(readme, encoding="utf-8")

    s2tab = scenarios[scenarios.scenario == "S2"][["method", "mean_coverage", "mean_width", "mean_interval_score", "material_undercoverage"]]
    s4tab = scenarios[scenarios.scenario == "S4"][["method", "mean_coverage", "mean_width", "mean_interval_score", "material_undercoverage"]]
    md = f"""# Stage 12 summary

## Outcome

**{scientific_class}**. GCCM preserved aggregate coverage but did not improve
S2 efficiency relative to Raw-CP. CMI remained distinctly better in S2.

## Feature behavior

- S2: x2 positive {frequencies['S2_x2_positive']}/20; x2 largest {frequencies['S2_x2_largest']}/20; g_proxy largest {frequencies['S2_g_proxy_largest']}/20; fallback {frequencies['S2_fallback']}/20.
- S4: false selection {frequencies['S4_false_selection_replicates']}/20; fallback {frequencies['S4_fallback']}/20; dominant counts {frequencies['S4_dominant_counts']}.
- Constant fallback maximum bound difference from Raw-CP: {max_fallback:.3g} (tolerance 1e-12).

## S2

{s2tab.to_markdown(index=False)}

## S4

{s4tab.to_markdown(index=False)}

## Interpretation

Removing geography from the final metric is not sufficient to rescue the
original GCCM signal. The Stage-11 gain is associated with CMI's better
residual-scale feature recovery, not causal-only localization in the abstract.
The next experiment should be a preregistered combined Geo+CMI normalized-CP
simulation ablation, retaining Raw, GeoOnly, CausalOnly-CMI, and Oracle.
"""
    (RECORD / "stage12_summary.md").write_text(md, encoding="utf-8")

    input_paths = [STAGE11A / "stage11a_summary.json", STAGE11 / "replicate_results.csv",
                   ROOT / "simulation_pilot_20260908" / "gccm_runtime" / "gccm_runtime_config_v1.json",
                   stage12.CORE, RUNNER, WRAPPER]
    input_manifest = {"inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size}
                                  for p in input_paths]}
    (RECORD / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n", encoding="utf-8")
    output_names = ["config.json", "gccm_artifact_inventory.csv", "preflight_summary.json", "test_unseal_audit.csv",
                    "gccm_weight_audit.csv", "replicate_results.csv", "k_tuning_results.csv", "scenario_summary.csv",
                    "paired_comparisons.csv", "stage12_summary.json", "stage12_summary.md", "environment.txt",
                    "gccm_provenance_audit.md", "README.md", "METHOD.md", "run_log.txt", "stderr.txt"]
    manifest = {"outputs": [{"path": n, "sha256": sha(RECORD/n), "bytes": (RECORD/n).stat().st_size}
                            for n in output_names]}
    (RECORD / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "scientific_class": scientific_class,
                      "fallback_max_abs_diff": max_fallback, "frequencies": frequencies}, indent=2))


if __name__ == "__main__":
    main()
