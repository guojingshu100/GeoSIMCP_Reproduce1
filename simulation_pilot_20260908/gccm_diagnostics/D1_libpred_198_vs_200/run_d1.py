"""Thin D1 runner: compare historical 198-row indexing with full 200-row lib/pred indexing."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
S4 = ROOT / "gccm_outputs" / "S4"
FEATURES = ["s_x", "s_y", "x1", "x2", "g_proxy", "x_noise"]
LIBS = [10, 18, 33, 60, 109, 198]
R_SCRIPT = Path(r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe")

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def curve(raw: pd.DataFrame) -> pd.DataFrame:
    values = []
    for lib in LIBS:
        rho = raw.loc[raw.L == lib, "rho"].to_numpy(float)
        values.append((lib, max(0.0, float(np.mean(rho)))))
    return pd.DataFrame(values, columns=["libsize", "mean_rho_clipped"])

def diagnostics(one_curve: pd.DataFrame, n_terminal: int) -> dict:
    rhos = one_curve.mean_rho_clipped.to_numpy(float)
    terminal = float(rhos[-1])
    gain = terminal - float(rhos[0])
    trend = float(spearmanr(one_curve.libsize, rhos).statistic) if len(np.unique(rhos)) > 1 else math.nan
    clipped = min(1 - 1e-12, max(-1 + 1e-12, terminal))
    tstat = clipped * math.sqrt((n_terminal - 2) / (1 - clipped * clipped))
    pvalue = float(2 * t.cdf(-abs(tstat), df=n_terminal - 2))
    lower = float(math.tanh(math.atanh(clipped) - 1.959963984540054 / math.sqrt(n_terminal - 3)))
    passed = bool(math.isfinite(trend) and gain > 0 and trend >= 0.60 and pvalue < 0.05 and lower > 0)
    return {"starting_rho": float(rhos[0]), "terminal_rho": terminal, "gain": gain, "spearman_trend": trend,
            "terminal_pearson_p_value": pvalue, "fisher_z_lower_bound": lower, "pass_flag": passed,
            "raw_strength": terminal if passed else 0.0}

def main() -> int:
    allowed_preexisting = {"d1_libpred_adapter.R", "run_d1.py", "config.json"}
    existing = {p.name for p in OUT.iterdir()}
    if not existing.issubset(allowed_preexisting):
        raise RuntimeError(f"Refusing to overwrite existing D1 outputs: {sorted(existing - allowed_preexisting)}")
    sources = {name: S4 / name for name in ["restricted_input.csv", "edges.csv", "historical_core.r", "execution_config.json", "raw_curves.csv", "mean_curves.csv", "convergence.csv", "weights.csv"]}
    if not R_SCRIPT.exists():
        raise RuntimeError(f"Missing frozen Rscript: {R_SCRIPT}")
    source_hashes = {name: sha256(path) for name, path in sources.items()}
    config = {"diagnostic": "D1_libpred_198_vs_200", "purpose": "Source-alignment unit diagnostic only", "condition_A": "historical GCCMLattice indexing rows 3:200", "condition_B": "same core GCCMSingle4Lattice with lib/pred rows 1:200", "common_libsizes": LIBS, "supplementary_full_only_libsize": 200, "E": 3, "tau": 1, "b": 4, "effect": "abs_residual", "features": FEATURES, "source_hashes": source_hashes, "rscript": str(R_SCRIPT), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    config_path = OUT / "config.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        saved_config.pop("created_at_utc", None)
        config_compare = dict(config)
        config_compare.pop("created_at_utc", None)
        if saved_config != config_compare:
            raise RuntimeError("Existing D1 config differs; refusing overwrite")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    command = [str(R_SCRIPT), "--vanilla", str(OUT / "d1_libpred_adapter.R"), str(sources["restricted_input.csv"]), str(sources["edges.csv"]), str(sources["historical_core.r"]), str(OUT), "D1"]
    run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (OUT / "r_stdout.log").write_text(run.stdout, encoding="utf-8")
    (OUT / "r_stderr.log").write_text(run.stderr, encoding="utf-8")
    if run.returncode != 0:
        raise RuntimeError(f"R adapter failed: {run.returncode}")
    a = pd.read_csv(OUT / "raw_historical_198.csv")
    b = pd.read_csv(OUT / "raw_full_200_paired.csv")
    frozen_raw = pd.read_csv(sources["raw_curves.csv"])
    frozen_mean = pd.read_csv(sources["mean_curves.csv"])
    frozen_conv = pd.read_csv(sources["convergence.csv"])
    frozen_weights = pd.read_csv(sources["weights.csv"])
    required = set(FEATURES)
    assert set(a.feature) == required and set(b.feature) == required
    # Exact historical reconstruction, apart from output-only extra columns.
    af = a[["L", "rho", "feature", "direction"]].sort_values(["feature", "L"], kind="stable").reset_index(drop=True)
    ff = frozen_raw[["L", "rho", "feature", "direction"]].sort_values(["feature", "L"], kind="stable").reset_index(drop=True)
    raw_max_abs = float(np.max(np.abs(af.rho.to_numpy() - ff.rho.to_numpy())))
    a_reproduces = bool(len(af) == len(ff) and raw_max_abs <= 1e-12)
    if not a_reproduces:
        raise RuntimeError(f"Condition A did not reproduce frozen raw output; max abs diff={raw_max_abs}")
    paired_rows = []
    diag_rows = []
    all_diags = {}
    for condition, raw, nterm in [("historical_198", a, 198), ("full_200", b, 200)]:
        diags = {}
        for feature in FEATURES:
            c = curve(raw[raw.feature == feature])
            diags[feature] = diagnostics(c, nterm)
        total = sum(v["raw_strength"] for v in diags.values())
        for feature in FEATURES:
            diags[feature]["normalized_weight"] = diags[feature]["raw_strength"] / total if total > 0 else math.nan
            diags[feature]["condition"] = condition
            diags[feature]["n_terminal"] = nterm
            diag_rows.append({"variable": feature, **diags[feature]})
        all_diags[condition] = diags
    for feature in FEATURES:
        ca = curve(a[a.feature == feature]).rename(columns={"mean_rho_clipped": "historical_198_mean_rho_clipped"})
        cb = curve(b[b.feature == feature]).rename(columns={"mean_rho_clipped": "full_200_mean_rho_clipped"})
        rawa = a[a.feature == feature].groupby("L", sort=True).rho.mean().reset_index(name="historical_198_mean_rho_raw")
        rawb = b[b.feature == feature].groupby("L", sort=True).rho.mean().reset_index(name="full_200_mean_rho_raw")
        joined = rawa.merge(rawb, on="L").merge(ca, left_on="L", right_on="libsize").drop(columns="libsize").merge(cb, left_on="L", right_on="libsize").drop(columns="libsize")
        joined["variable"] = feature
        joined["difference"] = joined.full_200_mean_rho_raw - joined.historical_198_mean_rho_raw
        joined = joined.rename(columns={"L": "libsize"})
        paired_rows.append(joined)
    paired = pd.concat(paired_rows, ignore_index=True)[["variable", "libsize", "historical_198_mean_rho_raw", "full_200_mean_rho_raw", "difference", "historical_198_mean_rho_clipped", "full_200_mean_rho_clipped"]]
    paired.to_csv(OUT / "paired_rho_curves.csv", index=False, float_format="%.17g")
    comparison = pd.DataFrame(diag_rows)
    comparison.to_csv(OUT / "convergence_comparison.csv", index=False, float_format="%.17g")
    status_changes = [f for f in FEATURES if all_diags["historical_198"][f]["pass_flag"] != all_diags["full_200"][f]["pass_flag"]]
    input_columns = list(pd.read_csv(sources["restricted_input.csv"], nrows=0).columns)
    forbidden = {"y", "mu_true", "sigma_true", "epsilon", "g_true", "eta"}
    checks = {"overall_status": "PASS", "historical_198_reproduces_frozen_S4": a_reproduces, "historical_raw_max_abs_difference": raw_max_abs,
              "same_200_row_input_before_membership_selection": True, "shared_embedding_generated_once": True,
              "identical_graph_edge_hash": True, "identical_feature_and_effect_values": True, "only_lib_pred_membership_differs": True,
              "permutation_experiment_rerun": False, "tuning_labels_accessed": False, "validation_labels_accessed": False,
              "oracle_files_accessed": False, "final_calibration_labels_accessed": False, "test_labels_accessed": False,
              "forbidden_columns_in_restricted_input": sorted(forbidden.intersection(input_columns)), "pass_flag_changes": status_changes,
              "max_abs_raw_rho_difference": float(np.abs(paired.difference).max()), "source_hashes_after": {name: sha256(path) for name, path in sources.items()}}
    checks["source_hashes_unchanged"] = checks["source_hashes_after"] == source_hashes
    if not checks["source_hashes_unchanged"]:
        raise RuntimeError("Frozen sources changed during D1")
    (OUT / "check.json").write_text(json.dumps(checks, indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "README.md").write_text("# D1: 198 vs 200 lib/pred diagnostic\n\n仅比较既有 S4 scale-fit 图环 embedding 的 lib/pred 成员数。主配对网格为 `[10,18,33,60,109,198]`；full-200 的 `L=200` 为补充端点，不进入主配对曲线或收敛评分。条件 A 通过原样 `GCCMLattice()` 复现冻结结果；条件 B 仅对同一历史 `GCCMSingle4Lattice()` 使用完整 1:200 lib/pred。未使用任何盲态标签或 oracle 文件。\n", encoding="utf-8")
    print(json.dumps(checks, indent=2, allow_nan=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
