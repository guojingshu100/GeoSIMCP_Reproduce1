"""Generate controlled mechanisms and run the fixed Stage-10B.3 spEDM diagnostic."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORD = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b3_gcmc_mechanism_null_v1"
STAGE10B = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b_gcmc_one_seed_smoke_v1"
STAGE10B2 = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b2_gcmc_output_semantics_audit_v1"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")
RLIB = ROOT / "environment/gcmc_spedm_1.12_r4.6.1/library"
R_RUNNER = HERE / "run_stage10b3_gcmc_mechanism_null_v1.R"
GRAPH_SHA = "7155d18b4f4263d373eca061707165630c9977f3e67b4786d9e6e5556e8d2159"
MECHANISMS = [
    "M0_IID_NULL",
    "M1_SPATIALLY_STRUCTURED_NULL",
    "M2_SAME_LOCATION_RESIDUAL_SCALE",
    "M3_SPATIAL_LAG_CAUSAL",
    "M4_DIRECT_SAME_LOCATION_MEAN",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x, ddof=1)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def moran(x: np.ndarray, w: np.ndarray) -> float:
    z = x - np.mean(x)
    s0 = float(w.sum())
    return float((len(x) / s0) * (z @ w @ z) / (z @ z))


def truth_row(mechanism: str, replicate_index: int, seed: int, x: np.ndarray, a: np.ndarray,
              lag_x: np.ndarray, w: np.ndarray, sigma: np.ndarray | None) -> dict:
    result = {
        "mechanism": mechanism,
        "replicate_index": replicate_index,
        "generation_seed": seed,
        "pearson_X_A": pearson(x, a),
        "spearman_X_A": float(spearmanr(x, a).statistic),
        "moran_X": moran(x, w),
        "moran_A": moran(a, w),
        "pearson_spatial_lag_X_A": pearson(lag_x, a),
        "moran_sigma": np.nan,
        "pearson_X_sigma": np.nan,
    }
    if sigma is not None:
        result["moran_sigma"] = moran(sigma, w)
        result["pearson_X_sigma"] = pearson(x, sigma)
    return result


def run_r(mode: str, log_name: str, stderr_name: str) -> tuple[list[str], int]:
    command = [str(RSCRIPT), "--vanilla", str(R_RUNNER), str(RECORD), str(STAGE10B), str(RLIB), mode]
    env = dict(os.environ)
    env["R_LIBS_USER"] = str(RLIB).replace("\\", "/")
    env["LC_ALL"] = "C"
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=7200)
    (RECORD / log_name).write_text("COMMAND: " + subprocess.list2cmdline(command) + "\n" + process.stdout,
                                   encoding="utf-8")
    (RECORD / stderr_name).write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"R {mode} failed ({process.returncode}): {process.stderr[-3000:]}")
    return command, process.returncode


def main() -> None:
    for name in ("controlled_mechanisms.csv", "gcmc_controlled_results.csv", "observed_naive_permutation_results.csv"):
        if (RECORD / name).exists():
            raise FileExistsError(f"refusing to overwrite {name}")
    config = json.loads((RECORD / "config.json").read_text(encoding="utf-8"))
    assert config["generation"]["replicates_per_mechanism"] == 50
    assert config["gcmc"]["E"] == 3 and config["gcmc"]["k"] == 25

    edge_path = STAGE10B / "inputs/delaunay_edges.csv"
    s2_path = STAGE10B / "inputs/S2_scale_fit_gcmc.csv"
    s4_path = STAGE10B / "inputs/S4_scale_fit_gcmc.csv"
    if sha256(edge_path) != GRAPH_SHA:
        raise AssertionError("frozen graph hash mismatch")
    base = pd.read_csv(s2_path, float_precision="round_trip")
    paired = pd.read_csv(s4_path, float_precision="round_trip")
    assert len(base) == len(paired) == 200
    pd.testing.assert_frame_equal(base[["sample_id", "s_x", "s_y"]], paired[["sample_id", "s_x", "s_y"]], check_exact=True)
    edges = pd.read_csv(edge_path)
    w_binary = np.zeros((200, 200), dtype=float)
    for row in edges.itertuples(index=False):
        u, v = int(row.u) - 1, int(row.v) - 1
        w_binary[u, v] = w_binary[v, u] = 1.0
    degree = w_binary.sum(axis=1)
    assert len(edges) == 583 and np.all(degree > 0)
    w = w_binary / degree[:, None]
    xy = base[["s_x", "s_y"]].to_numpy(float)
    edge_lengths = np.linalg.norm(xy[edges.u.to_numpy() - 1] - xy[edges.v.to_numpy() - 1], axis=1)
    median_edge = float(np.median(edge_lengths))
    rho = 2 * median_edge
    if not np.isclose(median_edge, config["generation"]["spatial_field"]["median_edge_length"], atol=1e-15):
        raise AssertionError("predeclared edge scale mismatch")
    distance = cdist(xy, xy)
    covariance = np.exp(-distance / rho)
    cholesky = np.linalg.cholesky(covariance + 1e-10 * np.eye(200))

    generated_rows: list[pd.DataFrame] = []
    truth_rows: list[dict] = []
    sample_id = base.sample_id.astype(str).to_numpy()
    for replicate_index, seed in enumerate(range(20310001, 20310051), start=1):
        rng = np.random.default_rng(seed)
        spatial_x = standardize(cholesky @ rng.normal(size=200))
        independent_spatial = standardize(cholesky @ rng.normal(size=200))
        m0_x = rng.normal(size=200)
        m0_a = np.abs(rng.normal(size=200))
        m2_eps = rng.normal(size=200)
        m3_eps = rng.normal(scale=0.5, size=200)
        m4_eps = rng.normal(scale=0.5, size=200)
        lag_spatial_x = w @ spatial_x
        lag_spatial_x_std = standardize(lag_spatial_x)

        sigma_m2 = np.exp(0.6 * spatial_x)
        data = {
            "M0_IID_NULL": (m0_x, m0_a, w @ m0_x, None),
            "M1_SPATIALLY_STRUCTURED_NULL": (spatial_x, np.abs(independent_spatial), lag_spatial_x, None),
            "M2_SAME_LOCATION_RESIDUAL_SCALE": (spatial_x, sigma_m2 * np.abs(m2_eps), lag_spatial_x, sigma_m2),
            "M3_SPATIAL_LAG_CAUSAL": (spatial_x, lag_spatial_x_std + m3_eps, lag_spatial_x, None),
            "M4_DIRECT_SAME_LOCATION_MEAN": (spatial_x, spatial_x + m4_eps, lag_spatial_x, None),
        }
        for mechanism in MECHANISMS:
            x, a, lag_x, sigma = data[mechanism]
            dataset_id = f"R{replicate_index:02d}_{mechanism}"
            generated_rows.append(pd.DataFrame({
                "dataset_id": dataset_id,
                "mechanism": mechanism,
                "replicate_index": replicate_index,
                "generation_seed": seed,
                "sample_id": sample_id,
                "s_x": xy[:, 0],
                "s_y": xy[:, 1],
                "X": x,
                "A": a,
                "sigma": np.nan if sigma is None else sigma,
                "spatial_lag_X": lag_x,
            }))
            truth_rows.append(truth_row(mechanism, replicate_index, seed, x, a, lag_x, w, sigma))
    controlled = pd.concat(generated_rows, ignore_index=True)
    truth = pd.DataFrame(truth_rows)
    assert len(controlled) == 50 * 5 * 200 and len(truth) == 50 * 5
    controlled.to_csv(RECORD / "controlled_mechanisms.csv", index=False, lineterminator="\n", float_format="%.17g")
    truth.to_csv(RECORD / "controlled_truth_diagnostics.csv", index=False, lineterminator="\n", float_format="%.17g")

    commands = []
    command, _ = run_r("controlled", "controlled_run_log.txt", "controlled_stderr.txt")
    commands.append(command)

    # Only after the controlled benchmark is complete, inspect existing observed Stage-10B tables.
    observed_rows = []
    for scenario, table in (("S2", base), ("S4", paired)):
        a_model = table.A_model.to_numpy(float)
        lag_a = w @ a_model
        for variable in ("x1", "x2", "g_proxy", "x_noise", "A_model"):
            values = table[variable].to_numpy(float)
            lag_values = w @ values
            observed_rows.append({
                "scenario": scenario,
                "variable": variable,
                "variance_ddof1": float(np.var(values, ddof=1)),
                "moran_I": moran(values, w),
                "pearson_with_s_x": pearson(values, xy[:, 0]),
                "pearson_with_s_y": pearson(values, xy[:, 1]),
                "pearson_with_A_model": pearson(values, a_model),
                "pearson_with_spatial_lag_A_model": pearson(values, lag_a),
                "pearson_spatial_lag_variable_with_A_model": pearson(lag_values, a_model),
            })
    pd.DataFrame(observed_rows).to_csv(RECORD / "observed_spatial_diagnostics.csv", index=False,
                                       lineterminator="\n", float_format="%.17g")

    command, _ = run_r("permutation", "permutation_run_log.txt", "permutation_stderr.txt")
    commands.append(command)
    combined_log = []
    for name in ("controlled_run_log.txt", "permutation_run_log.txt"):
        combined_log.append((RECORD / name).read_text(encoding="utf-8"))
    (RECORD / "run_log.txt").write_text("\n".join(combined_log), encoding="utf-8")
    combined_stderr = (RECORD / "controlled_stderr.txt").read_text(encoding="utf-8") + \
        (RECORD / "permutation_stderr.txt").read_text(encoding="utf-8")
    (RECORD / "stderr.txt").write_text(combined_stderr, encoding="utf-8")

    manifest = {
        "fixed_graph": {"path": rel(edge_path), "sha256": sha256(edge_path)},
        "frozen_coordinates": {"path": rel(s2_path), "sha256": sha256(s2_path), "columns_used_for_generation": ["sample_id", "s_x", "s_y"]},
        "observed_comparison_tables": [
            {"path": rel(path), "sha256": sha256(path), "use": "post-controlled-benchmark descriptive diagnostics only"}
            for path in (s2_path, s4_path)
        ],
        "stage10b2_semantics": {
            "path": rel(STAGE10B2 / "gcmc_output_semantics.json"),
            "sha256": sha256(STAGE10B2 / "gcmc_output_semantics.json")
        },
        "source_code": [
            {"path": rel(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            {"path": rel(R_RUNNER), "sha256": sha256(R_RUNNER)},
            {"path": rel(RECORD / "config.json"), "sha256": sha256(RECORD / "config.json")},
        ],
        "generated_data": {
            "controlled_mechanisms.csv": sha256(RECORD / "controlled_mechanisms.csv"),
            "controlled_truth_diagnostics.csv": sha256(RECORD / "controlled_truth_diagnostics.csv"),
        },
        "commands": [subprocess.list2cmdline(command) for command in commands],
        "label_access": {
            "synthetic_generation": "coordinates/graph only; no S2/S4 y or A_model",
            "observed_post_benchmark": "existing Stage-10B scale_fit A_model only",
            "not_read": ["train labels", "validation labels", "tuning_calibration labels", "final_calibration labels", "test labels"]
        }
    }
    (RECORD / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("STAGE10B3_EXECUTION_PASS")


if __name__ == "__main__":
    main()
