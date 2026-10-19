"""Generate Stage-17 controls and orchestrate the preregistered estimators."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RECORD = ROOT / "experiments/causal_strength_benchmark/20260918_stage17_causal_estimator_validation_v1"
GRAPH = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b_gcmc_one_seed_smoke_v1/inputs/delaunay_edges.csv"
COORDS = ROOT / "experiments/causal_strength_diagnostic/20260917_stage10b_gcmc_one_seed_smoke_v1/inputs/S2_scale_fit_gcmc.csv"
CMI_PYTHON = ROOT / "causal_strength_diagnostic_v1/.venv_cmi_v1/Scripts/python.exe"
CMI_WORKER = HERE / "stage17_cmi_worker_v1.py"
R_RUNNER = HERE / "run_stage17_spedm_v1.R"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")
RLIB = ROOT / "environment/gcmc_spedm_1.12_r4.6.1/library"
MECHANISMS = ["C0_INDEPENDENT_NULL", "C1_RESIDUAL_SCALE_CAUSAL", "C2_PURE_CONFOUNDING", "C3_REVERSE_DIRECTION", "C4_SPATIAL_LAG_CAUSAL", "C5_NONLINEAR_DIRECT"]
FEATURES = ["x1", "x2", "g_proxy", "x_noise", "Z"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=1)


def run_command(command: list[str], stdout_path: Path, stderr_path: Path, timeout: int) -> dict:
    started = time.perf_counter()
    env = dict(os.environ)
    env["R_LIBS_USER"] = str(RLIB).replace("\\", "/")
    env["LC_ALL"] = "C"
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
    stdout_path.write_text("COMMAND: " + subprocess.list2cmdline(command) + "\n" + process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"command failed {process.returncode}: {subprocess.list2cmdline(command)}\n{process.stderr[-4000:]}")
    return {"command": subprocess.list2cmdline(command), "returncode": process.returncode, "runtime_seconds": time.perf_counter() - started}


def main() -> None:
    started = time.perf_counter()
    seal = json.loads((RECORD / "PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    for name, expected in seal["files"].items():
        if sha256(RECORD / name) != expected:
            raise AssertionError(f"preregistration hash changed: {name}")
    seeds = json.loads((RECORD / "seed_manifest.json").read_text(encoding="utf-8"))["seeds"]
    if seeds != list(range(20420001, 20420051)):
        raise AssertionError("seed manifest mismatch")
    if (RECORD / "controlled_inputs.csv").exists():
        raise FileExistsError("refusing to overwrite controlled scientific inputs")

    coords = pd.read_csv(COORDS, float_precision="round_trip")[["sample_id", "s_x", "s_y"]]
    edges = pd.read_csv(GRAPH)
    if len(coords) != 200 or len(edges) != 583 or sha256(GRAPH) != "7155d18b4f4263d373eca061707165630c9977f3e67b4786d9e6e5556e8d2159":
        raise AssertionError("fixed support mismatch")
    xy = coords[["s_x", "s_y"]].to_numpy(float)
    w_binary = np.zeros((200, 200), dtype=float)
    for row in edges.itertuples(index=False):
        u, v = int(row.u) - 1, int(row.v) - 1
        w_binary[u, v] = w_binary[v, u] = 1.0
    degree = w_binary.sum(axis=1)
    w = w_binary / degree[:, None]
    edge_lengths = np.linalg.norm(xy[edges.u.to_numpy() - 1] - xy[edges.v.to_numpy() - 1], axis=1)
    rho = 2 * float(np.median(edge_lengths))
    covariance = np.exp(-cdist(xy, xy) / rho)
    cholesky = np.linalg.cholesky(covariance + 1e-10 * np.eye(200))
    coord_trend = standardize(xy[:, 0] + xy[:, 1])

    frames: list[pd.DataFrame] = []
    truth_rows: list[dict] = []
    for replicate_index, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        fields = [standardize(cholesky @ rng.normal(size=200)) for _ in range(8)]
        base_x2, g_spatial, z_nuisance, conf_z, reverse_driver, lag_x2, nonlinear_x2, extra = fields
        iid_x1 = rng.normal(size=200)
        iid_noise = rng.normal(size=200)
        eps = [rng.normal(size=200) for _ in range(6)]
        mechanisms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = {}

        x2 = base_x2
        gp = standardize(0.75 * x2 + 0.50 * g_spatial + 0.25 * coord_trend)
        mechanisms[MECHANISMS[0]] = (iid_x1, x2, gp, iid_noise, z_nuisance, np.abs(eps[0]), "none")

        x2 = base_x2
        gp = standardize(0.75 * x2 + 0.50 * g_spatial + 0.25 * coord_trend)
        a = np.exp(0.6 * x2) * np.abs(eps[1])
        mechanisms[MECHANISMS[1]] = (iid_x1, x2, gp, iid_noise, z_nuisance, a, "x2")

        z = conf_z
        x2 = standardize(0.8 * z + 0.6 * rng.normal(size=200))
        gp = standardize(0.75 * x2 + 0.50 * g_spatial + 0.25 * coord_trend)
        a = np.exp(0.6 * z) * np.abs(eps[2])
        mechanisms[MECHANISMS[2]] = (iid_x1, x2, gp, iid_noise, z, a, "Z")

        a_driver = reverse_driver
        a = np.exp(0.5 * a_driver)
        x2 = standardize(a_driver + 0.5 * rng.normal(size=200))
        gp = standardize(0.75 * x2 + 0.50 * g_spatial + 0.25 * coord_trend)
        mechanisms[MECHANISMS[3]] = (iid_x1, x2, gp, iid_noise, z_nuisance, a, "A_to_x2")

        x2 = lag_x2
        lag = standardize(w @ x2)
        gp = standardize(0.75 * x2 + 0.50 * g_spatial + 0.25 * coord_trend)
        a = np.exp(0.6 * lag) * np.abs(eps[4])
        mechanisms[MECHANISMS[4]] = (iid_x1, x2, gp, iid_noise, z_nuisance, a, "lag_x2")

        x2 = nonlinear_x2
        gp = standardize(0.75 * x2 + 0.50 * g_spatial + 0.25 * coord_trend)
        a = x2 ** 2 + 0.5 * np.abs(eps[5])
        mechanisms[MECHANISMS[5]] = (iid_x1, x2, gp, iid_noise, z_nuisance, a, "x2")

        for mechanism_index, mechanism in enumerate(MECHANISMS):
            x1, x2, gp, xn, z, a, direct = mechanisms[mechanism]
            dataset_id = f"{mechanism}_R{replicate_index:02d}"
            frame = pd.DataFrame({
                "dataset_id": dataset_id, "mechanism": mechanism, "replicate_index": replicate_index,
                "generation_seed": seed, "row_index": np.arange(1, 201), "sample_id": coords.sample_id.astype(str),
                "s_x": xy[:, 0], "s_y": xy[:, 1], "x1": x1, "x2": x2,
                "g_proxy": gp, "x_noise": xn, "Z": z, "A": a,
            })
            frames.append(frame)
            for feature in FEATURES:
                direct_cause = (direct == feature) or (direct == "lag_x2" and feature == "x2")
                truth_rows.append({
                    "dataset_id": dataset_id, "mechanism": mechanism, "replicate_index": replicate_index,
                    "generation_seed": seed, "feature": feature, "direct_forward_cause": direct_cause,
                    "reverse_direction_truth": mechanism == "C3_REVERSE_DIRECTION" and feature == "x2",
                    "confounded_noncausal": mechanism == "C2_PURE_CONFOUNDING" and feature == "x2",
                    "proxy_noncausal": feature == "g_proxy", "truth_label": direct,
                    "pearson_feature_A": float(np.corrcoef(frame[feature], a)[0, 1]),
                })
    controlled = pd.concat(frames, ignore_index=True)
    truth = pd.DataFrame(truth_rows)
    if len(controlled) != 6 * 50 * 200 or len(truth) != 6 * 50 * 5:
        raise AssertionError("generated row count mismatch")
    controlled.to_csv(RECORD / "controlled_inputs.csv", index=False, lineterminator="\n", float_format="%.17g")
    truth.to_csv(RECORD / "replicate_truth.csv", index=False, lineterminator="\n", float_format="%.17g")

    execution_dir = RECORD / "execution_parts"
    execution_dir.mkdir()
    commands: list[dict] = []
    cmi_jobs = []
    for mechanism in MECHANISMS:
        output = execution_dir / f"cmi_{mechanism}.csv"
        command = [str(CMI_PYTHON), str(CMI_WORKER), "--input", str(RECORD / "controlled_inputs.csv"), "--mechanism", mechanism, "--output", str(output)]
        cmi_jobs.append((command, execution_dir / f"cmi_{mechanism}.stdout.txt", execution_dir / f"cmi_{mechanism}.stderr.txt"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_command, *job, 21600) for job in cmi_jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result(); commands.append(result); print("CMI job completed", result["runtime_seconds"], flush=True)
    cmi = pd.concat([pd.read_csv(execution_dir / f"cmi_{m}.csv") for m in MECHANISMS], ignore_index=True)
    cmi.loc[cmi.method.eq("CMI_MSPACE")].to_csv(RECORD / "cmi_mspace_results.csv", index=False, lineterminator="\n", float_format="%.17g")
    cmi.loc[cmi.method.eq("CMI_FULL")].to_csv(RECORD / "cmi_full_results.csv", index=False, lineterminator="\n", float_format="%.17g")

    r_jobs = []
    for mechanism in MECHANISMS:
        output = execution_dir / f"spedm_{mechanism}.csv"
        command = [str(RSCRIPT), "--vanilla", str(R_RUNNER), str(RECORD / "controlled_inputs.csv"), str(GRAPH), str(output), mechanism, str(RLIB)]
        r_jobs.append((command, execution_dir / f"spedm_{mechanism}.stdout.txt", execution_dir / f"spedm_{mechanism}.stderr.txt"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_command, *job, 21600) for job in r_jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result(); commands.append(result); print("spEDM job completed", result["runtime_seconds"], flush=True)
    spedm = pd.concat([pd.read_csv(execution_dir / f"spedm_{m}.csv") for m in MECHANISMS], ignore_index=True)
    for method, filename in (("GPC", "gpc_results.csv"), ("SCT", "sct_results.csv"), ("SCPCM", "scpcm_results.csv")):
        spedm.loc[spedm.method.eq(method)].to_csv(RECORD / filename, index=False, lineterminator="\n", float_format="%.17g")

    (RECORD / "execution_commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    combined_stdout = []
    combined_stderr = []
    for path in sorted(execution_dir.glob("*.stdout.txt")):
        combined_stdout.append(f"===== {path.name} =====\n" + path.read_text(encoding="utf-8"))
    for path in sorted(execution_dir.glob("*.stderr.txt")):
        text = path.read_text(encoding="utf-8")
        if text.strip(): combined_stderr.append(f"===== {path.name} =====\n{text}")
    with (RECORD / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(combined_stdout))
        handle.write(f"\nScientific execution runtime seconds: {time.perf_counter()-started:.3f}\n")
    (RECORD / "stderr.txt").write_text("\n".join(combined_stderr), encoding="utf-8")
    checkpoint = {
        "status": "SCIENTIFIC_EXECUTION_COMPLETE", "controlled_sha256": sha256(RECORD / "controlled_inputs.csv"),
        "truth_sha256": sha256(RECORD / "replicate_truth.csv"), "commands": commands,
        "row_counts": {"controlled": len(controlled), "truth": len(truth), "cmi": len(cmi), "spedm": len(spedm)},
    }
    (RECORD / "execution_checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checkpoint, indent=2))


if __name__ == "__main__":
    main()
