"""Generate and validate the frozen paired S1--S4 GCCP simulation data."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


GENERATOR_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = GENERATOR_DIR / "config_v1.json"
DEFAULT_OUTPUT_ROOT = GENERATOR_DIR.parent
SCENARIOS = ("S1", "S2", "S3", "S4")
ORACLE_ONLY_COLUMNS = {"mu_true", "sigma_true", "epsilon", "g_true", "eta"}


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    values = np.asarray(z, dtype=float)
    result = np.empty_like(values)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def scale_function(z: np.ndarray | float) -> np.ndarray:
    """Frozen positive heteroscedastic scale h(z)."""
    return 0.6 + 1.4 * sigmoid(1.2 * np.asarray(z, dtype=float))


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def create_base_random_variables(n: int = 2000, seed: int = 20260907) -> pd.DataFrame:
    """Generate all common random variables exactly once in the frozen order."""
    rng = np.random.default_rng(seed)
    s_x = rng.uniform(0.0, 1.0, size=n)
    s_y = rng.uniform(0.0, 1.0, size=n)
    x1 = rng.standard_normal(size=n)
    x2 = rng.standard_normal(size=n)
    eta = rng.standard_normal(size=n)
    x_noise = rng.standard_normal(size=n)
    epsilon = rng.standard_normal(size=n)
    g_true = np.sin(2.0 * np.pi * s_x) + np.cos(2.0 * np.pi * s_y)
    g_proxy = g_true + 0.35 * eta

    return pd.DataFrame(
        {
            "sample_id": [f"sample_{i:04d}" for i in range(n)],
            "s_x": s_x,
            "s_y": s_y,
            "x1": x1,
            "x2": x2,
            "eta": eta,
            "x_noise": x_noise,
            "epsilon": epsilon,
            "g_true": g_true,
            "g_proxy": g_proxy,
        }
    )


def create_frozen_split(
    sample_ids: pd.Series | np.ndarray | list[str],
    split_sizes: dict[str, int],
    internal_role_sizes: dict[str, int],
    split_seed: int = 620907,
    internal_seed: int = 620908,
) -> pd.DataFrame:
    """Create the common outer split and local-calibration internal roles."""
    sample_ids_array = np.asarray(sample_ids, dtype=str)
    n = len(sample_ids_array)
    if sum(split_sizes.values()) != n:
        raise ValueError("Outer split sizes do not sum to n.")
    if sum(internal_role_sizes.values()) != split_sizes["local_calibration"]:
        raise ValueError("Internal role sizes do not sum to local_calibration n.")

    outer_rng = np.random.default_rng(split_seed)
    permutation = outer_rng.permutation(n)
    outer_split = np.empty(n, dtype=object)
    cursor = 0
    for split_name in ("train", "local_calibration", "validation", "final_calibration", "test"):
        size = int(split_sizes[split_name])
        outer_split[permutation[cursor : cursor + size]] = split_name
        cursor += size

    local_indices = np.flatnonzero(outer_split == "local_calibration")
    internal_rng = np.random.default_rng(internal_seed)
    internal_permutation = internal_rng.permutation(local_indices)
    local_role = np.full(n, "", dtype=object)
    cursor = 0
    for role_name in ("scale_fit", "tuning_calibration"):
        size = int(internal_role_sizes[role_name])
        chosen = internal_permutation[cursor : cursor + size]
        local_role[chosen] = role_name
        cursor += size

    return pd.DataFrame(
        {
            "sample_id": sample_ids_array,
            "split": outer_split,
            "local_calibration_role": local_role,
        }
    )


def generate_scenario(
    scenario: str,
    base: pd.DataFrame,
    split_assignment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one scenario while preserving all common variables and roles."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if not np.array_equal(base["sample_id"].to_numpy(), split_assignment["sample_id"].to_numpy()):
        raise ValueError("Base variables and split assignment are not aligned.")

    x1 = base["x1"].to_numpy(dtype=float)
    x2 = base["x2"].to_numpy(dtype=float)
    epsilon = base["epsilon"].to_numpy(dtype=float)
    g_true = base["g_true"].to_numpy(dtype=float)

    if scenario == "S1":
        mu_true = 2.0 * x1
        sigma_true = scale_function(x1)
    elif scenario == "S2":
        mu_true = 2.0 * x1
        sigma_true = scale_function(x2)
    elif scenario == "S3":
        mu_true = 2.0 * x1 + 1.5 * g_true
        sigma_true = scale_function(g_true)
    else:
        mu_true = 2.0 * x1
        sigma_true = np.full(len(base), 1.3, dtype=float)

    y = mu_true + sigma_true * epsilon
    shared_keys = {
        "sample_id": base["sample_id"].to_numpy(),
        "scenario": np.full(len(base), scenario, dtype=object),
        "split": split_assignment["split"].to_numpy(),
        "local_calibration_role": split_assignment["local_calibration_role"].to_numpy(),
    }
    observed = pd.DataFrame(
        {
            **shared_keys,
            "s_x": base["s_x"].to_numpy(),
            "s_y": base["s_y"].to_numpy(),
            "x1": x1,
            "x2": x2,
            "g_proxy": base["g_proxy"].to_numpy(),
            "x_noise": base["x_noise"].to_numpy(),
            "y": y,
        }
    )
    truth = pd.DataFrame(
        {
            **shared_keys,
            "mu_true": mu_true,
            "sigma_true": sigma_true,
            "epsilon": epsilon,
            "g_true": g_true,
            "eta": base["eta"].to_numpy(),
        }
    )
    return observed, truth


def generate_all_scenarios(config: dict[str, Any]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    seeds = config["random_seeds"]
    base = create_base_random_variables(config["n"], seeds["base_variables"])
    split_assignment = create_frozen_split(
        base["sample_id"],
        config["split_sizes"],
        config["local_calibration_internal_roles"],
        seeds["outer_split"],
        seeds["local_calibration_role"],
    )
    return {scenario: generate_scenario(scenario, base, split_assignment) for scenario in SCENARIOS}


def write_outputs(
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_root: Path | str,
    overwrite: bool = False,
) -> list[Path]:
    """Write deterministic CSV files beneath output_root/data."""
    data_dir = Path(output_root).resolve() / "data"
    paths = [data_dir / f"{scenario}_{kind}.csv" for scenario in SCENARIOS for kind in ("observed", "truth")]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(str(path) for path in existing))
    data_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        observed, truth = scenarios[scenario]
        observed.to_csv(
            data_dir / f"{scenario}_observed.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            float_format="%.17g",
        )
        truth.to_csv(
            data_dir / f"{scenario}_truth.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            float_format="%.17g",
        )
    return paths


def read_outputs(output_root: Path | str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    data_dir = Path(output_root).resolve() / "data"
    result: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for scenario in SCENARIOS:
        observed = pd.read_csv(data_dir / f"{scenario}_observed.csv", keep_default_na=False)
        truth = pd.read_csv(data_dir / f"{scenario}_truth.csv", keep_default_na=False)
        result[scenario] = (observed, truth)
    return result


def _assert_allclose(actual: Any, expected: Any, label: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=1e-12,
        atol=1e-12,
        err_msg=label,
    )


def validate_scenario(
    scenario: str,
    observed: pd.DataFrame,
    truth: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate one scenario from the saved observed and truth tables."""
    n = int(config["n"])
    assert list(observed.columns) == config["observed_columns"]
    assert list(truth.columns) == config["truth_columns"]
    assert len(observed) == n and len(truth) == n
    assert observed["sample_id"].is_unique and truth["sample_id"].is_unique

    key_columns = ["sample_id", "scenario", "split", "local_calibration_role"]
    pd.testing.assert_frame_equal(observed[key_columns], truth[key_columns], check_dtype=False)
    assert set(observed["scenario"]) == {scenario}

    observed_numeric = observed.drop(columns=key_columns).to_numpy(dtype=float)
    truth_numeric = truth.drop(columns=key_columns).to_numpy(dtype=float)
    assert np.isfinite(observed_numeric).all()
    assert np.isfinite(truth_numeric).all()
    assert (truth["sigma_true"].to_numpy(dtype=float) > 0.0).all()

    x1 = observed["x1"].to_numpy(dtype=float)
    x2 = observed["x2"].to_numpy(dtype=float)
    y = observed["y"].to_numpy(dtype=float)
    mu_true = truth["mu_true"].to_numpy(dtype=float)
    sigma_true = truth["sigma_true"].to_numpy(dtype=float)
    epsilon = truth["epsilon"].to_numpy(dtype=float)
    g_true = truth["g_true"].to_numpy(dtype=float)
    eta = truth["eta"].to_numpy(dtype=float)

    _assert_allclose(observed["g_proxy"], g_true + 0.35 * eta, f"{scenario} g_proxy equation")
    if scenario == "S1":
        _assert_allclose(mu_true, 2.0 * x1, "S1 mu equation")
        _assert_allclose(sigma_true, scale_function(x1), "S1 sigma equation")
    elif scenario == "S2":
        _assert_allclose(mu_true, 2.0 * x1, "S2 mu equation")
        _assert_allclose(sigma_true, scale_function(x2), "S2 sigma equation")
    elif scenario == "S3":
        _assert_allclose(mu_true, 2.0 * x1 + 1.5 * g_true, "S3 mu equation")
        _assert_allclose(sigma_true, scale_function(g_true), "S3 sigma equation")
    elif scenario == "S4":
        _assert_allclose(mu_true, 2.0 * x1, "S4 mu equation")
        _assert_allclose(sigma_true, np.full(n, 1.3), "S4 sigma equation")
    _assert_allclose(y, mu_true + sigma_true * epsilon, f"{scenario} y equation")

    return {
        "rows_observed": len(observed),
        "rows_truth": len(truth),
        "sample_id_unique": True,
        "all_numeric_finite": True,
        "observed_truth_alignment": True,
        "equations_verified_from_saved_files": True,
        "sigma_true_min": float(sigma_true.min()),
        "sigma_true_max": float(sigma_true.max()),
    }


def validate_split_contract(
    saved: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: dict[str, Any],
) -> dict[str, Any]:
    reference = saved["S1"][0]
    expected_counts = {key: int(value) for key, value in config["split_sizes"].items()}
    actual_counts = {key: int(value) for key, value in reference["split"].value_counts().to_dict().items()}
    assert actual_counts == expected_counts
    assert reference["sample_id"].nunique() == config["n"]
    assert len(reference) == config["n"]

    local = reference[reference["split"] == "local_calibration"]
    role_counts = {key: int(value) for key, value in local["local_calibration_role"].value_counts().to_dict().items()}
    expected_roles = {key: int(value) for key, value in config["local_calibration_internal_roles"].items()}
    assert role_counts == expected_roles
    assert (reference.loc[reference["split"] != "local_calibration", "local_calibration_role"] == "").all()

    reference_assignment = reference[["sample_id", "split", "local_calibration_role"]]
    for scenario in SCENARIOS[1:]:
        pd.testing.assert_frame_equal(
            reference_assignment,
            saved[scenario][0][["sample_id", "split", "local_calibration_role"]],
            check_dtype=False,
        )
    return {
        "split_counts": actual_counts,
        "local_calibration_role_counts": role_counts,
        "all_sample_ids_covered_once": True,
        "assignment_identical_across_scenarios": True,
        "non_local_roles_empty": True,
    }


def validate_paired_design(saved: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> dict[str, Any]:
    observed_common = [
        "sample_id",
        "split",
        "local_calibration_role",
        "s_x",
        "s_y",
        "x1",
        "x2",
        "g_proxy",
        "x_noise",
    ]
    truth_common = ["sample_id", "split", "local_calibration_role", "epsilon", "g_true", "eta"]
    ref_observed, ref_truth = saved["S1"]
    for scenario in SCENARIOS[1:]:
        observed, truth = saved[scenario]
        pd.testing.assert_frame_equal(ref_observed[observed_common], observed[observed_common], check_dtype=False)
        pd.testing.assert_frame_equal(ref_truth[truth_common], truth[truth_common], check_dtype=False)

    truths = {scenario: saved[scenario][1] for scenario in SCENARIOS}
    _assert_allclose(truths["S1"]["mu_true"], truths["S4"]["mu_true"], "S1/S4 mu equality")
    _assert_allclose(truths["S2"]["mu_true"], truths["S4"]["mu_true"], "S2/S4 mu equality")
    _assert_allclose(truths["S4"]["sigma_true"], np.full(len(ref_truth), 1.3), "S4 constant sigma")
    y_vectors = {scenario: saved[scenario][0]["y"].to_numpy(dtype=float) for scenario in SCENARIOS}
    distinct_pairs: dict[str, bool] = {}
    for left_index, left in enumerate(SCENARIOS):
        for right in SCENARIOS[left_index + 1 :]:
            key = f"{left}_vs_{right}"
            distinct_pairs[key] = not np.array_equal(y_vectors[left], y_vectors[right])
            assert distinct_pairs[key]
    return {
        "common_variables_identical": True,
        "split_and_roles_identical": True,
        "S1_S4_mu_identical": True,
        "S2_S4_mu_identical": True,
        "scenario_equations_verified_by_numerical_checks": True,
        "all_y_vectors_pairwise_distinct": all(distinct_pairs.values()),
        "y_pair_checks": distinct_pairs,
    }


def validate_leakage(saved: dict[str, tuple[pd.DataFrame, pd.DataFrame]], config: dict[str, Any]) -> dict[str, Any]:
    per_scenario: dict[str, Any] = {}
    for scenario in SCENARIOS:
        observed = saved[scenario][0]
        leaked = sorted(ORACLE_ONLY_COLUMNS.intersection(observed.columns))
        assert not leaked
        assert list(observed.columns) == config["observed_columns"]
        per_scenario[scenario] = {"oracle_columns_found": leaked, "status": "PASS"}
    return {"oracle_only_columns": sorted(ORACLE_ONLY_COLUMNS), "per_scenario": per_scenario}


def runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for module_name, output_name in (("scipy", "scipy"), ("sklearn", "sklearn")):
        try:
            module = importlib.import_module(module_name)
            versions[output_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[output_name] = None
    return versions


def validate_configuration(config: dict[str, Any]) -> dict[str, Any]:
    assert config["generator_version"] == "gccp_sim_s1_s4_v1"
    assert config["n"] == 2000 and config["alpha"] == 0.10
    assert config["random_seeds"] == {
        "base_variables": 20260907,
        "outer_split": 620907,
        "local_calibration_role": 620908,
    }
    required = {"generator_version", "generation_timestamp", "n", "alpha", "random_seeds", "equations", "split_sizes", "observed_columns", "truth_columns", "local_calibration_internal_roles", "environment"}
    assert required.issubset(config)
    return {
        "generator_version": config["generator_version"],
        "generation_timestamp": config["generation_timestamp"],
        "recorded_environment": config["environment"],
        "runtime_environment": runtime_versions(),
    }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reproducibility_check(config: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gccp_sim_repro_a_") as first_dir, tempfile.TemporaryDirectory(prefix="gccp_sim_repro_b_") as second_dir:
        first = Path(first_dir)
        second = Path(second_dir)
        write_outputs(generate_all_scenarios(config), first)
        write_outputs(generate_all_scenarios(config), second)
        first_hashes = {path.name: sha256_file(path) for path in sorted((first / "data").glob("*.csv"))}
        second_hashes = {path.name: sha256_file(path) for path in sorted((second / "data").glob("*.csv"))}
        assert first_hashes == second_hashes
    return {
        "separate_temporary_directories": True,
        "corresponding_sha256_identical": True,
        "file_count": len(first_hashes),
        "hashes": first_hashes,
    }


def _run_check(name: str, function: Callable[[], dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    try:
        detail = function()
        return name, {"status": "PASS", **detail}
    except Exception as error:
        return name, {"status": "FAIL", "error_type": type(error).__name__, "message": str(error)}


def run_pipeline(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate final outputs, validate saved files, and write generator_check.json."""
    output_root_path = Path(output_root).resolve()
    config_path_obj = Path(config_path).resolve()
    config = load_config(config_path_obj)
    scenarios = generate_all_scenarios(config)
    output_paths = write_outputs(scenarios, output_root_path, overwrite=overwrite)
    saved = read_outputs(output_root_path)

    checks: dict[str, dict[str, Any]] = {}
    check_calls: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("configuration_record", lambda: validate_configuration(config)),
        (
            "numerical_validity",
            lambda: {scenario: validate_scenario(scenario, *saved[scenario], config) for scenario in SCENARIOS},
        ),
        ("split_validity", lambda: validate_split_contract(saved, config)),
        ("paired_design", lambda: validate_paired_design(saved)),
        ("leakage_prevention", lambda: validate_leakage(saved, config)),
        ("reproducibility", lambda: reproducibility_check(config)),
    ]
    for name, function in check_calls:
        check_name, result = _run_check(name, function)
        checks[check_name] = result

    file_hashes = {
        str(path.relative_to(output_root_path)).replace("\\", "/"): sha256_file(path)
        for path in output_paths
    }
    file_hashes["generator/config_v1.json"] = sha256_file(config_path_obj)
    row_counts = {
        scenario: {"observed": len(saved[scenario][0]), "truth": len(saved[scenario][1])}
        for scenario in SCENARIOS
    }
    split_counts = {
        key: int(value) for key, value in saved["S1"][0]["split"].value_counts().to_dict().items()
    }
    sigma_ranges = {
        scenario: {
            "min": float(saved[scenario][1]["sigma_true"].min()),
            "max": float(saved[scenario][1]["sigma_true"].max()),
        }
        for scenario in SCENARIOS
    }
    overall_status = "PASS" if all(result["status"] == "PASS" for result in checks.values()) else "FAIL"
    report = {
        "generator_version": config["generator_version"],
        "generation_timestamp": config["generation_timestamp"],
        "overall_status": overall_status,
        "checks": checks,
        "files": [str(path.relative_to(output_root_path)).replace("\\", "/") for path in output_paths],
        "row_counts": row_counts,
        "split_counts": split_counts,
        "local_calibration_role_counts": {
            key: int(value)
            for key, value in saved["S1"][0].loc[
                saved["S1"][0]["split"] == "local_calibration", "local_calibration_role"
            ].value_counts().to_dict().items()
        },
        "sigma_true_ranges": sigma_ranges,
        "sha256": file_hashes,
        "reproducibility_result": checks["reproducibility"]["status"],
        "paired_design_result": checks["paired_design"]["status"],
        "leakage_check_result": checks["leakage_prevention"]["status"],
    }
    check_path = output_root_path / "generator_check.json"
    check_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if overall_status != "PASS":
        raise RuntimeError(f"Generator validation failed. See {check_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will contain data/ and generator_check.json.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated CSV/check files.")
    args = parser.parse_args()
    report = run_pipeline(args.output_dir, args.config, overwrite=args.overwrite)
    print(f"Generator version: {report['generator_version']}")
    print(f"Overall status: {report['overall_status']}")
    print("Split counts:", report["split_counts"])
    print("Local roles:", report["local_calibration_role_counts"])
    print("Sigma ranges:", report["sigma_true_ranges"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
