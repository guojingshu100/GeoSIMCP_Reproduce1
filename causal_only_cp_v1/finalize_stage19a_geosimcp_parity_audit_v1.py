from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments" / "geo_causal_cp" / "20260919_stage19a_geosimcp_parity_audit_v1"
SOURCE = ROOT / "external" / "geoconformal_stage19a_audit_26419ef"
OLD = ROOT / "geoconformal-main"
EXPECTED = "26419ef06d7e3711e6618f97f4f1ed02af16bcb7"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def git(*args: str) -> str:
    cmd = ["git", "-c", f"safe.directory={SOURCE.as_posix()}", "-C", str(SOURCE), *args]
    return subprocess.check_output(cmd, text=True).strip()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    head = git("rev-parse", "HEAD")
    if head != EXPECTED:
        raise AssertionError((head, EXPECTED))
    if git("branch", "--show-current") != "main":
        raise AssertionError("not main")
    if git("status", "--porcelain"):
        raise AssertionError("audited checkout is dirty")

    required_rel = [
        "README.md", "CHANGELOG.md", "setup.py",
        "geoconformal/GeoConformalSpatialRegression.py",
        "geoconformal/GeoSIMConformalSpatialRegression.py",
        "geoconformal/geocp/weights.py", "geoconformal/geocp/utils.py",
        "geoconformal/geocp/core.py", "tests/test_unified.py",
        "tests/test_atom.py", "tests/test_variants.py",
        "Codes_Manuscript_GeoSIMCP/Codes_GeoSIMCP/GeoConformal/GeoConformalSpatialRegression.py",
        "Codes_Manuscript_GeoSIMCP/Codes_GeoSIMCP/GeoConformal/GeoSIMCP.py",
        "Codes_Manuscript_GeoSIMCP/Codes_GeoSIMCP/GeoConformal/GeoSIMCP_non_zero.py",
        "Codes_Manuscript_GeoSIMCP/Codes_GeoSIMCP/GeoConformal/GeoSIMCPzhao.py",
    ]
    required = [SOURCE / rel for rel in required_rel]
    notebooks = sorted((SOURCE / "Codes_Manuscript_GeoSIMCP").rglob("*.ipynb"))
    project_inputs = [
        ROOT / "experiments/causal_only_cp/20260918_stage11_causalonly_cmi_normcp_v2/METHOD.md",
        ROOT / "experiments/causal_only_cp/20260918_stage11_causalonly_cmi_normcp_v2/config.json",
        ROOT / "experiments/causal_only_cp/20260919_stage18_cmi_full_cp_translation_v1/PREREGISTRATION.md",
        ROOT / "experiments/causal_only_cp/20260919_stage18_cmi_full_cp_translation_v1/config.json",
        ROOT / "experiments/causal_only_cp/20260919_stage18_cmi_full_cp_translation_v1/stage18_summary.json",
    ]
    for path in required + notebooks + project_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    core_compare = []
    for rel in required_rel[:11]:
        current = SOURCE / rel
        prior = OLD / rel
        core_compare.append({
            "path": rel,
            "current_sha256": sha(current),
            "prior_snapshot_exists": prior.is_file(),
            "prior_snapshot_sha256": sha(prior) if prior.is_file() else None,
            "match": prior.is_file() and sha(prior) == sha(current),
        })
    if any(item["match"] for item in core_compare):
        # This is informational, not an error. Preserve the branch for future
        # snapshots while recording the present observed count below.
        pass

    input_manifest = {
        "stage": "19A",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_computation": False,
        "repository": {
            "url": "https://github.com/pengtum/geoconformal",
            "git_url": "https://github.com/pengtum/geoconformal.git",
            "default_branch": "main",
            "commit": head,
            "commit_author_date": git("log", "-1", "--format=%aI"),
            "commit_committer_date": git("log", "-1", "--format=%cI"),
            "commit_subject": git("log", "-1", "--format=%s"),
            "checkout": SOURCE.relative_to(ROOT).as_posix(),
            "checkout_clean": True,
            "package_version": "0.3.0",
        },
        "official_source_files": [describe(p) for p in required],
        "manuscript_notebooks": [describe(p) for p in notebooks],
        "project_method_inputs": [describe(p) for p in project_inputs],
        "prior_non_git_snapshot_comparison": {
            "path": OLD.relative_to(ROOT).as_posix(),
            "files_compared": len(core_compare),
            "matches": sum(int(x["match"]) for x in core_compare),
            "details": core_compare,
        },
        "network_actions": [
            "git ls-remote of official GitHub repository",
            "shallow clone of main at the audited HEAD",
        ],
        "prohibitions_observed": [
            "no new seeds", "no scientific simulation", "no CP execution",
            "no CP performance comparison", "no Stage-19B execution",
        ],
    }
    dump(RECORD / "input_manifest.json", input_manifest)

    expected_outputs = [
        "README.md", "OFFICIAL_REPOSITORY_PROVENANCE.md",
        "OFFICIAL_GEOCP_ALGORITHM.md", "OFFICIAL_GEOSIMCP_ALGORITHM.md",
        "FINITE_SAMPLE_UPDATE_AUDIT.md", "HYPERPARAMETER_TUNING_AUDIT.md",
        "STAGE11_18_SHELL_DIFFERENCE.md", "MINIMAL_CHANGE_GEOCAUSALCP_SPEC.md",
        "COMPONENT_DIFFERENCE_MATRIX.csv", "STAGE19B_FAIR_COMPARISON_PLAN.md",
        "input_manifest.json", "run_log.txt", "stderr.txt",
    ]
    missing = [name for name in expected_outputs if not (RECORD / name).is_file()]
    if missing:
        raise AssertionError(f"missing outputs: {missing}")
    if (RECORD / "stderr.txt").stat().st_size != 0:
        raise AssertionError("stderr is not empty")
    entries = [describe(RECORD / name) for name in expected_outputs]
    dump(RECORD / "output_manifest.json", {
        "stage": "19A", "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_computation": False,
        "audited_commit": head,
        "files": entries,
    })
    print(json.dumps({
        "status": "COMPLETE", "audited_commit": head,
        "official_source_files": len(required),
        "manuscript_notebooks": len(notebooks),
        "required_outputs": len(expected_outputs) + 1,
        "stderr_bytes": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
