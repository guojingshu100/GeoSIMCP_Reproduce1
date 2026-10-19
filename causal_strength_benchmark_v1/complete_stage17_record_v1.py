"""Write the final Stage-17 scientific interpretation and output manifest."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/causal_strength_benchmark/20260918_stage17_causal_estimator_validation_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get(frame: pd.DataFrame, method: str, mechanism: str) -> pd.Series:
    rows = frame[(frame.method == method) & (frame.mechanism == mechanism)]
    if len(rows) != 1:
        raise AssertionError((method, mechanism, len(rows)))
    return rows.iloc[0]


def main() -> None:
    recovery = pd.read_csv(RECORD / "method_recovery_summary.csv")
    conf = pd.read_csv(RECORD / "confounding_summary.csv")
    direction = pd.read_csv(RECORD / "directionality_summary.csv")
    checkpoint = json.loads((RECORD / "execution_checkpoint.json").read_text(encoding="utf-8"))
    seal = json.loads((RECORD / "PREREGISTRATION_SEAL.json").read_text(encoding="utf-8"))
    for name, expected in seal["files"].items():
        if sha256(RECORD / name) != expected:
            raise AssertionError(f"sealed preregistration mismatch: {name}")
    if checkpoint["row_counts"] != {"controlled": 60000, "truth": 1500, "cmi": 3000, "spedm": 4500}:
        raise AssertionError("execution row counts changed")
    if (RECORD / "stderr.txt").stat().st_size != 0:
        raise AssertionError("final stderr is nonempty")

    expected = {
        ("CMI_MSPACE", "C1_RESIDUAL_SCALE_CAUSAL"): (49, 46),
        ("CMI_FULL", "C1_RESIDUAL_SCALE_CAUSAL"): (50, 39),
        ("GPC", "C1_RESIDUAL_SCALE_CAUSAL"): (17, 32),
        ("SCT", "C1_RESIDUAL_SCALE_CAUSAL"): (1, 0),
        ("SCPCM", "C1_RESIDUAL_SCALE_CAUSAL"): (31, 34),
    }
    for key, counts in expected.items():
        row = get(recovery, *key)
        if (int(row.x2_top1_count), int(row.x2_selected_count)) != counts:
            raise AssertionError(f"unexpected key result {key}")

    def r(method: str, mechanism: str, field: str) -> int | float:
        value = get(recovery, method, mechanism)[field]
        return int(value) if float(value).is_integer() else float(value)

    summary = {
        "stage": "17",
        "execution_status": "PASS",
        "scientific_scope": "causal-strength estimator validation only; no conformal prediction",
        "preregistration_composite_sha256": seal["composite_sha256"],
        "development_seed_block": [20420001, 20420050],
        "replicates_per_mechanism": 50,
        "successful_replicates": {method: {mechanism: 50 for mechanism in recovery.mechanism.unique()} for method in recovery.method.unique()},
        "primary_results": {
            "C1_x2_top1": {m: r(m, "C1_RESIDUAL_SCALE_CAUSAL", "x2_top1_count") for m in recovery.method.unique()},
            "C1_x2_selected": {m: r(m, "C1_RESIDUAL_SCALE_CAUSAL", "x2_selected_count") for m in recovery.method.unique()},
            "C0_no_signal": {m: r(m, "C0_INDEPENDENT_NULL", "no_signal_count") for m in recovery.method.unique()},
            "C1_g_proxy_false_selected": {m: r(m, "C1_RESIDUAL_SCALE_CAUSAL", "g_proxy_selected_count") for m in recovery.method.unique()},
            "C2_x2_false_selected": dict(zip(conf.loc[conf.role.eq("confounded_noncausal_x2"), "method"], conf.loc[conf.role.eq("confounded_noncausal_x2"), "selected_count"].astype(int))),
            "C2_g_proxy_false_selected": dict(zip(conf.loc[conf.role.eq("noncausal_spatial_proxy"), "method"], conf.loc[conf.role.eq("noncausal_spatial_proxy"), "selected_count"].astype(int))),
            "C2_Z_selected": dict(zip(conf.loc[conf.role.eq("observed_direct_driver_Z"), "method"], conf.loc[conf.role.eq("observed_direct_driver_Z"), "selected_count"].astype(int))),
            "C3_x2_selected_symmetric_methods": {m: r(m, "C3_REVERSE_DIRECTION", "x2_selected_count") for m in ("CMI_MSPACE", "CMI_FULL")},
        },
        "score_semantics": {
            "CMI_MSPACE": "raw Tigramite CMIknn I(X;A|s_x,s_y); symmetric; 999-surrogate p and five-feature BH q",
            "CMI_FULL": "raw Tigramite CMIknn I(X;A|s_x,s_y,all other candidates); symmetric; 999-surrogate p and five-feature BH q",
            "GPC": "sum of spEDM full-library forward positive+negative+dark pattern strengths; X->A is y_xmap_x; descriptive forward-over-reverse selection only; no calibrated full-library p",
            "SCT": "spEDM sc.test sc[1]/sc[2] for forward X->A statistic/bootstrap p; sc[3]/sc[4] reverse; separate directional five-feature BH",
            "SCPCM": "spEDM scpcm pxmap y_xmap_x_mean/sig forward and x_xmap_y_mean/sig reverse; xmap retained only as unpartialed provenance; separate directional five-feature BH",
        },
        "answers": {
            "existing_cmi_mspace_correct": "Yes as a coordinate-conditioned residual-scale-relevant dependence calculation; no as a claim of identified direct or directional causality.",
            "observed_confounding": "CMI-Mspace is not robust: false x2 selection was 33/50 in C2. CMI-Full reduced this to 1/50 while selecting direct Z in 43/50.",
            "direction": "Neither CMI variant identifies direction; both selected x2 in 50/50 reverse-direction replicates.",
            "cmi_full_defensibility": "CMI-Full is more defensible when the relevant observed alternative causes/proxies are available, with C1 selection reduced from 46/50 to 39/50 but proxy selection reduced from 38/50 to 2/50.",
            "directional_spatial_methods": "No tested directional spatial method jointly improved target recovery, null control, confounder robustness, and directionality.",
        },
        "recommendation": {
            "next_stage_estimators": ["CMI_FULL", "CMI_MSPACE"],
            "primary": "CMI_FULL",
            "secondary": "CMI_MSPACE as the established high-power conditional-dependence comparator/ablation",
            "claim": "Both remain conditional-dependence feature-strength estimators; neither supports identified directional-causality language.",
        },
        "rejected_for_primary_weighting": {
            "GPC": "0/50 null no-signal under an uncalibrated descriptive gate, weak C1 top-1 recovery (17/50), and near-chance directional preference.",
            "SCT": "excellent null control (50/50 no-signal) but essentially no C1 residual-scale power (0/50 x2 selected).",
            "SCPCM": "moderate C1 recovery (34/50 selected) but inflated C0 selections (9/50), confounded x2 selections (15/50), and unstable/bidirectional direction calls.",
            "GCCM_REFERENCE": "frozen Stage-15 negative reference: x2 positive 1/50 and top 0/50; not rerun here.",
        },
        "limitations": [
            "C2 conditions only on an observed confounder; unmeasured confounding remains unresolved.",
            "CMI-Full can lose power and can retain a correlated proxy under nonlinear mechanisms (C5 g_proxy selected 39/50).",
            "C4 spatial-lag detection was weak after BH for both CMI variants despite frequent raw x2 top ranks.",
            "GPC full-library output lacks calibrated selection significance, so its no-signal metric is descriptive.",
            "SCPCM is secondary and its documentation/method maturity limitation remains.",
            "Controlled synthetic mechanisms do not establish universal causal identification on observational spatial data.",
        ],
    }
    (RECORD / "stage17_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = f"""# Stage 17 causal-strength estimator validation

## Execution and claim boundary

The preregistration remained hash-stable (`{seal['composite_sha256']}`). Seeds 20420001–20420050 were new development seeds and were paired across six mechanisms. Every method completed 50/50 replicates per mechanism; final stderr and all six spEDM warning files are empty. No conformal prediction was run, and Stage-15 evidence was not modified.

CMI-Mspace and CMI-Full are symmetric conditional-dependence estimators. Their results must not be described as identified directional causality.

## Frozen mechanisms

- C0: independent null.
- C1: target same-location residual-scale causality, `x2 -> sigma -> A`.
- C2: observed pure confounding, `Z -> x2` and `Z -> sigma -> A`, with no direct `x2 -> A`.
- C3: reverse direction, `A_driver -> x2`, with no `x2 -> A`.
- C4: first-order spatial-lag residual-scale causality.
- C5: nonlinear non-monotone direct `x2 -> A`.

All use the frozen 200-node, 583-edge support. `g_proxy` is deliberately spatial and correlated with `x2`/geography but never enters the target structural equation.

## Recovery and control (counts out of 50)

| Method | C1 x2 top | C1 x2 selected | C1 proxy false | C0 no signal | C2 false x2 | C2 direct Z selected |
|---|---:|---:|---:|---:|---:|---:|
| CMI-Mspace | 49 | 46 | 38 | 50 | 33 | 47 |
| CMI-Full | 50 | 39 | 2 | 48 | 1 | 43 |
| GPC | 17 | 32* | 36* | 0* | 28* | 32* |
| SCT | 1 | 0 | 3 | 50 | 2 | 1 |
| SCPCM | 31 | 34 | 21 | 41 | 15 | 33 |

`*` GPC selection is only a descriptive forward-over-reverse preference because full-library GPC has no calibrated selection p-value.

CMI-Mspace recovers the intended C1 residual-scale driver, but its high C2 false-x2 rate and 50/50 x2 selection under C3 show that it is detecting conditional association rather than direct/directional causality. CMI-Full is substantially more robust to the observed C2 confounder and C1 proxy while retaining useful C1 recovery, though its selection power is lower.

## Directional methods

GPC preferred the correct direction for x2 in 32/50 C1, 23/50 C4, and 30/50 C5 replicates, and the reverse direction in 29/50 C3 replicates—insufficiently separated from chance and without calibrated full-library selection. SCT's direction preference was 35/50 for C1/C4/C5 but its BH-gated x2 selection was 0/50, 2/50, and 0/50. SCPCM showed useful C1 forward preference (44/50) but selected x2 in both forward and reverse directions frequently; in C3 it preferred the correct reverse direction only 23/50 times, and in C4 it selected forward/reverse x2 47/50 and 43/50. None jointly solved recovery, null calibration, confounding, and direction.

## Scientific answer

The existing CMI-Mspace calculation was **correct for its actual estimand**: coordinate-conditioned residual-scale-relevant conditional dependence. It was **not** evidence of identified direct or directional causality. Conditioning on all observed alternative candidates (CMI-Full) gives a more defensible direct-feature interpretation under the benchmark's observed-confounder design, but it still cannot establish direction or address unmeasured confounding.

## Recommendation

Carry at most two estimators into the next Geo-Causal CP development stage:

1. **CMI-Full** as the primary confounder/proxy-aware feature-strength estimator.
2. **CMI-Mspace** as the established higher-power conditional-dependence comparator and ablation.

Do not promote GPC, SCT, or SCPCM to primary residual-scale weighting from this evidence. Retain their negative/secondary diagnostic role. Any next-stage manuscript language must describe both recommended methods as causally motivated conditional-dependence feature strength, not identified directional causality.
"""
    (RECORD / "stage17_summary.md").write_text(md, encoding="utf-8")

    execution_inputs_path = RECORD / "execution_input_manifest.json"
    execution_inputs = json.loads(execution_inputs_path.read_text(encoding="utf-8"))
    completion_source = Path(__file__).resolve()
    completion_entry = {"path": str(completion_source.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(completion_source)}
    execution_inputs["completion"] = completion_entry
    execution_inputs_path.write_text(json.dumps(execution_inputs, indent=2) + "\n", encoding="utf-8")

    # Manifest all canonical record files except the manifest itself and transient bytecode.
    entries = []
    for path in sorted(RECORD.rglob("*")):
        if not path.is_file() or path.name == "output_manifest.json":
            continue
        entries.append({"path": str(path.relative_to(RECORD)).replace("\\", "/"), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    output_manifest = {
        "record": str(RECORD.relative_to(ROOT)).replace("\\", "/"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": "PASS",
        "preregistration_composite_sha256": seal["composite_sha256"],
        "scientific_outputs_complete": True,
        "final_stderr_empty": True,
        "files": entries,
    }
    (RECORD / "output_manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "files_manifested": len(entries), "record": str(RECORD), "recommendation": summary["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
