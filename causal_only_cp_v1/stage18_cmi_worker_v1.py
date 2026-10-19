"""Run both preregistered Stage-18 CMI variants on one frozen scale-fit table."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tigramite.independence_tests.cmiknn import CMIknn


FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
METHODS = ["CMI_MSPACE_999", "CMI_FULL_999"]


def bh(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(ordered); running = 1.0
    for i in range(len(ordered) - 1, -1, -1):
        running = min(running, min(1.0, ordered[i][1] * len(ordered) / (i + 1)))
        adjusted[i] = running
    return {name: float(value) for (name, _), value in zip(ordered, adjusted)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--scenario", choices=["S2", "S4"], required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    data = pd.read_csv(args.input, keep_default_na=False)
    expected = ["sample_id", "s_x", "s_y", *FEATURES, "A_model"]
    if list(data.columns) != expected or len(data) != 200 or not data.sample_id.is_unique:
        raise AssertionError("invalid frozen CMI input")
    matrix = data[[*FEATURES, "s_x", "s_y", "A_model"]].to_numpy(float)
    mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=0)
    if not np.isfinite(matrix).all() or np.any(sd <= 0):
        raise AssertionError("invalid numeric CMI input")
    z = (matrix - mean) / sd
    names = [*FEATURES, "s_x", "s_y", "A_model"]
    values = {name: z[:, i] for i, name in enumerate(names)}
    rows = []
    scenario_offset = 0 if args.scenario == "S2" else 100
    for method_index, method in enumerate(METHODS):
        local = []; pvalues = {}
        for feature_index, feature in enumerate(FEATURES):
            conds = ["s_x", "s_y"] if method == "CMI_MSPACE_999" else ["s_x", "s_y", *[x for x in FEATURES if x != feature]]
            array = np.vstack([values[feature], values["A_model"], *[values[x] for x in conds]])
            xyz = np.asarray([0, 1, *([2] * len(conds))], dtype=int)
            seed = 21900000 + 1000 * args.replicate_index + scenario_offset + 10 * method_index + feature_index
            test = CMIknn(knn=.10, shuffle_neighbors=5, significance="shuffle_test", transform=False,
                          workers=1, sig_samples=999, sig_blocklength=1, seed=seed)
            score = float(test.get_dependence_measure(array.copy(), xyz.copy()))
            pvalue = float(test.get_shuffle_significance(array.copy(), xyz.copy(), score))
            pvalues[feature] = pvalue
            local.append({"method": method, "feature": feature, "raw_cmi": score, "p_value": pvalue,
                          "conditioning": "+".join(conds), "shuffle_seed": seed, "k": 20, "surrogates": 999})
        qvalues = bh(pvalues)
        eligible = {row["feature"]: qvalues[row["feature"]] < .05 and row["raw_cmi"] > 0 for row in local}
        total = sum(row["raw_cmi"] for row in local if eligible[row["feature"]])
        status = "SIGNAL" if total > 0 else "NO_CAUSAL_SIGNAL"
        ranking = sorted(local, key=lambda row: (-row["raw_cmi"], row["feature"]))
        ranks = {row["feature"]: i for i, row in enumerate(ranking, 1)}
        for row in local:
            feature = row["feature"]
            row.update({"bh_q": qvalues[feature], "eligible": eligible[feature],
                        "weight": row["raw_cmi"] / total if eligible[feature] and total > 0 else 0.0,
                        "status": status, "rank": ranks[feature]})
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) != 8:
        raise AssertionError("unexpected CMI output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, lineterminator="\n", float_format="%.17g")
    metadata = {"input": str(args.input), "output": str(args.output), "replicate_index": args.replicate_index,
                "scenario": args.scenario, "runtime_seconds": time.perf_counter() - started,
                "statuses": {m: str(out.loc[out.method.eq(m), "status"].iloc[0]) for m in METHODS}}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
