"""CMI-Full999 worker for one frozen Stage-19B CMI-scale-fit table."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tigramite.independence_tests.cmiknn import CMIknn

FEATURES = ["x1", "x2", "g_proxy", "x_noise"]


def bh(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(ordered)
    running = 1.0
    for i in range(len(ordered) - 1, -1, -1):
        running = min(running, min(1.0, ordered[i][1] * len(ordered) / (i + 1)))
        adjusted[i] = running
    return {name: float(value) for (name, _), value in zip(ordered, adjusted)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replicate-index", type=int, required=True)
    p.add_argument("--scenario", choices=["S2", "S4"], required=True)
    a = p.parse_args()
    started = time.perf_counter()
    data = pd.read_csv(a.input, keep_default_na=False)
    expected = ["sample_id", "s_x", "s_y", *FEATURES, "A_model"]
    if list(data.columns) != expected or len(data) != 200 or not data.sample_id.is_unique:
        raise AssertionError("invalid frozen CMI input")
    matrix = data[[*FEATURES, "s_x", "s_y", "A_model"]].to_numpy(float)
    sd = matrix.std(axis=0, ddof=0)
    if not np.isfinite(matrix).all() or np.any(sd <= 0):
        raise AssertionError("invalid CMI matrix")
    z = (matrix - matrix.mean(axis=0)) / sd
    names = [*FEATURES, "s_x", "s_y", "A_model"]
    values = {name: z[:, i] for i, name in enumerate(names)}
    rows, pvalues = [], {}
    scenario_offset = 0 if a.scenario == "S2" else 100
    for feature_index, feature in enumerate(FEATURES):
        conds = ["s_x", "s_y", *[x for x in FEATURES if x != feature]]
        array = np.vstack([values[feature], values["A_model"], *[values[x] for x in conds]])
        xyz = np.asarray([0, 1, *([2] * len(conds))], dtype=int)
        seed = 21950000 + 1000 * a.replicate_index + scenario_offset + feature_index
        test = CMIknn(knn=.10, shuffle_neighbors=5, significance="shuffle_test", transform=False,
                      workers=1, sig_samples=999, sig_blocklength=1, seed=seed)
        score = float(test.get_dependence_measure(array.copy(), xyz.copy()))
        pvalue = float(test.get_shuffle_significance(array.copy(), xyz.copy(), score))
        pvalues[feature] = pvalue
        rows.append({"feature": feature, "raw_cmi": score, "p_value": pvalue,
                     "conditioning": "+".join(conds), "shuffle_seed": seed, "k": 20, "surrogates": 999})
    qvalues = bh(pvalues)
    eligible = {r["feature"]: qvalues[r["feature"]] < .05 and r["raw_cmi"] > 0 for r in rows}
    total = sum(r["raw_cmi"] for r in rows if eligible[r["feature"]])
    status = "SIGNAL" if total > 0 else "NO_CAUSAL_SIGNAL"
    ranks = {r["feature"]: i for i, r in enumerate(sorted(rows, key=lambda r: (-r["raw_cmi"], r["feature"])), 1)}
    for r in rows:
        f = r["feature"]
        r.update({"bh_q": qvalues[f], "eligible": eligible[f],
                  "weight": r["raw_cmi"] / total if eligible[f] and total > 0 else 0.0,
                  "status": status, "rank": ranks[f]})
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False, lineterminator="\n", float_format="%.17g")
    metadata = {"input": str(a.input), "output": str(a.output), "replicate_index": a.replicate_index,
                "scenario": a.scenario, "runtime_seconds": time.perf_counter() - started, "status": status}
    a.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
