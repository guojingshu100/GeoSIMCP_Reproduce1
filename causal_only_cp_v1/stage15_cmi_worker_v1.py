"""Isolated Tigramite CMI worker for the preregistered Stage-15 protocol."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tigramite.independence_tests.cmiknn import CMIknn


FEATURES = ["x1", "x2", "g_proxy", "x_noise"]
COORDS = ["s_x", "s_y"]


def bh(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(ordered)
    running = 1.0
    for i in range(len(ordered) - 1, -1, -1):
        running = min(running, min(1.0, ordered[i][1] * len(ordered) / (i + 1)))
        adjusted[i] = running
    return {name: float(value) for (name, _), value in zip(ordered, adjusted)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--surrogates", type=int, choices=[99, 999], required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    data = pd.read_csv(args.input, keep_default_na=False)
    expected = ["sample_id", *COORDS, *FEATURES, "A_model"]
    if list(data.columns) != expected or len(data) != 200 or not data.sample_id.is_unique:
        raise AssertionError("invalid frozen CMI input")
    matrix = data[[*FEATURES, *COORDS, "A_model"]].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise AssertionError("nonfinite CMI input")
    means, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=0)
    if np.any(sd <= 0):
        raise AssertionError("constant CMI input")
    z = (matrix - means) / sd
    values = {name: z[:, i] for i, name in enumerate([*FEATURES, *COORDS, "A_model"])}
    raw: dict[str, float] = {}
    pvals: dict[str, float] = {}
    for feature in FEATURES:
        array = np.vstack([values[feature], values["A_model"], values["s_x"], values["s_y"]])
        xyz = np.asarray([0, 1, 2, 2], dtype=int)
        test = CMIknn(knn=.10, shuffle_neighbors=5, significance="shuffle_test", transform=False,
                      workers=1, sig_samples=args.surrogates, sig_blocklength=1, seed=args.seed)
        score = float(test.get_dependence_measure(array.copy(), xyz.copy()))
        raw[feature] = score
        pvals[feature] = float(test.get_shuffle_significance(array.copy(), xyz.copy(), score))
    qvals = bh(pvals)
    eligible = {feature: qvals[feature] < .05 and raw[feature] > 0 for feature in FEATURES}
    total = sum(raw[feature] for feature in FEATURES if eligible[feature])
    status = "SIGNAL" if total > 0 else "NO_CAUSAL_SIGNAL"
    rows = []
    for feature in FEATURES:
        rows.append({"feature": feature, "raw_cmi": raw[feature], "p_value": pvals[feature],
                     "bh_q": qvals[feature], "eligible": eligible[feature],
                     "weight": raw[feature] / total if eligible[feature] and total > 0 else 0.0,
                     "status": status, "n": len(data), "k": 20, "surrogates": args.surrogates,
                     "shuffle_seed": args.seed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, lineterminator="\n", float_format="%.17g")
    metadata = {"input": str(args.input), "output": str(args.output), "seed": args.seed,
                "surrogates": args.surrogates, "runtime_seconds": time.perf_counter() - started,
                "status": status}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_seconds": metadata["runtime_seconds"], "status": status}))


if __name__ == "__main__":
    main()
