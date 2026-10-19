"""Run preregistered Stage-17 CMI-Mspace and CMI-Full for one mechanism."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tigramite.independence_tests.cmiknn import CMIknn


FEATURES = ["x1", "x2", "g_proxy", "x_noise", "Z"]
METHODS = ["CMI_MSPACE", "CMI_FULL"]
MECHANISMS = ["C0_INDEPENDENT_NULL", "C1_RESIDUAL_SCALE_CAUSAL", "C2_PURE_CONFOUNDING", "C3_REVERSE_DIRECTION", "C4_SPATIAL_LAG_CAUSAL", "C5_NONLINEAR_DIRECT"]


def bh(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(ordered)
    running = 1.0
    for i in range(len(ordered) - 1, -1, -1):
        running = min(running, min(1.0, ordered[i][1] * len(ordered) / (i + 1)))
        adjusted[i] = running
    return {name: float(value) for (name, _), value in zip(ordered, adjusted)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mechanism", choices=MECHANISMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    data = pd.read_csv(args.input, float_precision="round_trip")
    data = data.loc[data.mechanism.eq(args.mechanism)].copy()
    if len(data) != 50 * 200 or data.duplicated(["dataset_id", "sample_id"]).any():
        raise AssertionError("invalid Stage-17 controlled input partition")
    mechanism_index = MECHANISMS.index(args.mechanism)
    rows: list[dict] = []
    for dataset_id, group in data.groupby("dataset_id", sort=True):
        group = group.sort_values("row_index", kind="stable")
        replicate_index = int(group.replicate_index.iloc[0])
        matrix = group[[*FEATURES, "s_x", "s_y", "A"]].to_numpy(float)
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0, ddof=0)
        if not np.isfinite(matrix).all() or np.any(scales <= 0):
            raise AssertionError(f"nonfinite/constant input: {dataset_id}")
        z = (matrix - means) / scales
        names = [*FEATURES, "s_x", "s_y", "A"]
        values = {name: z[:, i] for i, name in enumerate(names)}
        for method_index, method in enumerate(METHODS):
            method_rows: list[dict] = []
            pvalues: dict[str, float] = {}
            for feature_index, feature in enumerate(FEATURES):
                conditioning = ["s_x", "s_y"] if method == "CMI_MSPACE" else ["s_x", "s_y", *[x for x in FEATURES if x != feature]]
                array = np.vstack([values[feature], values["A"], *[values[x] for x in conditioning]])
                xyz = np.asarray([0, 1, *([2] * len(conditioning))], dtype=int)
                shuffle_seed = 20500000 + mechanism_index * 100000 + replicate_index * 100 + method_index * 10 + feature_index
                estimator = CMIknn(knn=.10, shuffle_neighbors=5, significance="shuffle_test", transform=False,
                                   workers=1, sig_samples=999, sig_blocklength=1, seed=shuffle_seed)
                score = float(estimator.get_dependence_measure(array.copy(), xyz.copy()))
                pvalue = float(estimator.get_shuffle_significance(array.copy(), xyz.copy(), score))
                pvalues[feature] = pvalue
                method_rows.append({
                    "dataset_id": dataset_id, "mechanism": args.mechanism,
                    "replicate_index": replicate_index, "generation_seed": int(group.generation_seed.iloc[0]),
                    "method": method, "feature": feature, "raw_score": score, "p_value": pvalue,
                    "conditioning": "+".join(conditioning), "shuffle_seed": shuffle_seed,
                    "n": 200, "knn_k": 20, "surrogates": 999,
                })
            qvalues = bh(pvalues)
            for row in method_rows:
                row["bh_q"] = qvalues[row["feature"]]
                row["selected"] = bool(row["bh_q"] < .05 and row["raw_score"] > 0)
            ranking = sorted(method_rows, key=lambda row: (-row["raw_score"], row["feature"]))
            ranks = {row["feature"]: rank for rank, row in enumerate(ranking, start=1)}
            for row in method_rows:
                row["rank"] = ranks[row["feature"]]
                row["no_signal"] = not any(x["selected"] for x in method_rows)
                rows.append(row)
        if replicate_index % 10 == 0:
            print(f"{args.mechanism}: completed {replicate_index}/50", flush=True)
    out = pd.DataFrame(rows)
    if len(out) != 50 * 2 * 5:
        raise AssertionError("unexpected CMI output row count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, lineterminator="\n", float_format="%.17g")
    print(f"PASS {args.mechanism} rows={len(out)} runtime_seconds={time.perf_counter()-started:.3f}")


if __name__ == "__main__":
    main()
