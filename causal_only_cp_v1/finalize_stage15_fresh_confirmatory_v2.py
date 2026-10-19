"""Versioned reporting-only correction for Stage 15.

The v1 finalizer expected generic column names for GCCM. The frozen Stage-12
wrapper actually exports `weight` and `converged`. This wrapper replaces only
that reporting adapter, then delegates every other preregistered summary to
the unchanged v1 finalizer. Scientific results are not recomputed.
"""
from __future__ import annotations

import pandas as pd

import finalize_stage15_fresh_confirmatory_v1 as base


def feature_recovery(cmi: pd.DataFrame, weights: pd.DataFrame, gccm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in ["S2", "S4"]:
        ws = weights[weights.scenario == scenario]
        per = ws.pivot(index="replicate_index", columns="feature", values="weight")
        status = ws.groupby("replicate_index").status.first()
        selected = per > 0
        top = per.idxmax(axis=1).where(per.max(axis=1) > 0, "NO_SIGNAL")
        rows.append({"estimator": "CMI99", "scenario": scenario, "replicates": len(per),
                     "x2_selected": int(selected.x2.sum()), "x2_top": int((top == "x2").sum()),
                     "no_signal": int((status == "NO_CAUSAL_SIGNAL").sum()),
                     "any_false_feature_selected": int(selected[["x1", "g_proxy", "x_noise"]].any(axis=1).sum()),
                     "x1_selected": int(selected.x1.sum()), "g_proxy_selected": int(selected.g_proxy.sum()),
                     "x_noise_selected": int(selected.x_noise.sum())})
        gs = gccm[gccm.scenario == scenario]
        gp = gs.pivot(index="replicate_index", columns="feature", values="weight").fillna(0)
        usable = (gp > 0).sum(axis=1)
        gtop = gp.idxmax(axis=1).where(gp.max(axis=1) > 0, "NO_SIGNAL")
        rows.append({"estimator": "GCCM", "scenario": scenario, "replicates": len(gp),
                     "x2_selected": int((gp.x2 > 0).sum()), "x2_top": int((gtop == "x2").sum()),
                     "no_signal": int((usable == 0).sum()),
                     "any_false_feature_selected": int((gp[["x1", "g_proxy", "x_noise"]] > 0).any(axis=1).sum()),
                     "x1_selected": int((gp.x1 > 0).sum()), "g_proxy_selected": int((gp.g_proxy > 0).sum()),
                     "x_noise_selected": int((gp.x_noise > 0).sum())})
    return pd.DataFrame(rows)


base.feature_recovery = feature_recovery

if __name__ == "__main__":
    base.main()
