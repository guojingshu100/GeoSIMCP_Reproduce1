"""Final reporting adapter for Stage 15.

Relative to v2, this labels an S4 selection of *any* feature as false; under
S2 only non-x2 selections count toward the corresponding nuisance-feature
diagnostic. No scientific calculation or interval is changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import finalize_stage15_fresh_confirmatory_v2 as v2


def feature_recovery(cmi: pd.DataFrame, weights: pd.DataFrame, gccm: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for estimator, frame, value_col, status_col in [
        ("CMI99", weights, "weight", "status"),
        ("GCCM", gccm, "weight", "status"),
    ]:
        for scenario in ["S2", "S4"]:
            x=frame[frame.scenario==scenario]
            per=x.pivot(index="replicate_index",columns="feature",values=value_col).fillna(0)
            selected=per>0
            top=per.idxmax(axis=1).where(per.max(axis=1)>0,"NO_SIGNAL")
            no_signal=(~selected.any(axis=1))
            false_mask=selected.any(axis=1) if scenario=="S4" else selected[["x1","g_proxy","x_noise"]].any(axis=1)
            rows.append({"estimator":estimator,"scenario":scenario,"replicates":len(per),
                "x2_selected":int(selected.x2.sum()),"x2_top":int((top=="x2").sum()),
                "no_signal":int(no_signal.sum()),"false_feature_selection_replicates":int(false_mask.sum()),
                "x1_selected":int(selected.x1.sum()),"g_proxy_selected":int(selected.g_proxy.sum()),
                "x_noise_selected":int(selected.x_noise.sum())})
    return pd.DataFrame(rows)


v2.base.feature_recovery=feature_recovery

if __name__=="__main__":
    v2.base.main()
    # Record the actually successful reporting implementation, then refresh
    # the top-level manifest so its hash of input_manifest remains valid.
    manifest_path=v2.base.RECORD/"input_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    this_file=Path(__file__).resolve()
    rel=str(this_file.relative_to(v2.base.ROOT))
    manifest["inputs"]=[x for x in manifest["inputs"] if x.get("path")!=rel]
    manifest["inputs"].append({"path":rel,"sha256":v2.base.sha(this_file),"bytes":this_file.stat().st_size,
                               "role":"successful reporting finalizer v3"})
    v2.base.dump("input_manifest.json",manifest)
    outputs=[]
    for p in sorted(v2.base.RECORD.glob("*")):
        if p.is_file() and p.name not in {"output_manifest.json","run_log.txt","stderr.txt"}:
            outputs.append({"path":str(p.relative_to(v2.base.ROOT)),"sha256":v2.base.sha(p),"bytes":p.stat().st_size})
    v2.base.dump("output_manifest.json",{"outputs":outputs,"large_frozen_artifacts_referenced_by_nested_COMPLETE_manifests":True})
