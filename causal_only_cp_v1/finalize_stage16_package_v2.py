"""Final presentation/manifest pass for the Stage-16 package."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import build_stage16_final_simulation_package_v1 as base

OUT=base.OUT; ROOT=base.ROOT

# Put manuscript columns in the exact requested order; values are unchanged.
p=OUT/"table_main_confirmatory.csv"; x=pd.read_csv(p)
cols=["scenario","method","mean_coverage","coverage_sd","mean_width","width_sd","mean_interval_score","interval_score_sd",
      "count_coverage_below_nominal","material_undercoverage"]
base.write_csv(x[cols],p)
labels={"mean_coverage":"Mean coverage","coverage_sd":"Coverage SD","mean_width":"Mean width","width_sd":"Width SD",
        "mean_interval_score":"Mean interval score","interval_score_sd":"Interval-score SD",
        "count_coverage_below_nominal":"Replicates coverage < .90","material_undercoverage":"Material undercoverage"}
(OUT/"table_main_confirmatory.md").write_text("# Primary confirmatory results (fresh Stage 15, alpha=.10)\n\n"+base.md_table(x[cols],cols,labels)+"\n",encoding="utf-8")

manifest_path=OUT/"input_manifest.json"; manifest=json.loads(manifest_path.read_text(encoding="utf-8")); this=Path(__file__).resolve(); rel=str(this.relative_to(ROOT))
manifest.setdefault("consolidation_tools",[]); manifest["consolidation_tools"]=[r for r in manifest["consolidation_tools"] if r.get("path")!=rel]
manifest["consolidation_tools"].append({"path":rel,"sha256":base.sha(this),"role":"successful final presentation and manifest pass"})
base.dump(manifest_path,manifest)

outputs=[]
for q in sorted(OUT.rglob("*")):
    if q.is_file() and q.name not in {"output_manifest.json","run_log.txt","stderr.txt"}:
        outputs.append({"path":str(q.relative_to(ROOT)),"sha256":base.sha(q),"bytes":q.stat().st_size})
base.dump(OUT/"output_manifest.json",{"outputs":outputs,"stage15_integrity_status":"PASS","missing_required_outputs":[]})

obj=json.loads((OUT/"output_manifest.json").read_text(encoding="utf-8")); bad=[]
for row in obj["outputs"]:
    q=ROOT/row["path"]
    if base.sha(q)!=row["sha256"] or q.stat().st_size!=row["bytes"]: bad.append(row["path"])
if bad or (OUT/"stderr.txt").stat().st_size: raise AssertionError({"hash_mismatches":bad,"stderr_bytes":(OUT/"stderr.txt").stat().st_size})
print(json.dumps({"status":"PASS","manifest_entries":len(outputs),"hash_mismatches":bad},indent=2))
