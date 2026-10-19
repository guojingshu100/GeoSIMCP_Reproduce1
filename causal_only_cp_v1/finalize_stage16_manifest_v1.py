"""Refresh and verify the Stage-16 output manifest after figure rendering."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"experiments/manuscript_package/20260918_stage16_final_simulation_package_v1"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

required=[
 "README.md","final_integrity_audit.json","evidence_chain.csv","table_main_confirmatory.csv","table_main_confirmatory.md",
 "table_paired_effects.csv","table_paired_effects.md","table_feature_recovery.csv","table_scale_recovery.csv",
 "table_alpha_robustness.csv","MANUSCRIPT_METHOD_DRAFT.md","MANUSCRIPT_RESULTS_DRAFT.md","ABLATION_INTERPRETATION.md",
 "FINAL_CLAIM_BOUNDARY.md","FINAL_METHOD_FAMILY.md","FUTURE_HOLDOUT_PROTOCOL_TEMPLATE.md","FINAL_SIMULATION_HANDOFF.md",
 "input_manifest.json","run_log.txt","stderr.txt"]
missing=[x for x in required if not (OUT/x).exists()]
for letter in "ABCDE":
    for ext in ["png","pdf"]:
        p=OUT/f"figures/figure_{letter}.{ext}"
        if not p.exists(): missing.append(str(p.relative_to(OUT)))
if missing: raise FileNotFoundError(missing)
if (OUT/"stderr.txt").stat().st_size: raise RuntimeError("non-empty successful stderr")
outputs=[]
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name not in {"output_manifest.json","run_log.txt","stderr.txt"}:
        outputs.append({"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size})
(OUT/"output_manifest.json").write_text(json.dumps({"outputs":outputs,"stage15_integrity_status":"PASS","missing_required_outputs":missing},indent=2)+"\n",encoding="utf-8")
# Immediate independent readback verification.
obj=json.loads((OUT/"output_manifest.json").read_text(encoding="utf-8")); bad=[]
for row in obj["outputs"]:
    p=ROOT/row["path"]
    if sha(p)!=row["sha256"] or p.stat().st_size!=row["bytes"]: bad.append(row["path"])
if bad: raise AssertionError(bad)
print(json.dumps({"status":"PASS","manifest_entries":len(outputs),"hash_mismatches":bad,"stderr_bytes":0},indent=2))
