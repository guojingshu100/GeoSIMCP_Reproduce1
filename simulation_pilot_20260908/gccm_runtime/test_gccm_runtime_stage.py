"""Tests for real restricted GCCM artifacts and negative wrapper contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t, norm

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_gccm_scale_fit_v1 import PILOT,FEATURES,SCENARIOS,RSCRIPT,sha

class RuntimeTests(unittest.TestCase):
    def test_01_environment_smoke(self):
        s=json.loads((HERE/'smoke_check_v1.json').read_text())
        self.assertEqual(s['status'],'PASS')
        self.assertEqual(s['runs'][0]['hashes'],s['runs'][1]['hashes'])

    def test_02_frozen_grid(self):
        c=json.loads((HERE/'gccm_runtime_config_v1.json').read_text(encoding='utf-8'))
        self.assertTrue(c['execution_frozen'])
        self.assertEqual(c['valid_embedding_counts'],dict.fromkeys(SCENARIOS,200))
        self.assertEqual(c['library_sizes'],[10,18,33,60,109,198])
        self.assertEqual(c['direction'],'effect_xmap_cause')
        self.assertEqual(c['random_library_draws'],0)

    def test_03_restricted_inputs_and_alignment(self):
        cols=['node_id','sample_id','s_x','s_y','abs_residual']+['z_'+f for f in FEATURES]
        for s in SCENARIOS:
            p=PILOT/'gccm_outputs'/s
            inp=pd.read_csv(p/'restricted_input.csv')
            self.assertEqual(list(inp),cols); self.assertEqual(len(inp),200)
            prep=pd.read_csv(PILOT/'scale_outputs'/f'{s}_scale_fit_preparation.csv').sort_values('sample_id')
            self.assertEqual(inp.sample_id.tolist(),prep.sample_id.tolist())
            self.assertEqual(set(prep.local_calibration_role),{'scale_fit'})
            np.testing.assert_allclose(inp.abs_residual,prep.abs_residual,rtol=1e-14,atol=1e-14)
            emb=pd.read_csv(p/'embedding.csv')
            self.assertEqual(emb.sample_id.tolist(),inp.sample_id.tolist())

    def test_04_historical_convergence_independent_recalculation(self):
        for s in SCENARIOS:
            p=PILOT/'gccm_outputs'/s
            raw=pd.read_csv(p/'raw_curves.csv'); diag=pd.read_csv(p/'convergence.csv')
            self.assertEqual(len(raw),5946)
            self.assertEqual(set(raw.direction),{'effect_xmap_cause'})
            for _,d in diag.iterrows():
                g=raw[raw.feature==d.feature].groupby('L').rho.mean().clip(lower=0)
                mono=spearmanr(g.index,g.to_numpy()).statistic if g.nunique()>1 else np.nan
                rr=np.clip(g.iloc[-1],-1+1e-12,1-1e-12)
                pval=2*t.sf(abs(rr*np.sqrt(196/(1-rr*rr))),df=196)
                lower=np.tanh(np.arctanh(rr)-norm.ppf(.975)/np.sqrt(195))
                converged=bool(np.isfinite(mono) and g.iloc[-1]-g.iloc[0]>0 and mono>=.6 and pval<.05 and lower>0)
                self.assertEqual(bool(d.converged),converged)
                self.assertAlmostEqual(d.strength,float(g.iloc[-1]) if converged else 0,places=12)

    def test_05_weights_no_uniform_fallback(self):
        for s in SCENARIOS:
            w=pd.read_csv(PILOT/'gccm_outputs'/s/'weights.csv')
            self.assertEqual(w.feature.tolist(),FEATURES)
            if w.strength.sum()>0:
                np.testing.assert_allclose(w.weight,w.strength/w.strength.sum(),rtol=1e-12,atol=1e-14)
            else:
                self.assertTrue(w.weight.isna().all())
                self.assertEqual(set(w.status),{'NO_CONVERGED_FEATURES'})

    def test_06_reproduction_and_hash_integrity(self):
        c=json.loads((PILOT/'gccm_runtime_check_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(c['deterministic_reproduction_status'],'PASS')
        for s,matches in c['reproducibility_hash_matches'].items():
            self.assertTrue(all(matches.values()))
            for name,digest in c['per_scenario_GCCM_execution_status'][s]['hashes'].items():
                self.assertEqual(sha(PILOT/'gccm_outputs'/s/name),digest)
        for k in ['tuning_calibration_labels_accessed','validation_labels_accessed','final_calibration_labels_accessed','test_labels_accessed','oracle_files_read_or_hashed','primary_validation_comparison_executed']:
            self.assertFalse(c[k])

    def test_07_negative_wrapper_schemas_and_ids(self):
        # Reuse only restricted scale_fit inputs, never read a mixed-label file.
        source=PILOT/'gccm_outputs'/'S1'
        import shutil
        with tempfile.TemporaryDirectory(prefix='gccm_negative_') as td:
            dest=Path(td)
            for name in ['restricted_input.csv','edges.csv','historical_core.r','parameters.csv']:
                shutil.copyfile(source/name,dest/name)
            original=pd.read_csv(dest/'restricted_input.csv')
            for case in ['extra_label','wrong_node_order']:
                frame=original.copy()
                if case=='extra_label': frame['y']=0 # synthetic forbidden column; not an accessed label
                else: frame.loc[0,'node_id']=200
                frame.to_csv(dest/'restricted_input.csv',index=False)
                cmd=[str(RSCRIPT),'--vanilla',str(HERE/'gccm_point_graph_wrapper_v1.R'),str(dest/'restricted_input.csv'),str(dest/'edges.csv'),str(dest/'historical_core.r'),str(dest/'parameters.csv'),str(dest),'preflight']
                proc=subprocess.run(cmd,env={**os.environ,'LC_ALL':'C'},capture_output=True,timeout=30)
                self.assertNotEqual(proc.returncode,0,case)

if __name__=='__main__': unittest.main(verbosity=2)
