"""Restricted scale-fit GCCM integration; never reads generator label files."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

RUNTIME=Path(__file__).resolve().parent
PILOT=RUNTIME.parent
ROOT=PILOT.parent
FEATURES=['s_x','s_y','x1','x2','g_proxy','x_noise']
SCENARIOS=['S1','S2','S3','S4']
RSCRIPT=Path('C:/Program Files/R/R-4.6.1/bin/Rscript.exe')
CORE=ROOT/'geoconformal-main/GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/GCCM4Lattice.r'
CORE_HASH='3552d871539dc682519da04d07306a97e9138f30b22ecdbfcc2cf61efa38b12d'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def prepare_input(scenario,destination):
    assert scenario in SCENARIOS
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=False)
    manifest=read_json(PILOT/'scale_comparison_check_v1.json')
    assert manifest['non_gccm_preparation_status']=='PASS'
    prep=PILOT/'scale_outputs'/f'{scenario}_scale_fit_preparation.csv'
    scaler_path=PILOT/'scale_outputs'/f'{scenario}_standard_scaler.joblib'
    assert sha(prep)==manifest['output_file_hashes'][prep.name]
    assert sha(scaler_path)==manifest['scaler_file_hashes'][scaler_path.name]
    frame=pd.read_csv(prep,keep_default_na=False,float_precision='round_trip')
    assert len(frame)==200 and frame.sample_id.is_unique
    assert set(frame.split)=={'local_calibration'} and set(frame.local_calibration_role)=={'scale_fit'}
    assert not set(frame).intersection({'y','mu_true','sigma_true','epsilon','g_true','eta'})
    frame=frame.sort_values('sample_id').reset_index(drop=True)
    scaler=joblib.load(scaler_path)
    z=frame[['z_'+f for f in FEATURES]].to_numpy(float)
    raw=scaler.inverse_transform(z)
    # Coordinates recovered from the frozen scaler; never open observed CSV.
    xy=raw[:,:2]
    tri=Delaunay(xy)
    edges=sorted({tuple(sorted((int(t[i])+1,int(t[j])+1))) for t in tri.simplices for i,j in [(0,1),(1,2),(0,2)]})
    inp=pd.DataFrame({'node_id':np.arange(1,201),'sample_id':frame.sample_id,'s_x':xy[:,0],'s_y':xy[:,1],'abs_residual':frame.abs_residual})
    for f in FEATURES: inp['z_'+f]=frame['z_'+f]
    assert np.isfinite(inp.drop(columns='sample_id').to_numpy(float)).all()
    inp.to_csv(destination/'restricted_input.csv',index=False,float_format='%.17g',lineterminator='\n')
    pd.DataFrame(edges,columns=['u','v']).to_csv(destination/'edges.csv',index=False,lineterminator='\n')
    assert sha(CORE)==CORE_HASH
    shutil.copyfile(CORE,destination/'historical_core.r')
    return inp,edges

def invoke(destination,mode,grid):
    destination=Path(destination)
    params=[('E','3'),('tau','1'),('b','4'),('seed','42'),('direction','effect_xmap_cause'),('library_sizes',';'.join(map(str,grid)))]
    pd.DataFrame(params,columns=['name','value']).to_csv(destination/'parameters.csv',index=False,lineterminator='\n')
    cmd=[str(RSCRIPT),'--vanilla',str(RUNTIME/'gccm_point_graph_wrapper_v1.R'),str(destination/'restricted_input.csv'),str(destination/'edges.csv'),str(destination/'historical_core.r'),str(destination/'parameters.csv'),str(destination),mode]
    start=time.perf_counter()
    proc=subprocess.run(cmd,cwd=destination,env={**os.environ,'LC_ALL':'C'},capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=1800)
    (destination/'stdout.log').write_text(proc.stdout,encoding='utf-8')
    (destination/'stderr.log').write_text(proc.stderr,encoding='utf-8')
    metadata={'mode':mode,'returncode':proc.returncode,'runtime_seconds':time.perf_counter()-start,'command':cmd,'environment_override':{'LC_ALL':'C'},'hashes':{p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file()},'source_access':'restricted scale-fit input, graph, parameters and copied historical core only'}
    write_json(destination/'execution_metadata.json',metadata)
    if proc.returncode: raise RuntimeError(f'R return code {proc.returncode}: {proc.stderr}')
    return metadata

def verify_embedding(frame,edges,destination):
    neighbours=[set() for _ in range(200)]
    for u,v in edges: neighbours[u-1].add(v-1); neighbours[v-1].add(u-1)
    vals=frame.abs_residual.to_numpy(float); expected=[]
    for i in range(200):
        front={i}; seen={i}; row=[]
        for _ in range(3):
            front=set().union(*(neighbours[j] for j in front))-seen
            assert front
            seen|=front; row.append(vals[sorted(front)].mean())
        expected.append(row)
    saved=pd.read_csv(Path(destination)/'embedding.csv')
    assert saved.sample_id.tolist()==frame.sample_id.tolist()
    np.testing.assert_allclose(saved.iloc[:,2:].to_numpy(float),expected,rtol=2e-13,atol=2e-13)
    result=pd.read_csv(Path(destination)/'preflight.csv').iloc[0]
    assert result.valid_embeddings==200 and result.max_library==198

def preflight_and_freeze():
    assert read_json(RUNTIME/'smoke_check_v1.json')['status']=='PASS'
    old=read_json(RUNTIME/'gccm_runtime_config_v1.json')
    assert old.get('execution_frozen') is not True
    preflight=RUNTIME/'scale_fit_preflight_v1'
    preflight.mkdir(exist_ok=False)
    counts={}
    for s in SCENARIOS:
        frame,edges=prepare_input(s,preflight/s)
        invoke(preflight/s,'preflight',[10,18,33,60,109,198])
        verify_embedding(frame,edges,preflight/s)
        counts[s]=200
    old.update({'overall_status':'READY_FOR_ESTIMATION','rscript_absolute_path':str(RSCRIPT),'execution_frozen':True,'frozen_at':datetime.now().astimezone().isoformat(),'valid_embedding_counts':counts,'max_library':198,'library_sizes':[10,18,33,60,109,198],'rounding':'floor(x+0.5)','duplicate_rule':'fail if any duplicate, per original preregistration','random_library_draws':0,'backend':'sequential cores=NULL','direction':'effect_xmap_cause','seed':42,'wrapper_sha256':sha(RUNTIME/'gccm_point_graph_wrapper_v1.R'),'runner_sha256':sha(Path(__file__)),'preflight_status':'PASS','smoke_status':'PASS','coordinate_source':'frozen scale-fit standardized coordinates inverse-transformed with frozen scaler; no observed CSV read','all_zero_fallback':'NO_CONVERGED_FEATURES; normalized weights unavailable; never substitute uniform'})
    old['candidate_protocol']['status']='FROZEN_AFTER_R_PREFLIGHT'
    old['candidate_protocol']['full_execution_config_frozen']=True
    old['candidate_protocol']['actual_valid_embedding_count']=200
    write_json(RUNTIME/'gccm_runtime_config_v1.json',old)
    print('PREFLIGHT PASS; 200 embeddings per scenario; grid frozen',old['library_sizes'])

def execute(root):
    cfg=read_json(RUNTIME/'gccm_runtime_config_v1.json')
    assert cfg['execution_frozen'] and cfg['preflight_status']=='PASS'
    assert cfg['wrapper_sha256']==sha(RUNTIME/'gccm_point_graph_wrapper_v1.R')
    assert cfg['runner_sha256']==sha(Path(__file__))
    results={}
    for s in SCENARIOS:
        out=Path(root)/s
        frame,edges=prepare_input(s,out)
        shutil.copyfile(RUNTIME/'gccm_runtime_config_v1.json',out/'execution_config.json')
        meta=invoke(out,'estimate',cfg['library_sizes'])
        verify_embedding(frame,edges,out)
        weights=pd.read_csv(out/'weights.csv',keep_default_na=False)
        raw=pd.read_csv(out/'raw_curves.csv')
        assert len(raw)==5946 and weights.feature.tolist()==FEATURES
        has_weights=set(weights.status)=={'WEIGHTS_AVAILABLE'}
        if has_weights:
            np.testing.assert_allclose(weights.weight.astype(float).sum(),1.0,rtol=0,atol=1e-12)
        results[s]={'execution_status':'PASS','weights_status':'WEIGHTS_AVAILABLE' if has_weights else 'NO_CONVERGED_FEATURES','valid_embeddings':200,'library_grid':cfg['library_sizes'],'runtime_seconds':meta['runtime_seconds'],'returncode':meta['returncode'],'hashes':meta['hashes'],'warnings':(out/'stderr.log').read_text(encoding='utf-8'),'weights':weights.to_dict('records')}
        print(s,results[s]['weights_status'],'seconds',round(meta['runtime_seconds'],2),flush=True)
    return results

def run_all():
    formal=execute(PILOT/'gccm_outputs')
    with tempfile.TemporaryDirectory(prefix='gccm_repro_a_') as a,tempfile.TemporaryDirectory(prefix='gccm_repro_b_') as b:
        first=execute(Path(a)); second=execute(Path(b))
        keys=['restricted_input.csv','edges.csv','historical_core.r','parameters.csv','execution_config.json','embedding.csv','raw_curves.csv','mean_curves.csv','convergence.csv','weights.csv']
        matches={s:{key:formal[s]['hashes'][key]==first[s]['hashes'][key]==second[s]['hashes'][key] for key in keys} for s in SCENARIOS}
    reproducible=all(all(x.values()) for x in matches.values())
    valid_weights=all(v['weights_status']=='WEIGHTS_AVAILABLE' for v in formal.values())
    status='FAILED_REPRODUCIBILITY' if not reproducible else ('PASS' if valid_weights else 'BLOCKED_PROTOCOL')
    check={'overall_status':status,'rscript_status':'PASS','package_status':'PASS','historical_core_status':'PASS','point_graph_wrapper_status':'PASS','library_grid_frozen_before_execution':True,'smoke_test_status':'PASS','deterministic_reproduction_status':'PASS' if reproducible else 'FAIL','scale_fit_only_status':'PASS','tuning_calibration_labels_accessed':False,'validation_labels_accessed':False,'final_calibration_labels_accessed':False,'test_labels_accessed':False,'oracle_files_read_or_hashed':False,'primary_validation_comparison_executed':False,'per_scenario_GCCM_execution_status':formal,'reproducibility_hash_matches':matches,'independent_temporary_runs':{'a':first,'b':second},'execution_config_sha256':sha(RUNTIME/'gccm_runtime_config_v1.json'),'warnings':[] if valid_weights else ['At least one scenario has no converged feature; no normalized weight vector exists. No uniform fallback used.'],'data_access_allowlist':['scale_outputs/S*_scale_fit_preparation.csv','scale_outputs/S*_standard_scaler.joblib','scale_comparison_check_v1.json'],'access_flag_scope':'current runtime stage only'}
    write_json(PILOT/'gccm_runtime_check_v1.json',check)
    print('OVERALL',status)
    return check

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight: preflight_and_freeze()
    else: run_all()
