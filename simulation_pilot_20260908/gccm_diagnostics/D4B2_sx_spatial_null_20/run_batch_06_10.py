import csv, hashlib, json, subprocess, tempfile
from pathlib import Path
import numpy as np
from scipy import linalg

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / 'simulation_pilot_20260908'
OUT = PILOT / 'gccm_diagnostics' / 'D4B2_sx_spatial_null_20'
D4A = PILOT / 'gccm_diagnostics' / 'D4A_moran_spatial_surrogate_validation'
S4 = PILOT / 'gccm_outputs' / 'S4'
R = Path(r'C:\Program Files\R\R-4.6.1\bin\Rscript.exe')
WRAP = PILOT / 'gccm_runtime' / 'gccm_point_graph_wrapper_v1.R'
CORE = S4 / 'historical_core.r'

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def moran_mem(w):
    n=w.shape[0]; H=np.eye(n)-np.ones((n,n))/n
    c=H@w@H
    vals, vecs=linalg.eigh(c, check_finite=True)
    keep=np.abs(vals)>1e-10
    return vecs[:,keep].astype(np.float32), vals[keep].astype(np.float32)

def singleton_surrogate(x, mem, vals, seed):
    rng=np.random.RandomState(seed)
    x=np.asarray(x,float); z=x-x.mean(); coef=(mem.T@z)
    out=np.zeros_like(coef)
    for i,v in enumerate(coef):
        out[i] = np.sign(v) * abs(v) if v == 0 else (1 if rng.rand() >= .5 else -1)*abs(v)
    y=mem@out
    y=y-y.mean(); sx=y.std(ddof=1); ox=x.std(ddof=1)
    return y*(ox/sx) if sx>0 else y

def write_csv(path, rows, fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'preregistered_seed_manifest_6_20.json'
    if manifest.exists():
        m=json.loads(manifest.read_text(encoding='utf-8'))
    else:
        rng=np.random.default_rng(20260911)
        seeds=[int(x) for x in rng.integers(1,2**31-1,size=15,dtype=np.int64)]
        m={'diagnostic':'D4B2_sx_spatial_null_20','candidate':'s_x','future_ids':list(range(6,21)), 'seeds':seeds,
           'seed_generation':'numpy.random.default_rng(20260911), integers [1,2^31-1)',
           'frozen_before_new_results':True}
        manifest.write_text(json.dumps(m,indent=2),encoding='utf-8')
    assert m['future_ids']==list(range(6,21)) and len(m['seeds'])==15
    dat=list(csv.DictReader(open(S4/'restricted_input.csv',encoding='utf-8')))
    edges=list(csv.DictReader(open(S4/'edges.csv',encoding='utf-8'))); n=len(dat)
    W=np.zeros((n,n),float)
    for e in edges:
        u=int(e['u'])-1; v=int(e['v'])-1; W[u,v]=W[v,u]=1
    mem, vals=moran_mem(W)
    x=np.array([float(r['z_s_x']) for r in dat]);
    # BrainSpace-like singleton randomization: independently randomize signs of Moran eigen-coefficients.
    run_rows=[]; curve_rows=[]
    params=S4/'execution_config.json'
    # wrapper parameter file uses name/value rows
    for sid, seed in zip(range(6,11),m['seeds'][:5]):
        surrogate=singleton_surrogate(x,mem,vals,seed)
        with tempfile.TemporaryDirectory(prefix=f'd4b2_{sid}_') as td:
            td=Path(td); inp=td/'input.csv'; out=td/'out'; out.mkdir()
            fields=list(dat[0].keys())
            with open(inp,'w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
                for i,r in enumerate(dat):
                    q=dict(r); q['z_s_x']=repr(float(surrogate[i])); w.writerow(q)
            pf=td/'params.csv'; write_csv(pf,[{'name':'E','value':'3'},{'name':'tau','value':'1'},{'name':'b','value':'4'},{'name':'direction','value':'effect_xmap_cause'},{'name':'seed','value':str(seed)},{'name':'library_sizes','value':'10;18;33;60;109;198'}],['name','value'])
            cmd=[str(R),str(WRAP),str(inp),str(S4/'edges.csv'),str(CORE),str(pf),str(out),'estimate']
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
            if p.returncode!=0: raise RuntimeError(f'ID {sid} failed: {p.stdout}\n{p.stderr}')
            conv=list(csv.DictReader(open(out/'convergence.csv',encoding='utf-8')))[0]
            curve=list(csv.DictReader(open(out/'mean_curves.csv',encoding='utf-8')))
            mor=float(np.corrcoef(surrogate, W@surrogate)[0,1])
            for rr in curve: curve_rows.append({'surrogate_id':sid,'surrogate_seed':seed,'surrogate_morans_I':mor,**rr})
            run_rows.append({'surrogate_id':sid,'surrogate_seed':seed,'surrogate_morans_I':mor,**conv})
    write_csv(OUT/'batch_06_10_curves.csv',curve_rows,['surrogate_id','surrogate_seed','surrogate_morans_I','L','rho','feature','direction'])
    write_csv(OUT/'batch_06_10_summary.csv',run_rows,list(run_rows[0].keys()))
    old=list(csv.DictReader(open(PILOT/'gccm_diagnostics'/'D4B1_sx_spatial_null_smoke'/'sx_surrogate_summary.csv',encoding='utf-8')))
    old_norm=[{'rho_Lmax':r['terminal_rho'],'strength':r['raw_strength'],'converged':str(r['pass_flag'])} for r in old]
    new=old_norm+run_rows
    real=0.246623560555136
    vals2=np.array([float(r['rho_Lmax']) for r in new]); strengths=np.array([float(r['strength']) for r in new])
    interim={'n':10,'pass_count':sum(str(r['converged']).lower()=='true' for r in new),'terminal_rho_ge_real_count':int((vals2>=real).sum()),'terminal_rho_median':float(np.median(vals2)),'terminal_rho_max':float(vals2.max()),'real_sx_terminal_rho':real,'terminal_rho_range':[float(vals2.min()),float(vals2.max())],'raw_strength_ge_real_count':int((strengths>=real).sum())}
    (OUT/'interim_01_10_summary.json').write_text(json.dumps(interim,indent=2),encoding='utf-8')
    check={'overall_status':'PASS','future_seed_count':15,'future_ids_frozen':list(range(6,21)),'executed_ids':list(range(6,11)),'unexecuted_ids':list(range(11,21)),'new_gccm_runs':5,'library_sizes':[10,18,33,60,109,198],'graph_hash':sha(S4/'edges.csv'),'core_hash':sha(CORE),'restricted_input_hash':sha(S4/'restricted_input.csv'),'blinded_data_accessed':False,'permutations_1_5_rerun':False,'original_sx_rerun':False,'interim_summary':interim}
    (OUT/'check_batch_06_10.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text('''# D4B-2A：s_x 空间零假设批次 6–10\n\n本目录记录预注册的 6–20 种子以及本次仅执行的 6–10 五个 GCCM 运行。设置沿用 D4B-1：S4 scale_fit、历史 198 行 lib/pred、E=3、tau=1、b=4、固定库大小和收敛规则。ID 11–20 只预注册，尚未执行。未访问盲测标签。\n''',encoding='utf-8')
if __name__=='__main__': main()
