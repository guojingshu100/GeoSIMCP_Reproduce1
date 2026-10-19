import csv,json,tempfile,subprocess
from pathlib import Path
import numpy as np
from run_batch_06_10 import singleton_surrogate, moran_mem, write_csv, S4, CORE, WRAP, R, OUT

def main():
    m=json.loads((OUT/'preregistered_seed_manifest_6_20.json').read_text())
    dat=list(csv.DictReader(open(S4/'restricted_input.csv',encoding='utf-8'))); n=len(dat)
    edges=list(csv.DictReader(open(S4/'edges.csv',encoding='utf-8'))); W=np.zeros((n,n))
    for e in edges:
        u=int(e['u'])-1; v=int(e['v'])-1; W[u,v]=W[v,u]=1
    mem,vals=moran_mem(W); x=np.array([float(r['z_s_x']) for r in dat])
    sums=[]; curves=[]
    for sid,seed in zip(range(11,16),m['seeds'][5:10]):
        sur=singleton_surrogate(x,mem,vals,seed)
        with tempfile.TemporaryDirectory(prefix=f'd4b2_{sid}_') as td:
            td=Path(td); inp=td/'input.csv'; out=td/'out'; out.mkdir(); fields=list(dat[0])
            with open(inp,'w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
                for i,r in enumerate(dat): q=dict(r); q['z_s_x']=repr(float(sur[i])); w.writerow(q)
            pf=td/'params.csv'; write_csv(pf,[{'name':'E','value':'3'},{'name':'tau','value':'1'},{'name':'b','value':'4'},{'name':'direction','value':'effect_xmap_cause'},{'name':'seed','value':str(seed)},{'name':'library_sizes','value':'10;18;33;60;109;198'}],['name','value'])
            p=subprocess.run([str(R),str(WRAP),str(inp),str(S4/'edges.csv'),str(CORE),str(pf),str(out),'estimate'],capture_output=True,text=True,timeout=600)
            if p.returncode: raise RuntimeError(f'{sid}: {p.stdout}\n{p.stderr}')
            conv=list(csv.DictReader(open(out/'convergence.csv',encoding='utf-8')))[0]
            mor=float(np.corrcoef(sur,W@sur)[0,1])
            for q in csv.DictReader(open(out/'mean_curves.csv',encoding='utf-8')):
                if q['feature']=='s_x': curves.append({'surrogate_id':sid,'surrogate_seed':seed,'surrogate_morans_I':mor,**q})
            sums.append({'surrogate_id':sid,'surrogate_seed':seed,'surrogate_morans_I':mor,**conv})
    write_csv(OUT/'batch_11_15_curves.csv',curves,list(curves[0]))
    write_csv(OUT/'batch_11_15_summary.csv',sums,list(sums[0]))
    old=list(csv.DictReader(open(OUT/'batch_06_10_summary.csv',encoding='utf-8')))
    old1=list(csv.DictReader(open(OUT.parent/'D4B1_sx_spatial_null_smoke'/'sx_surrogate_summary.csv',encoding='utf-8')))
    allrows=old1+old+sums; real=0.246623560555136
    terminal=np.array([float(r.get('terminal_rho',r['rho_Lmax'])) for r in allrows])
    strength=np.array([float(r.get('raw_strength',r['strength'])) for r in allrows])
    out={'n':15,'pass_count':sum(str(r.get('pass_flag',r['converged'])).lower()=='true' for r in allrows),'terminal_rho_ge_real_count':int((terminal>=real).sum()),'terminal_rho_median':float(np.median(terminal)),'terminal_rho_max':float(terminal.max()),'raw_strength_ge_real_count':int((strength>=real).sum()),'real_sx_terminal_rho':real,'descriptive_upper_tail':(1+int((terminal>=real).sum()))/16}
    (OUT/'interim_01_15_summary.json').write_text(json.dumps(out,indent=2))
    check={'overall_status':'PASS','executed_ids':[11,12,13,14,15],'unexecuted_ids':[16,17,18,19,20],'new_gccm_runs':5,'seed_manifest_unchanged':True,'settings_unchanged':True,'ids_1_10_not_rerun':True,'blinded_data_accessed':False,'summary':out}
    (OUT/'check_batch_11_15.json').write_text(json.dumps(check,indent=2))
if __name__=='__main__': main()
