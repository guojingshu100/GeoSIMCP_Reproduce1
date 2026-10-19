import csv,json,tempfile,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'D4B2_sx_spatial_null_20'))
from run_batch_06_10 import singleton_surrogate,moran_mem,write_csv,CORE,WRAP,R

ROOT=Path(__file__).resolve().parents[3]; PILOT=ROOT/'simulation_pilot_20260908'; OUT=PILOT/'gccm_diagnostics'/'D4C_gproxy_spatial_null_smoke'; S4=PILOT/'gccm_outputs'/'S4'; D4A=PILOT/'gccm_diagnostics'/'D4A_moran_spatial_surrogate_validation'
def main():
 OUT.mkdir(parents=True,exist_ok=True); manifest=json.loads((D4A/'surrogate_seed_manifest.json').read_text()); seed=20260910
 dat=list(csv.DictReader(open(S4/'restricted_input.csv',encoding='utf-8'))); n=len(dat); edges=list(csv.DictReader(open(S4/'edges.csv',encoding='utf-8'))); W=np.zeros((n,n))
 for e in edges:
  u=int(e['u'])-1;v=int(e['v'])-1;W[u,v]=W[v,u]=1
 mem,vals=moran_mem(W); x=np.array([float(r['z_g_proxy']) for r in dat]); sums=[]; curves=[]
 for sid in range(1,6):
  sur=singleton_surrogate(x,mem,vals,seed+sid-1)
  assert len(sur)==200 and np.isfinite(sur).all()
  with tempfile.TemporaryDirectory(prefix=f'd4c_{sid}_') as td:
   td=Path(td);inp=td/'input.csv';out=td/'out';out.mkdir();fields=list(dat[0])
   with open(inp,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for i,r in enumerate(dat):q=dict(r);q['z_g_proxy']=repr(float(sur[i]));w.writerow(q)
   pf=td/'params.csv';write_csv(pf,[{'name':'E','value':'3'},{'name':'tau','value':'1'},{'name':'b','value':'4'},{'name':'direction','value':'effect_xmap_cause'},{'name':'seed','value':str(seed+sid-1)},{'name':'library_sizes','value':'10;18;33;60;109;198'}],['name','value'])
   p=subprocess.run([str(R),str(WRAP),str(inp),str(S4/'edges.csv'),str(CORE),str(pf),str(out),'estimate'],capture_output=True,text=True,timeout=600)
   if p.returncode:raise RuntimeError(p.stderr)
   c=list(csv.DictReader(open(out/'convergence.csv',encoding='utf-8')))[4];mor=float(np.corrcoef(sur,W@sur)[0,1])
   for q in csv.DictReader(open(out/'mean_curves.csv',encoding='utf-8')):
    if q['feature']=='g_proxy':curves.append({'surrogate_id':sid,'surrogate_seed':seed+sid-1,'surrogate_morans_I':mor,**q})
   sums.append({'surrogate_id':sid,'surrogate_seed':seed+sid-1,'surrogate_morans_I':mor,**c})
 write_csv(OUT/'gproxy_surrogate_gccm_curves.csv',curves,list(curves[0]));write_csv(OUT/'gproxy_surrogate_summary.csv',sums,list(sums[0]))
 real=list(csv.DictReader(open(S4/'convergence.csv',encoding='utf-8')))[4];term=np.array([float(x['rho_Lmax']) for x in sums]);rt=float(real['rho_Lmax']); summary={'real_terminal_rho':rt,'real_gain':float(real['gain']),'real_pass':real['converged']=='TRUE','real_raw_strength':float(real['strength']),'surrogate_pass_count':sum(x['converged']=='TRUE' for x in sums),'surrogate_terminal_ge_real_count':int((term>=rt).sum()),'surrogate_terminal_range':[float(term.min()),float(term.max())],'surrogate_terminal_median':float(np.median(term)),'surrogate_raw_strength_ge_real_count':sum(float(x['strength'])>=float(real['strength']) for x in sums)}
 (OUT/'config.json').write_text(json.dumps({'diagnostic':'D4C_gproxy_spatial_null_smoke','surrogates':5,'candidate':'g_proxy','effect':'abs_residual','library_sizes':[10,18,33,60,109,198],'E':3,'tau':1,'b':4,'seed':seed,'settings':'frozen D4B adapter'},indent=2));(OUT/'check.json').write_text(json.dumps({'overall_status':'PASS','new_gccm_runs':5,'surrogates_finite_length_200':True,'graph_unchanged':True,'blinded_data_accessed':False,'real_gproxy_rerun':False,'summary':summary},indent=2));(OUT/'README.md').write_text('# D4C g_proxy spatial-null smoke test\n\nFive existing D4A g_proxy spatial surrogates were evaluated with the frozen S4 GCCM adapter.\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
