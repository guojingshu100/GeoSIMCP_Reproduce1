import json, hashlib, math
from pathlib import Path
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
BASE=Path(r'F:\GeoSIMCP_Reproduce\simulation_pilot_20260908'); OUT=BASE/'gccm_diagnostics'/'D6R_S2_HGBR_clean_replication'; OUT.mkdir(exist_ok=True)
FEATS=['s_x','s_y','x1','x2','g_proxy','x_noise']; TARGET='abs_residual'
obs=pd.read_csv(BASE/'data'/'S2_observed.csv'); prep=pd.read_csv(BASE/'scale_outputs'/'S2_scale_fit_preparation.csv'); ids=prep['sample_id'].tolist(); tr=obs.set_index('sample_id').loc[ids].reset_index(); tr[TARGET]=abs(tr['y']-2*tr['x1'])
training_bytes=tr[['sample_id']+FEATS+[TARGET]].to_csv(index=False).encode(); train_hash=hashlib.sha256(training_bytes).hexdigest()
est=HistGradientBoostingRegressor(loss='quantile',quantile=0.5,early_stopping=False); est.fit(tr[FEATS],tr[TARGET])
(OUT/'config.json').write_text(json.dumps({'diagnostic':'D6R_S2_HGBR_clean_replication','alpha':0.1,'replicates':20,'calibration_pool_n':400,'evaluation_n':1000,'n_cal':[50,100,200,400],'methods':['CP','HGBR-M1-NCP','Oracle-NCP'],'D6G_excluded':True,'D6H_excluded':True},indent=2))
(OUT/'estimator_identity.json').write_text(json.dumps({'class':est.__class__.__name__,'module':est.__class__.__module__,'get_params':est.get_params(deep=True),'sklearn_version':sklearn.__version__,'features':FEATS,'target':TARGET,'scale_fit_n':200,'number_of_model_fits':1},indent=2))
(OUT/'training_data_identity.json').write_text(json.dumps({'source_scale_fit_path':str(BASE/'scale_outputs'/'S2_scale_fit_preparation.csv'),'row_count':200,'row_ids':ids,'training_content_sha256':train_hash},indent=2))
seeds=[int(x) for x in np.random.SeedSequence(20260911).generate_state(20,dtype=np.uint32)]; (OUT/'replicate_seed_manifest.json').write_text(json.dumps({'start_seed':20260911,'replicate_ids':list(range(1,21)),'seeds':seeds,'frozen_before_results':True},indent=2))
runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); (OUT/'runner_identity.json').write_text(json.dumps({'runner_path':str(Path(__file__).resolve()),'runner_sha256':runner_hash,'random_state':'HGBR default None; replicate NumPy Generator seeds','point_predictor_artifact':str(BASE/'predictor_outputs'/'S2_model.joblib'),'point_predictor_config':str(BASE/'predictor'/'predictor_config_v1.json'),'model_fit_count':1},indent=2))
model=joblib.load(BASE/'predictor_outputs'/'S2_model.joblib')
def make(seed,n):
 r=np.random.default_rng(seed); sx=r.uniform(0,1,n); sy=r.uniform(0,1,n); x1=r.normal(0,1,n); x2=r.normal(0,1,n); eta=r.normal(0,1,n); xn=r.normal(0,1,n); eps=r.normal(0,1,n); gt=np.sin(2*np.pi*sx)+np.cos(2*np.pi*sy); gp=gt+0.35*eta; sig=0.6+1.4/(1+np.exp(-1.2*x2)); y=2*x1+sig*eps; return pd.DataFrame({'s_x':sx,'s_y':sy,'x1':x1,'x2':x2,'g_proxy':gp,'x_noise':xn,'y':y,'sigma_true':sig})
def pred(d): return model.predict(d[FEATS])
M=[]; Q=[]
for rid,seed in enumerate(seeds,1):
 c=make(seed,400); e=make(seed+1000000007,1000); mc=pred(c); me=pred(e); cs=est.predict(c[FEATS]); es=est.predict(e[FEATS]);
 if not np.isfinite(cs).all() or not np.isfinite(es).all() or (cs<=0).any() or (es<=0).any(): raise RuntimeError('positivity check failed')
 qi=pd.qcut(e.x2,5,labels=False)+1
 for nc in [50,100,200,400]:
  k=math.ceil((nc+1)*0.90); methods={'CP':(abs(c.y-mc),abs(e.y-me),np.ones(1000)),'HGBR-M1-NCP':(abs(c.y-mc)/cs,abs(e.y-me)/es,es),'Oracle-NCP':(abs(c.y-mc)/c.sigma_true,abs(e.y-me)/e.sigma_true,e.sigma_true)}
  for name,(a,z,w) in methods.items():
   q=float(np.sort(a.iloc[:nc] if hasattr(a,'iloc') else a[:nc])[k-1]); half=q*w; lo=me-half; hi=me+half; covered=(e.y>=lo)&(e.y<=hi); iscore=2*half+20*(lo-e.y)*(e.y<lo)+20*(e.y-hi)*(e.y>hi); qc=[]; mw=[]
   for qid in range(1,6):
    ix=qi==qid; qc.append(float(covered[ix].mean())); mw.append(float((2*half[ix]).mean())); Q.append({'replicate_id':rid,'seed':seed,'n_cal':nc,'method':name,'x2_quintile':qid,'n':int(ix.sum()),'coverage':qc[-1],'mean_width':mw[-1]})
   M.append({'replicate_id':rid,'seed':seed,'n_cal':nc,'method':name,'q':q,'coverage':float(covered.mean()),'mean_width':float((2*half).mean()),'median_width':float(np.median(2*half)),'interval_score':float(iscore.mean()),'coverage_range':max(qc)-min(qc),'mean_abs_quintile_dev':float(np.mean(np.abs(np.array(qc)-.9))),'high_low_mean_width_ratio':max(mw)/min(mw)})
md=pd.DataFrame(M); qd=pd.DataFrame(Q); md.to_csv(OUT/'replicate_metrics.csv',index=False); qd.to_csv(OUT/'replicate_quintile_metrics.csv',index=False)
rows=[]
for (nc,name),g in md.groupby(['n_cal','method']): rows.append({'n_cal':nc,'method':name,'coverage_mean':g.coverage.mean(),'coverage_median':g.coverage.median(),'coverage_sd':g.coverage.std(ddof=1),'coverage_min':g.coverage.min(),'coverage_max':g.coverage.max(),'coverage_mean_abs_dev':np.mean(abs(g.coverage-.9)),'coverage_fraction_ge_087':np.mean(g.coverage>=.87),'coverage_fraction_ge_090':np.mean(g.coverage>=.9),'q_mean':g.q.mean(),'q_sd':g.q.std(ddof=1),'q_cv':g.q.std(ddof=1)/g.q.mean(),'mean_width':g.mean_width.mean(),'mean_interval_score':g.interval_score.mean(),'mean_x2_coverage_range':g.coverage_range.mean(),'mean_abs_x2_coverage_deviation':g.mean_abs_quintile_dev.mean()})
su=pd.DataFrame(rows); su.to_csv(OUT/'calibration_size_summary.csv',index=False)
eff=[]
for nc in [50,100,200,400]:
 cp=md[(md.n_cal==nc)&(md.method=='CP')].set_index('replicate_id'); hg=md[(md.n_cal==nc)&(md.method=='HGBR-M1-NCP')].set_index('replicate_id'); oc=md[(md.n_cal==nc)&(md.method=='Oracle-NCP')].set_index('replicate_id'); d=hg.interval_score-cp.interval_score; eff.append({'n_cal':nc,'mean_delta_IS_HGBR_CP':d.mean(),'median_delta_IS_HGBR_CP':d.median(),'sd_delta_IS_HGBR_CP':d.std(ddof=1),'n_HGBR_IS_lt_CP':int((d<0).sum()),'n_HGBR_IS_gt_CP':int((d>0).sum()),'mean_pct_IS_diff':float(np.mean(100*d/cp.interval_score)),'mean_delta_IS_Oracle_CP':float((oc.interval_score-cp.interval_score).mean())})
pd.DataFrame(eff).to_csv(OUT/'hgbt_vs_cp_efficiency.csv',index=False)
a=su[(su.n_cal==100)&(su.method=='HGBR-M1-NCP')].iloc[0]; b=su[(su.n_cal==100)&(su.method=='CP')].iloc[0]; aa=su[(su.n_cal==400)&(su.method=='HGBR-M1-NCP')].iloc[0]; bb=su[(su.n_cal==400)&(su.method=='CP')].iloc[0]; criteria={'1_mean_HGBR_coverage_885_925':bool(.885<=a.coverage_mean<=.925),'2_mean_range_lower_than_CP':bool(a.mean_x2_coverage_range<b.mean_x2_coverage_range),'3_mean_IS_le_CP':bool(a.mean_interval_score<=b.mean_interval_score),'4_n400_range_lower_than_CP':bool(aa.mean_x2_coverage_range<bb.mean_x2_coverage_range)}; cls='A' if all(criteria.values()) else ('B' if criteria['1_mean_HGBR_coverage_885_925'] and criteria['2_mean_range_lower_than_CP'] and not criteria['3_mean_IS_le_CP'] else ('C' if not criteria['1_mean_HGBR_coverage_885_925'] or not criteria['2_mean_range_lower_than_CP'] else 'D'))
(OUT/'summary.json').write_text(json.dumps({'status':'PASS','classification':cls,'criteria':criteria,'D6G_historical_not_used_as_HGBR_evidence':True,'D6H_historical_KNN_deviation_not_used_as_HGBR_evidence':True,'clean_replacement_for_D6G_D6H':True},indent=2)); (OUT/'check.json').write_text(json.dumps({'status':'PASS','new_hgbr_fits':1,'completed_replicates':20,'positivity_check':True,'final_calibration_accessed':False,'test_accessed':False,'gccm_run':False,'tuning_performed':False},indent=2)); (OUT/'README.md').write_text('# D6R S2 HGBR normalized-conformal clean replication\n')
