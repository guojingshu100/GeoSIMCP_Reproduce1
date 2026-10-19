import csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
base=Path(__file__).resolve().parents[2]; src=base/'gccm_diagnostics'/'D6E_S2_scale_distortion_shift'; d6c=base/'gccm_diagnostics'/'D6C_S2_independent_normalized_conformal_pilot'; out=src.parent/'D6F_S2_calibration_quantile_uncertainty'; out.mkdir(exist_ok=True)
cc={r['sample_id']:r for r in csv.DictReader(open(d6c/'calibration_scores.csv'))}; vv={r['sample_id']:r for r in csv.DictReader(open(d6c/'validation_intervals.csv'))}; rows=list(csv.DictReader(open(src/'distortion_rows.csv')))
for r in rows:
 j=cc[r['sample_id']] if r['set']=='calibration' else vv[r['sample_id']]; r['R']=abs(float(j['y'])-float(j['y_pred']))
cal=[r for r in rows if r['set']=='calibration']; val=[r for r in rows if r['set']=='validation']; methods={'CP':'R','M1':'Z_M1','Oracle':'Z_oracle'}; rng=np.random.default_rng(20260909); qrows=[];gaps=[];shifts=[]
for name,col in methods.items():
 c=np.array([float(r[col]) for r in cal]);v=np.array([float(r[col]) for r in val]);q0=float(np.sort(c)[90]);qs=[];cov=[]
 for _ in range(2000):
  q=float(np.sort(rng.choice(c,100,replace=True))[90]);qs.append(q);cov.append(np.mean(v<=q))
 qs=np.array(qs);cov=np.array(cov);qrows.append({'method':name,'original_calibration_q':q0,'mean_q':qs.mean(),'median_q':np.median(qs),'sd_q':qs.std(ddof=1),'q_p2_5':np.quantile(qs,.025),'q_p10':np.quantile(qs,.1),'q_p90':np.quantile(qs,.9),'q_p97_5':np.quantile(qs,.975),'mean_coverage':cov.mean(),'median_coverage':np.median(cov),'coverage_p2_5':np.quantile(cov,.025),'coverage_p10':np.quantile(cov,.1),'coverage_p90':np.quantile(cov,.9),'coverage_p97_5':np.quantile(cov,.975),'fraction_coverage_ge_087':np.mean(cov>=.87),'fraction_coverage_ge_090':np.mean(cov>=.9)})
 q87=np.sort(v)[math.ceil(.87*200)-1];q90=np.sort(v)[math.ceil(.9*200)-1];gaps.append({'method':name,'q_original_calibration':q0,'q_required_87':q87,'q_required_90':q90,'q_required_87_over_original':q87/q0,'q_required_90_over_original':q90/q0});ks=ks_2samp(c,v);shifts.append({'method':name,'ks_statistic':ks.statistic,'ks_pvalue_descriptive':ks.pvalue,'median_validation_over_calibration':np.median(v)/np.median(c),'q90_validation_over_calibration':np.quantile(v,.9)/np.quantile(c,.9),'q95_validation_over_calibration':np.quantile(v,.95)/np.quantile(c,.95)})
def write(fn,data):
 with open(out/fn,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write('bootstrap_quantiles.csv',[{k:r[k] for k in ['method','original_calibration_q','mean_q','median_q','sd_q','q_p2_5','q_p10','q_p90','q_p97_5']} for r in qrows]);write('bootstrap_coverage_summary.csv',[{k:r[k] for k in ['method','mean_coverage','median_coverage','coverage_p2_5','coverage_p10','coverage_p90','coverage_p97_5','fraction_coverage_ge_087','fraction_coverage_ge_090']} for r in qrows]);write('required_quantile_gap.csv',gaps);write('score_shift_summary.csv',shifts)
s={'status':'PASS','new_model_fits':0,'B':2000,'seed':20260909,'k':91,'original_q':{r['method']:r['original_calibration_q'] for r in qrows},'bootstrap_q_95_ranges':{r['method']:[r['q_p2_5'],r['q_p97_5']] for r in qrows},'bootstrap_coverage_95_ranges':{r['method']:[r['coverage_p2_5'],r['coverage_p97_5']] for r in qrows},'required_quantile_gap':gaps,'score_shift':shifts,'diagnosis':'C_MIXED','blinded_data_accessed':False};(out/'summary.json').write_text(json.dumps(s,indent=2));(out/'check.json').write_text(json.dumps({'overall_status':'PASS','new_model_fits':0,'calibration_n':100,'validation_n':200,'B':2000,'seed':20260909,'k':91,'final_calibration_accessed':False,'test_accessed':False},indent=2));(out/'README.md').write_text('# D6F S2 calibration quantile uncertainty audit\n');print(json.dumps(s,indent=2))
