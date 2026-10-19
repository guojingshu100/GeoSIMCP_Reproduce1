"""New-seed Stage4C analysis: replicate pairing, six primary tests, Holm control."""
from pathlib import Path
import csv,json,hashlib,shutil
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
METRICS=['coverage','width','interval_score','infinite_fraction','finite_fraction','finite_coverage']

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(v):
    a=np.asarray(v,float);n=len(a)
    if not np.isfinite(a).all():return float(np.mean(a)),float('nan'),float('nan')
    m=float(a.mean());se=0. if np.all(a==a[0]) else float(a.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan')
    return m,float(m-t.ppf(.975,n-1)*se),float(m+t.ppf(.975,n-1)*se)
def test_less(v):
    a=np.asarray(v,float)
    if not np.isfinite(a).all():return 1.,'NONFINITE_NOT_ESTIMABLE_RETAINED'
    if np.all(a==a[0]):return (0. if a[0]<0 else 1.),'CONSTANT_DIFFERENCE'
    statistic=float(a.mean()/(a.std(ddof=1)/np.sqrt(len(a))))
    return float(t.cdf(statistic,len(a)-1)),'ESTIMABLE'
def fmt(a,lo,hi):return f'{a:.5f} [{lo:.5f}, {hi:.5f}]'
def kfull(proto,s):return len(proto['full_parents'][s[1:]])
def kres(proto,s):return len(proto['residual_parents'][s[1:]])

def make(tag):
    proto=json.loads((HERE/'protocol_stage4c.json').read_text(encoding='utf8'))
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in proto['read_only_source_hashes'].items())
    assert all(hashlib.sha256((HERE/k).read_bytes()).hexdigest()==v for k,v in proto['code_sha256'].items())
    rows=read(HERE/'per_replicate_metrics.csv');masks=read(HERE/'random_mask_metrics.csv');ranking=read(HERE/'selector_rankings.csv')
    groups=defaultdict(list)
    for x in rows:groups[(x['scenario'],x['method'])].append(x)
    assert len({(x['scenario'],x['method'],x['replicate']) for x in rows})==len(rows)
    assert len({(x['scenario'],x['method'],x['replicate'],x['mask_id']) for x in masks})==len(masks)
    randomgroups=defaultdict(list)
    for x in masks:randomgroups[(x['scenario'],x['method'],x['replicate'])].append(x)
    for (s,m,r),g in randomgroups.items():
        assert len(g)==10
        avg=next(x for x in groups[(s,m+'-Mean10')] if x['replicate']==r)
        assert all(np.isclose(float(avg[k]),np.mean([float(x[k]) for x in g]),atol=1e-14,rtol=1e-14) for k in METRICS)
    summary=[]
    for (s,m),g in sorted(groups.items()):
        out=dict(scenario=s,method=m,replicates=len(g),inference_unit='replicate',data='FRESH_STAGE4C')
        for k in METRICS:
            mean,lo,hi=ci([float(x[k]) for x in g]);out.update({k+'_mean':mean,k+'_ci95_low':lo,k+'_ci95_high':hi})
        lo,hi=out['coverage_ci95_low'],out['coverage_ci95_high']
        out['coverage_vs_90']='CI_ENTIRELY_BELOW_90' if hi<.90 else 'CI_ENTIRELY_ABOVE_90' if lo>.90 else 'CI_INCLUDES_90_NOT_ESTABLISHED'
        a=np.array([float(x['coverage']) for x in g]);se=float(a.std(ddof=1)/np.sqrt(len(a)))
        out['coverage_one_sided95_lower']=float(a.mean()-t.ppf(.95,len(a)-1)*se)
        out['practical_89_screen_only']=out['coverage_one_sided95_lower']>=.89
        summary.append(out)
    lookup={(x['scenario'],x['method']):x for x in summary};pairs=[]
    for s in ['S%d'%i for i in range(6)]:
        kf,kr=kfull(proto,s),kres(proto,s)
        comparisons=[('GCCP-ResidualParent-RC','RandomMask-k%d-RC-Mean10'%kr),('GCCP-ResidualParent-RC','SupervisedSelector-k%d-RC'%kr),
            ('GCCP-OracleParent-RC','RandomMask-k%d-RC-Mean10'%kf),('GCCP-OracleParent-RC','SupervisedSelector-k%d-RC'%kf),
            ('GCCP-ResidualParent-RC','GCCP-OracleParent-RC')]
        comparisons += [(m,'GeoSIMCP-RC') for (case,m) in groups if case==s and m!='GeoSIMCP-RC']
        if s=='S5':comparisons.append(('GCCP-WrongPrior-RC','GCCP-OracleParent-RC'))
        for method,base in sorted(set(comparisons)):
            a={x['replicate']:x for x in groups[(s,method)]};b={x['replicate']:x for x in groups[(s,base)]};assert set(a)==set(b)
            ids=sorted(a,key=int)
            primary=s in ['S2','S3','S4'] and method=='GCCP-ResidualParent-RC' and base in ['RandomMask-k1-RC-Mean10','SupervisedSelector-k1-RC']
            out=dict(scenario=s,method=method,baseline=base,paired_replicates=len(ids),is_primary=primary,
                primary_family='S2_S4_residual_IS_6' if primary else 'SECONDARY_DESCRIPTIVE',ci='UNADJUSTED_TWO_SIDED_T95',p_one_sided_score='',p_holm_score='',holm_reject_final=False)
            for k in METRICS:
                d=[float(a[i][k])-float(b[i][k]) for i in ids];mean,lo,hi=ci(d)
                out.update({k+'_delta_mean':mean,k+'_delta_ci95_low':lo,k+'_delta_ci95_high':hi})
            if primary:
                p,status=test_less([float(a[i]['interval_score'])-float(b[i]['interval_score']) for i in ids]);out['p_one_sided_score']=p;out['test_status']=status
            else:out['test_status']='NOT_PRIMARY'
            pairs.append(out)
    primary=[x for x in pairs if x['is_primary']];assert len(primary)==6
    ordered=sorted(primary,key=lambda x:x['p_one_sided_score']);previous=0.
    final=all(x['paired_replicates']==100 for x in primary)
    for i,x in enumerate(ordered):
        adj=min(1.,max(previous,(6-i)*x['p_one_sided_score']));previous=adj
        x['p_holm_score']=adj;x['holm_reject_final']=bool(final and adj<=.05)
        x['analysis_stage']='FINAL_CONFIRMATORY_100' if final else 'INTERIM_DESCRIPTIVE_NO_REJECTION'
    # Keep identical column set for secondary rows.
    for x in pairs:x.setdefault('analysis_stage','SECONDARY_DESCRIPTIVE')
    write(HERE/'summary_metrics.csv',summary);write(HERE/'paired_differences.csv',pairs);write(HERE/'primary_holm_tests.csv',primary)
    # Feature selection frequencies use the recorded developer-data ranking, never final labels.
    frequency=[]
    for s in ['S%d'%i for i in range(6)]:
        for k in sorted({kfull(proto,s),kres(proto,s)}):
            g=groups[(s,'SupervisedSelector-k%d-RC'%k)]
            counter=Counter(n for x in g for n in x['supplied_mask'].split(';') if n)
            for name in sorted(counter):frequency.append(dict(scenario=s,k=k,feature=name,selected_replicates=counter[name],replicates=len(g),selection_frequency=counter[name]/len(g),target='Y_0.9_QUANTILE_DFIT_DLOC'))
            if not k:frequency.append(dict(scenario=s,k=0,feature='EMPTY_FALLBACK',selected_replicates=len(g),replicates=len(g),selection_frequency=1.,target='NO_SELECTION_K0'))
    write(HERE/'selection_frequency.csv',frequency)
    overlap=[]
    for (s,m,r),g in sorted(randomgroups.items()):
        overlap.append(dict(scenario=s,method=m,replicate=r,masks=10,distinct_masks=len({x['supplied_mask'] for x in g}),
            mean_parent_overlap=float(np.mean([float(x['parent_overlap']) for x in g])),scale_driver_fraction=float(np.mean([float(x['contains_scale_driver']) for x in g])),
            mean_width=float(np.mean([float(x['width']) for x in g])),mean_interval_score=float(np.mean([float(x['interval_score']) for x in g]))))
    write(HERE/'random_mask_overlap.csv',overlap)
    receipts=[json.loads(p.read_text(encoding='utf8')) for p in (HERE/'receipts/confirmatory').glob('r*_S*.json')]
    assert len(receipts)==len({(x['scenario'],x['replicate']) for x in rows})
    assert all(x['completed'] and x['q0_frozen_before_Dcal'] and x['intervals_frozen_before_Deval'] and not x['truth_in_predictor'] and all((r!='D_cal' or state=='q0_frozen') and (r!='D_eval' or state=='intervals_frozen') for r,state in x['label_access']) for x in receipts)
    done={(x['phase'],int(x['replicate']),x['scenario']) for x in receipts}
    for p in (HERE/'receipts/smoke').glob('r*_S*.json'):
        x=json.loads(p.read_text(encoding='utf8'))
        if x.get('completed'):done.add((x['phase'],x['replicate'],x['scenario']))
    seeds=read(HERE/'seed_manifest.csv')
    for x in seeds:
        complete=(x['phase'],int(x['replicate']),x['scenario']) in done
        x['actual_use']='UNUSED_BY_PROTOCOL' if complete and x['planned_use'].startswith(('RESERVED','EMPTY')) else 'USED' if complete else 'PENDING'
    write(HERE/'seed_manifest.csv',seeds)
    figs=HERE/'figures';figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,base,label in zip(axes,['RandomMask-k1-RC-Mean10','SupervisedSelector-k1-RC'],['Residual parents minus random k=1 mean','Residual parents minus supervised k=1']):
        g=[next(x for x in primary if x['scenario']==s and x['baseline']==base) for s in ['S2','S3','S4']]
        means=np.array([x['interval_score_delta_mean'] for x in g]);lo=np.array([x['interval_score_delta_ci95_low'] for x in g]);hi=np.array([x['interval_score_delta_ci95_high'] for x in g])
        ax.errorbar(range(3),means,yerr=np.maximum(0,np.array([means-lo,hi-means])),fmt='o',capsize=5,color='#2166AC')
        ax.axhline(0,color='black',ls='--',lw=.8);ax.set_xticks(range(3),[x['scenario']+(' *' if x['holm_reject_final'] else '') for x in g]);ax.set_ylabel('Paired interval score difference (two-sided95 CI)');ax.set_title(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fresh Stage4C primary comparisons; * Holm FWER .05 rejection at R=100 only')
    fig.savefig(figs/'causal_vs_supervised.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    palette=plt.get_cmap('tab10').colors
    for i,ax in enumerate(axes.flat):
        s='S%d'%i;kf,kr=kfull(proto,s),kres(proto,s)
        methods=['GeoSIMCP-RC','GCCP-OracleParent-RC','GCCP-ResidualParent-RC']
        methods += ['RandomMask-k%d-RC-Mean10'%k for k in sorted({kf,kr})]+['SupervisedSelector-k%d-RC'%k for k in sorted({kf,kr})]
        if s=='S5':methods+=['GCCP-WrongPrior-RC']
        for j,m in enumerate(methods):
            a=lookup[(s,m)];x=a['coverage_mean'];y=a['width_mean']
            ax.errorbar(x,y,xerr=[[max(0,x-a['coverage_ci95_low'])],[max(0,a['coverage_ci95_high']-x)]],yerr=[[max(0,y-a['width_ci95_low'])],[max(0,a['width_ci95_high']-y)]],fmt='o',color=palette[j],label=m)
        ax.axvline(.90,color='black',ls='--',lw=.8);ax.set_title(s);ax.set_xlabel('Coverage, replicate t95 CI');ax.set_ylabel('Full width, replicate t95 CI')
    handles,labels=axes.flat[2].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Fresh independent samples; same RF, roles, kernel parameters and final rank')
    fig.savefig(figs/'coverage_efficiency.png',dpi=170);plt.close(fig)
    counts={s:len(groups[(s,'GeoSIMCP-RC')]) for s in ['S%d'%i for i in range(6)]}
    report=['# Stage4C independent-seed selector confirmation','',f'Checkpoint: {tag}; complete repeats by scenario {counts}; metric rows {len(rows)}, single-mask rows {len(masks)}.',
        'Formal claims use only the planned R=100 six-test primary family. R=20 is descriptive. No reused Stage4/4B samples, outcome arrays or performance files; only code/protocol and two explicit seed manifests.',
        '', '## Selector candidate and interpretation',
        'New HistGradientBoostingRegressor loss=quantile q=.9, 200 iterations, learning_rate .05, max_leaf_nodes15, min_leaf20, L2 .1, early_stopping False; full parameters in protocol. Target is observed Y upper quantile, not absolute RF residual. Fit D_fit1600, evaluate base/permuted pinball on D_loc800. Ten registered permutations per each of20 non-coordinate attributes, top-k exact with feature-name tie-break; no sign filter. Coordinates remain fixed. No D_cal/D_eval access during selection.',
        'Common point predictor remains original RF300 fit on D_fit with all22 observed inputs. Selector only changes localization attributes. D_loc labels legitimately support both supervised selection and q0; independent D_cal is used only after selection/q0 freeze. This differs from the old unimplemented D_fit-internal residual selector and is not causal discovery.',
        '', '## Primary Interval Score comparisons',
        'Delta=GCCP-ResidualParent-RC minus comparator. One-sided paired t alternative mean delta<0. Holm over exactly six S2-S4 comparisons, FWER .05. Displayed95 CIs are unadjusted two-sided; no CI-of-coverage-difference equivalence claim.',
        '| Scenario | Comparator | Score delta [t95] | Raw one-sided p | Holm p | Confirmatory rejection |','|---|---|---|---|---|---|']
    for x in primary:report.append('| %s | %s | %s | %.6g | %.6g | %s |'%(x['scenario'],x['baseline'],fmt(x['interval_score_delta_mean'],x['interval_score_delta_ci95_low'],x['interval_score_delta_ci95_high']),x['p_one_sided_score'],x['p_holm_score'],x['holm_reject_final']))
    report+=['','## Absolute coverage and efficiency',
        '| Scenario | Method | Coverage [t95] | Width | Interval Score | Coverage vs .90 |','|---|---|---|---|---|---|']
    for a in summary:report.append('| %s | %s | %s | %.5f | %.5f | %s |'%(a['scenario'],a['method'],fmt(a['coverage_mean'],a['coverage_ci95_low'],a['coverage_ci95_high']),a['width_mean'],a['interval_score_mean'],a['coverage_vs_90']))
    report+=['A CI including .90 is compatible with the target but does not establish coverage>=.90. A CI wholly below .90 flags undercoverage. One-sided lower>=.89 is only the separately reported practical screen. Score-test rejection is not automatically a coverage-qualified efficiency or causal-identification conclusion.',
        '', '## Selection frequencies (developer labels only)',
        '| Scenario | k | Feature | Selected / repeats |','|---|---|---|---|']
    for a in frequency:report.append('| %s | %d | %s | %d / %d |'%(a['scenario'],a['k'],a['feature'],a['selected_replicates'],a['replicates']))
    maxinf=max(float(x['infinite_fraction']) for x in rows+masks)
    report+=['',f'All predeclared masks retained; ten metrics averaged within replicate, no mask-level pseudoreplication. Max infinite fraction {maxinf}; finite-conditional coverage and finite fractions are in metrics and summaries. No difficult-query removal.',
        'S0 validates exact ordinary fallback; all-k0 skips an irrelevant selector fit. S1 residual empty means ordinary GeoSIM fallback, not CP or an oracle-mean predictor. S5 correct full/residual parents and wrong {X1,X3} are distinct methods.',
        '', '## Limits of the conclusion',
        'Oracle residual sets are known SCM scale drivers, not proven full parents of fitted RF residuals. Beating this observed-Y upper-quantile selector can reflect better alignment with residual uncertainty; it does not show superiority over all supervised residual-aware selectors. Full-parent/other-scenario/width/feature-frequency analyses are secondary descriptive and not covered by the six-test Holm claim. No real-data causal or dependent-spatial validation.',
        '', '## Reproduction',
        'From project root: python experiments/gccp_stage4c_selector_confirmation_20261019_v1/experiment.py prepare; smoke --workers6; batch --start0 --stop20 --workers6; report --tag preliminary20; batch --start20 --stop100 --workers6; report --tag final100. Supply spaces between options and numbers (e.g. --workers 6). Prepare refuses overwrite. Exact repeat uses the recorded source, software, namespace and roots in a fresh directory.']
    (HERE/'RUN_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    integrity=['# Stage4C integrity',f'Checkpoint {tag}; counts {counts}; completed receipts {len(receipts)}.',
        'Independent fresh samples: confirmation roots2026102000..2026102099 and namespace2026101906; smoke root2026102999 stage0. All new entropy keys and roots differ from Stage4 and4B explicit manifests; model/selector uint32 seeds are unique and do not reuse old registered uint32 values. Scope-limited, not global project uniqueness certification.',
        'No old performance CSV, original/evaluation arrays, sealed labels, real data, network or installations. Read-only sources are the exact code/protocol and two seed manifests recorded in protocol. No old files modified; bytecode disabled. Private imported generator module receives a new RNG binding only in memory; SCM body unchanged.',
        'Every method within replicate/scenario shares same fresh4000 units and disjoint roles1600/800/800/800, same common RF and frozen D_fit transform. Twenty observed attributes plus positions only; no innovation/mu/sigma/truth columns in learner.',
        'Selector receives D_fit and D_loc arrays only. Permutation changes copies of one non-coordinate D_loc column; original common covariate hash unchanged. All rankings/top-k masks and all q0 hashes stored before D_cal access; all intervals stored before D_eval labels. Receipts verify access states. No outcome-driven parameter, k, algorithm, mask or seed replacement.',
        'Mask definitions frozen before new metrics in immutable random_mask_registry.json; child streams in seed_manifest.csv. Unique k families share one mask bank if k_full=k_residual. Cross-mask duplicates and accidental parent overlaps retained. Random means are per-replicate averages of metric values, not averaged intervals or extra independent replicates.',
        f'All six primary comparisons retained; Holm family size6, paired independent replicates. Final inference only at100; checkpoint20 descriptive. No bootstrap. Infinity max={maxinf}; finite-conditional metrics explicit.',
        'Software quantile support verified by the registered fresh six-scenario smoke, no learner substitution. k0 fallback/kernel parity and labels blocked before final stages were checked. S0 all method intervals exactly equal; S1 residual-empty equals ordinary matched baseline.',
        'Old source fingerprints and new locked executable/mask-registry fingerprints verified; no repetition of Stage2 eight-source audit. New cache contains only transformed observed covariates and frozen generated residual/evaluation arrays, written after interval freeze.']
    (HERE/'INTEGRITY_REPORT.md').write_text('\n\n'.join(integrity)+'\n',encoding='utf8')
    snap=HERE/'checkpoints'/tag;snap.mkdir(parents=True,exist_ok=False)
    for name in ['protocol_stage4c.json','seed_manifest.csv','per_replicate_metrics.csv','random_mask_metrics.csv','selector_rankings.csv','summary_metrics.csv','paired_differences.csv','primary_holm_tests.csv','selection_frequency.csv','random_mask_overlap.csv','RUN_REPORT.md','INTEGRITY_REPORT.md']:
        shutil.copy2(HERE/name,snap/name)
    shutil.copytree(figs,snap/'figures')
    status=dict(state='COMPLETED_100' if final else 'PRELIMINARY',counts=counts,metric_rows=len(rows),random_rows=len(masks),ranking_rows=len(ranking),primary_rejections=sum(x['holm_reject_final'] for x in primary),checkpoint=tag)
    (HERE/'execution_status.json').write_text(json.dumps(status,indent=2),encoding='utf8');print(json.dumps(status),flush=True)
    for x in primary:print(x['scenario'],x['baseline'],fmt(x['interval_score_delta_mean'],x['interval_score_delta_ci95_low'],x['interval_score_delta_ci95_high']),'Holm=',x['p_holm_score'],'final_reject=',x['holm_reject_final'],flush=True)
