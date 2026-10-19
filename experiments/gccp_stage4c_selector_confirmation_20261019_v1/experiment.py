"""Fresh-seed Stage4C confirmation; observed-Y quantile permutation selector candidate."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE.parent/'gccp_stage4_execution_20261019_v1'
OLD_B=HERE.parent/'gccp_stage4b_causal_specificity_20261019_v1'
os.environ['MPLCONFIGDIR']=str(HERE/'runtime_cache'/'matplotlib')
import json,csv,hashlib,time,argparse,functools,importlib.util,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import scipy,sklearn
from scipy.spatial.distance import cdist,pdist
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from threadpoolctl import threadpool_limits
spec=importlib.util.spec_from_file_location('stage4c_shared',OLD/'scientific.py')
sc=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sc;spec.loader.exec_module(sc)
NAMESPACE=2026101906
CONFIRM_BASE=2026102000
SMOKE_ROOT=2026102999
PURPOSES={'attributes':1,'coordinates':2,'innovation':3,'proxies':4,'split':5,'model':6,'mask':7,'permutation':8,'selector':9,'bootstrap':10}
RESIDUAL={0:[],1:[],2:['X2'],3:['Z'],4:['X2'],5:['X2']}
METRICS=['coverage','width','interval_score','infinite_fraction','finite_fraction','finite_coverage']
VALUES=['coverage','width','interval_score','infinite_fraction','finite_coverage','finite_n','finite_covered_n','n_eval','delta','rank','ess_mean','ess_min','pmax','fallback_n','q0_std','finite_fraction']

def root_seed(stage,r):return SMOKE_ROOT if stage==0 else CONFIRM_BASE+r
def entropy(stage,r,s,purpose,*children):return [root_seed(stage,r),NAMESPACE,stage,s,PURPOSES[purpose],*children]
def sequence(stage,r,s,purpose,*children):return np.random.SeedSequence(entropy(stage,r,s,purpose,*children))
def phase(stage):return 'smoke' if stage==0 else 'confirmatory'
# Private imported module instance only: reuse identical SCM body with fresh RNG streams.
sc.sequence=sequence
def full_parent(s):return [] if s==0 else ['X1'] if s==1 else ['X1','Z'] if s==3 else ['X1','X2']
def feature_names(s):return ['X%d'%(i+1) for i in range(20)] if s!=3 else ['X1','Z','P','Q']+['noise%d'%i for i in range(1,17)]
def ks(s):return sorted({len(full_parent(s)),len(RESIDUAL[s])})
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
def save_csv(path,rows,mode='w'):
    with Path(path).open(mode,encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));
        if f.tell()==0:w.writeheader()
        w.writerows(rows);f.flush();os.fsync(f.fileno())
def quantile_model(seed):
    return HistGradientBoostingRegressor(loss='quantile',quantile=.9,learning_rate=.05,max_iter=200,
        max_leaf_nodes=15,max_depth=None,min_samples_leaf=20,l2_regularization=.1,max_features=1.,
        max_bins=255,categorical_features=None,monotonic_cst=None,interaction_cst=None,warm_start=False,
        early_stopping=False,scoring='loss',validation_fraction=.1,n_iter_no_change=10,tol=1e-7,verbose=0,random_state=seed)

def select_features(xfit,yfit,xloc,yloc,names,stage,r,s):
    """Only developer labels supplied. Target=Y upper quantile, not RF residual quantile."""
    if max(ks(s))==0:
        row=dict(phase=phase(stage),replicate=r,scenario='S%d'%s,feature='NONE',rank=0,
            baseline_pinball='NOT_USED',importance='NOT_USED',ten_increments_json='[]',selected_full=False,
            selected_residual=False,status='SKIPPED_ALL_K_ZERO',target='Y_0.9_QUANTILE')
        return {0:[]},[row]
    seed=int(sequence(stage,r,s,'selector').generate_state(1)[0])
    model=quantile_model(seed)
    # Unsupported loss/quantile must raise; no replacement learner or fallback algorithm.
    model.fit(xfit,yfit)
    baseline=float(mean_pinball_loss(yloc,model.predict(xloc),alpha=.9))
    importance=[];increments=[];work=xloc.copy()
    for j in range(20):
        losses=[]
        for perm in range(10):
            order=np.random.default_rng(sequence(stage,r,s,'permutation',j,perm)).permutation(len(xloc))
            work[:,j]=xloc[order,j]
            losses.append(float(mean_pinball_loss(yloc,model.predict(work),alpha=.9))-baseline)
        work[:,j]=xloc[:,j]
        increments.append(losses);importance.append(float(np.mean(losses)))
    ordering=sorted(range(20),key=lambda j:(-importance[j],names[j]))
    selected={k:[names[j] for j in ordering[:k]] for k in ks(s)}
    ranking=[]
    for order,j in enumerate(ordering):
        ranking.append(dict(phase=phase(stage),replicate=r,scenario='S%d'%s,feature=names[j],rank=order+1,
            baseline_pinball=baseline,importance=importance[j],ten_increments_json=json.dumps(increments[j]),
            selected_full=names[j] in selected[len(full_parent(s))],selected_residual=names[j] in selected[len(RESIDUAL[s])],
            status='COMPUTED_DFIT_MODEL_DLOC_PERMUTATION',target='Y_0.9_QUANTILE'))
    return selected,ranking

class Geometry:
    """Reuse Stage4B precomputation; unchanged Stage4 matched kernel and step quantile."""
    def __init__(self,f,ids):
        fit,loc=ids['D_fit'],ids['D_loc'];query=np.r_[ids['D_cal'],ids['D_eval']]
        sd=f.x[fit].std(axis=0,ddof=0);keep=sd>1e-8
        z=(f.x[:,keep]-f.x[fit][:,keep].mean(axis=0))/sd[keep]
        self.names=[n for n,k in zip(f.names,keep) if k];self.zloc=z[loc];self.zquery=z[query]
        self.sloc=f.s[loc];self.squery=f.s[query]
        geo=pdist(f.s[fit]);geo=geo[geo>0];self.scale=float(np.median(geo)) if len(geo) else 1.
        self.g2=cdist(self.squery,self.sloc,'sqeuclidean')/self.scale**2;self.cache={}
    def q(self,mask,r):
        key=tuple(sorted(mask))
        if key not in self.cache:
            sel=[self.names.index(n) for n in mask if n in self.names]
            a2=cdist(self.zquery[:,sel],self.zloc[:,sel],'sqeuclidean')*(len(self.names)/len(sel)) if sel else cdist(self.zquery,self.zloc,'sqeuclidean')
            self.cache[key]=sc.kernel_q0(-.5*(.5*self.g2+.5*a2),r)
        return self.cache[key]

@functools.lru_cache(None)
def mask_registry():
    a=json.loads((HERE/'random_mask_registry.json').read_text(encoding='utf8'))
    out={}
    for x in a:out.setdefault((x['stage'],x['replicate'],x['scenario_index'],x['k']),[]).append(x)
    return out
def base_row(stage,r,s,method,mask,mask_id='NA'):
    a,b=set(mask),set(full_parent(s));d=set(RESIDUAL[s])
    return dict(protocol_id='gccp_stage4c_selector_confirmation_20261019_v1',phase=phase(stage),replicate=r,
        scenario='S%d'%s,method=method,mask_id=mask_id,supplied_mask=';'.join(mask),mask_size=len(mask),
        parent_overlap=len(a&b),jaccard=len(a&b)/len(a|b) if a|b else 1.,contains_scale_driver=int(bool(a&d)),
        status='SUCCESS',data_origin='FRESH_INDEPENDENT_STAGE4C')

def cell(stage,r,s):
    started=time.perf_counter()
    with threadpool_limits(limits=1):
        f,labels=sc.generate(stage,r,s)
        for role in ('D_cal','D_eval'):
            try:labels.get(role)
            except RuntimeError:pass
            else:raise RuntimeError('Final-label gate failed')
        original_feature_hash=sc.digest(f.x)
        pred=sc.fit_predict(f,labels,int(sequence(stage,r,s,'model').generate_state(1)[0]))
        ids=labels.ids;yfit=labels.get('D_fit');yloc=labels.get('D_loc')
        xfit=np.c_[f.x[ids['D_fit']],f.s[ids['D_fit']]];xloc=np.c_[f.x[ids['D_loc']],f.s[ids['D_loc']]]
        selected,ranking=select_features(xfit,yfit,xloc,yloc,list(f.names),stage,r,s)
        assert sc.digest(f.x)==original_feature_hash,'Permutation changed common covariates'
        rloc=np.abs(yloc-pred['D_loc']);geo=Geometry(f,ids)
        config=[('GeoSIMCP-RC',[]),('GCCP-OracleParent-RC',full_parent(s)),('GCCP-ResidualParent-RC',RESIDUAL[s])]
        config += [('SupervisedSelector-k%d-RC'%k,selected[k]) for k in ks(s)]
        if s==5:config.append(('GCCP-WrongPrior-RC',['X1','X3']))
        qmap={name:geo.q(mask,rloc) for name,mask in config}
        random_config={}
        for k in ks(s):
            regs=mask_registry()[(stage,r,s,k)];assert len(regs)==10
            random_config[k]=[(x['mask_id'],x['mask'],geo.q(x['mask'],rloc)) for x in regs]
        if stage==0:
            for name,mask in config[:3]:assert np.array_equal(qmap[name][0],sc.matched_q(f,ids,rloc,mask)[0]),'Kernel parity failure'
            assert sc.rank.exact_quantile(np.arange(800),.1)[1]==721
            assert np.isinf(sc.rank.exact_quantile(np.arange(2),.1)[0])
        receipt=dict(phase=phase(stage),replicate=r,scenario='S%d'%s,roles={role:{'n':len(ix),'id_hash':sc.digest(ix)} for role,ix in ids.items()},
            feature_hash=original_feature_hash,selected_masks=selected,ranking_hash=hashlib.sha256(json.dumps(ranking,sort_keys=True).encode()).hexdigest(),
            selector_target='Y_0.9_QUANTILE',selector_roles=['D_fit','D_loc'],q0_frozen_before_Dcal=True,truth_in_predictor=False,
            q0_hashes={name:sc.digest(x[0]) for name,x in qmap.items()},
            random_q0_hashes={str(k):[sc.digest(x[2][0]) for x in values] for k,values in random_config.items()})
        rd=HERE/'receipts'/phase(stage);rd.mkdir(parents=True,exist_ok=True);rp=rd/('r%03d_S%d.json'%(r,s));save_json(rp,receipt)
        labels.state='q0_frozen';rcal=np.abs(labels.get('D_cal')-pred['D_cal'])
        def make_radius(q):
            delta,k=sc.rank.exact_quantile(rcal-q[:800],.1);assert k==721
            return np.maximum(0,q[800:]+delta),delta,k
        radii={name:make_radius(x[0]) for name,x in qmap.items()}
        random_radii={k:[make_radius(x[2][0]) for x in values] for k,values in random_config.items()}
        receipt['interval_hashes']={name:sc.digest(x[0]) for name,x in radii.items()};receipt['intervals_frozen_before_Deval']=True
        save_json(rp,receipt);labels.state='intervals_frozen';y=labels.get('D_eval')
        rows=[];individual=[]
        for name,mask in config:
            radius,delta,k=radii[name];q,diag=qmap[name]
            out=base_row(stage,r,s,name,mask);out.update(sc.metrics(y,pred['D_eval'],radius));out.update(diag,delta=delta,rank=k,q0_std=float(q[800:].std()),finite_fraction=float(np.isfinite(radius).mean()));rows.append(out)
        for k,values in random_config.items():
            group=[]
            for (mask_id,mask,(q,diag)),(radius,delta,rank) in zip(values,random_radii[k]):
                out=base_row(stage,r,s,'RandomMask-k%d-RC'%k,mask,str(mask_id));out.update(sc.metrics(y,pred['D_eval'],radius));out.update(diag,delta=delta,rank=rank,q0_std=float(q[800:].std()),finite_fraction=float(np.isfinite(radius).mean()));group.append(out)
            avg=base_row(stage,r,s,'RandomMask-k%d-RC-Mean10'%k,values[0][1],'MEAN10');avg['supplied_mask']='ALL_10_PREDECLARED_MASK_METRICS_MEAN';avg['mask_size']=k
            for field in VALUES+['parent_overlap','jaccard','contains_scale_driver']:
                v=[float(x[field]) for x in group];avg[field]=v[0] if all(x==v[0] for x in v) else float(np.mean(v))
            rows.append(avg);individual.extend(group)
        if s==0:
            assert len(set(receipt['interval_hashes'].values()))==1
            assert all(x['width']==rows[0]['width'] and x['interval_score']==rows[0]['interval_score'] for x in rows+individual)
        if s==1:assert receipt['interval_hashes']['GCCP-ResidualParent-RC']==receipt['interval_hashes']['GeoSIMCP-RC']
        cache=HERE/'cache'/phase(stage);cache.mkdir(parents=True,exist_ok=True)
        np.savez(cache/('r%03d_S%d.npz'%(r,s)),z_loc=geo.zloc,z_query=geo.zquery,s_loc=geo.sloc,s_query=geo.squery,
            geography_scale=geo.scale,attribute_names=np.asarray(geo.names),residual_loc=rloc,residual_cal=rcal,pred_eval=pred['D_eval'],y_eval=y)
        receipt.update(completed=True,label_access=labels.access,seconds=time.perf_counter()-started,ranking_rows=len(ranking))
        save_json(rp,receipt);return rows,individual,ranking

def prepare():
    if (HERE/'protocol_stage4c.json').exists():raise RuntimeError('Refusing protocol overwrite')
    old_paths=[OLD/'seed_manifest.csv',OLD_B/'seed_manifest_stage4b.csv']
    old_keys=set();old_words=set();old_roots=set()
    for path in old_paths:
        for x in read_csv(path):
            e=tuple(json.loads(x.get('entropy',x.get('entropy_json'))));old_keys.add(e);old_roots.add(e[0])
            if x.get('generated_uint32'):old_words.add(int(x['generated_uint32']))
    assert not (set(range(CONFIRM_BASE,CONFIRM_BASE+100))|{SMOKE_ROOT})&old_roots
    seeds=[];masks=[];new_keys=set();model_words=set()
    def register(stage,r,s,purpose,children=(),mask=None,k='',mask_id='',planned='EXECUTE'):
        e=entropy(stage,r,s,purpose,*children);key=tuple(e)
        assert key not in old_keys and key not in new_keys;new_keys.add(key)
        word=''
        if purpose in ('model','selector'):
            word=int(np.random.SeedSequence(e).generate_state(1)[0]);assert word not in old_words and word not in model_words;model_words.add(word)
        seeds.append(dict(phase=phase(stage),stage=stage,replicate=r,scenario='S%d'%s,purpose=purpose,
            child_json=json.dumps(list(children)),entropy_json=json.dumps(e),generated_uint32=word,k=k,mask_id=mask_id,
            mask_json=json.dumps(mask) if mask is not None else '',planned_use=planned,actual_use='PENDING'))
    for stage,reps in [(0,range(1)),(1,range(100))]:
        for r in reps:
            for s in range(6):
                for purpose in ['attributes','coordinates','innovation','proxies','split','model','selector','bootstrap']:
                    planned='EXECUTE'
                    if purpose=='bootstrap' or purpose=='proxies' and s!=3 or purpose=='selector' and max(ks(s))==0:planned='RESERVED_NOT_USED'
                    register(stage,r,s,purpose,planned=planned)
                for j in range(20):
                    for perm in range(10):register(stage,r,s,'permutation',(j,perm),planned='EXECUTE' if max(ks(s)) else 'RESERVED_ALL_K_ZERO')
                for k in ks(s):
                    for i in range(10):
                        e=entropy(stage,r,s,'mask',k,i);gen=np.random.default_rng(np.random.SeedSequence(e))
                        indices=sorted(gen.choice(20,k,replace=False).tolist()) if k else []
                        mask=[feature_names(s)[j] for j in indices]
                        register(stage,r,s,'mask',(k,i),mask,k,i,'EXECUTE' if k else 'EMPTY_NO_RANDOM_DRAW')
                        masks.append(dict(stage=stage,replicate=r,scenario_index=s,k=k,mask_id=i,mask=mask))
    save_csv(HERE/'seed_manifest.csv',seeds,'x');save_json(HERE/'random_mask_registry.json',masks)
    sources=[OLD/'scientific.py',OLD/'protocol.json',OLD/'seed_manifest.csv',OLD_B/'experiment.py',OLD_B/'protocol_stage4b.json',OLD_B/'seed_manifest_stage4b.csv']
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    model=quantile_model(0).get_params();model['random_state']='distinct registered selector uint32 per cell'
    proto=dict(protocol_id='gccp_stage4c_selector_confirmation_20261019_v1',status='LOCKED_NEW_SEED_CONFIRMATORY_PROTOCOL',created_before_generation=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        new_samples=True,old_samples_used_for_debug=False,old_performance_files_read=False,N=4000,roles={'D_fit':1600,'D_loc':800,'D_cal':800,'D_eval':800},scenarios=['S%d'%s for s in range(6)],replicates_target=100,workers=6,
        generator='Reuse Stage4 generate() unchanged, private module RNG binding replaced by recorded fresh SeedSequence; no file changes',
        point_predictor='Identical shared Stage4 RF300 leaf5 all22 observed covariates, D_fit only, registered fresh model seed, n_jobs1',
        localizer='Same D_fit standardization/geographic median, p/k sum-square subset, lambda=.5,h1; empty set eta0 ordinary GeoSIM fallback; D_loc .9 step q0; D_cal independent exact rank721 signed Delta, no query atom',
        full_parents={str(s):full_parent(s) for s in range(6)},residual_parents={str(s):RESIDUAL[s] for s in range(6)},S5_wrong=['X1','X3'],
        residual_semantics='Oracle SCM noise-scale parents, not claimed complete parents of fitted RF residual distribution; no causal discovery',
        selector=dict(status='NEW_CANDIDATE_NOT_THE_OLD_UNIMPLEMENTED_FOREST_QUANTILE_SELECTOR',software_class='sklearn.ensemble.HistGradientBoostingRegressor',parameters=model,
            target='Observed Y .9 conditional quantile, not absolute RF residual',inputs='20 observed attributes plus 2 coordinates',fit_role='D_fit',importance_role='D_loc',
            importance='mean of ten held-out .9 pinball loss increases per attribute; only 20 non-coordinate columns permuted; each permutation resets original column; no variable filtering even if negative importance',
            top_k='single ranking; exact k_full and k_residual; descending importance, feature-name lexicographic tie break; k0 empty ordinary fallback',
            zero_k_skip='S0 all k0: no unnecessary quantile fit/permutations; reserved streams marked unused',
            final_labels='Never D_cal/D_eval for model, ranking, parameters, selection or retention',
            difference_from_Stage2='User-authorized new D_fit model / D_loc importance workflow, not original 800/400/400 D_fit-internal residual selector',
            unsupported_policy='Stop selector branch and preserve failure; do not install or substitute learner'),
        random=dict(masks_per_k_per_replicate=10,cardinalities='unique full/residual parent dimensions per scenario; shared family if same k',
            rule='Uniform subset of20 without replacement within mask; independent registered mask streams; duplicates across masks retained, no outcome selection; every mask reported',
            inference='Mean ten metric values within replicate, NOT mean interval endpoints and NOT 10R independent units'),
        randomness=dict(confirm_root_range=[CONFIRM_BASE,CONFIRM_BASE+99],smoke_root=SMOKE_ROOT,namespace=NAMESPACE,purposes=PURPOSES,
            entropy='[root,namespace,stage,scenario,purpose_id,*child_indices]; permutation children attribute0..19/permutation0..9; mask children k/mask0..9',
            scope=[str(p) for p in old_paths],checked_old_unique_keys=len(old_keys),new_unique_keys=len(new_keys),no_old_entropy_key_or_root_reuse=True,
            model_uint32_checked_unique=True,global_uniqueness_certified=False),
        statistics=dict(primary_scenarios=['S2','S3','S4'],primary_method='GCCP-ResidualParent-RC',
            primary_comparators=['RandomMask-k1-RC-Mean10','SupervisedSelector-k1-RC'],primary_metric='interval_score',
            primary_family_size=6,alternative='one-sided paired t: mean(Residual-comparator)<0; Holm FWER .05 over all six',
            confidence_intervals='two-sided paired/replicate t95, df R-1; unadjusted CIs labeled; no bootstrap',
            interim='20-repeat checkpoint descriptive only; planned 100-repeat analysis is sole confirmatory rejection point; no outcome-based stopping or tuning',
            secondary='Other scenarios, full-parent contrasts, selection frequencies, width differences descriptive and not primary-family-confirmed',
            invalid='Nonfinite primary differences: retain rows, set p=1, flag non-estimable; no silent omission',
            coverage='Each method absolute coverage t95 vs .90 separately: CI below flags undercoverage; CI includes .90 is compatible but not proof; coverage difference CI containing0 is not equivalence. One-sided lower>=.89 is secondary practical screen only.'),
        limitation='Observed-Y upper-quantile selector may prioritize mean predictors rather than RF residual-scale relevance; beating this candidate does not imply beating all supervised residual selectors.',
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform()),
        read_only_source_hashes=source_hashes,mask_registry_sha256=hashlib.sha256((HERE/'random_mask_registry.json').read_bytes()).hexdigest(),
        code_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['experiment.py','report.py']})
    save_json(HERE/'protocol_stage4c.json',proto);save_json(HERE/'execution_status.json',dict(state='PREPARED',seed_rows=len(seeds),mask_rows=len(masks)))
    print('LOCKED fresh protocol: %d streams, %d masks; no old root/key/model seed reuse in scoped lists'%(len(seeds),len(masks)),flush=True)

def batch(start,stop,workers,smoke=False):
    proto=json.loads((HERE/'protocol_stage4c.json').read_text(encoding='utf8'))
    assert all(hashlib.sha256((HERE/k).read_bytes()).hexdigest()==v for k,v in proto['code_sha256'].items())
    assert hashlib.sha256((HERE/'random_mask_registry.json').read_bytes()).hexdigest()==proto['mask_registry_sha256']
    if not smoke:assert json.loads((HERE/'smoke_status.json').read_text())['status']=='PASS'
    prefix='smoke_' if smoke else '';output=HERE/(prefix+'per_replicate_metrics.csv')
    done={(int(x['replicate']),int(x['scenario'][1:])) for x in read_csv(output)} if output.exists() else set()
    jobs=[(r,s) for r in range(start,stop) for s in range(6) if (r,s) not in done];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool,(HERE/'progress.log').open('a',encoding='utf8') as log:
        tasks={pool.submit(cell,0 if smoke else 1,r,s):(r,s) for r,s in jobs}
        try:
            for future in as_completed(tasks):
                r,s=tasks[future];rows,masks,ranks=future.result()
                save_csv(HERE/(prefix+'random_mask_metrics.csv'),masks,'a');save_csv(HERE/(prefix+'selector_rankings.csv'),ranks,'a');save_csv(output,rows,'a')
                done.add((r,s));line='%s r=%d S%d saved cells=%d elapsed=%.1fs'%(phase(0 if smoke else 1),r,s,len(done),time.perf_counter()-started)
                print(line,flush=True);log.write(line+'\n');log.flush();save_json(HERE/'execution_status.json',dict(state=phase(0 if smoke else 1),completed_cells=len(done),target_cells=600))
        except Exception as e:
            save_json(HERE/'failure.json',dict(job=tasks.get(future),error=repr(e),completed_cells=len(done),selector_substitution=False))
            for f in tasks:f.cancel()
            raise
    if smoke:save_json(HERE/'smoke_status.json',dict(status='PASS',complete_six_scenarios=len(done)==6,selector_support='quantile0.9 HGB fitted successfully',old_samples_used=False,kernel_parity=True,label_gates=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','smoke','batch','report']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=20);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--tag',default='current');a=ap.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='smoke':batch(0,1,a.workers,True)
    elif a.command=='batch':batch(a.start,a.stop,a.workers)
    else:
        import report
        report.make(a.tag)
