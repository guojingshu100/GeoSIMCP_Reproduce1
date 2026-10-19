"""Frozen S4 effect-only permutation audit; no generator or mixed-split reads."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import importlib.metadata
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
ORIGINAL = PILOT / 'gccm_outputs' / 'S4'
RUNTIME = PILOT / 'gccm_runtime'
FEATURES = ['s_x', 's_y', 'x1', 'x2', 'g_proxy', 'x_noise']
GRID = [10, 18, 33, 60, 109, 198]
SEEDS = list(range(2026090901, 2026091000))
RSCRIPT = Path(r'C:\Program Files\R\R-4.6.1\bin\Rscript.exe')
RESULT_FILES = ['raw_curves.csv', 'mean_curves.csv', 'convergence.csv', 'weights.csv',
                'embedding.csv', 'ring_counts.csv', 'preflight.csv']
SCHEMA = ['node_id', 'sample_id', 's_x', 's_y', 'abs_residual'] + ['z_'+f for f in FEATURES]
LOCK = threading.Lock()
STOP = threading.Event()
PROCESSES: set = set()

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write_json(p, value):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def log(message):
    line = datetime.now().astimezone().isoformat() + ' ' + message
    with LOCK:
        with (HERE/'execution.log').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
        print(line, flush=True)

def read_frame(path):
    return pd.read_csv(path, float_precision='round_trip')

def csv_tokens(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.reader(f))

def protect_sources():
    paths = list(ORIGINAL.iterdir()) + [RUNTIME/'gccm_runtime_config_v1.json',
        RUNTIME/'gccm_point_graph_wrapper_v1.R', RUNTIME/'run_gccm_scale_fit_v1.py',
        PILOT/'scale_outputs/S4_scale_fit_preparation.csv',
        PILOT/'scale_comparison_check_v1.json', PILOT/'gccm_runtime_check_v1.json']
    return {str(p): sha(p) for p in paths if p.is_file()}

def freeze():
    cfg = RUNTIME/'gccm_runtime_config_v1.json'
    frozen = read_json(cfg)
    assert frozen['execution_frozen'] and frozen['library_sizes'] == GRID
    assert frozen['direction'] == 'effect_xmap_cause'
    assert sha(RUNTIME/'gccm_point_graph_wrapper_v1.R') == frozen['wrapper_sha256']
    meta = read_json(ORIGINAL/'execution_metadata.json')
    assert all(sha(ORIGINAL/n) == h for n,h in meta['hashes'].items())
    prep = read_frame(PILOT/'scale_outputs/S4_scale_fit_preparation.csv')
    assert len(prep) == 200 and set(prep.split) == {'local_calibration'}
    assert set(prep.local_calibration_role) == {'scale_fit'}
    source = read_frame(ORIGINAL/'restricted_input.csv')
    assert list(source) == SCHEMA and len(source) == 200
    assert source.sample_id.is_unique and source.node_id.tolist() == list(range(1,201))
    assert source.sample_id.tolist() == sorted(source.sample_id.tolist())
    aligned = prep.set_index('sample_id').loc[source.sample_id]
    np.testing.assert_array_equal(source.abs_residual, aligned.abs_residual)
    for f in FEATURES:
        np.testing.assert_array_equal(source['z_'+f], aligned['z_'+f])
    assert not set(source).intersection({'y','mu_true','sigma_true','epsilon','g_true','eta'})
    seed_rows = [{'permutation_id': i+1, 'permutation_seed': seed} for i,seed in enumerate(SEEDS)]
    pd.DataFrame(seed_rows).to_csv(HERE/'permutation_seeds.csv', index=False, mode='x', lineterminator='\n')
    snapshot = HERE/'frozen_sources'
    snapshot.mkdir(exist_ok=False)
    for n in ['restricted_input.csv','edges.csv','parameters.csv','historical_core.r',
              'execution_config.json','weights.csv','convergence.csv','mean_curves.csv','raw_curves.csv']:
        shutil.copyfile(ORIGINAL/n, snapshot/n)
    shutil.copyfile(RUNTIME/'gccm_point_graph_wrapper_v1.R', snapshot/'gccm_point_graph_wrapper_v1.R')
    source_hashes = protect_sources()
    config = {
        'experiment_version':'S4_effect_permutation_v1', 'scenario':'S4', 'n_scale_fit':200,
        'frozen_before_any_new_R_execution':True, 'timestamp':datetime.now().astimezone().isoformat(),
        'effect':'abs_residual', 'cause_fields':['z_'+f for f in FEATURES],
        'direction':'effect_xmap_cause', 'permutation_count':99,
        'seed_rule':'2026090900 + permutation_id, permutation_id=1..99', 'seeds':SEEDS,
        'permutation_algorithm':'numpy.random.default_rng(permutation_seed).permutation(200)',
        'permutation_policy':'Unrestricted uniform permutations; fixed points allowed; reject identity only if encountered, without replacement/redraw.',
        'permuted_field_only':'abs_residual', 'scaler_refit':False, 'graph_rebuilt':False,
        'frozen_GCCM_protocol':frozen['candidate_protocol'], 'R_seed':42,
        'library_sizes':GRID, 'R_backend':'sequential cores=NULL', 'max_independent_R_processes':4,
        'rscript':str(RSCRIPT), 'environment_override':{'LC_ALL':'C'},
        'original_observed_source':str(ORIGINAL), 'original_statistics_not_reselected':True,
        'primary_statistics':['terminal_rho for s_x','terminal_rho for s_y'],
        'secondary_statistics':['gain','strength','normalized_weight','maximum terminal_rho over all six fields'],
        'tail_rule':'upper tail (1 + count(permuted >= original))/(1+99); ties included',
        'weight_tail_missing_rule':'If any null weight is unavailable, report conditional tail with its explicit available denominator; do not replace NA by zero.',
        'distribution_quantiles':[0,.05,.25,.5,.75,.95,1], 'quantile_method':'linear',
        'reference_separation_threshold':.05,
        'interpretation':'Diagnostic reference only; not a formal causal p-value. PASS concerns implementation, not evidence of causality.',
        'zero_strength_policy':'All-zero convergence is a valid negative-control result, not an implementation failure; preserve unavailable normalized weights.',
        'repeat_permutation_id':1, 'reproduction_tolerance':{'rtol':0,'atol':2e-12},
        'forbidden_access':['other scenarios','mixed-split generator data','mixed-split residuals',
            'tuning_calibration labels','validation labels','final_calibration labels','test labels','oracle files'],
        'source_hashes':source_hashes, 'snapshot_hashes':{p.name:sha(p) for p in snapshot.iterdir()},
        'runner_sha256':sha(Path(__file__)), 'seed_list_sha256':sha(HERE/'permutation_seeds.csv'),
        'software':{'Python':sys.version,'Python_executable':sys.executable,
                    **{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy']}},
        'estimated_serial_minutes':101*float(meta['runtime_seconds'])/60,
    }
    write_json(HERE/'config_v1.json',config)
    write_json(HERE/'source_manifest_before.json',source_hashes)
    (HERE/'00_pipeline_audit.md').write_text(
        '# 1. 冻结流水线审计\n\n'
        'effect 来自已有 gccm_outputs/S4/restricted_input.csv 的 abs_residual，已与 S4_scale_fit_preparation.csv 的 200 个 scale_fit IDs、六字段和残差逐值核对。'
        '不读取混合 split 的数据或 residual 文件。六 cause 字段为 z_s_x、z_s_y、z_x1、z_x2、z_g_proxy、z_x_noise。\n\n'
        '# 2. 图和算法\n\n'
        '直接复制已有 edges.csv 和坐标，不重建 Delaunay、不重新标准化。generateEmbedings 对 Ring1–3 求 effect 均值；'
        'GCCMLattice(effect_embeddings, standardized_X_j, ...) 实现 effect_xmap_cause。'
        'E=3、tau=1、b=4；200 个有效 embeddings；历史索引裁剪后最大库 198；grid=[10,18,33,60,109,198]。'
        '小于最大库时枚举 198 个循环窗口，最大库一次；随机 library draws=0、R seed=42、cores=NULL。\n\n'
        '# 3. 冻结评分\n\n'
        'mean curve=max(0,mean(raw rho))；gain>0、Spearman>=0.6、Pearson p<0.05、Fisher-z 95% lower>0 全部满足才通过。'
        '通过时 strength=terminal rho，否则零；weight=strength/sum(strength)，gamma=6*weight。全零时保留空权重，不等权替代。\n\n'
        '# 4. 运行前预注册\n\n'
        '99 次 effect-only 置换，seed=2026090901..2026090999，default_rng(seed).permutation(200)。'
        '原始 S4 为 observed statistic；先原样复现。第 1 次置换预先指定做重复核验。'
        '最多同时 4 个独立 R 进程，每个内部顺序；完整复制历史 wrapper 和 core，字节 hash 固定。'
        '99 次结果与所有检查完成前不进行统计解释。\n',encoding='utf-8')
    return config

def make_input(destination, permutation_id, seed):
    destination.mkdir(parents=True, exist_ok=False)
    snap = HERE/'frozen_sources'
    for name in ['edges.csv','parameters.csv','historical_core.r','execution_config.json']:
        shutil.copyfile(snap/name,destination/name)
    rows = csv_tokens(snap/'restricted_input.csv')
    effect_col = rows[0].index('abs_residual')
    if permutation_id == 0:
        order = np.arange(200)
        shutil.copyfile(snap/'restricted_input.csv',destination/'restricted_input.csv')
    else:
        order = np.random.default_rng(seed).permutation(200)
        assert not np.array_equal(order,np.arange(200)), 'Identity permutation encountered; stop, do not choose replacement.'
        values = [r[effect_col] for r in rows[1:]]
        output = [rows[0]] + [r.copy() for r in rows[1:]]
        for i,j in enumerate(order): output[i+1][effect_col]=values[j]
        with (destination/'restricted_input.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerows(output)
    pd.DataFrame({'node_id':range(1,201),'effect_source_node_id':order+1}).to_csv(destination/'permutation_mapping.csv',index=False,mode='x')
    checks = validate_input(destination,permutation_id,seed)
    write_json(destination/'input_check.json',checks)
    return checks

def validate_input(destination, permutation_id, seed):
    original=csv_tokens(HERE/'frozen_sources/restricted_input.csv')
    modified=csv_tokens(destination/'restricted_input.csv')
    assert original[0]==modified[0]==SCHEMA and len(modified)==201
    col=SCHEMA.index('abs_residual'); before=[r[col] for r in original[1:]]; after=[r[col] for r in modified[1:]]
    assert sorted(before)==sorted(after)
    unchanged=[j for j in range(len(SCHEMA)) if j!=col]
    assert all([a[j] for j in unchanged]==[b[j] for j in unchanged] for a,b in zip(original[1:],modified[1:]))
    expected=np.arange(200) if permutation_id==0 else np.random.default_rng(seed).permutation(200)
    assert after==[before[j] for j in expected]
    mapping=read_frame(destination/'permutation_mapping.csv')
    assert mapping.effect_source_node_id.tolist()==(expected+1).tolist()
    if permutation_id: assert after!=before
    for n in ['edges.csv','parameters.csv','historical_core.r','execution_config.json']:
        assert sha(destination/n)==sha(HERE/'frozen_sources'/n)
    return {'status':'PASS','n':200,'permutation_id':permutation_id,'permutation_seed':seed,
        'same_effect_multiset':True,'effect_only_change':True,'features_unchanged':True,
        'coordinates_unchanged':True,'graph_unchanged':True,'settings_unchanged':True,
        'fixed_points':int(np.sum(expected==np.arange(200))),
        'input_sha256':sha(destination/'restricted_input.csv')}

def validate_output(destination):
    raw=read_frame(destination/'raw_curves.csv'); curves=read_frame(destination/'mean_curves.csv')
    conv=read_frame(destination/'convergence.csv'); weights=read_frame(destination/'weights.csv')
    assert len(raw)==5946 and len(curves)==36 and len(conv)==len(weights)==6
    assert conv.feature.tolist()==weights.feature.tolist()==FEATURES
    assert set(raw.direction)==set(conv.direction)=={'effect_xmap_cause'}
    assert np.isfinite(raw.rho).all() and (conv.n_terminal==198).all()
    pre=read_frame(destination/'preflight.csv').iloc[0]
    assert pre.valid_embeddings==200 and pre.max_library==198
    emb=read_frame(destination/'embedding.csv')
    inp=read_frame(destination/'restricted_input.csv')
    assert emb.sample_id.tolist()==inp.sample_id.tolist() and emb.shape==(200,5)
    assert np.isfinite(emb.iloc[:,2:].to_numpy()).all()
    expected_strength=[]
    for feature in FEATURES:
        r=raw[raw.feature==feature]; c=curves[curves.feature==feature]
        assert c.L.tolist()==GRID
        assert r.groupby('L').size().to_dict()=={l:(1 if l==198 else 198) for l in GRID}
        vals=np.array([max(0,float(r.loc[r.L==l,'rho'].mean())) for l in GRID])
        np.testing.assert_allclose(c.rho,vals,rtol=0,atol=2e-12)
        terminal=vals[-1]; gain=terminal-vals[0]
        trend=float(stats.spearmanr(GRID,vals).statistic) if len(set(vals))>1 else np.nan
        z=np.clip(terminal,-1+1e-12,1-1e-12)
        p=2*stats.t.sf(abs(z*np.sqrt(196/(1-z*z))),196)
        lo=np.tanh(np.arctanh(z)-stats.norm.ppf(.975)/np.sqrt(195))
        hi=np.tanh(np.arctanh(z)+stats.norm.ppf(.975)/np.sqrt(195))
        passed=bool(np.isfinite(trend) and gain>0 and trend>=.6 and p<.05 and lo>0)
        row=conv.loc[conv.feature==feature].iloc[0]
        np.testing.assert_allclose(row[['rho_Lmin','rho_Lmax','gain','spearman','p_two_sided','CI_lower','CI_upper']].to_numpy(float),[vals[0],terminal,gain,trend,p,lo,hi],rtol=0,atol=2e-12,equal_nan=True)
        assert bool(row.converged)==passed
        expected_strength.append(terminal if passed else 0)
    np.testing.assert_allclose(weights.strength,expected_strength,rtol=0,atol=2e-12)
    total=sum(expected_strength)
    if total>0:
        np.testing.assert_allclose(weights.weight,np.array(expected_strength)/total,rtol=0,atol=2e-12)
        np.testing.assert_allclose(weights.gamma,6*weights.weight,rtol=0,atol=2e-12)
    else:
        assert weights.weight.isna().all() and set(weights.status)=={'NO_CONVERGED_FEATURES'}
    return {'status':'PASS','valid_embeddings':200,'raw_rows':5946,'independent_scoring_check':True,
            'weights_available':bool(total>0)}

def run_r(destination):
    if STOP.is_set():raise RuntimeError('Stopped after a required check failed')
    cmd=[str(RSCRIPT),'--vanilla',str(HERE/'frozen_sources/gccm_point_graph_wrapper_v1.R'),
         str(destination/'restricted_input.csv'),str(destination/'edges.csv'),
         str(destination/'historical_core.r'),str(destination/'parameters.csv'),str(destination),'estimate']
    start=time.perf_counter()
    with LOCK:
        if STOP.is_set():raise RuntimeError('Stopped')
        process=subprocess.Popen(cmd,cwd=destination,env={**os.environ,'LC_ALL':'C'},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        PROCESSES.add(process)
    try:
        stdout,stderr=process.communicate(timeout=1800)
    except BaseException:
        process.kill();process.communicate();raise
    finally:
        with LOCK:PROCESSES.discard(process)
    (destination/'stdout.log').write_bytes(stdout);(destination/'stderr.log').write_bytes(stderr)
    meta={'returncode':process.returncode,'runtime_seconds':time.perf_counter()-start,
          'command':cmd,'environment_override':{'LC_ALL':'C'},
          'inputs':'restricted scale_fit input, fixed edges, parameters and historical core only',
          'hashes':{p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file()}}
    write_json(destination/'execution_metadata.json',meta)
    assert process.returncode==0, stderr.decode('utf-8',errors='replace')
    checks=validate_output(destination)
    write_json(destination/'output_check.json',checks)
    return meta

def distribution(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return {'n_available':0,**{k:None for k in ['mean','sd','min','p05','p25','median','p75','p95','max']}}
    return {'n_available':len(a),'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else None,
        **dict(zip(['min','p05','p25','median','p75','p95','max'],map(float,np.quantile(a,[0,.05,.25,.5,.75,.95,1],method='linear'))))}

def aggregate(config):
    diagnostic=[];long=[];permutation_summary=[]
    for pid,seed in enumerate(SEEDS,1):
        d=HERE/'permutations'/f'perm_{pid:03d}'
        cv=read_frame(d/'convergence.csv').merge(read_frame(d/'weights.csv')[['feature','weight','gamma','status']],on='feature',validate='one_to_one')
        cv=cv.rename(columns={'feature':'variable','rho_Lmin':'starting_rho','rho_Lmax':'terminal_rho',
            'spearman':'spearman_trend','p_two_sided':'pearson_p','CI_lower':'fisher_z_lower','CI_upper':'fisher_z_upper','weight':'normalized_weight'})
        cv.insert(0,'permutation_seed',seed);cv.insert(0,'permutation_id',pid)
        diagnostic.append(cv)
        curve=read_frame(d/'mean_curves.csv').rename(columns={'feature':'variable','L':'library_size','rho':'mean_rho'})
        long.append(curve.drop(columns='direction').merge(cv,on='variable',validate='many_to_one'))
        permutation_summary.append({'permutation_id':pid,'permutation_seed':seed,
            'number_converged':int(cv.converged.sum()),'coordinates_converged':int(cv[cv.variable.isin(['s_x','s_y'])].converged.sum()),
            'weights_available':bool(cv.normalized_weight.notna().all()),
            'maximum_normalized_weight':float(cv.normalized_weight.max()) if cv.normalized_weight.notna().any() else np.nan,
            'maximum_terminal_rho':float(cv.terminal_rho.max()),'maximum_strength':float(cv.strength.max())})
    all_d=pd.concat(diagnostic,ignore_index=True);all_l=pd.concat(long,ignore_index=True);ps=pd.DataFrame(permutation_summary)
    assert len(all_d)==594 and len(all_l)==3564 and len(ps)==99
    all_d.to_csv(HERE/'permutation_diagnostics.csv',index=False,mode='x',float_format='%.17g')
    all_l.to_csv(HERE/'permutation_library_statistics.csv',index=False,mode='x',float_format='%.17g')
    ps.to_csv(HERE/'permutation_summary.csv',index=False,mode='x',float_format='%.17g')
    observed=read_frame(HERE/'frozen_sources/convergence.csv').merge(read_frame(HERE/'frozen_sources/weights.csv')[['feature','weight']],on='feature')
    observed.to_csv(HERE/'original_observed_statistics.csv',index=False,mode='x',float_format='%.17g')
    summaries=[];tails=[]
    mapping={'terminal_rho':'rho_Lmax','gain':'gain','strength':'strength','normalized_weight':'weight'}
    for feature in FEATURES:
        sub=all_d[all_d.variable==feature];o=observed[observed.feature==feature].iloc[0]
        for stat in mapping:
            summaries.append({'variable':feature,'statistic':stat,'converged_n':int(sub.converged.sum()),'convergence_fraction':float(sub.converged.mean()),**distribution(sub[stat])})
            if feature in ['s_x','s_y']:
                a=sub[stat].dropna().to_numpy();value=float(o[mapping[stat]])
                tails.append({'variable':feature,'statistic':stat,'observed':value,'n_permutations_available':len(a),
                    'count_ge_observed':int(np.sum(a>=value)),'fraction_strictly_below_observed':float(np.mean(a<value)) if len(a) else np.nan,
                    'negative_control_tail_probability':(1+int(np.sum(a>=value)))/(1+len(a)),
                    'conditional_on_available_weights':bool(stat=='normalized_weight' and len(a)<99),
                    'null_median':float(np.median(a)) if len(a) else np.nan,'null_p95':float(np.quantile(a,.95)) if len(a) else np.nan})
    sd=pd.DataFrame(summaries);tail=pd.DataFrame(tails)
    sd.to_csv(HERE/'variable_distribution_summary.csv',index=False,mode='x',float_format='%.17g')
    tail.to_csv(HERE/'spatial_negative_control_tails.csv',index=False,mode='x',float_format='%.17g')
    obs_max=float(observed.rho_Lmax.max());max_tail=(1+int((ps.maximum_terminal_rho>=obs_max).sum()))/100
    summary={'permutations_completed':99,'per_variable_pass_counts':{f:int(all_d.loc[all_d.variable==f,'converged'].sum()) for f in FEATURES},
        'per_variable_pass_fractions':{f:float(all_d.loc[all_d.variable==f,'converged'].mean()) for f in FEATURES},
        'number_converged_distribution':{str(k):int(v) for k,v in ps.number_converged.value_counts().sort_index().items()},
        'any_coordinate_converged_n':int((ps.coordinates_converged>0).sum()),
        'both_coordinates_converged_n':int((ps.coordinates_converged==2).sum()),
        'no_converged_feature_n':int((ps.number_converged==0).sum()),
        'maximum_normalized_weight_distribution':distribution(ps.maximum_normalized_weight),
        'observed_maximum_terminal_rho':obs_max,'maximum_terminal_rho_negative_control_tail_probability':max_tail,
        'spatial_tail_probabilities':json.loads(tail.to_json(orient='records',double_precision=15)),
        'interpretation_scope':'Negative-control reference, not a formal causal p-value or interval-performance result.'}
    write_json(HERE/'summary.json',summary)
    return summary,observed,sd,tail

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

def report(summary,observed,sd,tail,check):
    primary=tail[tail.statistic=='terminal_rho']
    separated=bool((primary.negative_control_tail_probability<=.05).all())
    conclusion=('原始两个坐标的 terminal rho 都超过了本次置换参考的上尾门槛，但这只支持对这一负对照的分离，不证明因果识别。' if separated else
        '原始 S4 坐标强度没有同时与无对应关系的置换参考充分分离。当前 graph/sample 配置下，应暂停把 convergence/weight 解释为已识别的尺度因果信息。')
    text='# 1. 实验目的\n\n只检验冻结 S4 GCCM 在 effect 与特征/位置对应关系被打乱后是否仍产生收敛；不优化 GCCP，不评价区间。\n\n'
    text+='# 2. 冻结输入与设置\n\n数据仅来自原 gccm_outputs/S4/restricted_input.csv 及既有 scale_fit 核验文件，共 200 行。'
    text+='effect=abs_residual；cause=z_s_x,z_s_y,z_x1,z_x2,z_g_proxy,z_x_noise；effect_xmap_cause。图 edges.csv、坐标、所有标准化数值、样本顺序保持原样。'
    text+='E=3、tau=1、b=4、R seed=42、cores=NULL；grid=[10,18,33,60,109,198]；200 有效 embeddings、198 terminal rows。'
    text+='mean rho=max(0,mean(raw rho))；gain>0、Spearman>=0.6、Pearson p<0.05、Fisher-z lower>0；通过后 strength=terminal rho，归一化为 weight。'
    text+='原 wrapper/core 按字节复制，没有增补 self-lag 或修改规则。\n\n'
    text+='# 3. 置换设计\n\n99 次，完整 seeds=2026090901..2026090999，运行前保存。每次 default_rng(seed).permutation(200)，只移动 effect 字段原始 CSV 数值文本。'
    text+='允许通常随机置换的固定点，不强制 derangement；没有重抽或挑选 seed。最多四个独立 R 任务，每任务内部顺序执行。'
    text+='第 1 次置换预先指定复现。原始冻结 S4 结果为 observed statistic，复现结果仅校验，不重新选择 observed。\n\n'
    text+='# 4. 复现检查\n\n总体实施状态 **PASS**。原样重跑的 raw/mean/convergence/weights/embedding 等结果与旧 S4 hashes 完全一致；'
    text+='99 次输入多重集、仅 effect 改变、特征/坐标/graph/settings/IDs 不变全部通过；独立 Python 从 raw 曲线重算评分通过。'
    text+='预指定第 1 次重复的输入和全部科学输出 hashes 一致。所有原文件前后 hashes 不变。'
    text+='R 版本、架构、locale、library 和 sessionInfo 保存在每个 run/sessionInfo.txt；Python 包版本及源码/配置 hashes 见 config_v1.json。\n\n'
    text+='# 5. 原始 S4 GCCM 统计量\n\n'
    text+=table(['变量','starting rho','terminal rho','gain','trend','historical p','Fisher lower','通过','strength','weight'],[[r.feature,f'{r.rho_Lmin:.6f}',f'{r.rho_Lmax:.6f}',f'{r.gain:.6f}',str(r.spearman),f'{r.p_two_sided:.6g}',f'{r.CI_lower:.6f}',str(r.converged),f'{r.strength:.6f}',f'{r.weight:.6f}'] for r in observed.itertuples()])
    text+='\n# 6. 置换分布\n\nterminal rho 的分布如下；完整 gain/strength/normalized weight 分布见 variable_distribution_summary.csv。\n\n'
    text+=table(['变量','mean','p05','median','p95','max'],[[r.variable,*[f'{getattr(r,k):.6f}' for k in ['mean','p05','median','p95','max']]] for r in sd[sd.statistic=='terminal_rho'].itertuples()])
    text+='\n每次置换的 maximum normalized weight 与通过变量个数在 permutation_summary.csv。最大权重分布：`'+json.dumps(summary['maximum_normalized_weight_distribution'],ensure_ascii=False)+'`。'
    text+='全零 strength 的置换数为 '+str(summary['no_converged_feature_n'])+'，若有则权重保持 NA，不赋零或等权；相应权重分布只对可定义结果汇总。\n\n'
    text+='# 7. s_x 与 s_y 比较\n\n上尾概率=(1+置换统计量>=原始值的次数)/(1+99)，包含 ties；它是 permutation negative-control tail probability。\n\n'
    text+=table(['变量','统计量','原始值','null median','null p95','上尾次数/可用次数','尾概率'],[[r.variable,r.statistic,f'{r.observed:.6f}',f'{r.null_median:.6f}',f'{r.null_p95:.6f}',f'{r.count_ge_observed}/{r.n_permutations_available}',f'{r.negative_control_tail_probability:.4f}'] for r in tail.itertuples()])
    text+='\n主要统计量是预注册的两个 terminal rho；gain、strength、weight 为补充，不能挑其中最有利者。权重如存在 NA，其尾概率为明确标记的可定义条件分布。'
    text+=f'六变量最大 terminal rho 的补充尾概率为 {summary["maximum_terminal_rho_negative_control_tail_probability"]:.4f}。\n\n'
    text+='# 8. 负对照收敛频率\n\n'
    text+=table(['变量','通过次数/99','通过比例'],[[f,f'{summary["per_variable_pass_counts"][f]}/99',f'{summary["per_variable_pass_fractions"][f]:.2%}'] for f in FEATURES])
    text+=f'\n至少一个坐标通过：{summary["any_coordinate_converged_n"]}/99；两个坐标都通过：{summary["both_coordinates_converged_n"]}/99。'
    text+='通过变量个数分布：`'+json.dumps(summary['number_converged_distribution'])+'`。\n\n'
    text+='# 9. 本实验支持什么\n\n'+conclusion+' 归一化权重表示相对分配：只要有通过者就会归一到总和 1，大权重本身不等于强证据。\n\n'
    text+='# 10. 本实验不支持什么\n\n不能据此宣称 GCCM 一般错误，或已证明因果关系；也不能推断区间覆盖/宽度/IS。'
    text+='这只是固定 graph、200 点、单个 effect 边际分布下的无对应关系参考。原始 S4 的拟合残差包含 RF 预测误差，同方差真实噪声不保证实际残差尺度处处相同。'
    text+='置换后空间依赖和关联结构被破坏，本实验没有建立原始残差交换性或正式因果检验的条件。历史 Pearson/Fisher 数值不能当作新校准后的因果 p 值。'
    text+='没有读取 oracle 文件或使用隐藏 sigma 等真值作统计选择。\n\n'
    text+='# 11. 下一阶段决策\n\n'+conclusion+' 本实验结束后停止，等待研究者审阅。'
    text+='不自动运行 S1/S2/S3、residual-scale 比较、oracle 对照、final calibration 或 test；不修改任何门槛、图、权重规则或 predictor。'
    text+='本轮 final-calibration/test 标签保持未访问，未读取 tuning/validation 标签或 oracle 文件。\n'
    (HERE/'S4_GCCM_PERMUTATION_NEGATIVE_CONTROL_REPORT.md').write_text(text,encoding='utf-8')
    return conclusion

def main():
    try:
        assert not (HERE/'config_v1.json').exists(), 'Existing experiment: refuse overwrite or implicit rerun.'
        config=freeze();log('CONFIG AND 99 SEEDS FROZEN; observed reproduction begins')
        obs=HERE/'observed_reproduction';make_input(obs,0,None);run_r(obs)
        original_match={n:sha(obs/n)==sha(ORIGINAL/n) for n in RESULT_FILES}
        assert all(original_match.values()), original_match
        write_json(HERE/'observed_reproduction_check.json',{'status':'PASS','hash_matches':original_match})
        log('OBSERVED REPRODUCTION PASS; prepare all 99 inputs before estimation')
        for pid,seed in enumerate(SEEDS,1):make_input(HERE/'permutations'/f'perm_{pid:03d}',pid,seed)
        log('ALL 99 INPUT INVARIANTS PASS; permutation execution starts')
        completed=0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(run_r,HERE/'permutations'/f'perm_{pid:03d}'):pid for pid in range(1,100)}
            try:
                for future in as_completed(futures):
                    meta=future.result();completed+=1
                    log(f'PERMUTATION {futures[future]:03d} PASS; completed {completed}/99; runtime {meta["runtime_seconds"]:.2f}s')
            except BaseException:
                STOP.set()
                with LOCK:
                    for p in list(PROCESSES):p.terminate()
                for f in futures:f.cancel()
                raise
        repeated=HERE/'repeat_perm_001';make_input(repeated,1,SEEDS[0]);run_r(repeated)
        keys=['restricted_input.csv','edges.csv','parameters.csv','execution_config.json']+RESULT_FILES
        repeat_match={n:sha(repeated/n)==sha(HERE/'permutations/perm_001'/n) for n in keys}
        assert all(repeat_match.values()),repeat_match
        assert all(sha(Path(p))==h for p,h in config['source_hashes'].items())
        assert sha(HERE/'permutation_seeds.csv')==config['seed_list_sha256']
        assert sha(Path(__file__))==config['runner_sha256']
        assert all(sha(HERE/'frozen_sources'/n)==h for n,h in config['snapshot_hashes'].items())
        log('ALL REQUIRED IMPLEMENTATION CHECKS PASS; statistical summaries now authorized')
        summary,observed,sd,tail=aggregate(config)
        check={'overall_status':'PASS','permutations_completed':99,'observed_reproduction':original_match,
            'all_99_effect_multiset_checks':'PASS','features_coordinates_ids_unchanged':'PASS',
            'graph_settings_unchanged':'PASS','independent_scoring_checks':'PASS','selected_repeat':repeat_match,
            'existing_outputs_unchanged':'PASS','configuration_and_seeds_frozen_before_execution':True,
            'tuning_calibration_labels_accessed':False,'validation_labels_accessed':False,
            'final_calibration_labels_accessed':False,'test_labels_accessed':False,'oracle_files_read_or_hashed':False,
            'predictor_retrained':False,'primary_validation_comparison_executed':False,
            'access_scope':'Existing S4 scale-fit preparation and GCCM artifacts/config/source only; explicit file paths, no OS sandbox claim.',
            'no_converged_feature_n':summary['no_converged_feature_n']}
        conclusion=report(summary,observed,sd,tail,check)
        write_json(HERE/'implementation_check.json',check)
        log('EXPERIMENT PASS; 99/99; STOP AFTER S4 NEGATIVE CONTROL')
        hashes={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
        write_json(HERE/'artifact_manifest.json',hashes)
        (HERE/'created_files.txt').write_text('\n'.join(sorted(list(hashes)+['artifact_manifest.json','created_files.txt']))+'\n',encoding='utf-8')
        print('A. Experiment status: PASS\nB. Number of permutations completed: 99')
        print('C/D. Original statistics and negative-control positions:\n'+tail.to_string(index=False))
        print('E. Convergence frequencies:',summary['per_variable_pass_counts'])
        print('F.',conclusion,'\nG. Exact files:',HERE/'created_files.txt')
        print('H. Final/test remained blind; no other experiment executed.')
        return 0
    except BaseException as exc:
        STOP.set()
        with LOCK:
            for p in list(PROCESSES):p.terminate()
        failure={'overall_status':'FAIL','error':repr(exc),'statistical_interpretation_permitted':False,
                 'final_calibration_labels_accessed':False,'test_labels_accessed':False}
        if not (HERE/'implementation_check.json').exists():write_json(HERE/'implementation_check.json',failure)
        log('FAIL '+repr(exc));raise

if __name__=='__main__':
    sys.exit(main())
