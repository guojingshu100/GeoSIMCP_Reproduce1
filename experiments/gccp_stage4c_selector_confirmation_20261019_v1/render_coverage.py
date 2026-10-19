"""Presentation-only renderer: fixed method-to-color mapping across different k panels.

Reads Stage4C summary CIs; never changes models, protocol, metrics or inference.
"""
import os,sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'runtime_cache'/'matplotlib')
import csv,shutil,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLES={
    'GeoSIMCP-RC':('#666666','o'),
    'GCCP-OracleParent-RC':('#2166AC','o'),
    'GCCP-ResidualParent-RC':('#B2182B','s'),
    'RandomMask-k0-RC-Mean10':('#BBBBBB','^'),
    'RandomMask-k1-RC-Mean10':('#1B9E77','^'),
    'RandomMask-k2-RC-Mean10':('#66A61E','^'),
    'SupervisedSelector-k0-RC':('#C3B1DB','D'),
    'SupervisedSelector-k1-RC':('#7570B3','D'),
    'SupervisedSelector-k2-RC':('#E7298A','D'),
    'GCCP-WrongPrior-RC':('#000000','x')}

def render(tag):
    with (HERE/'summary_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,3,figsize=(15,9.5),layout='constrained');handles={}
    for i,ax in enumerate(axes.flat):
        scenario='S%d'%i
        for method,(color,marker) in STYLES.items():
            matches=[r for r in rows if r['scenario']==scenario and r['method']==method]
            if not matches:continue
            a=matches[0];x=float(a['coverage_mean']);y=float(a['width_mean'])
            h=ax.errorbar(x,y,xerr=[[max(0,x-float(a['coverage_ci95_low']))],[max(0,float(a['coverage_ci95_high'])-x)]],
                yerr=[[max(0,y-float(a['width_ci95_low']))],[max(0,float(a['width_ci95_high'])-y)]],
                fmt=marker,color=color,label=method,ms=5,elinewidth=1.1)
            handles[method]=h
        ax.axvline(.90,color='black',ls='--',lw=.8);ax.set_title(scenario)
        ax.set_xlabel('Coverage, replicate t95 CI');ax.set_ylabel('Full width, replicate t95 CI')
    order=[m for m in STYLES if m in handles]
    fig.legend([handles[m] for m in order],order,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Fresh independent Stage4C samples; fixed method colors across all k\nSame RF, roles, kernel parameters and final rank; k0 uses ordinary fallback')
    target=HERE/'figures/coverage_efficiency.png';fig.savefig(target,dpi=170);plt.close(fig)
    if tag:shutil.copy2(target,HERE/'checkpoints'/tag/'figures/coverage_efficiency.png')
    print('Coverage figure rendered with consistent method identity; no metric/inference changes.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='');a=ap.parse_args();render(a.tag)
