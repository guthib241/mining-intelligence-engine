"""Per-observer calibrated propagation metric (vectorized)."""
import numpy as np, pandas as pd, glob, os
def load_observers(path='block-arrival-times/data', headers='headers.pkl'):
    H=pd.read_pickle(headers)[['height','hash']]
    rows=[pd.read_csv(f,header=None,names=['height','hash','ms']).assign(src=os.path.basename(f)[:-4]) for f in sorted(glob.glob(path+'/*.csv')) if 'stratum' not in f]
    A=pd.concat(rows,ignore_index=True).merge(H,on='height',suffixes=('','_hdr')); A=A[A.hash==A.hash_hdr].drop(columns='hash_hdr'); A['ts']=A.ms/1000
    return A
def calibrate(A):
    A['ref']=A.groupby('height').ts.transform('median'); A['dev']=A.ts-A.ref
    A['mo']=pd.to_datetime(A.ts,unit='s').dt.to_period('M')
    grp=A.groupby(['src','mo']); A['offset']=grp.dev.transform('median'); A['absdev']=(A.dev-A.offset).abs(); A['mad']=A.groupby(['src','mo']).absdev.transform('median')
    n=grp.dev.transform('size'); tot=A.groupby('mo').height.transform('nunique'); A['reliability']=n/tot
    A['ts_cal']=A.ts-A.offset
    off=A.groupby(['src','mo']).agg(offset=('offset','first'),mad=('mad','first'),reliability=('reliability','first'),n=('dev','size')).reset_index()
    return A, off
def block_metrics(A, reliable=0.8, mad_max=2.0, min_obs=3):
    R=A[(A.reliability>=reliable)&(A.mad<=mad_max)]
    g=R.groupby('height').ts_cal.agg(['size','min','max']); q=R.groupby('height').ts_cal.quantile([.25,.75]).unstack()
    g=g.join(q); g.columns=['n','cal_min','cal_max','cal_p25','cal_p75']; g=g[g.n>=min_obs]
    g['spread_cal']=g.cal_max-g.cal_min; g['iqr_cal']=g.cal_p75-g.cal_p25; return g
