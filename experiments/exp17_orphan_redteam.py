import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, glob, os, json
from lib.propagation import *
H=pd.read_pickle('headers.pkl')[['height','hash','time']]
dbg=[f for f in glob.glob('block-arrival-times/data/*.csv') if any(s in f for s in ['monitoring1','rs2','peer-observer','n-thumann','darosior','vostrnad'])]
Ad=pd.concat([pd.read_csv(f,header=None,names=['height','hash','ms']).assign(src=os.path.basename(f)[:-4]) for f in dbg]).merge(H,on='height',suffixes=('','_hdr'))
comp_h=set(Ad[Ad.hash!=Ad.hash_hdr].height); cov=Ad.groupby('height').src.nunique()
A=load_observers(); A,off=calibrate(A); g=pd.read_pickle('arrivals.pkl')
def run(Asub,label,exclude_adjacent=False,era=None):
    B=block_metrics(Asub); X=pd.DataFrame(index=cov.index); X['comp']=X.index.isin(comp_h); X['prev_spread']=B.spread_cal.reindex(X.index).shift(1)
    X['first']=g['first'].reindex(X.index); X['yr']=pd.to_datetime(X['first'],unit='s').dt.year
    if exclude_adjacent: X=X[~X.index.isin({h+1 for h in comp_h})]
    if era: X=X[X.yr.isin(era)]
    X=X.dropna(subset=['prev_spread']); hi=X.prev_spread>2
    y=X.comp.values; obs=y[hi].mean()-y[~hi].mean(); r=np.random.default_rng(0); null=[abs((yp:=r.permutation(y))[hi].mean()-yp[~hi].mean()) for _ in range(3000)]
    print(f"{label:38s} n={len(X):6d} comp={int(y.sum()):3d} hi-spread n={int(hi.sum()):5d} rate hi={y[hi].mean()*1000:.2f} lo={y[~hi].mean()*1000:.2f} /1000  perm p={np.mean(np.array(null)>=abs(obs)):.4f}")
run(A,"baseline (all observers)")
run(A,"exclude h where h-1 also competing",exclude_adjacent=True)
print("adjacent competing pairs:",sum(1 for h in comp_h if h-1 in comp_h))
run(A[A.src!='0xb10c_peer-observer-frank'],"drop frank")
run(A[~A.src.isin(['0xb10c_peer-observer-frank','n-thumann','vostrnad_node1'])],"drop frank,n-thumann,vostrnad")
run(A,"era 2021-2023 only",era=[2021,2022,2023])
run(A,"era 2024-2025 only",era=[2024,2025])
run(A[A.src!='0xb10c_peer-observer-frank'],"era 2024-25, drop frank",era=[2024,2025])
# Mechanism probe: is high prev_spread itself caused by a competing block at h-1 that we did NOT observe (only some sources log losers)? Compare spread at known competing heights vs others.
B=block_metrics(A); sc=B.spread_cal.reindex(list(comp_h)).dropna(); print(f"\ncalibrated spread AT competing heights: median {sc.median():.2f}s (P75 {sc.quantile(.75):.2f}) vs all blocks median {B.spread_cal.median():.2f}s -> races inflate spread by construction")
# temporal ordering check: does spread at h (not h-1) predict comp at h+1?  and does comp at h predict spread at h+1? (reverse direction)
X=pd.DataFrame(index=cov.index); X['comp']=X.index.isin(comp_h); X['sp']=B.spread_cal.reindex(X.index); X['sp_next']=X.sp.shift(-1)
print("reverse: spread at h+1 when h competing: median",round(X.sp_next[X.comp].median(),2),"vs",round(X.sp_next[~X.comp].median(),2))
