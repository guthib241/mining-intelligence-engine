import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats
from lib.difficulty import epoch_table, walk_forward
df=pd.read_pickle('headers.pkl'); nz=df.nonce.values.astype(np.uint64); t=df.time.values.astype(np.int64); D=df.difficulty.values
E,_,lr=epoch_table(t,D)
ep=np.arange(E*2016)//2016
F=pd.DataFrame({'ep':ep,'b7':((nz[:E*2016]>>np.uint64(7))&1),'b8':((nz[:E*2016]>>np.uint64(8))&1),'b31':((nz[:E*2016]>>np.uint64(31))&1),'hiE':((nz[:E*2016]&np.uint64(0xE0))==0xE0)}).groupby('ep').mean()
# hashrate growth per epoch (observed): g[e] = logH[e]-logH[e-1] using epoch spans
from lib.kalman import epoch_hash_obs
_,logH=epoch_hash_obs(t,D); g=np.diff(logH)   # g[e-1] = growth into epoch e
start=220; end=E-1
res=[]
for f in F.columns:
    x=F[f].values; dx=np.diff(x)     # dx[e-1] = change from epoch e-1 to e
    for lag in [-1,0,1,2]:
        # correlate dx at epoch e with g at epoch e+lag  (lag>0: fingerprint leads hashrate)
        a=[];b=[]
        for e in range(start,end-3):
            a.append(dx[e-1]); b.append(g[e-1+lag])
        r,p=stats.pearsonr(a,b); sr,sp=stats.spearmanr(a,b)
        res.append((f,lag,len(a),round(r,3),p,round(sr,3),sp))
r=pd.DataFrame(res,columns=['feat','lag','n','pearson','p','spearman','sp']); print(r.to_string())
print("Bonferroni threshold p<",0.05/len(r))
# Does level or change of fingerprint predict blend residual at k=0 (out-of-sample residual)?
wf=walk_forward(t,D,0,330); wf=wf.dropna(subset=['mom_fit']); resid=(wf.actual-wf.mom_fit).values; ee=wf.e.values
for f in ['b7','b31','hiE']:
    dx=np.diff(F[f].values); x=dx[ee-1]   # change into current epoch e (known at k=0? NO: fingerprint of epoch e is not known at k=0). Use dx[ee-2] (change into e-1), known.
    xk=dx[ee-2]
    print(f"{f}: Δ(e-1) vs k=0 residual: r={stats.pearsonr(xk,resid)[0]:+.3f} p={stats.pearsonr(xk,resid)[1]:.3f}")
