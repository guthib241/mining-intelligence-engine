"""Difficulty-adjustment forecasting pipeline. Pure functions over (timestamps, difficulty) arrays so synthetic data can use the identical code path."""
import numpy as np, pandas as pd
def epoch_table(t, D):
    n=len(t); E=n//2016; ep=np.arange(E)
    D_e=D[ep*2016]; t0=t[ep*2016]
    lr=np.log(D_e[1:]/D_e[:-1])            # actual log adjustment into epoch e+1
    return E, t0, lr
def features(t, k, e, lr):
    base=e*2016
    if k>=30:
        elapsed=t[base+k]-t[base]; pace=elapsed/k
        extrap=np.log(np.clip(2016*600/(pace*2015),0.25,4))
    else: extrap=np.nan
    return extrap, lr[e-1]
def walk_forward(t, D, k, start, window=60, min_train=20, target=None):
    E,t0,lr=epoch_table(t,D)
    y = lr if target is None else target
    rows=[]
    for e in range(start,E-1):
        ex,mom=features(t,k,e,lr); rows.append((e,y[e],ex,mom))
    r=pd.DataFrame(rows,columns=['e','actual','extrap','mom'])
    r['naive']=0.0
    def fit(cols):
        out=[]
        for i in range(len(r)):
            tr=r.iloc[max(0,i-window):i]
            if len(tr)<min_train or (('extrap' in cols) and k<30): out.append(np.nan); continue
            X=np.c_[np.ones(len(tr)),tr[cols].values]; beta=np.linalg.lstsq(X,tr.actual.values,rcond=None)[0]
            out.append(beta@np.r_[1,r[cols].iloc[i].values])
        return out
    r['drift']=fit([]); r['mom_fit']=fit(['mom']); r['blend']=fit(['extrap','mom']); r['extrap_fit']=fit(['extrap'])
    return r
def mae_table(r, cols=('naive','extrap','drift','mom_fit','extrap_fit','blend')):
    m=r.blend.notna() if r.blend.notna().any() else r.mom_fit.notna()
    return {c: round(float(np.abs(r.actual[m]-r[c][m]).mean()*100),2) for c in cols if r[c][m].notna().any()}
