"""Difficulty forecasters + conformal quantiles. All models: inputs known at block e*2016+k only."""
import numpy as np, pandas as pd
from lib.difficulty import epoch_table
QS=[1,5,10,25,50,75,90,95,99]
def pace_extrap(t,D,e,k,win=None,halflife=None):
    """log-adjustment implied by observed pace. win: use only last `win` blocks; halflife: EWMA of block intervals."""
    base=e*2016
    if k<30: return np.nan
    if halflife:
        iv=np.diff(t[base:base+k+1]).astype(float); w=0.5**(np.arange(k)[::-1]/halflife); pace=np.sum(w*iv)/np.sum(w)
    elif win and k>win: pace=(t[base+k]-t[base+k-win])/win
    else: pace=(t[base+k]-t[base])/k
    elapsed=t[base+k]-t[base]; rem=(2015-k)*pace
    return float(np.log(np.clip(2016*600/(elapsed+rem),0.25,4)))
def build(t,D,k,start,models):
    E,_,lr=epoch_table(t,D); rows=[]
    for e in range(start,E-1):
        r={'e':e,'actual':lr[e],'mom':lr[e-1]}
        for name,fn in models.items(): r[name]=fn(e)
        rows.append(r)
    return pd.DataFrame(rows)
def walk(df,feature_sets,window=60,min_train=20):
    """Walk-forward OLS for each feature set; also conformal residual quantiles from the training window."""
    out={}
    for name,cols in feature_sets.items():
        pred=[];q=[]
        for i in range(len(df)):
            tr=df.iloc[max(0,i-window):i].dropna(subset=cols+['actual']) if cols else df.iloc[max(0,i-window):i]
            if len(tr)<min_train or (cols and pd.isna(df[cols].iloc[i]).any()): pred.append(np.nan); q.append([np.nan]*len(QS)); continue
            X=np.c_[np.ones(len(tr)),tr[cols].values] if cols else np.ones((len(tr),1))
            beta=np.linalg.lstsq(X,tr.actual.values,rcond=None)[0]
            fit=X@beta; res=tr.actual.values-fit
            p=beta@np.r_[1,df[cols].iloc[i].values] if cols else beta[0]
            pred.append(p); q.append(list(p+np.percentile(res,QS)))
        out[name]=np.array(pred); out[name+'_q']=np.array(q)
    return out
def pinball(y,q,tau): d=y-q; return np.mean(np.maximum(tau*d,(tau-1)*d))
def evaluate(df,out,names):
    m=np.all([~np.isnan(out[n]) for n in names],axis=0); y=df.actual.values[m]; res={}
    for n in names:
        p=out[n][m]; Q=out[n+'_q'][m]
        cov90=np.mean((y>=Q[:,1])&(y<=Q[:,7])); cov50=np.mean((y>=Q[:,3])&(y<=Q[:,5]))
        pb=np.mean([pinball(y,Q[:,j],QS[j]/100) for j in range(len(QS))])
        res[n]=dict(n=int(m.sum()),MAE=round(float(np.abs(y-p).mean())*100,2),RMSE=round(float(np.sqrt(np.mean((y-p)**2)))*100,2),cov50=round(float(cov50),2),cov90=round(float(cov90),2),pinball=round(float(pb)*100,3),width90=round(float(np.mean(Q[:,7]-Q[:,1]))*100,2))
    return res,m
