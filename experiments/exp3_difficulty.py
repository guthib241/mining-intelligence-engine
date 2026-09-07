import pandas as pd, numpy as np
from scipy import stats
df=pd.read_pickle('headers.pkl')
t=df.time.values.astype(np.int64); D=df.difficulty.values
n=len(df); E=n//2016
# per-epoch: difficulty, start time, end time (bitcoin uses time[first]..time[last of 2015] i.e. 2015 intervals)
ep=np.arange(E)
D_e=D[ep*2016]; t0=t[ep*2016]; t_last=t[ep*2016+2015]
span=(t_last-t0).astype(float)
ratio=D_e[1:]/D_e[:-1]   # actual adjustment ratio into epoch e+1
# sanity: implied = clamp(2016*600/span,0.25,4)
implied=np.clip(2016*600/span[:-1],0.25,4)
print("adjustment formula check, median rel err:",np.median(np.abs(implied/ratio-1)), " 2016+ max:",np.max(np.abs(implied[220:]/ratio[220:]-1)))
# --- Momentum test: is log(ratio) autocorrelated across epochs? (regime: post-2016 = epoch >= 220)
lr=np.log(ratio)
for start,name in [(0,'all'),(220,'2016+'),(330,'2020+'),(400,'2023+')]:
    x=lr[start:]; r=np.corrcoef(x[:-1],x[1:])[0,1]; print(f"log-adj autocorr lag1 [{name}] n={len(x)} r={r:+.3f} p={stats.pearsonr(x[:-1],x[1:]).pvalue:.4f}")
# --- Forecasting at k blocks into epoch. Walk-forward: models fit on epochs < e only.
def forecast_eval(k, start=330):
    rows=[]
    for e in range(start,E-1):
        base=e*2016
        elapsed=(t[base+k]-t[base]) if k>0 else 0
        actual=lr[e]              # log(D_{e+1}/D_e)
        naive=0.0
        # pace extrapolation: remaining blocks at expected 600s? no: at observed pace
        if k>=30:
            pace=elapsed/k; extrap=np.log(np.clip(2016*600/(pace*2015),0.25,4))
        else: extrap=np.nan
        # momentum: previous epoch's adjustment
        mom=lr[e-1]
        # fitted blend on past epochs: regress actual on [extrap,mom] using epochs [start-100, e)
        rows.append((e,actual,naive,extrap,mom))
    r=pd.DataFrame(rows,columns=['e','actual','naive','extrap','mom'])
    # walk-forward OLS: actual ~ a + b*extrap + c*mom, trained on prior epochs
    preds=[]
    for i in range(len(r)):
        tr=r.iloc[max(0,i-60):i]
        if len(tr)<20 or k<30: preds.append(np.nan); continue
        X=np.c_[np.ones(len(tr)),tr.extrap,tr.mom]; beta=np.linalg.lstsq(X,tr.actual,rcond=None)[0]
        preds.append(beta@[1,r.extrap.iloc[i],r.mom.iloc[i]])
    r['blend']=preds
    # momentum-only walk-forward (for k<30 too)
    pm=[]
    for i in range(len(r)):
        tr=r.iloc[max(0,i-60):i]
        if len(tr)<20: pm.append(np.nan); continue
        X=np.c_[np.ones(len(tr)),tr.mom]; beta=np.linalg.lstsq(X,tr.actual,rcond=None)[0]
        pm.append(beta@[1,r.mom.iloc[i]])
    r['mom_fit']=pm
    out={}
    for c in ['naive','extrap','mom_fit','blend']:
        m=r[c].notna()&r.blend.notna() if k>=30 else r[c].notna()&r.mom_fit.notna()
        err=np.abs(r.actual[m]-r[c][m]); out[c]=round(float(err.mean())*100,2)
    return out,r
print("\nMean abs error in % of difficulty, forecast made k blocks into epoch (walk-forward, epochs 2020+):")
print(f"{'k':>5} {'naive':>7} {'extrap':>7} {'mom':>7} {'blend':>7}")
res={}
for k in [0,100,300,500,1000,1500]:
    o,r=forecast_eval(k); res[k]=r; print(f"{k:5d} {o['naive']:7} {o.get('extrap','-'):>7} {o['mom_fit']:7} {o.get('blend','-'):>7}")
# statistical test: blend vs extrap at k=300 paired
r=res[300].dropna(); e1=np.abs(r.actual-r.extrap); e2=np.abs(r.actual-r.blend)
print("\nk=300 paired test blend<extrap: n=",len(r)," wilcoxon p=",stats.wilcoxon(e1,e2,alternative='greater').pvalue, " win rate=",round(float((e2<e1).mean()),3))
r=res[0].dropna(subset=['mom_fit']); e1=np.abs(r.actual-r.naive); e2=np.abs(r.actual-r.mom_fit)
print("k=0   paired test mom<naive:    n=",len(r)," wilcoxon p=",stats.wilcoxon(e1,e2,alternative='greater').pvalue, " win rate=",round(float((e2<e1).mean()),3))

# --- Decompose k=0: drift-only vs drift+momentum
r=res[0].copy()
dr=[]
for i in range(len(r)):
    tr=r.iloc[max(0,i-60):i]; dr.append(tr.actual.mean() if len(tr)>=20 else np.nan)
r['drift']=dr; r=r.dropna(subset=['drift','mom_fit'])
for c in ['naive','drift','mom_fit']: print(f"k=0 {c:8s} MAE={np.abs(r.actual-r[c]).mean()*100:.2f}%")
# --- Poisson noise floor: sd of log adjustment from block-time randomness alone
print("Poisson floor on full-epoch adjustment sd ~", round(100/np.sqrt(2015),2),"%  (naive MAE implies total sd ~",round(3.61*1.25,2),"%)")
# --- Regime robustness: blend at k=500 by year bucket
o,r=forecast_eval(500); r=r.dropna()
yr=pd.to_datetime(t[r.e.values*2016],unit='s').year
for y in sorted(set(yr)):
    m=yr==y; print(f"  {y}: n={m.sum():2d} naive={np.abs(r.actual[m]-r.naive[m]).mean()*100:.2f}  extrap={np.abs(r.actual[m]-r.extrap[m]).mean()*100:.2f}  blend={np.abs(r.actual[m]-r.blend[m]).mean()*100:.2f}")
