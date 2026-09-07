import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from lib.difficulty import walk_forward, mae_table, epoch_table
from lib.kalman import forecast
from scipy import stats
df=pd.read_pickle('headers.pkl'); t=df.time.values.astype(np.int64); D=df.difficulty.values
E,_,lr=epoch_table(t,D)
def run(t,D,k,start,q):
    E,_,lr=epoch_table(t,D)
    return np.array([forecast(t,D,k,e,q) for e in range(start,E-1)]), lr[start:E-1]
# tune q on 2016-2019 epochs (220..329), evaluate on 330+
print("q tuning (train epochs 220-329, k=0):")
best=None
for q in [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2]:
    p,y=run(t,D,0,220,q); p,y=p[:110],y[:110]; m=np.abs(p-y).mean()*100
    print(f"  q={q:g}: MAE={m:.2f}"); best=(m,q) if best is None or m<best[0] else best
q=best[1]; print("chosen q",q)
print("\nTEST epochs 330+ (2020-2025):  naive / blend(EXP3) / kalman")
res={}
for k in [0,100,300,500,1000,1500]:
    r=walk_forward(t,D,k,330); mt=mae_table(r)
    p,y=run(t,D,k,330,q); m=r.blend.notna() if k>=30 else r.mom_fit.notna()
    kal=np.abs(p[m.values]-y[m.values]).mean()*100
    res[k]=(r,p,y,m)
    print(f"  k={k:5d}: naive {mt['naive']:.2f}  blend {mt.get('blend',mt['mom_fit']):.2f}  kalman {kal:.2f}")
# paired tests k=0 and k=500
for k in [0,500]:
    r,p,y,m=res[k]; comp=r.blend if k>=30 else r.mom_fit
    e_b=np.abs(comp[m]-r.actual[m]).values; e_k=np.abs(p[m.values]-y[m.values]); e_n=np.abs(r.actual[m]).values
    print(f"k={k}: kalman vs blend wilcoxon p={stats.wilcoxon(e_b,e_k,alternative='greater').pvalue:.4f} win={np.mean(e_k<e_b):.2f};  kalman vs naive p={stats.wilcoxon(e_n,e_k,alternative='greater').pvalue:.2e}")
# regime by year, k=0 and k=500
for k in [0,500]:
    r,p,y,m=res[k]; yr=pd.to_datetime(t[r.e.values*2016],unit='s').year
    print(f"k={k} by year:")
    for Y in sorted(set(yr)):
        mm=(yr==Y)&m.values; comp=(r.blend if k>=30 else r.mom_fit).values
        print(f"   {Y}: n={mm.sum():2d} naive={np.abs(r.actual.values[mm]).mean()*100:.2f} blend={np.abs(comp[mm]-r.actual.values[mm]).mean()*100:.2f} kalman={np.abs(p[mm]-y[mm]).mean()*100:.2f}")
# synthetic constant hashrate check
sys.path.insert(0,'experiments')
rng=np.random.default_rng(1)
def synth(hash_path, seed):
    r=np.random.default_rng(seed); Dc=1.0; ts=[0]; Ds=[]
    for e in range(len(hash_path)):
        H=hash_path[e]; iv=r.exponential(600*Dc/H,2016)
        for x in iv: ts.append(ts[-1]+x)
        Ds+=[Dc]*2016; span=ts[-1]-ts[-2016-1]-iv[-1]; Dc=Dc*np.clip(2016*600/span,0.25,4)
    return np.array(ts[1:]).astype(np.int64), np.array(Ds)
ts,Ds=synth(np.ones(400),1)
print("\nSynthetic constant-hashrate (floor at k=0 = 1.78):")
for k in [0,500]:
    p,y=run(ts,Ds,k,60,q); print(f"  k={k}: kalman {np.abs(p-y).mean()*100:.2f}  (naive 2.83, blend {mae_table(walk_forward(ts,Ds,k,60)).get('blend','-')})")
# error distribution at k=0 real
r,p,y,m=res[0]; err=(p[m.values]-y[m.values])*100
print("\nk=0 real signed error % percentiles:",np.percentile(err,[1,5,10,25,50,75,90,95,99]).round(2), "mean",err.mean().round(2),"sd",err.std().round(2))
