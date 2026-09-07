import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json
from scipy import stats
z=np.load('headers_compact.npz'); t=z['time'].astype(np.int64); D=z['difficulty']
# daily hashrate proxy: 144-block windows, H_proxy = D * 144*600 / span  (log). Poisson sd of log ~ 1/sqrt(144)=8.3%
n=len(t)//144*144; W=t[:n].reshape(-1,144); Dw=D[:n].reshape(-1,144)[:,0]
span=(W[:,-1]-W[:,0]).astype(float); logH=np.log(Dw)+np.log(143*600/span)
day=pd.to_datetime(W[:,0],unit='s')
S=pd.DataFrame({'day':day,'logH':logH}); S=S[S.day>=pd.Timestamp('2018-01-01')].reset_index(drop=True)
S['dH']=S.logH.diff()               # daily change (noisy, sd~12% from Poisson alone)
# smoothed weekly change target: mean logH next 7 windows minus mean of last 7 (reduces Poisson noise to ~3%)
S['H7_fwd']=S.logH[::-1].rolling(7).mean()[::-1].shift(-1); S['H7_bwd']=S.logH.rolling(7).mean()
S['target']=S.H7_fwd-S.H7_bwd
print("daily windows:",len(S),"; dH sd",S.dH.std().round(3),"(Poisson-only ~.118); weekly target sd",S.target.std().round(3))
print("dH autocorr lag1..3:",[round(S.dH.autocorr(l),3) for l in (1,2,3)]," (Poisson MA(1) expects lag1 ≈ -0.5)")
px=pd.read_csv('btc/btc_prices.csv'); px.columns=['date','o','h','l','c','v']; px['date']=pd.to_datetime(px.date); lp=np.log(px.set_index('date').c)
d0=S.day.dt.normalize()-pd.Timedelta(days=1)
for nlag in [7,14,28,56,90]:
    S[f'r{nlag}']=[float(lp.asof(d)-lp.asof(d-pd.Timedelta(days=nlag))) if d-pd.Timedelta(days=nlag)>=lp.index[0] else np.nan for d in d0]
S['Hmom']=S.H7_bwd-S.logH.rolling(14).mean()   # recent hashrate momentum (known)
V=S.dropna(subset=['target','r7','r14','r28','r56','r90','Hmom']); V=V[V.day<pd.Timestamp('2025-12-01')]
print("n weekly-target rows:",len(V))
# overlapping windows -> use Newey-West via block bootstrap on correlation (block=14)
def block_boot_r(x,y,B=500,blk=14,seed=0):
    r=np.random.default_rng(seed); n=len(x); obs=np.corrcoef(x,y)[0,1]; rs=[]
    for _ in range(B):
        idx=np.concatenate([np.arange(s,min(s+blk,n)) for s in r.integers(0,n,n//blk+1)])[:n]
        rs.append(np.corrcoef(x[idx],y[idx])[0,1])
    return obs, float(np.std(rs))
print("\nPrice return features (known at day t) vs next-week hashrate change (block-bootstrap se):")
for f in ['r7','r14','r28','r56','r90','Hmom']:
    obs,se=block_boot_r(V[f].values,V.target.values); print(f"  {f:5s} r={obs:+.3f} se={se:.3f} z={obs/se:+.2f}")
# Walk-forward test: predict target with [Hmom] vs [Hmom, r56], 365-day window, from 2020
V=V.reset_index(drop=True); pred_b=[];pred_m=[]
for i in range(len(V)):
    tr=V.iloc[max(0,i-365):i]; tr=tr[tr.day<V.day.iloc[i]-pd.Timedelta(days=8)]   # gap to avoid target overlap leakage
    if len(tr)<200 or V.day.iloc[i]<pd.Timestamp('2020-01-01'): pred_b.append(np.nan); pred_m.append(np.nan); continue
    Xb=np.c_[np.ones(len(tr)),tr.Hmom]; bb=np.linalg.lstsq(Xb,tr.target,rcond=None)[0]; pred_b.append(bb@[1,V.Hmom.iloc[i]])
    Xm=np.c_[np.ones(len(tr)),tr.Hmom,tr.r56]; bm=np.linalg.lstsq(Xm,tr.target,rcond=None)[0]; pred_m.append(bm@[1,V.Hmom.iloc[i],V.r56.iloc[i]])
V['pb']=pred_b; V['pm']=pred_m; T=V.dropna(subset=['pb'])
e0=np.abs(T.target); eb=np.abs(T.target-T.pb); em=np.abs(T.target-T.pm)
print(f"\nwalk-forward 2020+ n={len(T)}: MAE zero {e0.mean()*100:.2f}%  Hmom {eb.mean()*100:.2f}%  Hmom+r56 {em.mean()*100:.2f}%")
# block-wise paired test (14-day blocks) to respect overlap
blocks=np.arange(len(T))//14; db=pd.DataFrame({'b':blocks,'d':(eb-em).values}).groupby('b').d.mean()
print(f"  paired (14d blocks n={len(db)}): mean improvement {db.mean()*100:+.3f}pp, t={db.mean()/db.std()*np.sqrt(len(db)):+.2f}, Wilcoxon p={stats.wilcoxon(db,alternative='greater').pvalue:.3f}")
json.dump({"n":int(len(T)),"MAE_zero":float(e0.mean()),"MAE_Hmom":float(eb.mean()),"MAE_Hmom_r56":float(em.mean())},open('research_ledger/exp13.json','w'))
