import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats
from lib.difficulty import epoch_table, walk_forward
from lib.kalman import epoch_hash_obs
px=pd.read_csv('btc/btc_prices.csv'); px.columns=['date','o','h','l','c','v']; px['date']=pd.to_datetime(px.date); px=px.set_index('date').sort_index()
# validation
print("rows",len(px),"gaps(days>1):",(px.index.to_series().diff().dt.days>1).sum(),"dupes:",px.index.duplicated().sum())
for d,exp in [('2020-03-12','~4900-5000 close'),('2021-11-09','~66-69k'),('2024-03-13','~73k'),('2022-11-09','~15.9k')]:
    print(" ",d,px.loc[d,'c'],exp)
print("neg/zero prices:",(px.c<=0).sum()," max daily |ret|:",np.abs(np.log(px.c).diff()).max().round(3))
df=pd.read_pickle('headers.pkl'); t=df.time.values.astype(np.int64); D=df.difficulty.values
E,_,lr=epoch_table(t,D); _,logH=epoch_hash_obs(t,D); g=np.diff(logH)   # g[e-1]=growth of avg hashrate from epoch e-1 to e
lp=np.log(px.c)
def feat_at(ts):
    d=pd.Timestamp(ts,unit='s').normalize()-pd.Timedelta(days=1)   # last full day before epoch start (no lookahead)
    if d<lp.index[0]+pd.Timedelta(days=200) or d>lp.index[-1]: return None
    r=lambda n: float(lp.asof(d)-lp.asof(d-pd.Timedelta(days=n)))
    ma200=float(lp.asof(d)-lp[:d].tail(200).mean())
    return dict(r14=r(14),r28=r(28),r56=r(56),r90=r(90),r180=r(180),ma200=ma200)
rows=[]
for e in range(1,E-1):
    f=feat_at(t[e*2016])
    if f is None: continue
    f.update(e=e,g_next=g[e-1],g_next2=g[e] if e<E-2 else np.nan,g_prev=g[e-2]); rows.append(f)
R=pd.DataFrame(rows); print("epochs with features:",len(R), "range",R.e.min(),R.e.max())
feats=['r14','r28','r56','r90','r180','ma200']
print("\nCorrelation of price features (known at epoch start) with hashrate growth during this epoch (g_next) and next (g_next2):")
for f in feats:
    a=stats.pearsonr(R[f],R.g_next); b=stats.spearmanr(R[f],R.g_next); c=stats.pearsonr(R[f].iloc[:-1],R.g_next2.dropna())
    print(f"  {f:6s} g_next: r={a[0]:+.3f} p={a[1]:.4f} (spearman {b[0]:+.3f} p={b[1]:.4f})   g_next2: r={c[0]:+.3f} p={c[1]:.4f}")
print("Bonferroni thr:",0.05/(len(feats)*2))
# partial: control for g_prev (hashrate momentum)
import numpy.linalg as la
for f in ['r56','r90','r180']:
    X=np.c_[np.ones(len(R)),R.g_prev,R[f]]; b=la.lstsq(X,R.g_next,rcond=None)[0]; res=R.g_next-X@b
    se=np.sqrt(np.sum(res**2)/(len(R)-3)*la.inv(X.T@X)[2,2]); print(f"  {f} coef={b[2]:+.4f} t={b[2]/se:+.2f} (controlling g_prev)")
# Walk-forward: does adding best price feature to champion blend at k=0 and k=500 reduce MAE?  test epochs 330+
for k in [0,500]:
    wf=walk_forward(t,D,k,330); wf=wf.merge(R[['e','r90','r180','ma200']],on='e')
    base='blend' if k>=30 else 'mom_fit'; wf=wf.dropna(subset=[base])
    cols=['mom'] if k<30 else ['extrap','mom']
    pred=[]
    for i in range(len(wf)):
        tr=wf.iloc[max(0,i-60):i]
        if len(tr)<20: pred.append(np.nan); continue
        X=np.c_[np.ones(len(tr)),tr[cols+['r90']].values]; beta=la.lstsq(X,tr.actual.values,rcond=None)[0]
        pred.append(beta@np.r_[1,wf[cols+['r90']].iloc[i].values])
    wf['plus_price']=pred; m=wf.plus_price.notna()
    e0=np.abs(wf.actual[m]-wf[base][m]); e1=np.abs(wf.actual[m]-wf.plus_price[m])
    print(f"k={k}: champion MAE={e0.mean()*100:.2f}  +r90 MAE={e1.mean()*100:.2f}  wilcoxon p={stats.wilcoxon(e0,e1,alternative='greater').pvalue:.3f} win={np.mean(e1<e0):.2f} n={m.sum()}")
