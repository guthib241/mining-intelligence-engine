import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json
from scipy import stats
from lib.forecast import *
z=np.load('headers_compact.npz'); t=z['time'].astype(np.int64); D=z['difficulty']
halv_ep={2012:105000//2016,2016:420000//2016,2020:630000//2016,2024:840000//2016}   # epoch index containing the halving block
print("halving epochs:",halv_ep)
out={}
for k in [0,500]:
    d=build(t,D,k,100,{'extrap':lambda e:pace_extrap(t,D,e,k)}); o=walk(d,{'champion':['extrap','mom'] if k>=30 else ['mom']})
    m=~np.isnan(o['champion']); d=d[m].copy(); d['r']=(d.actual-o['champion'][m])*100
    tab={}
    for yr,he in halv_ep.items():
        seg=d[(d.e>=he-3)&(d.e<=he+6)].set_index(d.e[(d.e>=he-3)&(d.e<=he+6)]-he).r
        tab[yr]={int(i):round(float(v),2) for i,v in seg.items()}
    T=pd.DataFrame(tab); print(f"\nk={k}: champion residual (pp) by epochs from halving (row) x halving year (col)"); print(T.round(1).to_string())
    post=T.loc[0:4]; print("  epochs 0-4 mean per halving:",post.mean().round(2).to_dict(),"| sign consistency (all 4 negative?):",bool((post.mean()<0).all()))
    # per-epoch-offset sign test across the 4 events
    signs=(post<0).sum(axis=1); print("  #negative of 4 events at offsets 0..4:",signs.to_dict())
    # TEMPORAL HOLDOUT: fit mean shift over offsets 0-4 using 2012+2016+2020, apply to 2024 (untouched P1 test)
    shift=float(post[[2012,2016,2020]].stack().mean()); r24=post[2024].dropna()
    mae_before=r24.abs().mean(); mae_after=(r24-shift).abs().mean()
    print(f"  shift fitted on 2012/16/20 offsets 0-4: {shift:+.2f}pp -> 2024 held-out epochs 0-4: MAE {mae_before:.2f} -> {mae_after:.2f} pp (n={len(r24)})")
    # effect on full P1 score (109 epochs): only 5 epochs touched
    p1=d[d.e>=350].copy(); adj=p1.r.copy(); he=halv_ep[2024]; sel=(p1.e>=he)&(p1.e<=he+4); adj[sel]=adj[sel]-shift
    print(f"  P1 overall MAE: champion {p1.r.abs().mean():.3f} -> +halving rule {adj.abs().mean():.3f} pp  (paired on 5 touched epochs: Δ={ (p1.r[sel].abs()-adj[sel].abs()).mean():+.2f}pp)")
    # NULL: apply the same shift to 5 random non-halving epochs, 2000 draws -> distribution of MAE change on those 5
    r=np.random.default_rng(0); nul=[]
    cand=p1.index[~sel]
    for _ in range(2000):
        idx=r.choice(cand,5,replace=False); nul.append((p1.r[idx].abs()-(p1.r[idx]-shift).abs()).mean())
    obs=(p1.r[sel].abs()-adj[sel].abs()).mean(); print(f"  null (shift applied to random 5 epochs): P(Δ>=obs)={np.mean(np.array(nul)>=obs):.4f}")
    out[k]=dict(table=T.to_dict(),shift=shift,mae_2024_before=float(mae_before),mae_2024_after=float(mae_after),p1_before=float(p1.r.abs().mean()),p1_after=float(adj.abs().mean()),null_p=float(np.mean(np.array(nul)>=obs)))
# hashrate (not difficulty) around halvings: observed log-hashrate change over epochs 0-4 vs the 10 epochs before
from lib.kalman import epoch_hash_obs
_,logH=epoch_hash_obs(t,D)
for yr,he in halv_ep.items():
    pre=np.diff(logH[he-10:he]).mean()*100; post=np.diff(logH[he:he+5]).mean()*100; print(f"  hashrate growth/epoch: {yr} pre(10ep) {pre:+.2f}%  post(5ep) {post:+.2f}%")
json.dump(out,open('reports/exp23_halving.json','w'),indent=1,default=float)
