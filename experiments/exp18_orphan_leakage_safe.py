import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, glob, os, json
from lib.propagation import *
H=pd.read_pickle('headers.pkl')[['height','hash','time']]
dbg=[f for f in glob.glob('block-arrival-times/data/*.csv') if any(s in f for s in ['monitoring1','rs2','peer-observer','n-thumann','darosior','vostrnad'])]
Ad=pd.concat([pd.read_csv(f,header=None,names=['height','hash','ms']).assign(src=os.path.basename(f)[:-4]) for f in dbg]).merge(H,on='height',suffixes=('','_hdr'))
comp_h=set(Ad[Ad.hash!=Ad.hash_hdr].height); cov=Ad.groupby('height').src.nunique()
A=load_observers(); A,off=calibrate(A); A=A[~A.src.isin(['0xb10c_peer-observer-frank','n-thumann','vostrnad_node1'])]
R=A[(A.reliability>=0.8)&(A.mad<=2.0)]
first_next=R.groupby('height').ts.min().rename('first_next'); first_next.index=first_next.index-1   # first arrival of block h+1, indexed at h
R=R.join(first_next,on='height'); R['before']=R.ts<R.first_next
# leakage-safe spread of block h: over arrivals that occurred BEFORE block h+1's first arrival, and censored at that moment
Rb=R[R.before]; g=Rb.groupby('height').ts_cal.agg(['size','min','max']); g=g[g['size']>=3]; g['spread_safe']=g['max']-g['min']
# also: number of reliable observers that had NOT yet received h when h+1 arrived  (late fraction) -- strongest pre-outcome propagation signal
tot=R.groupby('height').size().rename('n_tot'); nb=Rb.groupby('height').size().rename('n_before'); late=(1-nb/tot).rename('late_frac')
X=pd.DataFrame(index=cov.index); X['comp']=X.index.isin(comp_h)
X['prev_spread_safe']=g.spread_safe.reindex(X.index).shift(1); X['prev_late']=late.reindex(X.index).shift(1); X['prev_ntot']=tot.reindex(X.index).shift(1)
X=X[X.prev_ntot>=3]
def test(mask,label,n=3000):
    d=X.dropna(subset=[mask.name]) if hasattr(mask,'name') else X; m=mask.loc[d.index].values; y=d.comp.values
    obs=y[m].mean()-y[~m].mean(); r=np.random.default_rng(0); null=[abs((yp:=r.permutation(y))[m].mean()-yp[~m].mean()) for _ in range(n)]
    print(f"{label:45s} n={len(d):6d} flagged={int(m.sum()):6d} rate flagged={y[m].mean()*1000:.2f} other={y[~m].mean()*1000:.2f} /1000  perm p={np.mean(np.array(null)>=abs(obs)):.4f}")
Xs=X.dropna(subset=['prev_spread_safe']); test((Xs.prev_spread_safe>2).rename('prev_spread_safe'),"LEAKAGE-SAFE prev spread > 2s")
test((Xs.prev_spread_safe>0.5).rename('prev_spread_safe'),"LEAKAGE-SAFE prev spread > 0.5s")
Xl=X.dropna(subset=['prev_late']); test((Xl.prev_late>0).rename('prev_late'),"prev block not yet at all observers when h arrived")
test((Xl.prev_late>=0.5).rename('prev_late'),"prev block missing at >=50% of observers")
# how often does the previous block still not have full propagation when the next arrives, and interval dependence
print("\nP(prev_late>0) =",round(float((Xl.prev_late>0).mean()),4)," -> that is essentially 'block h arrived within propagation time of h-1' = a race condition by definition")
g2=pd.read_pickle('arrivals.pkl'); iv=g2['first'].diff().reindex(Xl.index)
print("interval h-1 -> h when prev_late>0: median",round(float(iv[Xl.prev_late>0].median()),1),"s ; otherwise",round(float(iv[Xl.prev_late==0].median()),1),"s")
# Given that, the honest test: does prev_spread_safe carry information BEYOND a short interval? condition on interval>30s
m=(iv>30); Xc=Xs[m.reindex(Xs.index).fillna(False)]
y=Xc.comp.values; hi=(Xc.prev_spread_safe>2).values
r=np.random.default_rng(1); obs=y[hi].mean()-y[~hi].mean(); null=[abs((yp:=r.permutation(y))[hi].mean()-yp[~hi].mean()) for _ in range(3000)]
print(f"\nconditional on interval>30s: n={len(Xc)} flagged={int(hi.sum())} rate hi={y[hi].mean()*1000:.2f} lo={y[~hi].mean()*1000:.2f} perm p={np.mean(np.array(null)>=abs(obs)):.4f}")
