import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from lib.difficulty import walk_forward, mae_table
df=pd.read_pickle('headers.pkl'); t=df.time.values.astype(np.int64); D=df.difficulty.values
rng=np.random.default_rng(0)
print("=== A. REAL DATA via library (must match sprint-1 numbers: k500 naive 3.61 blend 2.95) ===")
for k in [0,500,1000]: print(k, mae_table(walk_forward(t,D,k,330)))
print("\n=== B. SHUFFLED TARGETS (epoch-permuted actuals; expect fitted models <= no better than drift/naive) ===")
from lib.difficulty import epoch_table
E,_,lr=epoch_table(t,D)
for seed in range(3):
    perm=lr.copy(); idx=np.arange(330,E-1); perm[idx]=rng.permutation(lr[idx])
    print(f"seed{seed} k500", mae_table(walk_forward(t,D,500,330,target=perm)))
print("\n=== C. SYNTHETIC constant hashrate, 400 epochs (pure Poisson; expect naive MAE≈1.8% [=0.8*2.23], blend→theory sqrt((2016-k)/2016)*1.8) ===")
def synth(hash_path, seed):
    r=np.random.default_rng(seed); Dc=1.0; ts=[0]; Ds=[]
    for e in range(len(hash_path)):
        H=hash_path[e]; mean=600*Dc/H
        iv=r.exponential(mean,2016)
        for x in iv: ts.append(ts[-1]+x)
        Ds+= [Dc]*2016
        span=ts[-1]-ts[-2016-1]-iv[-1]   # first 2015 intervals
        Dc=Dc*np.clip(2016*600/span,0.25,4)
    return np.array(ts[1:]).astype(np.int64), np.array(Ds)
ts,Ds=synth(np.ones(400),1)
for k in [0,500,1000,1500]:
    print(k, mae_table(walk_forward(ts,Ds,k,60)), " theory-blend≈", round(1.8*np.sqrt((2016-k)/2016),2))
print("\n=== D. SYNTHETIC random-walk hashrate (log-growth ~N(0.02,0.04) per epoch) — sanity that drift is learned ===")
g=np.exp(np.cumsum(rng.normal(0.02,0.04,400))); ts,Ds=synth(g,2)
for k in [0,500]: print(k, mae_table(walk_forward(ts,Ds,k,60)))
