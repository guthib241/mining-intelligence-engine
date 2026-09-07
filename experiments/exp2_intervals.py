import pandas as pd, numpy as np
from scipy import stats
df=pd.read_pickle('headers.pkl')
t=df.time.values.astype(np.int64); dt=np.diff(t)
h=np.arange(1,len(t))
# Recent era: last 150k blocks (~2.9 yrs), hashrate drift per block negligible
sel=slice(len(dt)-150000,len(dt)); d=dt[sel].astype(float)
print("N",len(d),"mean",d.mean().round(1),"std",d.std().round(1),"(exp: std==mean)","neg intervals:",(d<0).sum(),"median",np.median(d))
# 1. Memoryless test: E[remaining | elapsed>t] should be constant = mean
print("\nElapsed t -> mean remaining time (exp: flat ~mean):")
for th in [0,300,600,1200,1800,3600]:
    m=d>th; print(f"  >{th:5d}s: n={m.sum():6d}  E[rem]={ (d[m]-th).mean():.1f}")
# 2. Lag autocorr of intervals
for lag in [1,2,3,5,10]:
    print(f"interval autocorr lag{lag}: {np.corrcoef(d[:-lag],d[lag:])[0,1]:+.4f}  2se={2/np.sqrt(len(d)):.4f}")
# 3. Timestamp-noise-robust version: intervals over k blocks vs raw
# use smoothed 'true' arrival: cumulative time over 6 blocks; test whether previous 6-block window predicts next single interval
w=np.convolve(d,np.ones(6),'valid')[:-1]; nxt=d[6:]
print("prev-6-block total time vs next interval corr:",np.corrcoef(w,nxt)[0,1].round(4))
# 4. Does previous interval predict next? bin it
prev=d[:-1]; nx=d[1:]
for lo,hi in [(-1e9,0),(0,120),(120,600),(600,1200),(1200,3600),(3600,1e9)]:
    m=(prev>=lo)&(prev<hi); print(f"  prev in [{lo:.0f},{hi:.0f}): n={m.sum():6d} next mean={nx[m].mean():.1f}")
# 5. distribution shape: exponential fit KS on positive intervals
pos=d[d>0]; print("KS vs exponential p=",stats.kstest(pos,'expon',args=(0,pos.mean())).pvalue)
# 6. within-epoch position effect: intervals early vs late in epoch (hashrate growth)
pos_in_epoch=(h[sel]%2016)
early=d[pos_in_epoch<500].mean(); late=d[pos_in_epoch>=1500].mean(); print("epoch early mean",early.round(1),"late mean",late.round(1))
