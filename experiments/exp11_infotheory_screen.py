import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json
from lib.infotheory import mi_null
z=np.load('headers_compact.npz'); t=z['time'].astype(np.int64); nz=z['nonce'].astype(np.uint64); ver=z['version']; D=z['difficulty']
n=len(t); sel=slice(n-150000,n-1)   # recent era
iv=np.diff(t).astype(float)
feats={'nonce_bit7':((nz>>np.uint64(7))&1).astype(float),'nonce_lowbyte':(nz&np.uint64(0xff)).astype(float),'nonce_top4':(nz>>np.uint64(28)).astype(float),
       'version_asicboost_bits':((ver>>13)&0xffff).astype(float),'prev_interval':np.r_[np.nan,iv],'hdr_ts_mod600':(t%600).astype(float)}
print("Block level (last 150k blocks): MI(feature_t ; next interval), permutation null n=200")
rows=[]
for k,f in feats.items():
    x=f[sel]; y=iv[sel]; m=~np.isnan(x)&~np.isnan(y); x=x[m][:-0 or None]; y=y[m]
    # feature at block i vs interval i->i+1: iv[i] = t[i+1]-t[i]; sel already aligned (iv index i = interval after block i)
    obs,p,mu,sd=mi_null(x,y,n=200); rows.append((k,obs,p,mu,sd)); print(f"  {k:24s} MI={obs:.5f} null={mu:.5f}±{sd:.5f} p={p:.3f}")
# Epoch level: per-epoch feature means vs next-epoch log adjustment (n~230)
E=n//2016; ep=np.arange(E*2016)//2016
Dl=np.log(D[np.arange(E)*2016]); lr=np.diff(Dl)
print("\nEpoch level (epochs 220+): MI(epoch-mean feature_e ; adjustment into e+1)")
for k in ['nonce_bit7','version_asicboost_bits','nonce_top4']:
    fe=pd.Series(feats[k][:E*2016]).groupby(ep).mean().values
    x=np.diff(fe)[219:E-2]; y=lr[220:E-1]   # change in feature into epoch e vs adjustment into e+1
    obs,p,mu,sd=mi_null(x,y,n=500,bx=5,by=5); rows.append((k+'_epoch',obs,p,mu,sd)); print(f"  Δ{k:24s} MI={obs:.4f} null={mu:.4f}±{sd:.4f} p={p:.3f}")
json.dump([dict(feature=r[0],MI=r[1],p=r[2],null_mean=r[3],null_sd=r[4]) for r in rows],open('research_ledger/exp11_mi.json','w'),indent=1)
# version field descriptive: overt ASICBoost adoption over time
q=pd.PeriodIndex(pd.to_datetime(t,unit='s'),freq='Y'); vb=pd.Series(((ver>>13)&0xffff)!=0).groupby(q).mean()
print("\nFraction of blocks with nonzero version bits 13-28 (overt ASICBoost / version rolling) by year:"); print(vb[vb.index>=pd.Period('2017')].round(3).to_string())
