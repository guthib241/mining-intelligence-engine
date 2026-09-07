import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
df=pd.read_pickle('headers.pkl'); nz=df.nonce.values.astype(np.uint64); t=df.time.values
n=len(nz); bits=((nz[:,None]>>np.arange(32,dtype=np.uint64))&1).astype(np.float32)
# 1. per-bit imbalance by era (2-year windows)
yrs=pd.to_datetime(t,unit='s').year
print("Per-bit P(bit=1) deviation from .5, by era  (bits 31..24 shown = top byte; 2se≈",round(1/np.sqrt(60000),4),")")
print("era        "+" ".join(f"b{b:2d}" for b in range(31,15,-1)))
for y0 in range(2012,2026,2):
    m=(yrs>=y0)&(yrs<y0+2)
    if m.sum()<5000: continue
    dev=bits[m].mean(0)-0.5
    print(f"{y0}-{y0+1} n={m.sum():6d} "+" ".join(f"{dev[b]:+.2f}" for b in range(31,15,-1)))
# 2. which bits are most informative overall in 2022+?
m=yrs>=2022; dev=bits[m].mean(0)-0.5; order=np.argsort(-np.abs(dev))
print("\n2022+ most-imbalanced bits:",[(int(b),round(float(dev[b]),3)) for b in order[:8]])
# 3. Top-byte value histogram 2024+: look for structured depletion
m=yrs>=2024; top=(nz[m]>>np.uint64(24)).astype(int); h=np.bincount(top,minlength=256)/m.sum()*256
print("\n2024+ top-byte relative frequency (1.0=uniform), 16x16 grid rows=top nibble:")
for i in range(16): print(f"{i:x}: "+" ".join(f"{h[i*16+j]:.2f}" for j in range(16)))
# 4. low bits / low byte check
low=(nz[m]&np.uint64(0xff)).astype(int); hl=np.bincount(low,minlength=256)/m.sum()*256
print("\n2024+ low-byte rel freq min/max:",hl.min().round(2),hl.max().round(2))
