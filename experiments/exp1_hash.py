import pandas as pd, numpy as np
from scipy import stats
df=pd.read_pickle('headers.pkl')
H=np.array([int(h,16) for h in df.hash],dtype=object)
T=np.array([int(t) for t in df.target],dtype=object)
# u = hash/target in [0,1) -- how far below target the hash landed. Should be Uniform(0,1) and iid.
u=np.array([float(h)/float(t) for h,t in zip(H,T)])
print("N",len(u))
print("u mean/std (expect .5,.2887):",u.mean().round(4),u.std().round(4))
ks=stats.kstest(u,'uniform'); print("KS vs uniform p=",ks.pvalue)
# lag autocorrelation of u
for lag in [1,2,3,10,100,2016]:
    r=np.corrcoef(u[:-lag],u[lag:])[0,1]; print(f"autocorr lag{lag}: {r:+.5f}  (2se={2/np.sqrt(len(u)):.5f})")
# Low 32 bits of hash (well below target) as random bits
low=np.array([int(h)&0xffffffff for h in H],dtype=np.uint64)
bitsarr=((low[:,None]>>np.arange(32,dtype=np.uint64))&1).astype(np.int8)
print("bit balance |mean-0.5| max:",np.abs(bitsarr.mean(0)-0.5).max(), " 2se=",2*0.5/np.sqrt(len(u)))
# Can previous header fields predict next block's low hash bit? Mutual info via chi2
prev_nonce_bit=(df.nonce.values[:-1]&1); nxt_bit=bitsarr[1:,0]
ct=pd.crosstab(prev_nonce_bit,nxt_bit); print("prev nonce parity -> next hash bit0 chi2 p=",stats.chi2_contingency(ct)[1])
prev_u_hi=(u[:-1]>0.5).astype(int)
ct=pd.crosstab(prev_u_hi,nxt_bit); print("prev u>0.5 -> next hash bit0 chi2 p=",stats.chi2_contingency(ct)[1])
# Time-of-day / timestamp -> u
tod=(df.time.values%86400)//3600
print("u by hour-of-day ANOVA p=",stats.f_oneway(*[u[tod==h] for h in range(24)]).pvalue)
# Nonce distribution: uniform?  (known: ASIC artifacts)
nz=df.nonce.values.astype(np.uint64)
hist,_=np.histogram(nz,bins=16,range=(0,2**32))
print("nonce top-4-bit histogram (recent 100k):",np.histogram(nz[-100000:],bins=16,range=(0,2**32))[0])
print("nonce top-4-bit histogram (2012-2013 blocks 200k-260k):",np.histogram(nz[200000:260000],bins=16,range=(0,2**32))[0])
