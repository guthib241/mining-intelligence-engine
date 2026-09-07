import numpy as np, hashlib, struct, sys, pandas as pd
def load(paths):
    raw=b''.join(open(p,'rb').read() for p in paths)
    n=len(raw)//80; a=np.frombuffer(raw[:n*80],dtype=np.uint8).reshape(n,80)
    ver=a[:,0:4].copy().view('<i4').ravel()
    prev=a[:,4:36]; merkle=a[:,36:68]
    ts=a[:,68:72].copy().view('<u4').ravel()
    bits=a[:,72:76].copy().view('<u4').ravel()
    nonce=a[:,76:80].copy().view('<u4').ravel()
    return raw,n,ver,prev,merkle,ts,bits,nonce
def dsha(b): return hashlib.sha256(hashlib.sha256(b).digest()).digest()
def bits_to_target(b):
    e=b>>24; m=b&0xffffff
    return m<<(8*(e-3)) if e>3 else m>>(8*(3-e))
paths=['btc-archive/btc.archive.bin']
import os
cur=[f for f in os.listdir('btc-current') if f.endswith('.bin')]
print('current files',cur)
paths+= ['btc-current/'+f for f in cur]
raw,n,ver,prev,merkle,ts,bits,nonce=load(paths)
print('headers',n)
hashes=np.empty((n,32),dtype=np.uint8)
for i in range(n): hashes[i]=np.frombuffer(dsha(raw[i*80:(i+1)*80]),dtype=np.uint8)
print('genesis hash', hashes[0][::-1].tobytes().hex())
links_ok=(prev[1:]==hashes[:-1]).all(axis=1)
print('bad links', (~links_ok).sum())
# PoW check
tmax=2**256; bad=0
hv=np.array([int.from_bytes(hashes[i][::-1].tobytes(),'big') for i in range(n)],dtype=object)
for i in range(n):
    if hv[i]>bits_to_target(int(bits[i])): bad+=1
print('pow failures',bad)
df=pd.DataFrame({'height':np.arange(n),'version':ver,'time':ts,'bits':bits,'nonce':nonce,
    'hash':[hashes[i][::-1].tobytes().hex() for i in range(n)],
    'merkle':[merkle[i][::-1].tobytes().hex() for i in range(n)]})
df['target']=[float(bits_to_target(int(b))) for b in bits]
df['difficulty']=float(bits_to_target(0x1d00ffff))/df['target']
df.to_pickle('headers.pkl')
print(df.tail(3)); print(pd.to_datetime(df.time.iloc[-1],unit='s'))
