import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json, time
from lib.propagation import *
T0=time.time(); A=load_observers(); A,off=calibrate(A); print("observers",A.src.nunique(),"rows",len(A),"calib",round(time.time()-T0,1),"s")
summ=A.groupby('src').dev.describe(percentiles=[.05,.5,.95,.99])[['count','5%','50%','95%','99%']].round(2)
summ['rel_med']=off.groupby('src').reliability.median().round(2); summ['mad_med']=off.groupby('src').mad.median().round(2); print(summ.to_string())
print("\nobserver-months flagged (rel<.8 or MAD>2s):",int(((off.reliability<0.8)|(off.mad>2)).sum()),"of",len(off))
print(off[off.mad>2].sort_values('mad',ascending=False).head(5)[['src','mo','offset','mad','reliability']].to_string(index=False))
ref=A.groupby('height').ref.first()
B=block_metrics(A); B['q']=pd.to_datetime(ref.loc[B.index],unit='s').dt.to_period('Q')
qt=B.groupby('q').agg(n=('spread_cal','size'),sp50=('spread_cal','median'),sp90=('spread_cal',lambda x:x.quantile(.9)),iqr50=('iqr_cal','median')).round(2)
print("\nCALIBRATED spread quarterly (reliable observers only):"); print(qt[qt.n>1000].to_string())
A24=A[A.mo>=pd.Period('2024-01')]; base=block_metrics(A24); base['q']=pd.to_datetime(ref.loc[base.index],unit='s').dt.to_period('Q'); bq=base.groupby('q').spread_cal.median()
loo={}
for s in A24.src.unique():
    Bs=block_metrics(A24[A24.src!=s]); Bs['q']=pd.to_datetime(ref.loc[Bs.index],unit='s').dt.to_period('Q'); qs=Bs.groupby('q').spread_cal.median()
    c=qs.index.intersection(bq.index); loo[s]=float(np.abs(qs.loc[c]-bq.loc[c]).max()) if len(c) else np.nan
print("\nLeave-one-out max |Δ quarterly median spread| 2024+:",{k:round(v,2) for k,v in sorted(loo.items(),key=lambda x:-x[1])})
B.to_pickle('prop_metrics.pkl'); off.to_pickle('observer_offsets.pkl'); print("runtime",round(time.time()-T0,1),"s")
