import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, glob, os
from scipy import stats
H=pd.read_pickle('headers.pkl')[['height','hash','time']]
rows=[]
for f in sorted(glob.glob('block-arrival-times/data/*.csv')):
    if 'stratum' in f: continue
    d=pd.read_csv(f,header=None,names=['height','hash','ms']); d['src']=os.path.basename(f)[:-4]; rows.append(d)
A=pd.concat(rows); A['ts']=A.ms/1000
j=A.merge(H,on='height',how='inner',suffixes=('','_hdr')); comp=j[j.hash!=j.hash_hdr]
print("competing-block observations:",len(comp)," distinct heights:",comp.height.nunique()," distinct losing hashes:",comp.hash.nunique())
yr=pd.to_datetime(comp.ts,unit='s').dt.year
g=pd.read_pickle('arrivals.pkl'); gy=pd.to_datetime(g['first'],unit='s').dt.year
cov=pd.Series(gy).value_counts().sort_index()
per=(comp.groupby('height').first().assign(yr=lambda x:pd.to_datetime(x.ts,unit='s').dt.year).yr.value_counts().sort_index())
tab=pd.DataFrame({'blocks_observed':cov,'competing_heights':per}).fillna(0); tab['rate_per_1000']=(tab.competing_heights/tab.blocks_observed*1000).round(2)
print(tab.to_string())
# propagation spread trend (>=3 sources), yearly P50/P90
g3=g[g.n_src>=3]; sp=pd.DataFrame({'yr':pd.to_datetime(g3['first'],unit='s').dt.year,'spread':g3.spread})
print("\nmulti-node spread (s) by year, blocks with >=3 sources:"); print(sp.groupby('yr').spread.describe(percentiles=[.5,.9,.99])[['count','50%','90%','99%']].round(2).to_string())
# is competing-block occurrence associated with large spread (slow propagation) at that height? matched comparison
g['comp']=g.index.isin(comp.height.unique())
m3=g[g.n_src>=3]
a=m3.spread[m3.comp]; b=m3.spread[~m3.comp]
print(f"\nspread at competing heights: n={len(a)} median {a.median():.2f}s vs others median {b.median():.2f}s; Mann-Whitney p={stats.mannwhitneyu(a,b,alternative='greater').pvalue:.2e}")
# prior interval at competing heights (short intervals -> race)
iv=g['first'].diff(); print(f"interval before competing block: median {iv[g.comp].median():.0f}s vs {iv[~g.comp].median():.0f}s; MW p={stats.mannwhitneyu(iv[g.comp].dropna(),iv[~g.comp].dropna(),alternative='less').pvalue:.2e}")
