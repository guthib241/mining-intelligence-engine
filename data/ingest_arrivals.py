import pandas as pd, numpy as np, glob, os, hashlib, json
rows=[]
for f in sorted(glob.glob('block-arrival-times/data/*.csv')):
    src=os.path.basename(f)[:-4]
    d=pd.read_csv(f,header=None,names=['height','hash','ms'],dtype={'height':'int64','hash':'str','ms':'int64'})
    d['src']=src; rows.append(d)
A=pd.concat(rows,ignore_index=True); A['ts']=A.ms/1000.0
cov=A.groupby('src').agg(n=('height','size'),hmin=('height','min'),hmax=('height','max'),tmin=('ts','min'),tmax=('ts','max'))
cov['tmin']=pd.to_datetime(cov.tmin,unit='s').dt.date; cov['tmax']=pd.to_datetime(cov.tmax,unit='s').dt.date
print(cov.to_string())
H=pd.read_pickle('headers.pkl')[['height','hash','time']]
# per-block node arrivals (exclude stratum_* which are not node arrivals)
N=A[~A.src.str.startswith('stratum')]
# validate hashes against headers
j=N.merge(H,on='height',how='left',suffixes=('','_hdr'))
bad=(j.hash!=j.hash_hdr)&j.hash_hdr.notna(); print("hash mismatches vs headers:",int(bad.sum()),"of",len(j)," (blocks beyond header range:",int(j.hash_hdr.isna().sum()),")")
N=j[~bad & j.hash_hdr.notna()]
g=N.groupby('height').agg(n_src=('src','nunique'),first=('ts','min'),last=('ts','max'),med=('ts','median'),hdr_time=('time','first'))
g['spread']=g['last']-g['first']; g['hdr_err']=g.hdr_time-g['first']
S=A[A.src.str.startswith('stratum')].pivot_table(index='height',columns='src',values='ts',aggfunc='min')
g=g.join(S,how='left')
g.to_pickle('arrivals.pkl')
print("\nblocks with >=1 node arrival:",len(g)," height range",g.index.min(),g.index.max()," with>=3 sources:",int((g.n_src>=3).sum()))
print("stratum coverage: empty",int(g.stratum_work_empty.notna().sum())," not_empty",int(g.stratum_work_not_empty.notna().sum()))
print("header-timestamp error (hdr - first arrival) percentiles s:",np.percentile(g.hdr_err,[1,5,25,50,75,95,99]).round(1))
print("multi-node spread (last-first) s, blocks with>=3 src:",np.percentile(g.spread[g.n_src>=3],[50,75,90,95,99]).round(2))
json.dump({"dataset_id":"BLOCK-ARRIVALS-CC0","source":"github bitcoin-data/block-arrival-times","license":"CC0-1.0","retrieved":"2026-09-07","records_raw":len(A),"blocks":len(g),"height_range":[int(g.index.min()),int(g.index.max())],"sources":int(A.src.nunique()),"validation":[f"hash mismatches vs verified headers: {int(bad.sum())}"],"limitations":["coverage uneven across heights/sources","arrival = first-connected at that node, node-specific latency","stratum files: pool job-issuance times, source semantics per repo docs"],"authority":"MEDIUM-HIGH (multi-source, CI QA'd)","sha256":hashlib.sha256(open('block-arrival-times/data/KIT_monitor1.csv','rb').read()).hexdigest()},open('research_state/manifest_arrivals.json','w'),indent=1)
