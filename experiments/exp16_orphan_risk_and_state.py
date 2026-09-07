import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, glob, os, json
from scipy import stats
H=pd.read_pickle('headers.pkl')[['height','hash','time']]
dbg=[f for f in glob.glob('block-arrival-times/data/*.csv') if any(s in f for s in ['monitoring1','rs2','peer-observer','n-thumann','darosior','vostrnad'])]
A=pd.concat([pd.read_csv(f,header=None,names=['height','hash','ms']).assign(src=os.path.basename(f)[:-4]) for f in dbg]).merge(H,on='height',suffixes=('','_hdr'))
cov=A.groupby('height').src.nunique(); comp_h=set(A[A.hash!=A.hash_hdr].height)
B=pd.read_pickle('prop_metrics.pkl'); g=pd.read_pickle('arrivals.pkl')
X=pd.DataFrame(index=cov.index); X['comp']=X.index.isin(comp_h); X['n_dbg']=cov
X['first']=g['first'].reindex(X.index); X['prev_iv']=X['first'].diff(); X['prev_spread']=B.spread_cal.reindex(X.index).shift(1); X['prev_iqr']=B.iqr_cal.reindex(X.index).shift(1)
X['hour']=pd.to_datetime(X['first'],unit='s').dt.hour; X['q']=pd.to_datetime(X['first'],unit='s').dt.to_period('Q'); X['epoch_pos']=X.index%2016
X=X.dropna(subset=['prev_iv']); print("blocks with debug-log coverage:",len(X),"competing:",int(X.comp.sum()),"rate/1000:",round(X.comp.mean()*1000,3))
print("\n§4 Pre-outcome features vs competing-block occurrence (rate per 1000 by feature bin; permutation p on rate difference):")
def binrate(col,bins,labels):
    b=pd.cut(X[col],bins,labels=labels); r=X.groupby(b,observed=True).comp.agg(['size','mean']); r['per1000']=(r['mean']*1000).round(2); return r[['size','per1000']]
print(binrate('prev_iv',[-1e9,60,300,900,1e9],['<60s','60-300','300-900','>900']))
Xs=X.dropna(subset=['prev_spread']); print(binrate('prev_spread',[-1,0.2,0.5,2,1e9],['<0.2','0.2-0.5','0.5-2','>2']) if len(Xs) else "no spread")
def perm_p(col,thr,n=2000,seed=0):
    d=X.dropna(subset=[col]); y=d.comp.values; x=d[col].values>thr; obs=y[x].mean()-y[~x].mean(); r=np.random.default_rng(seed); null=[]
    for _ in range(n): yp=r.permutation(y); null.append(yp[x].mean()-yp[~x].mean())
    return obs*1000, float((np.abs(null)>=abs(obs)).mean())
for col,thr in [('prev_iv',60),('prev_iv',900),('prev_spread',0.5),('prev_spread',2.0),('epoch_pos',1008)]:
    o,p=perm_p(col,thr); print(f"  {col}>{thr}: Δrate={o:+.3f}/1000 perm p={p:.3f}")
print("  Bonferroni threshold:",0.05/5)
# hour-of-day chi2
ct=pd.crosstab(X.hour//6,X.comp); print("  6h-bucket chi2 p=",round(stats.chi2_contingency(ct)[1],3))
# ------- §5 network state: monthly state vector; change points; do they align with observer-set changes? -------
print("\n§5 Network state regimes")
Bm=B.copy(); Bm['mo']=pd.to_datetime(g['first'].reindex(Bm.index),unit='s').dt.to_period('M')
S=Bm.groupby('mo').agg(sp50=('spread_cal','median'),sp90=('spread_cal',lambda x:x.quantile(.9)),iqr50=('iqr_cal','median'),n=('spread_cal','size'))
gm=g.copy(); gm['mo']=pd.to_datetime(gm['first'],unit='s').dt.to_period('M'); gm['iv']=gm['first'].diff()
S=S.join(gm.groupby('mo').agg(hdr_err=('hdr_err','median'),iv_mean=('iv','mean'),n_src=('n_src','median'))); S=S[S.n>500].dropna()
Z=(S[['sp50','sp90','iqr50','hdr_err','iv_mean']]-S[['sp50','sp90','iqr50','hdr_err','iv_mean']].mean())/S[['sp50','sp90','iqr50','hdr_err','iv_mean']].std()
def binseg(Zv,min_size=6,max_cp=6):
    cps=[]; segs=[(0,len(Zv))]
    for _ in range(max_cp):
        best=None
        for (a,b) in segs:
            if b-a<2*min_size: continue
            tot=((Zv[a:b]-Zv[a:b].mean(0))**2).sum()
            for c in range(a+min_size,b-min_size):
                cost=((Zv[a:c]-Zv[a:c].mean(0))**2).sum()+((Zv[c:b]-Zv[c:b].mean(0))**2).sum()
                if best is None or tot-cost>best[0]: best=(tot-cost,c,(a,b))
        if best is None or best[0]<0.15*((Zv-Zv.mean(0))**2).sum(): break
        cps.append(best[1]); a,b=best[2]; segs.remove((a,b)); segs+= [(a,best[1]),(best[1],b)]
    return sorted(cps)
cps=binseg(Z.values); months=list(S.index); cp_months=[str(months[c]) for c in cps]
print("change points (binary segmentation, multivariate):",cp_months)
# observer-set transitions
obs=pd.read_pickle('observer_offsets.pkl'); first=obs.groupby('src').mo.min(); last=obs.groupby('src').mo.max()
trans=sorted(set(list(first.astype(str))+list((last+1).astype(str))))
print("observer start/end months:",trans)
near=[c for c in cp_months if any(abs((pd.Period(c)-pd.Period(t)).n)<=2 for t in trans)]
print(f"change points within ±2 months of an observer-set transition: {len(near)}/{len(cp_months)}")
# bootstrap stability of segmentation
r=np.random.default_rng(0); agree=[]
for _ in range(30):
    idx=np.sort(r.choice(len(Z),len(Z),replace=True)); Zb=Z.values[np.unique(idx)]
    agree.append(len(set(binseg(Zb)) & set(cps))/max(1,len(cps)))
print("bootstrap change-point recovery rate:",round(float(np.mean(agree)),2))
json.dump({"change_points":cp_months,"observer_transitions":trans,"aligned":len(near),"total":len(cp_months)},open('research_ledger/exp16_state.json','w'),indent=1)
