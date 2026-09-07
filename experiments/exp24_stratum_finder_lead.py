import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, glob, os, json, time
from scipy import stats
T0=time.time()
H=pd.read_pickle('headers.pkl')[['height','hash','time']]
g=pd.read_pickle('arrivals.pkl')
dbg=[f for f in glob.glob('block-arrival-times/data/*.csv') if any(s in f for s in ['monitoring1','rs2','peer-observer','n-thumann','darosior','vostrnad'])]
Ad=pd.concat([pd.read_csv(f,header=None,names=['height','hash','ms']).assign(src=os.path.basename(f)[:-4]) for f in dbg]).merge(H,on='height',suffixes=('','_hdr'))
comp_h=set(Ad[Ad.hash!=Ad.hash_hdr].height); cov=Ad.groupby('height').size()
S=g.dropna(subset=['stratum_work_empty']).copy(); S=S[S.n_src>=3]
S['L']=S.stratum_work_empty-S['first']            # M3 finder lead: first job - first observer arrival
S['gap']=S.stratum_work_not_empty-S.stratum_work_empty
print("blocks with stratum+>=3 observers:",len(S))
print("L percentiles (s):",{p:round(float(np.percentile(S.L,p)),2) for p in [1,5,10,25,50,75,90,95,99]})
# bimodality: 2-component Gaussian mixture via EM vs single Gaussian (BIC)
x=S.L.clip(-5,5).values
def gmm2(x,it=200):
    mu=np.array([-1.0,0.0]); sd=np.array([0.5,0.3]); w=np.array([0.3,0.7])
    for _ in range(it):
        p=np.c_[w[0]*stats.norm.pdf(x,mu[0],sd[0]),w[1]*stats.norm.pdf(x,mu[1],sd[1])]; r=p/p.sum(1,keepdims=True)
        w=r.mean(0); mu=(r*x[:,None]).sum(0)/r.sum(0); sd=np.sqrt((r*(x[:,None]-mu)**2).sum(0)/r.sum(0))+1e-4
    ll=np.log(p.sum(1)).sum(); return w,mu,sd,ll
w,mu,sd,ll2=gmm2(x); ll1=stats.norm.logpdf(x,x.mean(),x.std()).sum(); n=len(x)
print(f"GMM2: weights {w.round(2)} means {mu.round(2)} sds {sd.round(2)}; ΔBIC(1-2 comp)={(-2*ll1+2*np.log(n))-(-2*ll2+5*np.log(n)):.0f} (>10 => 2 components)")
S['delay_net']=(-S.L).clip(lower=0)       # network delay to fastest observer when finder emits at creation
print("delay_net (s) percentiles:",{p:round(float(np.percentile(S.delay_net,p)),2) for p in [50,75,90,95,99]})
# H020 component: delay_net vs template-build gap and vs interval
iv=S['first'].diff(); print("spearman delay_net~gap:",round(stats.spearmanr(S.delay_net,S.gap)[0],3)," delay_net~interval:",round(stats.spearmanr(S.delay_net.iloc[1:],iv.iloc[1:])[0],3))
# Orphan replication with DISJOINT feature source: feature = delay_net(h-1) [stratum + first arrival, both of h-1]; leakage-safe: require first(h-1) < first(h)
X=pd.DataFrame(index=cov.index); X['comp']=X.index.isin(comp_h)
X['f']=S.delay_net.reindex(X.index).shift(1); X['ok']=(g['first'].reindex(X.index).diff()>0)
X=X[X.ok].dropna(subset=['f']); print("\norphan test: n",len(X),"competing",int(X.comp.sum()))
pos=X.f[X.comp].values; neg=X.f[~X.comp].values
if len(pos)>=5:
    auc=float(np.mean([np.mean(neg<pp)+0.5*np.mean(neg==pp) for pp in pos])); r=np.random.default_rng(0)
    aucs=[np.mean([np.mean(neg<pp)+0.5*np.mean(neg==pp) for pp in r.choice(pos,len(pos))]) for _ in range(300)]
    print(f"AUC delay_net(h-1)->comp(h) = {auc:.3f} CI95 [{np.percentile(aucs,2.5):.3f},{np.percentile(aucs,97.5):.3f}]")
    for th in [0.5,1,2]:
        hi=X.f>th; a=int(X.comp[hi].sum()); b=int(hi.sum())-a; c=int(X.comp[~hi].sum()); d=int((~hi).sum())-c
        print(f"  delay_net>{th}s: flagged {a+b} comp {a} ({a/max(a+b,1)*1000:.2f}/1000) vs {c/(c+d)*1000:.2f}/1000 Fisher p={stats.fisher_exact([[a,b],[c,d]],alternative='greater')[1]:.3f}")
    # perm null for AUC
    yv=X.comp.values; fv=X.f.values; nul=[]
    for _ in range(300):
        yp=r.permutation(yv); pp_=fv[yp]; nn=fv[~yp]; nul.append(np.mean([np.mean(nn<q)+0.5*np.mean(nn==q) for q in pp_]))
    print(f"  permutation null AUC 95% = [{np.percentile(nul,2.5):.3f},{np.percentile(nul,97.5):.3f}]")
else: auc=None; print("too few events")
# does the finder-lead signal reveal which blocks were found by a 'monitored' pool? cross-check: L<-0.5 fraction over time
S['mo']=pd.to_datetime(S['first'],unit='s').dt.to_period('M'); print("\nfraction L<-0.5s by month:",S.groupby('mo').L.apply(lambda v:(v<-0.5).mean()).round(2).to_dict())
json.dump({"n":int(len(S)),"L_pct":{p:float(np.percentile(S.L,p)) for p in [1,5,25,50,75,95,99]},"gmm":{"w":w.tolist(),"mu":mu.tolist(),"sd":sd.tolist(),"dBIC":float((-2*ll1+2*np.log(n))-(-2*ll2+5*np.log(n)))},"orphan_auc":auc,"runtime_s":round(time.time()-T0,1)},open('reports/hidden_signal_screen_005.json','w'),indent=1)
print("runtime",round(time.time()-T0,1),"s")
