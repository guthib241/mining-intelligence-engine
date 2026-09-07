import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json, time
from scipy import stats
from lib.forecast import *
from lib.difficulty import epoch_table
z=np.load('headers_compact.npz'); t=z['time'].astype(np.int64); D=z['difficulty']; n=len(t)
g=pd.read_pickle('arrivals.pkl')
ta=np.full(n,np.nan); ta[g.index.values[g.index.values<n]]=g['first'].values[g.index.values<n]
E,_,lr=epoch_table(t,D)
def pace_arr(e,k):
    base=e*2016
    if k<30 or np.isnan(ta[base]) or np.isnan(ta[base+k]): return np.nan
    elapsed_hdr=t[base+k]-t[base]; pace=(ta[base+k]-ta[base])/k          # hashrate estimate from true arrivals; elapsed for the formula stays header-based (that is what consensus uses)
    return float(np.log(np.clip(2016*600/(elapsed_hdr+(2015-k)*pace),0.25,4)))
T0=time.time(); tables={}
print("H016: arrival-based pace vs header pace. Protocol P1. Same epochs only (arrival endpoints available).")
for k in [50,100,200,300,500]:
    d=build(t,D,k,330,{'extrap':lambda e:pace_extrap(t,D,e,k),'extrap_arr':lambda e:pace_arr(e,k)})
    out=walk(d,{'drift':[],'champion':['extrap','mom'],'cand':['extrap_arr','mom']})
    m=~np.isnan(out['champion'])&~np.isnan(out['cand']); y=d.actual.values[m]
    rows={}
    for nm in ['drift','champion','cand']:
        p=out[nm][m]; Q=out[nm+'_q'][m]; err=(y-p)*100
        rows[nm]=dict(MAE=np.abs(err).mean(),RMSE=np.sqrt((err**2).mean()),mean_err=err.mean(),sd=err.std(),**{f"P{q}":np.percentile(err,q) for q in [1,5,10,25,50,75,90,95,99]},cov50=np.mean((y>=Q[:,3])&(y<=Q[:,5])),cov90=np.mean((y>=Q[:,1])&(y<=Q[:,7])),width90=np.mean(Q[:,7]-Q[:,1])*100)
    ec=np.abs(y-out['champion'][m]); ea=np.abs(y-out['cand'][m]); dlt=ec-ea
    tables[k]=dict(n=int(m.sum()),rows=rows,paired=dict(mean_delta_pp=float(dlt.mean()*100),ci95=[float(x*100) for x in stats.t.interval(0.95,len(dlt)-1,loc=dlt.mean(),scale=stats.sem(dlt))],wilcoxon_p=float(stats.wilcoxon(ec,ea).pvalue) if not np.allclose(ec,ea) else 1.0,win=float(np.mean(ea<ec))))
    r=rows; print(f"\nk={k} n={m.sum()} | {'metric':8s} {'drift':>8s} {'champ':>8s} {'cand':>8s} {'Δ(abs pp)':>10s} {'Δ rel':>8s}")
    for met in ['MAE','RMSE','mean_err','sd','P5','P50','P95','P99','cov50','cov90','width90']:
        c,a=r['champion'][met],r['cand'][met]; print(f"  {met:14s} {r['drift'][met]:8.3f} {c:8.3f} {a:8.3f} {c-a:10.3f} {((c-a)/abs(c)*100 if c else 0):7.2f}%")
    pt=tables[k]['paired']; print(f"  paired Δ|err| champ-cand = {pt['mean_delta_pp']:+.4f}pp CI95 [{pt['ci95'][0]:+.3f},{pt['ci95'][1]:+.3f}] wilcoxon p={pt['wilcoxon_p']:.3f} cand-win {pt['win']:.2f}")
    # regime: by year
    yr=pd.to_datetime(t[d.e.values[m]*2016],unit='s').year
    print("  by year MAE champ/cand:",{int(Y):(round(np.abs(y[yr==Y]-out['champion'][m][yr==Y]).mean()*100,2),round(np.abs(y[yr==Y]-out['cand'][m][yr==Y]).mean()*100,2)) for Y in sorted(set(yr))})
print("runtime",round(time.time()-T0,1),"s")
json.dump({str(k):v for k,v in tables.items()},open('research_ledger/exp14_h016.json','w'),indent=1,default=float)
# ---------------- H017: header-timestamp error as infrastructure fingerprint ----------------
print("\nH017: hdr_err = header_time - first_arrival (s)")
gg=g[g.n_src>=2].copy(); gg['q']=pd.to_datetime(gg['first'],unit='s').dt.to_period('Q'); gg=gg[gg.hdr_err.abs()<600]
qs=gg.groupby('q').hdr_err.describe(percentiles=[.05,.25,.5,.75,.95])[['count','5%','25%','50%','75%','95%']].round(1)
print(qs[qs['count']>2000].to_string())
# stability: KS between adjacent quarters; autocorrelation block-to-block (pool persistence proxy); bimodality
qq=[q for q in qs.index if qs.loc[q,'count']>2000]; ks=[stats.ks_2samp(gg.hdr_err[gg.q==qq[i]],gg.hdr_err[gg.q==qq[i+1]]).statistic for i in range(len(qq)-1)]
print("adjacent-quarter KS statistic: median",round(float(np.median(ks)),3),"max",round(float(np.max(ks)),3))
print("block-to-block autocorr of hdr_err (lag1,2,5):",[round(gg.hdr_err.autocorr(l),3) for l in (1,2,5)])
h,edges=np.histogram(gg.hdr_err,bins=np.arange(-120,61,5)); print("histogram 5s bins (-120..60):",list(h))
