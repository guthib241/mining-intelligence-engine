import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json, time
from scipy import stats
from lib.forecast import *
df=pd.read_pickle('headers.pkl'); t=df.time.values.astype(np.int64); D=df.difficulty.values
t0=time.time(); results={}
for k in [300,500,1000,1500]:
    models={'extrap':lambda e:pace_extrap(t,D,e,k),'win300':lambda e:pace_extrap(t,D,e,k,win=300),'win150':lambda e:pace_extrap(t,D,e,k,win=150),'ewma200':lambda e:pace_extrap(t,D,e,k,halflife=200),'ewma500':lambda e:pace_extrap(t,D,e,k,halflife=500)}
    d=build(t,D,k,330,models); d['naive']=0.0
    fs={'naive':[],'extrap_raw':None,'champion':['extrap','mom'],'win300':['win300','mom'],'win150':['win150','mom'],'ewma200':['ewma200','mom'],'ewma500':['ewma500','mom'],'both':['extrap','win300','mom']}
    out=walk(d,{k2:v for k2,v in fs.items() if v is not None})
    # raw standard estimator (no fit) with its own empirical residual quantiles from window
    out['extrap_raw']=d.extrap.values; qs=[]
    for i in range(len(d)):
        tr=d.iloc[max(0,i-60):i]
        qs.append(list(d.extrap.iloc[i]+np.percentile((tr.actual-tr.extrap).dropna(),QS)) if len(tr)>=20 else [np.nan]*9)
    out['extrap_raw_q']=np.array(qs)
    res,m=evaluate(d,out,list(fs.keys())); results[k]=res
    print(f"\n=== k={k} (n={res['naive']['n']}) ===")
    for n,r in res.items(): print(f"  {n:11s} MAE {r['MAE']:5.2f} RMSE {r['RMSE']:5.2f} cov50 {r['cov50']:.2f} cov90 {r['cov90']:.2f} width90 {r['width90']:5.2f} pinball {r['pinball']:.3f}")
    y=d.actual.values[m]; eb=np.abs(y-out['champion'][m])
    for c in ['win300','ewma200','both']:
        ec=np.abs(y-out[c][m]); print(f"  {c} vs champion: wilcoxon p={stats.wilcoxon(eb,ec,alternative='greater').pvalue:.3f} win={np.mean(ec<eb):.2f}")
    # contradiction engine: model disagreement vs error magnitude
    P=np.c_[out['champion'][m],out['win300'][m],out['ewma200'][m],out['extrap_raw'][m]]; dis=P.std(1)
    print(f"  disagreement sd vs |champion err| spearman r={stats.spearmanr(dis,eb)[0]:+.3f} p={stats.spearmanr(dis,eb)[1]:.3f}")
    if k==500:
        Q=out['champion_q'][m]; print("  champion k=500 forecast-dist example (last test epoch, % adj):",[round(x*100,2) for x in Q[-1]],"actual",round(y[-1]*100,2))
json.dump({str(k):v for k,v in results.items()},open('research_ledger/exp8_results.json','w'),indent=1)
print("runtime s",round(time.time()-t0,1))
