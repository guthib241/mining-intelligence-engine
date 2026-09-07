"""EXP25: 14-day-ahead mining-revenue uncertainty attribution (price / fees / difficulty) + VOI per component."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json, time
T0=time.time()
f=pd.read_csv('data/external/mempool_fees_3y_stride4.csv'); f['date']=pd.to_datetime(f.ts,unit='s').dt.normalize()
z=np.load('headers_compact.npz'); ht=pd.to_datetime(z['time'],unit='s'); D=pd.Series(z['difficulty'],index=ht); Dd=D.resample('D').last().ffill()
px=pd.read_csv('btc/btc_prices.csv'); px.columns=['date','o','h','l','c','v']; px['date']=pd.to_datetime(px.date); P=px.set_index('date').c
HALV=pd.Timestamp('2024-04-20')
f['S']=np.where(f.date<HALV,6.25,3.125); f['F']=f.avgFees_sat/1e8; f['P']=[P.asof(d) for d in f.date]; f['D']=[Dd.asof(d) for d in f.date]
f['fee_share']=f.F/(f.S+f.F)
f['lP']=np.log(f.P); f['lR']=np.log(f.S+f.F); f['lD']=np.log(f.D); f['lRev']=f.lP+f.lR-f.lD
H=7   # 7 rows = 14 days
for c in ['lP','lR','lD','lRev']: f['d'+c]=f[c].shift(-H)-f[c]
# exclude windows spanning the halving (deterministic subsidy step)
v=f[(f.date+pd.Timedelta(days=14)<HALV)|(f.date>=HALV)].dropna(subset=['dlRev'])
print("n windows",len(v)," period",v.date.min().date(),"->",v.date.max().date())
print("fee share of block reward: median %.3f mean %.3f P10 %.3f P90 %.3f max %.3f"%(f.fee_share.median(),f.fee_share.mean(),f.fee_share.quantile(.1),f.fee_share.quantile(.9),f.fee_share.max()))
def decomp(v,label):
    X=v[['dlP','dlR','dlD']]; C=X.cov(); tot=v.dlRev.var()
    sh={'price':C.loc['dlP','dlP']/tot,'fees':C.loc['dlR','dlR']/tot,'difficulty':C.loc['dlD','dlD']/tot,'cov_terms':(2*C.loc['dlP','dlR']-2*C.loc['dlP','dlD']-2*C.loc['dlR','dlD'])/tot}
    print(f"  {label:12s} sd(14d log rev)={np.sqrt(tot)*100:5.1f}%  shares: price {sh['price']:.2f} fees {sh['fees']:.2f} diff {sh['difficulty']:.2f} cov {sh['cov_terms']:+.2f} | E|Δ|: P {v.dlP.abs().mean()*100:4.1f}% F {v.dlR.abs().mean()*100:4.1f}% D {v.dlD.abs().mean()*100:4.1f}%")
    return sh
print("\nVARIANCE ATTRIBUTION of 14-day-ahead log revenue per hash (naive = no change for all components):")
res={'all':decomp(v,'all'),'pre_halving':decomp(v[v.date<HALV],'2023-24 pre'),'post_halving':decomp(v[v.date>=HALV],'post-halving'),'2025':decomp(v[v.date>='2025-01-01'],'2025')}
# block bootstrap CI on fee share of variance (block = 14 rows = 28 d)
r=np.random.default_rng(0); n=len(v); B=[]
for _ in range(500):
    idx=np.concatenate([np.arange(s,min(s+14,n)) for s in r.integers(0,n,n//14+1)])[:n]; b=v.iloc[idx]; C=b[['dlP','dlR','dlD']].cov(); B.append(C.loc['dlR','dlR']/b.dlRev.var())
print("  fees share CI95 (block bootstrap):",np.percentile(B,[2.5,97.5]).round(3))
# ---- Fee-share predictability at 14d: persistence vs 28d mean vs expanding mean (OOS from 2024-07) ----
f['lF']=np.log(f.lR.pipe(np.exp)-f.S)   # log fees per block (BTC)
f['tgt']=f.lF.shift(-H); f['pers']=f.lF; f['m28']=f.lF.rolling(14).mean(); f['m90']=f.lF.rolling(45).mean(); f['exp']=f.lF.expanding().mean()
o=f[(f.date>='2024-07-01')].dropna(subset=['tgt','m90'])
print("\nFEE (BTC/block) 14d-ahead log forecast, OOS 2024-07+, n=",len(o))
for m in ['pers','m28','m90','exp']:
    e=(o.tgt-o[m]); print(f"  {m:5s} MAE {e.abs().mean():.3f}  RMSE {np.sqrt((e**2).mean()):.3f}  bias {e.mean():+.3f}")
ac=[round(f.lF.autocorr(l),2) for l in (1,3,7,14)]; print("  log-fee autocorr at 2d/6d/14d/28d:",ac)
# ---- VOI per component for forward-selling 1 EH/s 14d revenue (USD): mispricing = revenue x E|Δ| ----
from lib.econ import rev_per_day
rev14=[rev_per_day(1e6,d,p,s+fb)*14 for d,p,s,fb in zip(v.D,v.P,v.S,v.F)]
v=v.assign(rev14=rev14)
voi={c:float((v.rev14*(np.exp(v[c].abs())-1)).mean()) for c in ['dlP','dlR','dlD']}
print("\nVOI (perfect info on ONE component, forward-selling 1 EH/s 14d revenue, $/epoch): price %.0f  fees %.0f  difficulty %.0f"%(voi['dlP'],voi['dlR'],voi['dlD']))
print("  vs fee predictability: best model (m90) reduces fee MAE from persistence %.3f to %.3f -> capturable fee VOI ~ $%.0f/epoch"%((o.tgt-o.pers).abs().mean(),(o.tgt-o.m90).abs().mean(),voi['dlR']*max(0,1-(o.tgt-o.m90).abs().mean()/(o.tgt-o.pers).abs().mean())))
json.dump({"n":int(len(v)),"variance_shares":res,"fees_share_CI95":[float(x) for x in np.percentile(B,[2.5,97.5])],"fee_forecast_OOS":{m:float((o.tgt-o[m]).abs().mean()) for m in ['pers','m28','m90','exp']},"VOI_per_component_usd_per_EH_epoch":voi,"fee_share_stats":{"median":float(f.fee_share.median()),"mean":float(f.fee_share.mean()),"p90":float(f.fee_share.quantile(.9)),"max":float(f.fee_share.max())},"runtime_s":round(time.time()-T0,1)},open('reports/voi_target_map.json','w'),indent=1)
print("runtime",round(time.time()-T0,1),"s")
