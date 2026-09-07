import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json, time
from scipy import stats
from lib.forecast import *
T0=time.time()
z=np.load('headers_compact.npz'); t=z['time'].astype(np.int64); D=z['difficulty']; nz=z['nonce'].astype(np.uint64); ver=z['version']; n=len(t)
E=n//2016
px=pd.read_csv('btc/btc_prices.csv'); px.columns=['date','o','h','l','c','v']; px['date']=pd.to_datetime(px.date); lp=np.log(px.set_index('date').c)
g=pd.read_pickle('arrivals.pkl'); ep_arr=pd.Series(g.hdr_err.values,index=g.index)
halvings=[420000,630000,840000]
res={}
for k in [0,500]:
    d=build(t,D,k,330,{'extrap':lambda e:pace_extrap(t,D,e,k)}); out=walk(d,{'champion':['extrap','mom'] if k>=30 else ['mom']})
    m=~np.isnan(out['champion']); d=d[m].copy(); d['pred']=out['champion'][m]; d['r']=d.actual-d.pred
    tr_sd=[]; 
    rows=[]
    for e in d.e:
        base=e*2016; start=t[base]; day=pd.Timestamp(start,unit='s').normalize()-pd.Timedelta(days=1)
        iv=np.diff(t[base:base+max(k,144)+1]).astype(float)
        hd=min(((base-h)//2016 for h in halvings),key=abs) if any(abs(base-h)<2016*6 for h in halvings) else np.nan
        rows.append(dict(
          halv=hd, disp=iv.var()/iv.mean()**2 if k>=144 else np.nan, prev_sign=np.sign(d.mom[d.e==e].iloc[0]),
          pvol=float(lp[:day].tail(28).diff().std()) if day>lp.index[0] else np.nan, pret=float(lp.asof(day)-lp.asof(day-pd.Timedelta(days=28))) if day>lp.index[0] else np.nan,
          fp7=float(((nz[base-2016:base]>>np.uint64(7))&1).mean()-((nz[base-4032:base-2016]>>np.uint64(7))&1).mean()),
          verb=float((((ver[base-2016:base]>>13)&0xffff)!=0).mean()), tserr=float(ep_arr.reindex(range(base-2016,base)).median()) if ep_arr.index.max()>base else np.nan,
          month=pd.Timestamp(start,unit='s').month, pos_prev_res=np.nan, year=pd.Timestamp(start,unit='s').year, epoch_len=float(t[base]-t[base-2016])))
    F=pd.DataFrame(rows,index=d.index); d=pd.concat([d,F],axis=1)
    d['S_prev']=d.r.shift(1).abs()   # M1 proxy: previous |residual| (variance clustering)
    d['pos_prev_res']=d.r.shift(1)   # residual autocorrelation
    d['season']=np.cos(2*np.pi*(d.month-1)/12)
    cov=['halv','disp','prev_sign','pvol','pret','fp7','verb','tserr','season','S_prev','pos_prev_res','epoch_len']
    res[k]={}
    print(f"\n=== k={k} n={len(d)} residual sd {d.r.std()*100:.2f}pp mean {d.r.mean()*100:+.2f}pp ===")
    for c in cov:
        x=d[c]; y=d.r; mm=x.notna()&y.notna()
        if mm.sum()<8 or x[mm].nunique()<2: res[k][c]=dict(n=int(mm.sum()),note="insufficient"); print(f"  {c:12s} n={mm.sum():3d} insufficient"); continue
        tgt = y.abs() if c=='S_prev' else y
        rho,p=stats.spearmanr(x[mm],tgt[mm]); 
        # split consistency
        e1=d.year[mm]<=2023; r1=stats.spearmanr(x[mm][e1],tgt[mm][e1])[0] if e1.sum()>6 else np.nan; r2=stats.spearmanr(x[mm][~e1],tgt[mm][~e1])[0] if (~e1).sum()>6 else np.nan
        res[k][c]=dict(n=int(mm.sum()),rho=float(rho),p=float(p),rho_2021_23=float(r1) if r1==r1 else None,rho_2024_25=float(r2) if r2==r2 else None,target="|r|" if c=='S_prev' else "r")
        print(f"  {c:12s} n={mm.sum():3d} rho={rho:+.3f} p={p:.4f}  split {r1:+.2f}/{r2:+.2f}")
    # halving detail: residual by epochs-after-halving
    hv=d.dropna(subset=['halv']); print("  halving: mean residual (pp) by epochs from halving:",hv.groupby('halv').r.mean().mul(100).round(2).to_dict())
    res[k]['halving_table']={str(int(a)):float(b) for a,b in hv.groupby('halv').r.mean().items()}
    # M5 error complementarity
    d2=build(t,D,k,330,{'extrap':lambda e:pace_extrap(t,D,e,k),'ewma':lambda e:pace_extrap(t,D,e,k,halflife=500)}); o2=walk(d2,{'drift':[],'std':['extrap'],'ewma':['ewma','mom']}) if k>=30 else walk(d2,{'drift':[]})
    errs={'champ':d.r.values}
    for nm in ([ 'drift','std','ewma'] if k>=30 else ['drift']):
        errs[nm]=(d2.actual.values-o2[nm])[m]
    C={a+'~'+b:round(float(np.corrcoef(errs[a],errs[b])[0,1]),3) for a in errs for b in errs if a<b}; res[k]['error_corr']=C; print("  M5 error correlations:",C)
bonf=0.05/24; surv=[(k,c,v['rho'],v['p']) for k in res for c,v in res[k].items() if isinstance(v,dict) and 'p' in v and v['p']<bonf]
print("\nBonferroni threshold",round(bonf,5),"survivors:",surv); print("runtime",round(time.time()-T0,1),"s")
json.dump({"protocol":"P1","tests":24,"bonferroni":bonf,"survivors":surv,"by_horizon":res,"decision":"see ledger"},open('reports/champion_residual_map.json','w'),indent=1,default=float)
