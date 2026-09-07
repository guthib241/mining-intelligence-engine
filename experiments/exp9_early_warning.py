import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats
from lib.forecast import *
from lib.difficulty import epoch_table
df=pd.read_pickle('headers.pkl'); t=df.time.values.astype(np.int64); D=df.difficulty.values
E,_,lr=epoch_table(t,D)
# Daily (144-block) hashrate proxy within epoch; CUSUM on log(pace) relative to epoch-start pace
def cusum_flag(e,k,h=3.0,drift=0.0):
    base=e*2016; iv=np.diff(t[base:base+k+1]).astype(float)
    if k<288: return 0.0,0.0
    days=iv[:k//144*144].reshape(-1,144).mean(1); z=(np.log(days)-np.log(days[:2].mean()))/ (1/np.sqrt(144))  # log-pace change in sd units (poisson sd of a 144-block mean ~ 1/sqrt(144)=8.3%)
    s_pos=0;s_neg=0;mx=0
    for zz in z[2:]:
        s_pos=max(0,s_pos+zz-drift); s_neg=max(0,s_neg-zz-drift); mx=max(mx,s_pos,s_neg)
    return mx, z[-1]
k=500
models={'extrap':lambda e:pace_extrap(t,D,e,k),'win300':lambda e:pace_extrap(t,D,e,k,win=300)}
d=build(t,D,k,330,models)
out=walk(d,{'champion':['extrap','mom'],'win300':['win300','mom']})
m=~np.isnan(out['champion']); d=d[m].copy(); d['champ']=out['champion'][m]; d['win']=out['win300'][m]
d['cus']=[cusum_flag(e,k)[0] for e in d.e]; d['e_ch']=np.abs(d.actual-d.champ); d['e_win']=np.abs(d.actual-d.win)
d['date']=pd.to_datetime(t[d.e.values*2016],unit='s').date
big=d[np.abs(d.actual)>0.08]
print("Epochs with |adjustment|>8% (shock epochs), k=500:"); print(big[['e','date','actual','champ','win','cus']].assign(actual=lambda x:(x.actual*100).round(1),champ=lambda x:(x.champ*100).round(1),win=lambda x:(x.win*100).round(1),cus=lambda x:x.cus.round(1)).to_string(index=False))
print(f"shock epochs n={len(big)}: champion MAE {big.e_ch.mean()*100:.2f}  win300 MAE {big.e_win.mean()*100:.2f}")
print(f"non-shock  n={len(d)-len(big)}: champion MAE {d[np.abs(d.actual)<=0.08].e_ch.mean()*100:.2f}  win300 {d[np.abs(d.actual)<=0.08].e_win.mean()*100:.2f}")
# Does CUSUM score predict champion error magnitude (uncertainty early warning)?
print("CUSUM vs |champion err| spearman:",stats.spearmanr(d.cus,d.e_ch))
# Switching rule: use win300 when CUSUM>thr else champion. Threshold chosen on epochs <330 (train) - do train eval
dtr=build(t,D,k,220,models); otr=walk(dtr,{'champion':['extrap','mom'],'win300':['win300','mom']}); mt=~np.isnan(otr['champion'])&(dtr.e.values<330)
dtr=dtr[mt].copy(); dtr['champ']=otr['champion'][mt]; dtr['win']=otr['win300'][mt]; dtr['cus']=[cusum_flag(e,k)[0] for e in dtr.e]
for thr in [2,3,4,5,6]:
    sw=np.where(dtr.cus>thr,dtr.win,dtr.champ); print(f"  train thr={thr}: switch MAE {np.abs(dtr.actual-sw).mean()*100:.2f} vs champion {np.abs(dtr.actual-dtr.champ).mean()*100:.2f} (n_switch={int((dtr.cus>thr).sum())})")
thr=4; sw=np.where(d.cus>thr,d.win,d.champ); print(f"TEST thr={thr}: switch MAE {np.abs(d.actual-sw).mean()*100:.2f} vs champion {d.e_ch.mean()*100:.2f} (n_switch={int((d.cus>thr).sum())})")
