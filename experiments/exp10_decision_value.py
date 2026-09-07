import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json
from lib.forecast import *
from lib.econ import *
df=pd.read_pickle('headers.pkl'); t=df.time.values.astype(np.int64); D=df.difficulty.values
k=500; d=build(t,D,k,330,{'extrap':lambda e:pace_extrap(t,D,e,k)}); out=walk(d,{'champion':['extrap','mom'],'drift':[]})
m=~np.isnan(out['champion']); d=d[m].copy(); d['champ']=out['champion'][m]; d['drift']=out['drift'][m]; d['naive']=0.0
d['D_now']=D[d.e.values*2016]; d['D_next']=D[(d.e.values+1)*2016]
# Decision: at k=500 of epoch e, run or curtail fleet for epoch e+1 (14 days). Realised profit uses actual D_next.
# Parameter grid chosen to span thin-margin conditions. Fleet 1 PH/s = 1000 TH/s.
grid=[]; H=1000
for eta in [18,25,32]:                       # J/TH: new gen / mid / old
  for E in [0.03,0.045,0.06,0.08]:
    for P in [40000,60000,90000]:
      R=3.125+0.15
      val={}
      for f in ['naive','drift','champ','perfect']:
          tot=0;changed=0
          for _,r in d.iterrows():
              Dn_hat=r.D_now*np.exp(r.actual if f=='perfect' else r[f])
              run=profit_per_day(H,Dn_hat,P,R,eta,E)>0
              real=profit_per_day(H,r.D_next,P,R,eta,E)*14
              tot+= real if run else 0.0
          val[f]=tot/len(d)
      # count epochs where champion and naive decide differently
      diff=sum(1 for _,r in d.iterrows() if (profit_per_day(H,r.D_now*np.exp(r.champ),P,R,eta,E)>0)!=(profit_per_day(H,r.D_now,P,R,eta,E)>0))
      grid.append(dict(eta=eta,E=E,P=P,naive=val['naive'],drift=val['drift'],champ=val['champ'],perfect=val['perfect'],n_decisions_changed=diff))
G=pd.DataFrame(grid); G['champ_minus_naive']=G.champ-G.naive; G['perfect_minus_naive']=G.perfect-G.naive
print("Realised 14-day profit per PH/s ($), mean over 109 test epochs, by decision rule:")
show=G[(G.n_decisions_changed>0)].sort_values('champ_minus_naive',ascending=False)
print(show.round(0).to_string(index=False))
print("\ncells where forecast changed at least one decision:",int((G.n_decisions_changed>0).sum()),"of",len(G))
print("total value champ-naive across thin-margin cells ($/PH/epoch):",G.champ_minus_naive[G.n_decisions_changed>0].mean().round(0)," perfect-naive:",G.perfect_minus_naive[G.n_decisions_changed>0].mean().round(0))
# Forecast-distribution economics for the latest test epoch (parametric example)
Q=out['champion_q'][m][-1]; qd={p:q*100 for p,q in zip(QS,Q)}
ex=monte_carlo(d.D_now.iloc[-1],qd,P=90000,R=3.275,eta=25,E=0.06)
print("\nExample MC (latest epoch, P=90k, eta=25, E=$0.06, 1 PH/s, 14d): mean $%.0f, P(loss)=%.2f, P10 $%.0f, P50 $%.0f, P90 $%.0f"%(ex['mean'],ex['P_loss'],ex['pct'][10],ex['pct'][50],ex['pct'][90]))
print("break-even electricity at P=90k, eta=25: $%.3f/kWh ; break-even price at E=0.06, eta=25: $%.0f"%(breakeven_E(d.D_now.iloc[-1],90000,3.275,25),breakeven_P(d.D_now.iloc[-1],3.275,25,0.06)))
G.to_csv('research_ledger/exp10_decision_grid.csv',index=False)
