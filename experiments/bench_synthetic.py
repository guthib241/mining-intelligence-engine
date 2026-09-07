"""Synthetic recovery benchmark A-J. Runs the SAME walk() evaluator used on real data. Detection rule: feature model beats intercept-only by Wilcoxon p<0.01 AND >=5% MAE reduction, out-of-sample walk-forward."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, time, json, tracemalloc
from scipy import stats
from lib.forecast import walk
rng=np.random.default_rng(42); N=400
def detect(df,cols):
    out=walk(df,{'base':[],'model':cols}); m=~np.isnan(out['model'])&~np.isnan(out['base']); y=df.actual.values[m]
    eb=np.abs(y-out['base'][m]); em=np.abs(y-out['model'][m]); p=stats.wilcoxon(eb,em,alternative='greater').pvalue if not np.allclose(eb,em) else 1.0
    return dict(mae_base=round(eb.mean(),4),mae_model=round(em.mean(),4),p=float(p),detected=bool(p<0.01 and em.mean()<0.95*eb.mean()))
S={}
x=rng.normal(size=N); n=rng.normal(size=N)
S['A_pure_noise']=(pd.DataFrame({'actual':n,'x':x}),['x'],False)
S['B_weak_linear']=(pd.DataFrame({'actual':0.3*x+n,'x':x}),['x'],True)
S['C_weak_nonlinear']=(pd.DataFrame({'actual':0.5*x**2-0.5+n,'x':x,'x2':x**2}),['x','x2'],True)
xl=np.roll(x,2); S['D_delayed']=(pd.DataFrame({'actual':0.4*xl+n,'x_lag2':xl}),['x_lag2'],True)
reg=(np.arange(N)//100)%2; S['E_regime_switch']=(pd.DataFrame({'actual':np.where(reg==1,0.5,-0.5)*x+n,'x':x}),['x'],None)   # sign flips -> linear walk-forward should mostly fail; informative either way
S['F_structural_break']=(pd.DataFrame({'actual':np.where(np.arange(N)<250,0.5*x,0.0)+n,'x':x}),['x'],None)
S['G_heteroskedastic']=(pd.DataFrame({'actual':0.3*x+n*(1+np.abs(x)),'x':x}),['x'],True)
z=rng.normal(size=N); S['H_confounded']=(pd.DataFrame({'actual':0.6*z+n,'x':0.6*z+rng.normal(size=N)}),['x'],'confounded')   # x predicts y via z; detected but not causal
# I multiple-testing trap: 50 noise features; pick in-sample best on first 150 rows, then walk-forward on all
F=rng.normal(size=(N,50)); yI=rng.normal(size=N); best=int(np.argmax([abs(np.corrcoef(F[:150,j],yI[:150])[0,1]) for j in range(50)]))
dI=pd.DataFrame({'actual':yI,'fbest':F[:,best]}); S['I_multiple_testing']=(dI,['fbest'],False)
# J leakage trap: feature is next period's target (look-ahead)
yJ=rng.normal(size=N); S['J_leakage']=(pd.DataFrame({'actual':yJ,'leak':np.roll(yJ,-1)}),['leak'],'leak')
res={}
tracemalloc.start(); t0=time.time()
for k,(df,cols,truth) in S.items():
    r=detect(df,cols); r['expected']=truth
    # leakage check: does feature correlate more with FUTURE actual than with current? (feature at t vs actual at t+1)
    r['leak_check_r_future']=round(float(np.corrcoef(df[cols[0]].values[:-1],df.actual.values[1:])[0,1]),3)
    res[k]=r; print(f"{k:22s} base {r['mae_base']:.3f} model {r['mae_model']:.3f} p={r['p']:.1e} detected={r['detected']} expected={truth} leak_r_future={r['leak_check_r_future']}")
print("runtime",round(time.time()-t0,2),"s  peak mem MB",round(tracemalloc.get_traced_memory()[1]/1e6,1))
json.dump(res,open('research_ledger/bench_synthetic.json','w'),indent=1)
# Verdict
ok = (not res['A_pure_noise']['detected']) and res['B_weak_linear']['detected'] and (not res['I_multiple_testing']['detected'])
print("PIPELINE VERDICT:", "PASS" if ok else "FAIL - halt real-data discovery")
