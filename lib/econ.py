"""Parametric mining economics. Units: hashrate in TH/s, efficiency J/TH, electricity $/kWh, price $/BTC, reward BTC/block."""
import numpy as np
def rev_per_day(H_ths, D, P, R):            # network hashrate = D*2^32/600 H/s
    share=H_ths*1e12/(D*2**32/600); return 144*R*P*share
def cost_per_day(H_ths, eta, E, uptime=1.0): return H_ths*eta*24/1000*E*uptime   # W*24h/1000 = kWh
def profit_per_day(H_ths,D,P,R,eta,E,uptime=1.0,pool_fee=0.02):
    return rev_per_day(H_ths,D,P,R)*uptime*(1-pool_fee)-cost_per_day(H_ths,eta,E,uptime)
def breakeven_E(D,P,R,eta,pool_fee=0.02): return rev_per_day(1,D,P,R)*(1-pool_fee)/(eta*24/1000)
def breakeven_P(D,R,eta,E,pool_fee=0.02): return cost_per_day(1,eta,E)/(rev_per_day(1,D,1,R)*(1-pool_fee))
def monte_carlo(D_now, adj_quantiles_logpct, P, R, eta, E, H_ths=1000, n=20000, seed=0, **kw):
    """adj_quantiles_logpct: dict pct->log adjustment (%). Samples adjustment by inverse-CDF interpolation."""
    rng=np.random.default_rng(seed); ps=np.array(sorted(adj_quantiles_logpct)); qs=np.array([adj_quantiles_logpct[p] for p in ps])/100
    u=rng.uniform(ps[0]/100,ps[-1]/100,n); adj=np.interp(u,ps/100,qs); D_next=D_now*np.exp(adj)
    prof=profit_per_day(H_ths,D_next,P,R,eta,E,**kw)*14
    return dict(mean=prof.mean(),P_loss=float((prof<0).mean()),pct={p:float(np.percentile(prof,p)) for p in [1,5,10,25,50,75,90,95,99]})
