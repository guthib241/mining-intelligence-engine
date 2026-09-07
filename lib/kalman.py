"""Kalman local-level+drift model of log-hashrate from epoch spans. Observation noise is analytic (Poisson). No fitted regression weights."""
import numpy as np
S_EPOCH=1/np.sqrt(2015)
def epoch_hash_obs(t,D):
    n=len(t); E=n//2016; ep=np.arange(E)
    t0=t[ep*2016]; tl=t[ep*2016+2015]; span=(tl-t0).astype(float)
    logH=np.log(D[ep*2016])+np.log(2015*600/span)   # log(H/const): avg hashrate during epoch e, observed with sd S_EPOCH
    return E,logH
def forecast(t,D,k,e,q,drift_win=26,fixed_drift=None):
    """Forecast log(D_{e+1}/D_e) given data up to block e*2016+k. Filter runs on epochs < e (complete), then partial epoch e."""
    E,logH=epoch_hash_obs(t,D)
    # drift: trailing mean of observed log-hashrate change
    d = fixed_drift if fixed_drift is not None else float(np.mean(np.diff(logH[max(0,e-drift_win):e])))
    # filter over complete epochs 0..e-1
    x,P=logH[0],S_EPOCH**2
    for i in range(1,e):
        xp,Pp=x+d,P+q
        K=Pp/(Pp+S_EPOCH**2); x=xp+K*(logH[i]-xp); P=(1-K)*Pp
    # predict epoch e hashrate
    x,P=x+d,P+q
    base=e*2016
    if k>=30:
        elapsed=float(t[base+k]-t[base]); z=np.log(D[base])+np.log(k*600/elapsed); s2=1/k
        K=P/(P+s2); x=x+K*(z-x); P=(1-K)*P
    else: elapsed=0.0
    # expected full span = elapsed + remaining blocks at rate exp(x)
    H=np.exp(x); rem=(2015-k)*600*D[base]/H if k>=30 else 2015*600*D[base]/H
    return float(np.log(np.clip(2016*600/(elapsed+rem),0.25,4)))
