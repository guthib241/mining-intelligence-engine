import numpy as np
def mi_binned(x,y,bx=8,by=8):
    """Plug-in mutual information (nats) with quantile bins."""
    qx=np.digitize(x,np.quantile(x,np.linspace(0,1,bx+1)[1:-1])); qy=np.digitize(y,np.quantile(y,np.linspace(0,1,by+1)[1:-1]))
    j=np.zeros((bx,by)); np.add.at(j,(qx,qy),1); p=j/j.sum(); px=p.sum(1,keepdims=True); py=p.sum(0,keepdims=True)
    nz=p>0; return float((p[nz]*np.log(p[nz]/(px@py)[nz])).sum())
def mi_null(x,y,n=200,seed=0,**kw):
    r=np.random.default_rng(seed); obs=mi_binned(x,y,**kw); null=np.array([mi_binned(x,r.permutation(y),**kw) for _ in range(n)])
    return obs, float((null>=obs).mean()), float(null.mean()), float(null.std())
