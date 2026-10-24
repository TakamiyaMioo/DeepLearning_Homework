import numpy as np
from .mlp_numpy import NumpyMLP
def check(seed=42,epsilon=1e-5,threshold=1e-6):
    rng=np.random.default_rng(seed);X=rng.normal(size=(4,4)).astype(np.float64);y=np.array([0,2,1,0]);m=NumpyMLP(4,5,3,rng);_,g=m.loss_grads(X,y);rows=[]
    for k,a in m.params.items():
        num=np.zeros_like(a)
        for ix in np.ndindex(a.shape):
            v=a[ix];a[ix]=v+epsilon;hi=m.loss(X,y);a[ix]=v-epsilon;lo=m.loss(X,y);a[ix]=v;num[ix]=(hi-lo)/(2*epsilon)
        rel=np.linalg.norm(g[k]-num)/(np.linalg.norm(g[k])+np.linalg.norm(num)+1e-12)
        rows.append({'parameter':k,'maximum_absolute_error':float(np.max(np.abs(g[k]-num))),'relative_error':float(rel),'checked_elements':a.size,'passed':bool(rel<threshold)})
    return rows
