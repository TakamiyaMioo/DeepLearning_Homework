import numpy as np
from numpy_bp.mlp_numpy import NumpyMLP
from numpy_bp.gradient_check import check
def test_gradient():
    m=NumpyMLP(4,5,3,np.random.default_rng(3));loss,g=m.loss_grads(np.ones((2,4)),np.array([0,1]));assert np.isfinite(loss) and all(v.shape==m.params[k].shape for k,v in g.items());assert all(r['passed'] for r in check())
