import numpy as np
class NumpyMLP:
    def __init__(self,input_dim,hidden_dim,num_classes,rng):
        self.params={'W1':rng.normal(0,.1,(input_dim,hidden_dim)),'b1':np.zeros(hidden_dim),'W2':rng.normal(0,.1,(hidden_dim,num_classes)),'b2':np.zeros(num_classes)}
    def loss_grads(self,X,y):
        p=self.params; a=np.tanh(X@p['W1']+p['b1']); z=a@p['W2']+p['b2']; z=z-z.max(axis=1,keepdims=True); lp=z-np.log(np.exp(z).sum(axis=1,keepdims=True)); probs=np.exp(lp); n=len(y); loss=-lp[np.arange(n),y].mean(); dz=probs.copy(); dz[np.arange(n),y]-=1; dz/=n; da=dz@p['W2'].T; dz1=da*(1-a*a)
        return loss,{'W1':X.T@dz1,'b1':dz1.sum(0),'W2':a.T@dz,'b2':dz.sum(0)}
    def loss(self,X,y): return self.loss_grads(X,y)[0]
