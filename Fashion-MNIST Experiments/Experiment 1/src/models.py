from torch import nn
class LinearClassifier(nn.Module):
    def __init__(self): super().__init__(); self.fc=nn.Linear(784,10)
    def forward(self,x): return self.fc(x.reshape(x.shape[0],-1))
class TwoLayerLinear(nn.Module):
    def __init__(self,hidden_dim=128): super().__init__(); self.fc1=nn.Linear(784,hidden_dim); self.fc2=nn.Linear(hidden_dim,10)
    def forward(self,x): return self.fc2(self.fc1(x.reshape(x.shape[0],-1)))
class ReluMLP(nn.Module):
    def __init__(self,hidden_dim=128): super().__init__(); self.fc1=nn.Linear(784,hidden_dim); self.relu=nn.ReLU(); self.fc2=nn.Linear(hidden_dim,10)
    def forward(self,x): return self.fc2(self.relu(self.fc1(x.reshape(x.shape[0],-1))))
def build_model(name,hidden_dim=128):
    t={'linear':LinearClassifier,'two_layer_linear':lambda:TwoLayerLinear(hidden_dim),'relu_mlp':lambda:ReluMLP(hidden_dim)}
    if name not in t: raise ValueError(name)
    return t[name]()
def parameter_count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
