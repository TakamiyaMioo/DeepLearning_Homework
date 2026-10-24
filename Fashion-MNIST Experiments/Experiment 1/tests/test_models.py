import torch
from src.models import build_model,parameter_count
def test_shapes_and_params():
    x=torch.randn(2,1,28,28)
    for n in ['linear','two_layer_linear','relu_mlp']: assert build_model(n)(x).shape==(2,10)
    assert parameter_count(build_model('linear'))==7850
