import argparse,torch
from src.utils import load_config
from src.models import build_model
from src.data import get_loaders
from src.evaluator import evaluate
p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['linear','two_layer_linear','relu_mlp']);p.add_argument('--checkpoint',required=True);p.add_argument('--config',default='configs/base.yaml');p.add_argument('--download',action='store_true')
if __name__=='__main__':
 a=p.parse_args();c=load_config(a.config);_,_,test=get_loaders(c['dataset']['root'],c['training']['batch_size'],c['split_seed'],a.download);d=torch.device('cuda' if torch.cuda.is_available() and c['device']=='auto' else 'cpu');m=build_model(a.model,c['model']['hidden_dim']).to(d);m.load_state_dict(torch.load(a.checkpoint,map_location=d,weights_only=False)['model_state_dict']);print(evaluate(m,test,d))
