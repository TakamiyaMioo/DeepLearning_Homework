import argparse
from src.utils import load_config
from src.trainer import fit
p=argparse.ArgumentParser();p.add_argument('--model',choices=['linear','two_layer_linear','relu_mlp'],default='linear');p.add_argument('--config',default='configs/base.yaml');g=p.add_mutually_exclusive_group();g.add_argument('--download',action='store_true');g.add_argument('--no-download',action='store_true')
if __name__=='__main__':
 a=p.parse_args();print(fit(a.model,load_config(a.config),a.download)[0])
