import argparse
from src.utils import load_config,save_csv,save_json
from src.trainer import fit
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/base.yaml');p.add_argument('--download',action='store_true')
if __name__=='__main__':
 a=p.parse_args();rows=[]
 for name in ['linear','two_layer_linear','relu_mlp']:rows.append(fit(name,load_config(a.config),a.download)[0])
 save_csv(rows,'results/comparison/model_comparison.csv');save_json(rows,'results/comparison/model_comparison.json')
