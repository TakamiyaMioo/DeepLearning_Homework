import argparse
from src.utils import load_config,save_csv
from src.trainer import fit
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/diagnostic.yaml');p.add_argument('--download',action='store_true')
if __name__=='__main__':
 a=p.parse_args();d=load_config(a.config);cfg=load_config(d['config']);rows=[]
 for label,lr in [('control',d['control_learning_rate']),('treatment',d['treatment_learning_rate'])]:
  _,hist=fit(d['model_name'],cfg,a.download,lr);rows += [dict(condition=label,**r) for r in hist]
 save_csv(rows,'results/diagnostic/diagnostic_comparison.csv');print('Use real logs to fill observed problem, hypothesis, evidence and conclusion.')
