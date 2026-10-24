import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/base.yaml');p.add_argument('--download',action='store_true')
if __name__=='__main__':
 a=p.parse_args();subprocess.run([sys.executable,'-m','numpy_bp.run_gradient_check'],check=True);subprocess.run([sys.executable,'-m','scripts.compare_models','--config',a.config]+(['--download'] if a.download else []),check=True)
