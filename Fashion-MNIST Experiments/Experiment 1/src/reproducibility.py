import random,numpy as np,torch
def seed_everything(seed,deterministic=True):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if deterministic:torch.use_deterministic_algorithms(True,warn_only=True)
