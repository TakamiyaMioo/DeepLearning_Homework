import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_grid(images, path, nrow=8):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image((images + 1) / 2, path, nrow=nrow)


def save_loss_curve(history, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4)); plt.plot(history); plt.xlabel("step"); plt.ylabel("MSE loss"); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

