import argparse
from pathlib import Path
import torch
from src.diffusion import GaussianDiffusion
from src.model import TinyUNet
from src.utils import save_grid


def main(a):
    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(a.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    model = TinyUNet(base=train_args.get("base", 32)).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    diffusion = GaussianDiffusion(a.timesteps, device=device)
    save_grid(diffusion.sample(model, a.num_samples, device=device), a.output, nrow=8)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--output", default="outputs/sample.png"); p.add_argument("--num-samples", type=int, default=64); p.add_argument("--timesteps", type=int, default=1000); p.add_argument("--device", default=None); main(p.parse_args())

