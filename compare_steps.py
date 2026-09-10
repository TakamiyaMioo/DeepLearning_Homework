import argparse
import torch
from src.diffusion import GaussianDiffusion
from src.model import TinyUNet
from src.utils import save_grid


def main(a):
    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(a.checkpoint, map_location=device); cfg = ckpt.get("args", {})
    model = TinyUNet(base=cfg.get("base", 32)).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    diffusion = GaussianDiffusion(timesteps=cfg.get("timesteps", 1000), device=device)
    for steps in (1000, 200, 50):
        save_grid(diffusion.sample_subsequence(model, a.num_samples, steps, device=device), f"{a.output_dir}/steps_{steps}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--output-dir", default="outputs/compare"); p.add_argument("--num-samples", type=int, default=64); p.add_argument("--device", default=None); main(p.parse_args())
