"""Generate report figures and timing data from a trained checkpoint (no training)."""
import argparse
import json
import time
from pathlib import Path

import torch

from src.diffusion import GaussianDiffusion
from src.model import TinyUNet
from src.utils import save_grid


def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    model = TinyUNet(base=cfg.get("base", 32)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(timesteps=cfg.get("timesteps", 1000), device=device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    results = {"device": str(device), "num_samples": args.num_samples, "checkpoint_step": ckpt.get("step")}
    for steps in (1000, 200, 50):
        torch.manual_seed(args.seed)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        images = diffusion.sample_subsequence(model, args.num_samples, steps, device=device)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        save_grid(images.cpu(), out / f"steps_{steps}.png")
        results[f"steps_{steps}_seconds"] = round(elapsed, 4)
        results[f"steps_{steps}_seconds_per_image"] = round(elapsed / args.num_samples, 4)
    (out / "timing.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/train/epoch_020.pt")
    p.add_argument("--output-dir", default="ClassReport/figures")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default=None)
    main(p.parse_args())
