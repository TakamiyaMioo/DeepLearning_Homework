import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.diffusion import GaussianDiffusion
from src.model import TinyUNet, count_parameters
from src.utils import save_loss_curve, save_grid, seed_everything


def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = TinyUNet(base=args.base).to(device)
    print(f"device={device}, trainable parameters={count_parameters(model):,}")
    assert count_parameters(model) <= 5_000_000
    diffusion = GaussianDiffusion(args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    history, step = [], 0
    model.train()
    for epoch in range(args.epochs):
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for x0, _ in bar:
            x0 = x0.to(device, non_blocking=True)
            t = torch.randint(0, args.timesteps, (x0.size(0),), device=device)
            xt, noise = diffusion.q_sample(x0, t)
            loss = nn.functional.mse_loss(model(xt, t), noise)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            history.append(loss.item()); step += 1; bar.set_postfix(loss=f"{loss.item():.4f}")
            if args.save_every > 0 and step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / "latest.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / f"epoch_{epoch + 1:03d}.pt")
    save_loss_curve(history, out / "loss.png")
    model.eval(); save_grid(diffusion.sample(model, 64, device=device), out / "samples_final.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data"); p.add_argument("--output-dir", default="outputs/train")
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--base", type=int, default=32); p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=1000); p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default=None)
    main(p.parse_args())

