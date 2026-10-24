import torch


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = self.alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1 - self.alpha_bars).sqrt()

    def _extract(self, values, t, x_shape):
        return values.gather(0, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        a = self._extract(self.sqrt_alpha_bars, t, x0.shape)
        b = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return a * x0 + b * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, t):
        beta = self._extract(self.betas, t, x.shape)
        alpha = self._extract(self.alphas, t, x.shape)
        abar = self._extract(self.alpha_bars, t, x.shape)
        pred_noise = model(x, t)
        mean = (x - beta * pred_noise / (1 - abar).sqrt()) / alpha.sqrt()
        if t[0].item() == 0:
            return mean
        posterior_var = beta * (1 - self._extract(self.alpha_bars, t - 1, x.shape)) / (1 - abar)
        return mean + posterior_var.clamp_min(1e-20).sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, model, n, image_size=28, device=None):
        device = device or self.device
        x = torch.randn(n, 1, image_size, image_size, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_subsequence(self, model, n, sampling_steps, image_size=28, device=None):
        """DDPM ancestral sampling over an evenly spaced subset of the base schedule."""
        device = device or self.device
        indices = torch.linspace(self.timesteps - 1, 0, sampling_steps, device=device).long()
        x = torch.randn(n, 1, image_size, image_size, device=device)
        for i, current in enumerate(indices):
            t = torch.full((n,), current.item(), device=device, dtype=torch.long)
            abar = self._extract(self.alpha_bars, t, x.shape)
            pred_noise = model(x, t)
            if i + 1 < len(indices):
                prev = indices[i + 1]
                abar_prev = self._extract(self.alpha_bars, torch.full_like(t, prev), x.shape)
                alpha_skip = abar / abar_prev
                beta_skip = 1 - alpha_skip
                x0_pred = (x - (1 - abar).sqrt() * pred_noise) / abar.sqrt()
                mean = (abar_prev.sqrt() * beta_skip / (1 - abar) * x0_pred
                        + alpha_skip.sqrt() * (1 - abar_prev) / (1 - abar) * x)
                variance = ((1 - abar_prev) / (1 - abar) * beta_skip).clamp_min(1e-20)
                x = mean + variance.sqrt() * torch.randn_like(x)
            else:
                x = (x - (1 - abar).sqrt() * pred_noise) / abar.sqrt()
        return x.clamp(-1, 1)
