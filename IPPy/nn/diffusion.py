import math
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm_groups(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = torch.arange(half_dim, device=t.device, dtype=torch.float32) / max(
        half_dim, 1
    )
    freqs = torch.exp(-math.log(10000.0) * exponent)
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float32)
    alpha_bar = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, 0.999)


def extract(coefficients: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    out = coefficients.to(t.device)[t]
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


class ResBlock(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        time_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(ch_in), ch_in)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, ch_out))

        self.norm2 = nn.GroupNorm(_group_norm_groups(ch_out), ch_out)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)

        self.skip = (
            nn.Identity()
            if ch_in == ch_out
            else nn.Conv2d(ch_in, ch_out, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_norm_groups(channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_in = x
        x = self.norm(x).view(b, c, h * w).transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return x + h_in


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        dropout: float = 0.1,
        attn_levels: tuple[int, ...] = (1, 2),
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.init = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        channels = [base_ch * mult for mult in channel_mults]

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_ch
        skips = []
        for level, ch_out in enumerate(channels):
            block1 = ResBlock(ch, ch_out, time_dim, dropout=dropout)
            block2 = ResBlock(ch_out, ch_out, time_dim, dropout=dropout)
            attn = (
                SelfAttention2d(ch_out, num_heads=num_heads)
                if level in attn_levels
                else nn.Identity()
            )
            self.down_blocks.append(nn.ModuleList([block1, block2, attn]))
            skips.append(ch_out)
            ch = ch_out
            if level != len(channels) - 1:
                self.downsamples.append(Downsample(ch))

        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout=dropout)
        self.mid_attn = SelfAttention2d(ch, num_heads=num_heads)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            ch_skip = skips[level]
            block1 = ResBlock(ch + ch_skip, ch_skip, time_dim, dropout=dropout)
            block2 = ResBlock(ch_skip, ch_skip, time_dim, dropout=dropout)
            attn = (
                SelfAttention2d(ch_skip, num_heads=num_heads)
                if level in attn_levels
                else nn.Identity()
            )
            self.up_blocks.append(nn.ModuleList([block1, block2, attn]))
            ch = ch_skip
            if level != 0:
                self.upsamples.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(_group_norm_groups(ch), ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))

        h = self.init(x)
        skips = []
        for level, (block1, block2, attn) in enumerate(self.down_blocks):
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            h = attn(h)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for level, (block1, block2, attn) in enumerate(self.up_blocks):
            skip = skips.pop()
            if level > 0:
                h = self.upsamples[level - 1](h)
            h = torch.cat([h, skip], dim=1)
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            h = attn(h)

        return self.out_conv(self.out_act(self.out_norm(h)))


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        shadow_state = self.shadow.state_dict()
        model_state = model.state_dict()
        for name, value in model_state.items():
            if not value.dtype.is_floating_point:
                shadow_state[name].copy_(value)
                continue
            shadow_state[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": self.shadow.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.shadow.load_state_dict(state_dict["shadow"])


def denormalize_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def predict_x0_from_eps(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    return (
        x_t - extract((1 - alpha_bars).sqrt(), t, x_t.shape) * eps_pred
    ) / extract(alpha_bars.sqrt(), t, x_t.shape)


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    num_samples: int,
    image_shape: tuple[int, int, int],
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    x = torch.randn(num_samples, *image_shape, device=device)
    alpha_bars_prev = torch.cat(
        [torch.ones(1, device=device), alpha_bars[:-1].to(device)], dim=0
    )
    posterior_var = betas.to(device) * (1 - alpha_bars_prev) / (1 - alpha_bars.to(device))
    posterior_var = posterior_var.clamp(min=1e-20)

    for step in reversed(range(len(betas))):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        x0_hat = predict_x0_from_eps(x, eps_pred, t, alpha_bars).clamp(-1.0, 1.0)

        alpha_t = alphas[step].to(device)
        alpha_bar_t = alpha_bars[step].to(device)
        alpha_bar_prev_t = alpha_bars_prev[step]
        beta_t = betas[step].to(device)

        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev_t) / (1 - alpha_bar_t)
        coef_xt = (1 - alpha_bar_prev_t) * torch.sqrt(alpha_t) / (1 - alpha_bar_t)
        mean = coef_x0 * x0_hat + coef_xt * x

        if step > 0:
            x = mean + torch.sqrt(posterior_var[step]) * torch.randn_like(x)
        else:
            x = mean

    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    num_samples: int,
    image_shape: tuple[int, int, int],
    alpha_bars: torch.Tensor,
    device: str | torch.device,
    sample_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    schedule = torch.linspace(
        len(alpha_bars) - 1, 0, sample_steps, dtype=torch.long, device=device
    )
    x = torch.randn(num_samples, *image_shape, device=device)

    for i in range(len(schedule) - 1):
        t_current = int(schedule[i].item())
        t_next = int(schedule[i + 1].item())
        t = torch.full((num_samples,), t_current, device=device, dtype=torch.long)

        eps_pred = model(x, t)
        alpha_bar_t = alpha_bars[t_current].to(device)
        alpha_bar_next = alpha_bars[t_next].to(device)
        x0_hat = predict_x0_from_eps(x, eps_pred, t, alpha_bars).clamp(-1.0, 1.0)

        sigma_t = eta * torch.sqrt(
            ((1 - alpha_bar_next) / (1 - alpha_bar_t))
            * (1 - alpha_bar_t / alpha_bar_next)
        )
        coeff_eps = torch.sqrt(torch.clamp(1 - alpha_bar_next - sigma_t**2, min=0.0))
        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        x = torch.sqrt(alpha_bar_next) * x0_hat + coeff_eps * eps_pred + sigma_t * noise

    t_final = torch.full(
        (num_samples,), int(schedule[-1].item()), device=device, dtype=torch.long
    )
    eps_final = model(x, t_final)
    return predict_x0_from_eps(x, eps_final, t_final, alpha_bars).clamp(-1.0, 1.0)
