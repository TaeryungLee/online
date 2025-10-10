import math
import torch
import torch.nn as nn
from models.denoiser_init import DenoiserInit

# ---------- EDM utilities ----------
def edm_sigma_from_ramp(ramp, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """
    EDM 보간식: ramp=0 -> sigma_max (고노이즈), ramp=1 -> sigma_min (저노이즈)
    ramp: scalar or tensor
    """
    if not torch.is_tensor(ramp):
        ramp = torch.as_tensor(ramp, dtype=torch.float32)
    r = ramp.to(torch.float32)
    inv_rho = 1.0 / rho
    smin = sigma_min ** inv_rho
    smax = sigma_max ** inv_rho
    return (smax + r * (smin - smax)) ** rho

def edm_sigma_steps(num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu"):
    """
    EDM 스텝열 (내림차순 σ가 되도록 ramp는 0→1 증가).
    """
    ramp = torch.linspace(0, 1, num_steps, device=device)
    return edm_sigma_from_ramp(ramp, sigma_min, sigma_max, rho)

@torch.no_grad()
def ode_drift_from_x0(x, sigma, x0):
    """
    VE-ODE drift (x0-pred): dx/dσ = (x - x0) / σ
    sigma: x와 브로드캐스트 가능한 shape (보통 [B,1,1,...])
    """
    return (x - x0) / sigma.clamp_min(1e-12)

def cfg_combine(x0_cond, x0_uncond, scale: float):
    if scale is None or scale == 1.0:
        return x0_cond
    return x0_uncond + scale * (x0_cond - x0_uncond)


class DiffusionInit(nn.Module):
    """Uniform-schedule Diffusion Init (initial x0 generator)"""
    def __init__(self, cfg):
        super().__init__()
        self.denoiser = DenoiserInit(cfg)   # (x_noisy, sigma, condition) -> x0_pred

        # schedule / loss params
        self.num_timesteps = int(getattr(cfg, "num_timesteps", 18))
        self.sigma_data    = float(getattr(cfg, "sigma_data", 0.5))
        self.sigma_min     = float(getattr(cfg, "sigma_min", 0.002))
        self.sigma_max     = float(getattr(cfg, "sigma_max", 80.0))
        self.rho           = float(getattr(cfg, "rho", 7.0))

        # optional: small stochasticity (off by default)
        self.heun_churn    = float(getattr(cfg, "heun_churn", 0.0))

    # ---------------- TRAIN ----------------
    def forward(self, target, condition):
        """
        target: clean x0 tensor (B, C, T, ...)  -- 전 프레임 동일 σ 적용
        condition: any (B, ...)
        반환: loss_mean, pred_xstart
        """
        B = target.shape[0]
        device = target.device

        # t는 추론 격자 {i/(N-1)}에서 샘플 (스케줄 미스매치 제거)
        N = max(int(self.num_timesteps), 2)
        idx = torch.randint(low=0, high=N, size=(B,), device=device)      # [B] ∈ {0..N-1}
        t = idx.to(torch.float32) / (N - 1)                                # [B] ∈ {i/(N-1)}

        # σ (배치 스칼라) → target과 브로드캐스트
        sigma = edm_sigma_from_ramp(t, self.sigma_min, self.sigma_max, self.rho)  # [B]
        sigma_b = sigma.view(B, *([1] * (target.dim() - 1)))                      # [B,1,1,...]

        # 노이즈 주입 (VE)
        noise = torch.randn_like(target)
        x_noisy = target + sigma_b * noise

        # x0 예측
        pred_xstart = self.denoiser(x_noisy, sigma, condition)   # (B, C, T, ...)

        # EDM 가중 MSE: weight = (σ^2 + σ_data^2) / (σ σ_data)^2
        w = (sigma_b**2 + (self.sigma_data**2)) / ((sigma_b * self.sigma_data)**2)
        loss = w * (pred_xstart - target) ** 2
        loss = loss.view(B, -1).mean(dim=1)                      # per-sample
        return loss.mean(), pred_xstart

    # ---------------- SAMPLE ----------------
    @torch.no_grad()
    def sample(
        self,
        condition,
        shape,                 # (B, C, T, ...)
        cfg: float = 1.0,
        num_steps: int = None
    ):
        """
        Heun ODE (EDM), uniform σ per-sample across all frames.
        - σ 스케줄: Karras/EDM ρ-스케줄
        - 초기화: σ_0 * N(0,1) (σ_0 = sigma_steps[0])
        """
        device = condition.device if torch.is_tensor(condition) else ("cuda" if torch.cuda.is_available() else "cpu")
        B = shape[0]
        N = int(num_steps or self.num_timesteps)
        N = max(N, 2)

        # σ 스텝 (공유 스칼라 σ; 배치/프레임 전역 동일)
        sigmas = edm_sigma_steps(N, self.sigma_min, self.sigma_max, self.rho, device=device)  # [N]
        sigma0 = sigmas[0].view(1, *([1] * (len(shape) - 1)))                                 # [1,1,1,...]

        # 초기 상태
        x = torch.randn(shape, device=device) * sigma0

        for i in range(N - 1):
            sigma_i   = sigmas[i]
            sigma_next= sigmas[i + 1]
            sb   = sigma_i.view(1, *([1] * (len(shape) - 1)))       # [1,1,1,...]
            sb_n = sigma_next.view(1, *([1] * (len(shape) - 1)))
            h = (sb_n - sb)

            # (옵션) EDM churn (기본 0)
            if self.heun_churn > 0:
                gamma = min(self.heun_churn, math.sqrt(2) - 1.0)
                eps = torch.randn_like(x)
                x = x + eps * (sb * gamma)
                sb = sb * (1.0 + gamma)

            # Heun 1단계: Euler
            # denoiser의 σ 인자는 배치 스칼라가 필요하므로 [B]로 맞춰서 전달
            sigma_batch = torch.full((B,), float(sigma_i), device=device)
            x0_c = self.denoiser(x, sigma_batch, condition)
            if cfg != 1.0:
                x0_u = self.denoiser(x, sigma_batch, None)
                x0   = cfg_combine(x0_c, x0_u, cfg)
            else:
                x0 = x0_c

            d_cur = ode_drift_from_x0(x, sb, x0)
            x_eul = x + h * d_cur

            if i == N - 2:
                x = x_eul
                break

            # Heun 보정
            sigma_batch_n = torch.full((B,), float(sigma_next), device=device)
            x0_c_n = self.denoiser(x_eul, sigma_batch_n, condition)
            if cfg != 1.0:
                x0_u_n = self.denoiser(x_eul, sigma_batch_n, None)
                x0_n   = cfg_combine(x0_c_n, x0_u_n, cfg)
            else:
                x0_n = x0_c_n

            d_next = ode_drift_from_x0(x_eul, sb_n, x0_n)
            x = x + 0.5 * h * (d_cur + d_next)

        # σ→0 한계에서 x ≈ x0
        return x
