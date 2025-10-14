import math
import torch
import torch.nn as nn
from models.denoiser_init import DenoiserInit
from tqdm import tqdm
from diffusers import EDMEulerScheduler


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
        self.cfg = cfg
        self.denoiser = DenoiserInit(cfg)   # (x_noisy, sigma, condition) -> x0_pred

        # schedule / loss params
        self.num_timesteps = int(getattr(cfg, "num_timesteps", 18))
        self.sigma_data    = float(getattr(cfg, "sigma_data", 0.5))
        self.sigma_min     = float(getattr(cfg, "sigma_min", 0.002))
        self.sigma_max     = float(getattr(cfg, "sigma_max", 80.0))
        self.rho           = float(getattr(cfg, "rho_init", 7.0))

        # optional: small stochasticity (off by default)
        self.P_mean     = float(getattr(cfg, "P_mean", -1.2))
        self.P_std      = float(getattr(cfg, "P_std",  1.2))
        # self.scheduler = EDMEulerScheduler(
        #     sigma_min=self.sigma_min,
        #     sigma_max=self.sigma_max,
        #     sigma_data=self.sigma_data,
        #     rho=self.rho,                  # Karras rho
        #     # prediction_type="epsilon"    # 기본값: epsilon
        # )
        
        null_dim = self.cfg.text_dim if self.cfg.text_dim is not None else self.cfg.latent_dim
        self.null_ctx = nn.Parameter(torch.zeros(1, 1, null_dim))  # L_null=1
        nn.init.normal_(self.null_ctx, std=0.02)


    # ---------------- TRAIN ----------------
    def forward(self, target, condition, condition_len):
        """
        target: clean x0 tensor (B, C, T, ...)  -- 전 프레임 동일 σ 적용
        condition: any (B, ...)
        반환: loss_mean, pred_xstart
        """
        B = target.shape[0]
        device = target.device
        rnd = torch.randn(B, 1, 1, 1, device=device, dtype=target.dtype)
        sigma = torch.exp(self.P_mean + self.P_std * rnd).view(B)  # [B]


        # 노이즈 주입 (VE)
        noise = torch.randn_like(target)
        sigma_b = sigma.view(B, *([1] * (target.dim() - 1)))          # [B,1,1,...]
        x_noisy = target + sigma_b * noise

        # x0 예측 (CFG during training controlled by self.cfg.cfg)

        # uncond용 null context / 길이
        null_ctx = self.null_ctx.expand(B, condition.shape[1], -1)           # [B, 1, C_txt]
        len_uncond = torch.ones(B, device=device, dtype=torch.long)   # [B]

        # 배치 결합 (x, sigma, condition, condition_len)
        x_cat         = torch.cat([x_noisy, x_noisy], dim=0)          # [2B, ...]
        sigma_cat     = torch.cat([sigma,   sigma],   dim=0)          # [2B]
        cond_cat      = torch.cat([condition, null_ctx], dim=0) if condition is not None else torch.cat([null_ctx, null_ctx], dim=0)
        condlen_cat   = torch.cat([condition_len, len_uncond], dim=0) if condition_len is not None else torch.cat([len_uncond, len_uncond], dim=0)

        # 한 번의 forward로 x0_u, x0_c 동시 계산
        x0_both = self.denoiser(x_cat, sigma_cat, cond_cat, condlen_cat)  # [2B, ...]
        x0_c = x0_both[:B]   # 주의: cond 먼저/나중 순서는 위 cat 순서에 맞춰 선택
        x0_u = x0_both[B:]

        if self.cfg.cfg != 1.0:
            pred_xstart = x0_u + self.cfg.cfg * (x0_c - x0_u)
        else:
            pred_xstart = x0_c

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
        condition_len,
        motion_length,
        num_steps: int = None
    ):
        """
        Heun ODE (EDM), uniform σ per-sample across all frames.
        - σ 스케줄: Karras/EDM ρ-스케줄
        - 초기화: σ_0 * N(0,1) (σ_0 = sigma_steps[0])
        """
        device = condition.device if torch.is_tensor(condition) else ("cuda" if torch.cuda.is_available() else "cpu")
        B = condition.shape[0]
        N = int(num_steps or self.num_timesteps)
        shape = (B, motion_length, self.cfg.latent_dim)

        # σ 스텝 (공유 스칼라 σ; 배치/프레임 전역 동일)
        sigmas = edm_sigma_steps(N, self.sigma_min, self.sigma_max, self.rho, device=device)  # [N]
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1], device=device)])
        sigma0 = sigmas[0].view(1, *([1] * (len(shape) - 1)))                                 # [1,1,1,...]

        cfg = self.cfg.cfg
        
        # 초기 상태
        x = torch.randn(shape, device=device) * sigma0

        # uncond
        null_ctx = self.null_ctx.expand(B, condition.shape[1], -1)
        len_uncond = torch.ones(B, device=device, dtype=torch.long)

        for i in tqdm(range(N), desc="Sampling initial window", leave=False):
            sigma_i   = sigmas[i]
            sigma_next= sigmas[i + 1]
            sb   = sigma_i.view(1, *([1] * (len(shape) - 1)))       # [1,1,1,...]
            sb_n = sigma_next.view(1, *([1] * (len(shape) - 1)))
            h = (sb_n - sb)

            # Heun 1단계: Euler
            # denoiser의 σ 인자는 배치 스칼라가 필요하므로 [B]로 맞춰서 전달
            # x0 predict한 이후 그 방향으로 적분

            sigma_cat = torch.full((2*B,), float(sigma_i), device=device, dtype=x.dtype)
            x_cat       = torch.cat([x, x], dim=0)   # [2B, T, C]
            cond_cat    = torch.cat([condition, null_ctx], dim=0)
            condlen_cat = torch.cat([condition_len, len_uncond], dim=0)

            x0_both = self.denoiser(x_cat, sigma_cat, cond_cat, condlen_cat)  # [2B, T, C]
            x0_c, x0_u = x0_both[:B], x0_both[B:]

            x0 = x0_u + cfg * (x0_c - x0_u)

            d_cur = ode_drift_from_x0(x, sb, x0)
            x_eul = x + h * d_cur
            
            if i < N - 1:
                # Heun 보정
                sigma_cat_n = torch.full((2*B,), float(sigma_next), device=device, dtype=x.dtype)
                # x0_c_n = self.denoiser(x_eul, sigma_cat_n, condition, condition_len)
                # if cfg != 1.0:
                #     x0_u_n = self.denoiser(x_eul, sigma_cat_n, None, condition_len)
                #     x0_n   = cfg_combine(x0_c_n, x0_u_n, cfg)
                # else:
                #     x0_n = x0_c_n
                x_eul_cat = torch.cat([x_eul, x_eul], dim=0)

                x0_both_n = self.denoiser(x_eul_cat, sigma_cat_n, cond_cat, condlen_cat)  # [2B, T, C]
                x0_c_n, x0_u_n = x0_both_n[:B], x0_both_n[B:]
                x0_n = x0_u_n + cfg * (x0_c_n - x0_u_n)

                d_next = ode_drift_from_x0(x_eul, sb_n, x0_n)
                x = x + 0.5 * h * (d_cur + d_next)

        return x

    # # ---------------- SAMPLE ----------------
    # @torch.no_grad()
    # def sample(
    #     self,
    #     condition,
    #     condition_len,
    #     motion_length,
    #     num_steps: int = None
    # ):
    #     """
    #     Heun ODE (EDM), uniform σ per-sample across all frames.
    #     - σ 스케줄: Karras/EDM ρ-스케줄
    #     - 초기화: σ_0 * N(0,1) (σ_0 = sigma_steps[0])
    #     """
    #     device = condition.device if torch.is_tensor(condition) else ("cuda" if torch.cuda.is_available() else "cpu")
    #     B = condition.shape[0]
    #     N = int(num_steps or self.num_timesteps)
    #     shape = (B, motion_length, self.cfg.latent_dim)
    #     cfg = self.cfg.cfg

    #     # σ 스텝 (공유 스칼라 σ; 배치/프레임 전역 동일)
    #     sigmas = edm_sigma_steps(N, self.sigma_min, self.sigma_max, self.rho, device=device)  # [N]
    #     sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1], device=device)])
    #     sigma0 = sigmas[0].view(1, *([1] * (len(shape) - 1)))                                 # [1,1,1,...]
        
    #     # 초기 상태
    #     x = torch.randn(shape, device=device) * sigma0

    #     null_ctx = self.null_ctx.expand(B, condition.shape[1], -1)
    #     len_uncond = torch.ones(B, device=device, dtype=torch.long)

    #     for i in tqdm(range(N), desc="Sampling initial window", leave=False):
    #         sigma_i   = sigmas[i]
    #         sigma_next= sigmas[i + 1]
    #         sb   = sigma_i.view(1, *([1] * (len(shape) - 1)))       # [1,1,1,...]
    #         sb_n = sigma_next.view(1, *([1] * (len(shape) - 1)))
    #         h = (sb_n - sb)

    #         # Heun 1단계: Euler
    #         # denoiser의 σ 인자는 배치 스칼라가 필요하므로 [B]로 맞춰서 전달
    #         sigma_batch = torch.full((B,), float(sigma_i), device=device)
    #         # x0 predict한 이후 그 방향으로 적분
    #         x0_c = self.denoiser(x, sigma_batch, condition, condition_len)
    #         if cfg != 1.0:
    #             x0_u = self.denoiser(x, sigma_batch, null_ctx, len_uncond)
    #             x0   = cfg_combine(x0_c, x0_u, cfg)
    #         else:
    #             x0 = x0_c
    #         # return (x - x0) / sigma.clamp_min(1e-12)
    #         d_cur = ode_drift_from_x0(x, sb, x0)
    #         x_eul = x + h * d_cur
            
    #         if i < N - 1:
    #             # Heun 보정
    #             sigma_batch_n = torch.full((B,), float(sigma_next), device=device)
    #             x0_c_n = self.denoiser(x_eul, sigma_batch_n, condition, condition_len)
    #             if cfg != 1.0:
    #                 x0_u_n = self.denoiser(x_eul, sigma_batch_n, null_ctx, len_uncond)
    #                 x0_n   = cfg_combine(x0_c_n, x0_u_n, cfg)
    #             else:
    #                 x0_n = x0_c_n

    #             d_next = ode_drift_from_x0(x_eul, sb_n, x0_n)
    #             x = x + 0.5 * h * (d_cur + d_next)

    #     # σ→0 한계에서 x ≈ x0
    #     return x






    # @torch.no_grad()
    # def sample(
    #     self,
    #     condition,
    #     condition_len,
    #     motion_length,
    #     num_steps: int = None
    # ):
    #     """
    #     diffusers EDMEulerScheduler 기반 + 배치 결합 1패스 CFG
    #     입력/출력 시그니처는 기존과 동일.
    #     """
    #     device = condition.device if torch.is_tensor(condition) else ("cuda" if torch.cuda.is_available() else "cpu")
    #     dtype  = condition.dtype  if torch.is_tensor(condition) else torch.float32

    #     B = condition.shape[0]
    #     N = int(num_steps or self.num_timesteps)
    #     N = max(N, 2)
    #     shape = (B, motion_length, self.cfg.latent_dim)
    #     cfg = self.cfg.cfg

    #     # --- 스케줄러 파라미터 동기화 & timesteps 설정 ---
    #     self.scheduler.set_timesteps(N, device=device)
    #     self.scheduler.is_scale_input_called = True

    #     sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)   # [N+1] (마지막이 0 근처)
    #     t_steps = self.scheduler.timesteps.to(device=device, dtype=dtype)
    #     sigma0 = sigmas[0].view(1, 1, 1)

    #     # --- 초기 상태: x ~ N(0,1) * sigma_max ---
    #     x = torch.randn(shape, device=device, dtype=dtype) * sigma0

    #     # --- uncond용 null context 준비 ---
    #     null_ctx = self.null_ctx.expand(B, condition.shape[1], -1)
    #     len_uncond = torch.ones(B, device=device, dtype=torch.long)

    #     # --- 샘플링 루프 ---
    #     for i in range(len(sigmas) - 1):
    #         sigma_i = sigmas[i]                                     # 스칼라 텐서
    #         t_i = t_steps[i]
    #         sb = sigma_i.view(1, 1, 1)                              # [1,1,1]
    #         sigma_cat = torch.full((2 * B,), float(sigma_i), device=device, dtype=dtype)  # [2B]

    #         # 배치 결합: cond / uncond를 한 번에 추론
    #         x_cat       = torch.cat([x, x], dim=0)   # [2B, T, C]
    #         cond_cat    = torch.cat([condition, null_ctx], dim=0) if condition is not None else torch.cat([null_ctx, null_ctx], dim=0)
    #         condlen_cat = torch.cat([condition_len, len_uncond], dim=0) if condition_len is not None else torch.cat([len_uncond, len_uncond], dim=0)

    #         # 한 번의 forward로 x0_c, x0_u 동시 계산
    #         x0_both = self.denoiser(x_cat, sigma_cat, cond_cat, condlen_cat)  # [2B, T, C]
    #         x0_c, x0_u = x0_both[:B], x0_both[B:]

    #         # CFG 결합
    #         if cfg != 1.0:
    #             x0 = x0_u + cfg * (x0_c - x0_u)
    #         else:
    #             x0 = x0_c

    #         # x0 -> epsilon 변환 (EDMEuler는 epsilon을 기대)
    #         eps = (x - x0) / sb.clamp_min(1e-12)

    #         # diffusers 한 스텝
    #         out = self.scheduler.step(
    #             model_output=eps,
    #             timestep=t_i,   # EDMEuler는 sigma값을 timestep으로 사용
    #             sample=x,
    #             return_dict=True
    #         )
    #         x = out.prev_sample

    #     # sigma -> 0 한계에서 x ≈ x0
    #     return x