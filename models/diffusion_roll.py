# diffusion_roll.py
import math
import torch
import torch.nn as nn
from models.denoiser_roll import DenoiserRoll


# =========================
# Ramp & EDM sigma utilities
# =========================



def _erdm_ramp(t, w, W: int, k: int):
    """
    ERDM ramp: r = 1 - ((w - t(k+1) - k) / (W - 2k - 1))
    - t: [B,1] in [0,1] (continuous) or [B] (자동 확장)
    - w: [1,W] (0..W-1)
    반환: [B,W] in [0,1]
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    if t.dim() == 1:
        t = t[:, None]  # [B,1]
    device = t.device
    w = torch.as_tensor(w, dtype=torch.float32, device=device)  # [1,W]
    denom = max(W - 2*k - 1, 1e-6)
    r = 1 - ((w - t*(k + 1) - k) / denom)          # [B,W]
    return r.clamp_(0.0, 1.0)

def erdm_ramp(t, w, W: int, k: int):
    zero_ramp = _erdm_ramp(torch.zeros_like(t), w, W, k)
    one_ramp = _erdm_ramp(torch.ones_like(t), w, W, k)
    return zero_ramp + (one_ramp - zero_ramp) * t[:, None]


def edm_sigma_from_ramp(ramp, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """
    EDM 보간식:
      ramp=0 -> sigma_max (고노이즈), ramp=1 -> sigma_min (저노이즈)
    """
    if not torch.is_tensor(ramp):
        ramp = torch.as_tensor(ramp, dtype=torch.float32)
    r = ramp.to(dtype=torch.float32)
    inv_rho = 1.0 / rho
    smin = sigma_min ** inv_rho
    smax = sigma_max ** inv_rho
    sigmas = (smax + r * (smin - smax)) ** rho
    return sigmas

def make_sigma_map(sigma_bw, target_shape, time_dim: int):
    """
    sigma_bw: [B,W] 프레임별 σ
    target_shape: e.g. (B,C,W,J) or (B,C,T) ...
    time_dim: 시간축 인덱스 (기본 2)
    반환: target과 브로드캐스트 가능한 σ 맵 (target과 동일 rank)
    """
    B = target_shape[0]
    W = target_shape[time_dim]
    assert sigma_bw.shape == (B, W), f"sigma_bw must be [B,W], got {tuple(sigma_bw.shape)}"
    # shape like [B,1,...,W,...,1]
    shape = [1] * len(target_shape)
    shape[0] = B
    shape[time_dim] = W
    return sigma_bw.view(*shape)


# ================
# ODE helper funcs
# ================

@torch.no_grad()
def ode_drift_from_x0(x, sigma, x0):
    """
    VE ODE drift for x0-pred: dx/dσ = (x - x0) / σ
    sigma: broadcastable to x
    """
    return (x - x0) / sigma.clamp_min(1e-12)


def cfg_combine(x0_cond, x0_uncond, scale: float):
    if scale is None or scale == 1.0:
        return x0_cond
    return x0_uncond + scale * (x0_cond - x0_uncond)


# ==========================
# Main module w/ ERDM+EDM
# ==========================

class DiffusionRoll(nn.Module):
    """
    Diffusion Roll with:
      - ERDM ramp schedule (train: t~U[0,1] per-sample / sample: t progresses 0->1)
      - EDM-weighted MSE loss (x0 regression)
      - Heun ODE sampler (EDM) using x0-pred drift
    Denoiser: (x_noisy, sigma, condition) -> x0_pred
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.denoiser = DenoiserRoll(cfg)  # (x, sigma, condition) -> x0_pred

        # Shapes
        self.time_dim   = 1  # (B,C,T,...)에서 T의 인덱스
        self.W          = int(getattr(cfg, "window_size", 64))        # window size (훈련/샘플 모두 동일 예상)
        self.k          = int(getattr(cfg, "overlap_size", 8))-1         # overlap (0 < k < W)

        # EDM params
        self.sigma_data = float(getattr(cfg, "sigma_data", 0.5))
        self.sigma_min  = float(getattr(cfg, "sigma_min", 0.002))
        self.sigma_max  = float(getattr(cfg, "sigma_max", 80.0))
        self.rho        = float(getattr(cfg, "rho", 7.0))

        # Sampler
        self.num_timesteps = int(getattr(cfg, "num_timesteps", 32))

        # null context for CFG (matches text input dim)
        null_dim = self.cfg.text_dim if getattr(self.cfg, "text_dim", None) is not None else self.cfg.latent_dim
        self.null_ctx = nn.Parameter(torch.zeros(1, 1, null_dim))
        nn.init.normal_(self.null_ctx, std=0.02)


    # ---------- TRAIN ----------
    def forward(self, target, condition, condition_len, text_timing=None):
        """
        target: x0 clean, shape (B, C, T, ...)
        condition: any (B, ...)
        반환: (loss_mean, pred_x0)
        """
        B = target.shape[0]
        T = target.shape[self.time_dim]
        device = target.device

        # 1) t sample
        N = int(self.num_timesteps)
        idx = torch.randint(low=0, high=N, size=(B,), device=target.device)   # [B] 정수 인덱스
        t = idx.to(torch.float32) / N      

        # 2) w = [0..W-1] (공유)
        w = torch.arange(self.W, device=device, dtype=torch.float32)[None, :]  # [1,W]

        # 3) ramp & frame-wise σ: [B,W]
        ramp_bw = erdm_ramp(t, w, self.W, self.k)                                      # [B,W]
        sigma_bw = edm_sigma_from_ramp(ramp_bw, self.sigma_min, self.sigma_max, self.rho)  # [B,W]

        # 4) σ를 입력과 브로드캐스트 가능하게 확장
        sigma_map = make_sigma_map(sigma_bw, target.shape, self.time_dim)               # [B,1,...,W,...,1]

        # 5) 노이즈 주입 (VE)
        noise = torch.randn_like(target)
        x_noisy = target + sigma_map * noise

        # 6) x0 예측 with CFG (batch-concat cond/uncond, single forward)
        cfg_scale = self.cfg.cfg
        B = target.shape[0]
        # prepare uncond null context and lengths
        null_ctx = self.null_ctx.expand(B, condition.shape[1], -1)
        len_uncond = torch.ones(B, device=device, dtype=torch.long)

        x_cat         = torch.cat([x_noisy,     x_noisy],     dim=0)
        sigma_cat     = torch.cat([sigma_map,   sigma_map],   dim=0)
        cond_cat      = torch.cat([condition,   null_ctx],    dim=0)
        condlen_cat   = torch.cat([condition_len, len_uncond], dim=0)
        if text_timing is not None:
            text_timing_cat = torch.cat([text_timing, text_timing], dim=0)
        else:
            text_timing_cat = None

        x0_both = self.denoiser(x_cat, sigma_cat, cond_cat, condlen_cat, text_timing_cat)
        x0_c, x0_u = x0_both[:B], x0_both[B:]
        pred_x0 = x0_u + cfg_scale * (x0_c - x0_u) if cfg_scale != 1.0 else x0_c

        # 7) EDM weighted MSE
        # weight = (σ^2 + σ_data^2) / (σ * σ_data)^2  (elementwise)
        w_edm = (sigma_map**2 + (self.sigma_data**2)) / ((sigma_map * self.sigma_data)**2)
        loss = w_edm * (pred_x0 - target) ** 2
        loss = loss.view(B, -1).mean(dim=1)                                             # per-sample
        return loss.mean(), pred_x0


    @torch.no_grad()
    def stream_rollout(
        self,
        init_x0,                  
        condition,
        condition_len,
        total_frames: int,         # 최종 생성할 총 프레임 수 (>= W 권장)
        num_steps: int = None,
    ):
        """
        k//2 프레임씩 윈도우를 뒤로 밀면서 스트리밍 롤아웃.
        - 각 윈도우는 Heun ODE(EDM) + frame-wise σ 스케줄로 clean을 생성.
        - 매 루프 후 앞쪽 hop 프레임을 외부로 내보내고, 뒤쪽엔 노이즈 hop 프레임을 덧붙여 다음 윈도우 시작.
        반환: (long_x0)  (B, C, total_frames, ...)
        """
        cfg_scale = self.cfg.cfg
        shape = init_x0.shape
        B = shape[0]
        T = shape[self.time_dim]
        assert T == self.W, f"init_x0 time length {T} must equal W {self.W}"
        device = init_x0.device

        hop = self.k + 1
        N = int(num_steps or self.num_timesteps)


        # 결과 버퍼
        chunks = []
        produced = 0

        # 현재 clean 윈도우
        cur_x0 = init_x0.clone()

        # w 인덱스(프레임별 σ 계산용)
        w = torch.arange(self.W, device=device, dtype=torch.float32)[None, :]  # [1,W]

        # ---- 1) 현재 윈도우에서 Heun ODE로 다음 clean 윈도우 생성 ----
        # i=0의 frame-wise σ
        t0 = torch.zeros(B, device=device)
        ramp0 = erdm_ramp(t0, w, self.W, self.k)                               # [B,W]
        sigma0 = edm_sigma_from_ramp(ramp0, self.sigma_min, self.sigma_max, self.rho)  # [B,W]
        sigma0_map = make_sigma_map(sigma0, cur_x0.shape, self.time_dim)       # [B,1,...,W,...,1]

        eps0 = torch.randn_like(cur_x0)
        x = cur_x0 + (sigma0_map * eps0)

        null_ctx = self.null_ctx.expand(B, condition.shape[1], -1)
        len_uncond = torch.ones(B, device=device, dtype=torch.long)

        cond_cat = torch.cat([condition, null_ctx], dim=0)
        condlen_cat = torch.cat([condition_len, len_uncond], dim=0)
        while produced < total_frames:
            # text_timing_cat = torch.cat([text_timing, text_timing], dim=0)

            # Heun 적분
            for i in range(N):
                t_i   = torch.full((B,), i / (N),     device=device)
                t_nxt = torch.full((B,), (i + 1) / (N), device=device)

                ramp_i   = erdm_ramp(t_i,   w, self.W, self.k)                 # [B,W]
                ramp_nxt = erdm_ramp(t_nxt, w, self.W, self.k)                 # [B,W]

                sigma_i   = edm_sigma_from_ramp(ramp_i,   self.sigma_min, self.sigma_max, self.rho)  # [B,W]
                sigma_nxt = edm_sigma_from_ramp(ramp_nxt, self.sigma_min, self.sigma_max, self.rho)  # [B,W]

                sb   = make_sigma_map(sigma_i,   cur_x0.shape, self.time_dim)
                sb_n = make_sigma_map(sigma_nxt, cur_x0.shape, self.time_dim)

                dt = sb_n - sb
                breakpoint()

                # Euler
                x_cat = torch.cat([x, x], dim=0)
                sigma_cat = torch.cat([sb, sb], dim=0)


                x0_both = self.denoiser(x_cat, sigma_cat, cond_cat, condlen_cat, None)
                x0_c, x0_u = x0_both[:B], x0_both[B:]
                x0 = x0_u + cfg_scale * (x0_c - x0_u)

                d_cur  = ode_drift_from_x0(x, sb, x0)
                x_eul  = x + d_cur * dt

                if i < N - 1:  
                    # Heun correction
                    x_eul_cat = torch.cat([x, x], dim=0)
                    sigma_nxt_cat = torch.cat([sb, sb], dim=0)
                    x0_c_n = self.denoiser(x_eul_cat, sigma_nxt_cat, cond_cat, condlen_cat, None)
                    x0_u_n, x0_c_n = x0_c_n[:B], x0_c_n[B:]
                    x0_n = x0_u_n + cfg_scale * (x0_c_n - x0_u_n)

                    d_next = ode_drift_from_x0(x_eul, sb_n, x0_n)
                    x = x + 0.5 * dt * (d_cur + d_next)

            # 최종 clean 윈도우
            next_x0 = x

            # ---- 2) 앞쪽 hop 프레임을 외부로 배출 ----
            # 남은 필요 길이만큼 잘라 배출
            emit_len = min(hop, total_frames - produced)
            # emit_chunk = next_x0.narrow(self.time_dim, 0, emit_len)  # (B,emit_len,C,...)
            emit_chunk = x[:, :emit_len]
            chunks.append(emit_chunk)
            produced += emit_len
            if produced >= total_frames:
                break

            # ---- 3) 윈도우 슬라이드: 앞 hop을 버리고, 뒤에 노이즈 hop 프레임 붙이기 ----
            # prefix: next_x0[:, :, hop:, ...]  (길이 W - hop)

            # prefix = next_x0.narrow(self.time_dim, hop, self.W - hop)
            prefix = next_x0[:, hop:]
            # tail noise clean (B, hop, C, ...)
            tail_shape = list(x.shape)
            tail_shape[self.time_dim] = hop
            tail_noise = torch.randn(tail_shape, device=device)

            # 다음 루프의 cur_x0
            cur_x0 = torch.cat([prefix, tail_noise], dim=self.time_dim)

        # 마지막: 아직 window의 남은 프레임이 있고, total_frames을 채우지 못했다면(보통 위에서 종료됨) 처리
        # (일반적으로 위 while에서 정확히 채움)

        # 결과 연결
        long_x0 = torch.cat(chunks, dim=self.time_dim)  # (B, C, total_frames, ...)
        return long_x0






    @torch.no_grad()
    def sample(
        self,
        condition,
        init_x0,                 # (B, C, T, ...) 초기 윈도우의 clean 결과
        cfg_scale: float = 1.0,
        num_steps: int = None,
        heun_churn: float = None,
        warm_start_noise: float = 1.0,  # 0이면 x=init_x0에서 시작, 1이면 표준 노이즈 주입
    ):
        """
        Heun ODE (EDM) with frame-wise σ schedule, starting FROM given init_x0.
        - t progression: t_i = i/(N-1)
        - i=0에서 frame-wise σ를 sigma0_map으로 만들고, x = init_x0 + warm_start_noise*sigma0_map*ε 로 시작
        """
        assert init_x0 is not None, "init_x0 (clean) must be provided to start sampling."
        shape = init_x0.shape
        N = int(num_steps or self.num_timesteps)
        churn = self.heun_churn_default if heun_churn is None else heun_churn

        B = shape[0]
        T = shape[self.time_dim]
        if T != self.W:
            raise ValueError(f"init_x0 time length {T} must equal W {self.W}")

        device = init_x0.device
        w = torch.arange(self.W, device=device, dtype=torch.float32)[None, :]  # [1,W]

        # i=0 ramp/σ (frame-wise)
        t0 = torch.zeros(B, device=device)                                     # [B]
        ramp0 = erdm_ramp(t0, w, self.W, self.k)                               # [B,W]
        sigma0 = edm_sigma_from_ramp(ramp0, self.sigma_min, self.sigma_max, self.rho)  # [B,W]
        sigma0_map = make_sigma_map(sigma0, shape, self.time_dim)              # [B,1,...,W,...,1]

        # warm-start around init_x0
        if warm_start_noise == 0.0:
            x = init_x0.clone()
        else:
            eps0 = torch.randn_like(init_x0)
            x = init_x0 + (warm_start_noise * sigma0_map * eps0)

        for i in range(N - 1):
            t_i   = torch.full((B,), i / (N - 1),     device=device)           # [B]
            t_nxt = torch.full((B,), (i + 1) / (N - 1), device=device)         # [B]

            ramp_i   = erdm_ramp(t_i,   w, self.W, self.k)                     # [B,W]
            ramp_nxt = erdm_ramp(t_nxt, w, self.W, self.k)                     # [B,W]

            sigma_i   = edm_sigma_from_ramp(ramp_i,   self.sigma_min, self.sigma_max, self.rho)  # [B,W]
            sigma_nxt = edm_sigma_from_ramp(ramp_nxt, self.sigma_min, self.sigma_max, self.rho)  # [B,W]

            sb   = make_sigma_map(sigma_i,   shape, self.time_dim)             # [B,1,...,W,...,1]
            sb_n = make_sigma_map(sigma_nxt, shape, self.time_dim)

            h = (sb_n - sb)                                                    # frame-wise Δσ (음수)

            # (옵션) EDM churn
            if churn > 0:
                gamma = min(churn, math.sqrt(2) - 1.0)
                eps = torch.randn_like(x)
                x = x + eps * (sb * gamma)
                sb = sb * (1.0 + gamma)

            # Heun 1단계: Euler
            x0_c = self.denoiser(x, sb, condition)
            if cfg_scale != 1.0:
                x0_u = self.denoiser(x, sb, None)
                x0   = cfg_combine(x0_c, x0_u, cfg_scale)
            else:
                x0 = x0_c

            d_cur  = ode_drift_from_x0(x, sb, x0)
            x_eul  = x + h * d_cur

            if i == N - 2:
                x = x_eul
                break

            # Heun 보정
            x0_c_n = self.denoiser(x_eul, sb_n, condition)
            if cfg_scale != 1.0:
                x0_u_n = self.denoiser(x_eul, sb_n, None)
                x0_n   = cfg_combine(x0_c_n, x0_u_n, cfg_scale)
            else:
                x0_n = x0_c_n

            d_next = ode_drift_from_x0(x_eul, sb_n, x0_n)
            x = x + 0.5 * h * (d_cur + d_next)

        return x

