import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class ConditionEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class Block01(nn.Module):
    def __init__(self, cfg, num_heads, latent_dim, dropout, norm, activation, conv_mlp):
        super().__init__()                
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Cross-attention (text as KV)
        self.cross_attn_text = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention (text as KV, PE[text timing] as Q)
        self.cross_attn_text_time = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.text_time_PE = PositionalEncoding(latent_dim, dropout=dropout, max_len=1000, batch_first=True)(torch.zeros((1, 1000, latent_dim)))

        # Normalizations
        self.norm_sa = nn.LayerNorm(latent_dim)
        self.norm_ca_text = nn.LayerNorm(latent_dim)
        self.norm_ca_text_time = nn.LayerNorm(latent_dim)
        self.norm_ff = nn.LayerNorm(latent_dim)

        # AdaLN modulations
        self.mod_sa_sigma = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim)
        )
        self.mod_ca_texttime = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim)
        )
        self.mod_ff_sigma = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim)
        )

        # Feed-forward network
        hidden_multiplier = int(getattr(cfg, "denoiser_ff_mult", 4))
        hidden_dim = hidden_multiplier * latent_dim
        # TODO: conv mlp
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU() if getattr(cfg, "denoiser_activation", "gelu") == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x, sigma_enc, condition, condition_len, text_timing):
        B, T, C = x.shape

        # ----- Self-Attention (causal first, then Norm+AdaLN) -----
        attn_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        sa_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(sa_out)
        x_sa = self.norm_sa(x)
        gam_beta_sa = self.mod_sa_sigma(sigma_enc).unsqueeze(1)  # [B, 2C]
        gamma_sa, beta_sa = gam_beta_sa.chunk(2, dim=-1)
        x = x_sa * (1 + gamma_sa) + beta_sa
        
        # ----- Cross-Attention (text) -----
        if condition is not None:
            L = condition.shape[1]
            if condition_len is not None:
                idxs = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]
                key_padding_mask = idxs >= condition_len.view(-1, 1)  # [B, L] True -> PAD
            else:
                key_padding_mask = None
            ca_out, _ = self.cross_attn_text(x, condition, condition, key_padding_mask=key_padding_mask)
            x = x + self.dropout(ca_out)
            x_ca = self.norm_ca_text(x)

            # TODO: remove this, just use adaln with sigma_enc
            max_len_pe = self.text_time_PE.shape[1]
            starts = text_timing.to(torch.long).clamp(min=0, max=max_len_pe - T)
            ar = torch.arange(T, device=x.device).view(1, T)  # [1, T]
            idx = starts.view(-1, 1) + ar                      # [B, T]
            pe_exp = self.text_time_PE.to(x.device).expand(B, -1, -1)  # [B, max_len, C]
            text_time_PE = torch.gather(pe_exp, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # [B, T, C]
            text_time_ca_out, _ = self.cross_attn_text_time(text_time_PE, condition, condition, key_padding_mask=key_padding_mask)

            gam_beta_ca = self.mod_ca_texttime(text_time_ca_out)  # [B, 2C]
            gamma_ca, beta_ca = gam_beta_ca.chunk(2, dim=-1)
            x = x_ca * (1 + gamma_ca) + beta_ca

        # ----- Feed-Forward + Norm+AdaLN -----
        
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x_ff = self.norm_ff(x)
        gam_beta_ff = self.mod_ff_sigma(sigma_enc)
        gamma_ff, beta_ff = gam_beta_ff.chunk(2, dim=-1)
        x = x_ff * (1 + gamma_ff) + beta_ff
        

        return x



class DenoiserInit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sigma_data    = float(getattr(cfg, "sigma_data", 0.5))

        # model/sequence dims for positional encoding
        self.latent_dim = int(getattr(cfg, "latent_dim", 128))
        self.max_len = int(getattr(cfg, "window_size", 512))

        self.num_layers = int(getattr(cfg, "denoiser_num_layers", 12))
        self.num_heads = int(getattr(cfg, "denoiser_num_heads", 8))
        self.dropout = float(getattr(cfg, "denoiser_dropout", 0.1))
        self.norm = getattr(cfg, "denoiser_norm", "ln")
        self.activation = getattr(cfg, "denoiser_activation", "gelu")
        self.conv_mlp = getattr(cfg, "denoiser_conv_mlp", True)

        # positional encoding for inputs shaped as [B, T, C]
        self.pos_enc = PositionalEncoding(self.latent_dim, dropout=self.dropout, max_len=self.max_len, batch_first=True)
        self.sigma_enc = ConditionEmbedding(self.latent_dim)
        self.time_enc = ConditionEmbedding(self.latent_dim)
        self.text_proj = None
        self.text_in_dim = getattr(cfg, "text_dim", None)
        if self.text_in_dim is not None and self.text_in_dim != self.latent_dim:
            self.text_proj = nn.Linear(self.text_in_dim, self.latent_dim)

        denoiser_block = getattr(cfg, "denoiser_block", 1)
        block_name = f"Block{str(denoiser_block).zfill(2)}"
        block = eval(block_name)

        self.blocks = nn.ModuleList([block(cfg, self.num_heads, self.latent_dim, self.dropout, self.norm, self.activation, self.conv_mlp) for _ in range(self.num_layers)])


    def forward(self, x, sigma, condition, condition_len, text_timing=None):
        # return self.denoiser(x, sigma, condition)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        # input
        x_in = x
        c_in_b = c_in.view(c_in.shape[0], *([1] * (x.dim() - 1)))
        x = x * c_in_b
        # Positional encoding
        x = self.pos_enc(x)
        # Condition encoding
        sigma_enc = self.sigma_enc(c_noise)


        if text_timing is None:
            text_timing = torch.zeros_like(condition_len)
        # text_timing = self.time_enc(text_timing.to(torch.float32))

        # Project text condition if needed
        cond_proj = None
        if condition is not None:
            if self.text_proj is not None:
                cond_proj = self.text_proj(condition)
            else:
                cond_proj = condition

        # Blocks
        for block in self.blocks:
            x = block(x, sigma_enc, cond_proj, condition_len, text_timing)

        c_skip_b = c_skip.view(c_skip.shape[0], *([1] * (x.dim() - 1)))
        c_out_b = c_out.view(c_out.shape[0], *([1] * (x.dim() - 1)))
        x_out = c_skip_b * x_in + c_out_b * x
        return x_out