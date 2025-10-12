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
    def __init__(self, cfg, num_layers, num_heads, latent_dim, dropout, norm, activation, conv_mlp):
        super().__init__()                

        # causal self attention
        # adaln
    
    def forward(self, x, sigma_enc, condition, text_timing=None):
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

        denoiser_block = getattr(cfg, "denoiser_block", 1)
        block_name = f"Block{str(denoiser_block).zfill(2)}"
        block = eval(block_name)

        self.blocks = nn.ModuleList([block(cfg, self.num_layers, self.num_heads, self.latent_dim, self.dropout, self.norm, self.activation, self.conv_mlp) for _ in range(self.num_layers)])


    def forward(self, x, sigma, condition, text_timing=None):
        # return self.denoiser(x, sigma, condition)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        breakpoint()
        # input
        x_in = x
        x = x * c_in
        # Positional encoding
        x = self.pos_enc(x)
        # Condition encoding
        sigma_enc = self.sigma_enc(c_noise)

        # Blocks
        for block in self.blocks:
            x = block(x, sigma_enc, condition, text_timing)

        x_out = c_skip * x_in + c_out * x

        return x_out