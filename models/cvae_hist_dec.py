import pdb
from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from models.dart.mld.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from models.dart.mld.models.operator import PositionalEncoding
from models.dart.mld.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from models.dart.mld.models.operator.position_encoding import build_position_encoding
from models.dart.mld.utils.temos_utils import lengths_to_mask



class CVAE_hist_dec(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: tuple = [1, 256],
                 h_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 nfuture: int = 8,
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = 1
        self.latent_dim = latent_dim
        self.h_dim = h_dim
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = False
        self.pe_type = "mld"
        self.nfuture = nfuture

        self.query_pos_encoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding)


        # posterior network
        encoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        # prior network
        prior_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        prior_norm = nn.LayerNorm(self.h_dim)
        self.prior_encoder = SkipTransformerEncoder(prior_layer, num_layers,
                                              prior_norm)
        self.prior_encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)



        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")
        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)

        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.h_dim))
        self.global_motion_token_prior = nn.Parameter(
            torch.randn(self.latent_size * 2, self.h_dim))

        self.skel_embedding = nn.Linear(input_feats, self.h_dim)
        self.final_layer = nn.Linear(self.h_dim, output_feats)

        self.register_buffer('latent_mean', torch.tensor(0))
        self.register_buffer('latent_std', torch.tensor(1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def encode(
            self,
            future_motion, history_motion,
            scale_latent: bool = False,
    ) -> Union[Tensor, Distribution]:
        device = future_motion.device
        bs, nfuture, nfeats = future_motion.shape
        nhistory = history_motion.shape[1]

        x = torch.cat((history_motion, future_motion), dim=1)  # [bs, H+F, nfeats]
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, h_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq)[:dist.shape[0]]  # [2*latent_size, bs, h_dim]
        dist = self.encoder_latent_proj(dist)  # [2*latent_size, bs, latent_dim]

        # content distribution
        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]
        logvar = torch.clamp(logvar, min=-10, max=10)  # avoid numerical issues, otherwise denoiser rollout can break
        # if torch.isnan(mu).any() or torch.isinf(mu).any() or torch.isnan(logvar).any() or torch.isinf(logvar).any():
        #     pdb.set_trace()

        # # resampling
        # std = logvar.exp().pow(0.5)
        # dist = torch.distributions.Normal(mu, std)
        # latent = dist.rsample()
        # if scale_latent:  # only used during denoiser training
        #     latent = latent / self.latent_std
        # return latent, dist
        return mu.transpose(1, 0), logvar.transpose(1, 0)

    def prior(self, future_motion):
        device = future_motion.device
        bs, nfuture, nfeats = future_motion.shape

        x = future_motion
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, h_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token_prior[:, None, :], (1, bs, 1))

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        xseq = self.query_pos_encoder(xseq)
        dist = self.prior_encoder(xseq)[:dist.shape[0]]  # [2*latent_size, bs, h_dim]
        dist = self.prior_encoder_latent_proj(dist)  # [2*latent_size, bs, latent_dim]

        # content distribution
        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]
        logvar = torch.clamp(logvar, min=-10, max=10)  # avoid numerical issues, otherwise denoiser rollout can break

        return mu.transpose(1, 0), logvar.transpose(1, 0)

    def forward(self, future_motion, history_motion):
        mu, logvar = self.encode(future_motion, history_motion)
        mu_prior, logvar_prior = self.prior(future_motion)

        # KL( N(mu, sigma^2) || N(mu_prior, sigma_prior^2) ) for diagonal Gaussians
        # logvar/logvar_prior represent log(sigma^2)
        kl_prior = 0.5 * (
            (logvar_prior - logvar)
            + (logvar.exp() + (mu - mu_prior) ** 2) / logvar_prior.exp()
            - 1
        )
        # sum over latent slots and dims, mean over batch
        kl_prior = kl_prior.sum(dim=-1).mean()

        z = self.reparameterize(mu, logvar)
        x_out = self.decode(z, self.nfuture)

        return x_out, mu_prior, logvar_prior, kl_prior

    def decode(self, z, nfuture, scale_latent = False):
        bs = z.shape[0]
        if scale_latent:  # only used during denoiser training
            z = z * self.latent_std
        z = self.decoder_latent_proj(z).transpose(1, 0)  # [latent_size, bs, latent_dim] => [latent_size, bs, h_dim]
        queries = torch.zeros(nfuture, bs, self.h_dim, device=z.device)


        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                xseq)[-nfuture:]

        elif self.arch == "encoder_decoder":
            xseq = torch.cat((z, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                tgt=xseq,
                memory=z,
            )[-nfuture:]

        output = self.final_layer(output)
        # Pytorch Transformer: [Sequence, Batch size, ...]
        return output.permute(1, 0, 2)



# wrapper for mld_vae
class CVAEWrapper_hist_dec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cvae = CVAE_hist_dec(cfg.dim_pose,
            cfg.latent_dim,
            cfg.hidden_size,
            cfg.width,
            cfg.depth,
            cfg.n_heads,
            nfuture=cfg.future,
        )

        self.history = cfg.history
        self.future = cfg.future


    def forward(self, future_motion, history_motion):
        """
        Forward pass for training
        Args:
            x: (B, T, D) - input motion sequence
        Returns:
            x_out: (B, T, D) - output motion sequence
            mu: (B, T, D_latent) - mean of the latent distribution
            logvar: (B, T, D_latent) - log variance of the latent distribution
        """

        return self.cvae.forward(future_motion, history_motion)


    def forward_long(self, x):
        primitives = self.encode_long(x)
        x_out = self.decode_long(primitives)
        # pad zeros at the end if decoded sequence is shorter than input along time dimension
        if x_out.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - x_out.shape[1]
            pad = torch.zeros(x_out.shape[0], pad_len, x_out.shape[2], device=x_out.device, dtype=x_out.dtype)
            x_out = torch.cat([x_out, pad], dim=1)

        return x_out


    def encode_long(self, x):
        """
        Encode the input motion sequence into series of latent vectors (Inference 용도)
        Args:
            x: (B, T, D) - input motion sequence
        Returns:
            x_encoder: (B, T_comp, D_latent) - series of latent vectors with shape (T, D_latent)
        """

        num_primitive = (x.shape[1]) // (self.future)
        primitives = []
        with torch.no_grad():
            for i in range(num_primitive):
                future_motion = x[:, i * self.future: i * self.future + self.future]
                # history_motion = x[:, self.future + i*self.history: self.future + (i+1)*self.history]
                mu, logvar = self.cvae.prior(future_motion)
                latent = self.cvae.reparameterize(mu, logvar)
                primitives.append(latent)

        return torch.stack(primitives, dim=1)




    def decode_long(self, z):
        num_primitive = z.shape[1]
        x_out = []
        with torch.no_grad():
            for i in range(num_primitive):
                latent = z[:, i]
                future_motion = self.cvae.decode(latent, self.future)
                x_out.append(future_motion)

        return torch.cat(x_out, dim=1)