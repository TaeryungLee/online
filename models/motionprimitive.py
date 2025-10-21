from models.DART.model.mld_vae import AutoMldVae
import torch
from torch import nn

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import logger,rotate_half,Cache,BaseModelOutputWithPast,DynamicCache,repeat_kv,_flash_attention_forward



# wrapper for mld_vae
class MotionPrimitive(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vae = AutoMldVae(nfeats, latent_dim, h_dim, ff_size, num_layers, num_heads, dropout, arch, normalize_before, activation, position_embedding)

        self.history = 2
        self.future = 8

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        Encode the input motion sequence into series of latent vectors (Inference 용도)
        Args:
            x: (B, T, D) - input motion sequence
        Returns:
            x_encoder: (B, T_comp, D_latent) - series of latent vectors with shape (T, D_latent)
        """

        num_primitive = (x.shape[1] - self.history) // (self.future)
        primitives = []
        with torch.no_grad():
            for i in range(num_primitive):
                future_motion = x[:, i * self.future: i * self.future + self.future]
                history_motion = x[:, self.future + i*self.history: self.future + (i+1)*self.history]
                latent, dist = self.vae.encode(future_motion, history_motion)
                primitives.append(latent.detach().cpu())

        return torch.stack(primitives, dim=1)


    def forward(self, x):
        """
        Forward pass for training
        Args:
            x: (B, T, D) - input motion sequence
        Returns:
            x_out: (B, T, D) - output motion sequence
            mu: (B, T, D_latent) - mean of the latent distribution
            logvar: (B, T, D_latent) - log variance of the latent distribution
        """

        assert x.shape[1] == self.history + self.future
        
        # Encode
        _, dist = self.vae.encode(x[:, :self.history], x[:, self.history:])
        mu, logvar = dist.mean, dist.stddev
        x_encoder = self.reparameterize(mu, logvar)
        
        # decoder
        x_out = self.vae.decode(x_encoder, x[:, :self.history], self.future)
        return x_out, mu, logvar


    def forward_decoder(self, x):
        x_out = self.decoder(x)
        return x_out



    def encode(
            self,
            future_motion, history_motion,
            scale_latent: bool = False,
    ) -> Union[Tensor, Distribution]:
        return
    def decode(self, z: Tensor, history_motion, nfuture,
            scale_latent: bool = False,
            ):
        return