from models.dart.model.mld_vae import AutoMldVae
import torch
from torch import nn


# wrapper for mld_vae
class MotionPrimitive(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vae = AutoMldVae(cfg.dim_pose,
            cfg.latent_dim,
            cfg.hidden_size,
            cfg.width,
            cfg.depth,
            cfg.n_heads,
        )

        self.history = 2
        self.future = 8

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_long(self, x):
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
                history_motion = x[:, i*self.future: self.history + i*self.future]
                future_motion = x[:, self.history + i * self.future: self.history + (i+1) * self.future]
                mu, logvar = self.vae.encode(future_motion, history_motion)
                latent = self.reparameterize(mu, logvar)
                primitives.append(latent)

        return torch.stack(primitives, dim=1)


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
        
        # Encode
        mu, logvar = self.vae.encode(future_motion, history_motion)
        x_encoder = self.reparameterize(mu, logvar)
        
        # decoder
        x_out = self.vae.decode(x_encoder, history_motion, self.future, scale_latent=True)
        return x_out, mu, logvar


    def forward_decoder(self, z, history_motion):
        """
            Forward pass for decoding
            Args:
                z: (B, T_latent, D_latent) - input latent vectors
                history_motion: (B, H, D) - input history motion sequence
            Returns:
                x_out: (B, T = T_latent*F + H, D) - output motion sequence
        """
        x_out = self.vae.decode(z, history_motion, self.future)
        return x_out


    def forward_long(self, x):
        primitives = self.encode_long(x)
        x_out = self.decode_long(primitives, x[:, :self.history])
        # pad zeros at the end if decoded sequence is shorter than input along time dimension
        if x_out.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - x_out.shape[1]
            pad = torch.zeros(x_out.shape[0], pad_len, x_out.shape[2], device=x_out.device, dtype=x_out.dtype)
            x_out = torch.cat([x_out, pad], dim=1)

        return x_out


    def decode_long(self, z, initial_history_motion):
        num_primitive = z.shape[1]
        x_out = [initial_history_motion]
        with torch.no_grad():
            for i in range(num_primitive):
                latent = z[:, i]
                history_motion = x_out[-1][:, -self.history:]
                future_motion = self.vae.decode(latent, history_motion, self.future)
                x_out.append(future_motion)

        return torch.cat(x_out, dim=1)