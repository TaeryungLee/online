import torch.nn as nn
from models.causal_cnn import CausalEncoder, CausalDecoder
from models.enc_dec import ConvCausalEncoder, ConvCausalDecoder, LocalCausalDecoder, MLPEncoder, DecoderConfig, ConvCausalEncoder_NoComp, LocalCausalEncoder
import torch


# Causal TAE:
class Causal_TAE(nn.Module):
    def __init__(self,
                 hidden_size=1024,
                 down_t=2,
                 stride_t=2,
                 width=1024,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range = [-30,20]
                 ):
        
        super().__init__()

        self.decode_proj = nn.Linear(latent_dim, width)  

        self.encoder = ConvCausalEncoder(272, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, latent_dim=latent_dim, clip_range=clip_range)
        self.decoder = ConvCausalDecoder(272, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
    


    def preprocess(self, x):
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        x = x.permute(0,2,1)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def encode(self, x):
        x_in = self.preprocess(x)
        mu, logvar = self.encoder(x_in)
        x_encoder = self.reparameterize(mu, logvar)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  
        return x_encoder, mu, logvar

    def forward(self, x):
        breakpoint()
        x_in = self.preprocess(x)       
        # Encode
        mu, logvar = self.encoder(x_in)  
        x_encoder = self.reparameterize(mu, logvar)
        x_encoder = self.decode_proj(x_encoder) 
        # decoder
        x_decoder = self.decoder(x_encoder)
        x_out = self.postprocess(x_decoder)  
        return x_out, mu, logvar


    def forward_decoder(self, x):         
        # decoder
        x_width = self.decode_proj(x)           
        x_decoder = self.decoder(x_width)
        x_out = self.postprocess(x_decoder)
        return x_out









class LatentSpaceVAE(nn.Module):
    def __init__(self,
        cfg,
        hidden_size=512,
        depth=12,
        n_heads=8,
        attn_window=16,
        activation='relu',
        norm=None,
        latent_dim=16,
        clip_range = []
    ):
        
        super().__init__()

        transformer_config = DecoderConfig(
            d_model=hidden_size, n_heads=n_heads, depth=depth, attn_window=attn_window, rope_fraction=1.0,
            dropout=0.1, mlp_expand=4, final_norm=True, bias=True
        )

        if cfg.encoder == 'mlp':
            self.encoder = MLPEncoder(272, latent_dim, hidden_size, depth, expand=4, act=activation, dropout=0.1, norm=norm, layerscale_init=1e-5, final_norm=True, clip_range=clip_range)
        elif cfg.encoder == 'conv1d':
            self.encoder = ConvCausalEncoder_NoComp(272, hidden_size, cfg.down_t, hidden_size, 3, cfg.dilation_growth_rate, activation=activation, norm=norm, latent_dim=latent_dim, clip_range=clip_range)
        elif cfg.encoder == 'transformer':
            self.encoder = LocalCausalEncoder(272, latent_dim, transformer_config, clip_range=clip_range)
        
        self.decoder = LocalCausalDecoder(latent_dim, 272, transformer_config)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    


    def encode(self, x):
        mu, logvar = self.encoder(x)
        x_encoder = self.reparameterize(mu, logvar)
        return x_encoder


    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        x_encoder = self.reparameterize(mu, logvar)
        # decoder
        x_out = self.decoder(x_encoder)
        return x_out, mu, logvar


    def forward_decoder(self, x):
        x_out = self.decoder(x)
        return x_out



