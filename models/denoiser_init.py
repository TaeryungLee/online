import torch
import torch.nn as nn




class DenoiserInit(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x, sigma, condition):
        # return self.denoiser(x, sigma, condition)
        return x