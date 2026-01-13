import torch
import torch.nn as nn
import numpy as np
from .base_embedding import BaseEmbedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# TODO: Finish this and put in transformer. text, condition 공통.
class MotionMaskEmbedding(BaseEmbedding):
    def __init__(self,
                 num_embed=512,
                 content_seq_len=49,
                 embed_dim=512, 
                 trainable=True,
                 pos_emb_type='embedding'

        ):
        super().__init__()

        self.content_seq_len = content_seq_len
        self.num_embed = num_embed + 1
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.pos_emb_type = pos_emb_type

        # assert self.pos_emb_type in ['embedding', 'parameter']
        
        self.emb = nn.Embedding(self.num_embed, embed_dim)
        # if self.pos_emb_type == 'embedding':
        #     self.height_emb = nn.Embedding(self.spatial_size[0], embed_dim) # height   
        #     self.width_emb = nn.Embedding(self.spatial_size[1], embed_dim) # width
        # else:
        #     self.height_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim)) # height #32,1024
        #     self.width_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim)) # width   #32,1024
        self.pos_emb = PositionalEncoding(embed_dim, 0.0)
        self._set_trainable()

    def forward(self, index, **kwargs):
        assert index.dim() == 2 # B x L
        try:
            index[index < 0] = 0  
            emb = self.emb(index)
        except:
            raise RuntimeError('IndexError: index out of range in self, max index {}, num embed {}'.format(index.max(), self.num_embed))
        
        emb = self.pos_emb(emb)
        # add col and row embedding
        # if emb.shape[1] > 0:
        # # if False:
        #     if self.pos_emb_type == 'embedding':
        #         height_emb = self.height_emb(torch.arange(self.spatial_size[0], device=index.device).view(1, self.spatial_size[0])).unsqueeze(2) # 1 x H x D -> 1 x H x 1 x D
        #         width_emb = self.width_emb(torch.arange(self.spatial_size[1], device=index.device).view(1, self.spatial_size[1])).unsqueeze(1) # 1 x W x D -> 1 x 1 x W x D
        #     else:
        #         height_emb = self.height_emb.unsqueeze(2) # 1 x H x D -> 1 x H x 1 x D
        #         width_emb = self.width_emb.unsqueeze(1) # 1 x W x D -> 1 x 1 x W x D
        #     pos_emb = (height_emb + width_emb).view(1, self.spatial_size[0] * self.spatial_size[1], -1) # 1 x H x W x D -> 1 x L xD
        #     emb = emb + pos_emb[:, :emb.shape[1], :]

        return emb
