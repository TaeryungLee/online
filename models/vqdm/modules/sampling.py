import torch
from einops import rearrange 
from torch.nn import functional as F



def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def sample_from_logits(logits, temperature=1.0, top_k=None, sample=False):
    """ Samples from logits with top_k and temperature.
    Input is of shape [batch_size, time, nb_books, softmax_size]"""

    batch, emb, time = logits.shape
    # logits = logits[:, -1, ...] # Take last time step.
    # Get logits at the final step, put book dimension in batch size
    logits = rearrange(logits, 'batch emb time -> (batch time) emb')
    # scale by temperature
    logits = logits / temperature
    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample:
        iz = torch.multinomial(probs, num_samples=1)
    else:
        _, iz = torch.topk(probs, k=1, dim=-1)
    iz = rearrange(iz, '(batch time) one_dim  -> batch time one_dim', batch=batch, time=time, one_dim=1).squeeze(2)
    return iz
