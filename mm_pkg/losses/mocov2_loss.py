import torch
import torch.distributed as dist
import torch.nn.functional as F
from ..model_utils.misc_utils import *


def mocov2_loss(query, key, queue, temperature):
    # positive logits: Nx1
    pos_logits = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1) 
    # negative logits: NxK
    neg_logits = torch.einsum("nc,ck->nk", [query, queue])
 
    # logits: Nx(1+K)
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    # apply temperature
    logits /= temperature

    # labels
    labels = torch.zeros(logits.shape[0], device=query.device, dtype=torch.long)
    # loss
    loss = F.cross_entropy(logits, labels)
    return loss

