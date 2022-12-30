import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from ..model_utils.misc_utils import gather, all_gather, get_rank


# modified from https://github.com/facebookresearch/SLIP/blob/main/losses.py
class CLIP_Loss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, image_embed, text_embed):
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            all_gather(image_embed), all_gather(text_embed)

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        logit_scale = logit_scale.exp()
        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t() 
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        return loss

