import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from ..model_utils.misc_utils import gather


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def clip_loss(image_embeddings, text_embeddings, temperature):
    image_embeddings, text_embeddings = gather(image_embeddings), gather(text_embeddings)
    # normalize
    image_embeddings, text_embeddings = F.normalize(image_embeddings, dim=-1), F.normalize(text_embeddings, dim=-1)
    # compute loss
    logits = (text_embeddings @ image_embeddings.T) / temperature
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
        (images_similarity + texts_similarity) / 2.0 * temperature, dim=-1
    )
    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (images_loss + texts_loss) / 2.0    # shape: (batch_size)
    return loss

