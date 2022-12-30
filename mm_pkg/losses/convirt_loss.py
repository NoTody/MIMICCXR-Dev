import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from ..model_utils.misc_utils import gather, all_gather


class ConVIRT_Loss(nn.Module):
    def __init__(self, batch_size, alpha=0.75, temperature=0.1, world_size=1):
        super(ConVIRT_Loss, self).__init__()
        self.batch_size = batch_size
        self.alpha = alpha
        self.temperature = temperature
        self.world_size = world_size
        self.similarity_f = nn.CosineSimilarity(dim=2)
        
    def NT_Xent(self, z_i, z_j):
        N = self.batch_size * self.world_size

        similarity_matrix = self.similarity_f(z_i.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature

        nominator = torch.exp(torch.diag(similarity_matrix))
        denominator = torch.sum(torch.exp(similarity_matrix), dim=1)

        loss_partial = -torch.log(nominator / denominator)
        loss = torch.sum(loss_partial) / N

        return loss

    def forward(self, z_img, z_text):
        z_img, z_text = all_gather(z_img), all_gather(z_text) 
        loss_a, loss_b = self.NT_Xent(z_img, z_text), self.NT_Xent(z_text, z_img)
        return self.alpha * loss_a + (1 - self.alpha) * loss_b

