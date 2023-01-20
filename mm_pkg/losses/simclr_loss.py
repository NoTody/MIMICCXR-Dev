import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_utils.misc_utils import gather, get_rank


class SIMCLR_Loss(nn.Module):

    def __init__(self, batch_size, temperature=0.1, world_size=1):
        super().__init__()
        self.tau = temperature
        self.all_batch_size = batch_size * world_size
        self.mask = (~torch.eye(self.all_batch_size * 2, self.all_batch_size * 2, dtype=bool)).float()
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        k_i, k_j = gather(z_i), gather(z_j)
        # nomralize
        k_i = F.normalize(k_i, dim=-1, p=2)
        k_j = F.normalize(k_j, dim=-1, p=2)

        # calculate similarity matrix for two views
        k_cat = torch.cat([k_i, k_j], dim=0)
        similarity_matrix = self.similarity_f(k_cat.unsqueeze(1), k_cat.unsqueeze(0))

        # get positive pairs
        sim_ij = torch.diag(similarity_matrix, self.all_batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.all_batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.tau)
        denominator = self.mask.to(similarity_matrix.device) * torch.exp(similarity_matrix / self.tau)
        
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))

        loss = torch.sum(all_losses) / (2 * self.all_batch_size)

        return loss

