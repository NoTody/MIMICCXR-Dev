from ..losses.clip_loss import cross_entropy
from ..losses.convirt_loss import ConVIRT_Loss
from ..losses.mocov2_loss import mocov2_loss
from ..losses.simclr_loss import NT_Xent 
from ..losses.vicreg_loss import vicreg_loss


__all__ = [
    "clip_loss",
    "convirt_loss",
    "mocov2_loss",
    "simclr_loss",
    "vicreg_loss",
]
