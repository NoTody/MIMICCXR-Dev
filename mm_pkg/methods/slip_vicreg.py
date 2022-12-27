import argparse
from ..methods.base import *
from ..methods.base import BASE_SLIP
from ..losses.clip_loss import clip_loss
from ..losses.vicreg_loss import vicreg_loss


class SLIP_VICREG(BASE_SLIP):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model()


    def _build_model(self):
        super()._build_model()

        # vicreg projector
        self.vicreg_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.vicreg_proj_hidden_dim),
            nn.BatchNorm1d(self.hparams.vicreg_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.vicreg_proj_hidden_dim, self.hparams.vicreg_proj_hidden_dim),
            nn.BatchNorm1d(self.hparams.vicreg_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.vicreg_proj_hidden_dim, self.hparams.vicreg_proj_output_dim),
        )


    def shared_forward(self, batch, batch_idx):
        images_ssl1, images_ssl2, mm_loss = super().shared_forward(batch, batch_idx)

        # vicreg
        feat1, feat2 = self.img_backbone(images_ssl1), self.img_backbone(images_ssl2)
        z1, z2 = self.vicreg_projector(feat1), self.vicreg_projector(feat2)
        ssl_loss = vicreg_loss(z1, z2, invariance_lamb=self.hparams.invariance_lamb, 
                variance_mu=self.hparams.variance_mu, covairance_v=self.hparams.covariance_v)

        # slip final loss
        loss = mm_loss + self.hparams.ssl_scale * ssl_loss
        return {"loss": loss, "mm_loss": mm_loss, "ssl_loss": ssl_loss}


    @property
    def learnable_params(self):
        return [
            {"type": "backbone", "params": self.img_backbone.parameters()},
            {"type": "backbone", "params": self.text_backbone.parameters()},
            {"type": "projector", "params": self.img_projector.parameters()},
            {"type": "projector", "params": self.text_projector.parameters()},
            {"type": "projector", "params": self.vicreg_projector.parameters()},
        ]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("slip_vicreg")

        # vicreg hyper-parameters
        parser.add_argument("--invariance_lamb", type=float, default=25)
        parser.add_argument("--variance_mu", type=float, default=25)
        parser.add_argument("--covariance_v", type=float, default=1.)
        parser.add_argument("--vicreg_proj_output_dim", type=int, default=8192)
        parser.add_argument("--vicreg_proj_hidden_dim", type=int, default=8192)

        return parent_parser


