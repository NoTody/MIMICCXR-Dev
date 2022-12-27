from ..methods.base import *
from ..methods.base import BASE_SLIP
from ..losses.simclr_loss import NT_Xent


class SLIP_SIMCLR(BASE_SLIP):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model()


    def _build_model(self):
        super()._build_model()
        
        # simclr objective
        self.criterion = NT_Xent(self.hparams.batch_size, self.hparams.temperature_ssl, \
                        self.hparams.gpus * self.hparams.num_nodes)

        # simclr projector
        self.simclr_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.simclr_proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.hparams.simclr_proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.simclr_proj_hidden_dim, self.hparams.simclr_proj_output_dim, bias=True),
        )


    def shared_forward(self, batch, batch_idx):
        images_ssl1, images_ssl2, mm_loss = super().shared_forward(batch, batch_idx)

        # simclr 
        feat1, feat2 = self.img_backbone(images_ssl1), self.img_backbone(images_ssl2)
        z1, z2 = self.simclr_projector(feat1), self.simclr_projector(feat2)
        ssl_loss = self.criterion(z1, z2)

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
            {"type": "projector", "params": self.simclr_projector.parameters()},
        ]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("slip_simclr")

        # simclr projector
        parser.add_argument("--simclr_proj_hidden_dim", type=int, default=512)
        parser.add_argument("--simclr_proj_output_dim", type=int, default=128)

        return parent_parser


