from ..methods.base import *
from ..methods.base import BASE_SSL
from ..losses.simclr_loss import NT_Xent, SIMCLR_Loss


class SIMCLR(BASE_SSL):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model()


    def _build_model(self):
        super()._build_model()

        # simclr objective        
        self.criterion = SIMCLR_Loss(self.hparams.temperature_ssl, self.hparams.gpus * self.hparams.num_nodes)

        # simclr projector
        self.simclr_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.simclr_proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.hparams.simclr_proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.simclr_proj_hidden_dim, self.hparams.simclr_proj_output_dim, bias=True),
        )


    def shared_forward(self, batch, batch_idx):
        images_ssl1, images_ssl2 = batch
        # only use first image for clip
        images_ssl1, images_ssl2 = torch.stack((images_ssl1)), torch.stack((images_ssl2))

        # simclr 
        feat1, feat2 = self.img_backbone(images_ssl1), self.img_backbone(images_ssl2)
        z1, z2 = self.simclr_projector(feat1), self.simclr_projector(feat2)
        ssl_loss = self.criterion(z1, z2)

        return {"loss": ssl_loss}


    @property
    def learnable_params(self):
        return [
            {"type": "backbone", "params": self.img_backbone.parameters()},
            {"type": "projector", "params": self.simclr_projector.parameters()},
        ]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("simclr")

        # simclr projector
        parser.add_argument("--simclr_proj_hidden_dim", type=int, default=512)
        parser.add_argument("--simclr_proj_output_dim", type=int, default=128)

        return parent_parser


