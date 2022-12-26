from ..methods.base import *
from ..methods.base import BASE_SSL
from ..losses.simclr_loss import NT_Xent


class SIMCLR(BASE_SSL):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model(self.hparams.img_backbone, self.hparams.text_backbone, 
                    self.hparams.dropout)


    def _build_model(self, img_backbone, text_backbone, dropout):
        self.img_backbone = self.img_backbones[img_backbone]

        # simclr objective        
        self.criterion = NT_Xent(self.hparams.batch_size, self.hparams.temperature, \
                                self.hparams.gpus * self.hparams.num_nodes)

        # simclr projector
        self.simclr_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.simclr_proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.hparams.simclr_proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.simclr_proj_hidden_dim, self.hparams.simclr_proj_output_dim, bias=True),
        )


    def shared_forward(self, batch, batch_idx, mode="train"):
        images_ssl1, images_ssl2 = batch
        # only use first image for clip
        images_ssl1, images_ssl2 = torch.stack((images_ssl1)), torch.stack((images_ssl2))

        # simclr 
        feat1, feat2 = self.img_backbone(images_ssl1), self.img_backbone(images_ssl2)
        z1, z2 = self.simclr_projector(feat1), self.simclr_projector(feat2)
        ssl_loss = self.criterion(z1, z2)

        return {"loss": ssl_loss}


    def training_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx, "train")
        loss = shared_out["loss"]
        self.log("train_loss", loss, on_epoch=False, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx, "val")
        loss = shared_out["loss"]
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)


    @property
    def learnable_params(self):
        return [
            {"type": "backbone", "params": self.img_backbone.parameters()},
            {"type": "projector", "params": self.simclr_projector.parameters()},
        ]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("simclr")

        parser.add_argument("--img_embedding_dim", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--temperature", type=float, default=0.1)

        # simclr projector
        parser.add_argument("--simclr_proj_hidden_dim", type=int, default=512)
        parser.add_argument("--simclr_proj_output_dim", type=int, default=128)

        return parent_parser


