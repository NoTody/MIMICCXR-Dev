from ..methods.base import *
from ..methods.base import BASE
from ..losses.vicreg_loss import vicreg_loss


class VICREG(BASE):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model(self.hparams.img_backbone, self.hparams.text_backbone, 
                    self.hparams.dropout)


    def _build_model(self, img_backbone, text_backbone, dropout):
        self.img_backbone = self.img_backbones[img_backbone]

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


    def shared_forward(self, batch, batch_idx, mode="train"):
        images1, images2 = batch
        #print(f"images: {images1}\nshape: {len(images1)}")
        # only use first image for clip
        images1, images2 = torch.stack((images1)), torch.stack((images2))

        # vicreg
        feat1, feat2 = self.img_backbone(images1), self.img_backbone(images2)
        z1, z2 = self.vicreg_projector(feat1), self.vicreg_projector(feat2)
        ssl_loss = vicreg_loss(z1, z2, invariance_lamb=self.hparams.invariance_lamb, 
                variance_mu=self.hparams.variance_mu, covairance_v=self.hparams.covariance_v)

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
            {"type": "projector", "params": self.vicreg_projector.parameters()},
        ]


    # collate_fn for tokenizing input
    def collate_fn_batch_encoding(self, batch):
        images1, images2, texts = zip(*batch)
        return images1, images2


    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_mem,
                          shuffle=True, drop_last=True, collate_fn=self.collate_fn_batch_encoding)


    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_mem,
                          shuffle=True, drop_last=True, collate_fn=self.collate_fn_batch_encoding)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("vicreg")

        parser.add_argument("--img_embedding_dim", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--temperature", type=float, default=1.0)

        # vicreg projector
        parser.add_argument("--invariance_lamb", type=float, default=25.)
        parser.add_argument("--variance_mu", type=float, default=25.)
        parser.add_argument("--covariance_v", type=float, default=1.)
        parser.add_argument("--vicreg_proj_output_dim", type=int, default=8192)
        parser.add_argument("--vicreg_proj_hidden_dim", type=int, default=8192)
        parser.add_argument("--ssl_scale", type=float, default=1.0)

        return parent_parser


