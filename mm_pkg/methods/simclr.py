from ..methods.base import *
from ..methods.base import BASE
from ..losses.simclr_loss import NT_Xent


class SIMCLR(BASE):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model(self.hparams.img_backbone, self.hparams.text_backbone, 
                    self.hparams.dropout)


    def _build_model(self, img_backbone, text_backbone, dropout):
        self.img_backbone = self.img_backbones[img_backbone]

        # simclr objective        
        self.criterion = NT_Xent(self.hparams.batch_size, self.hparams.temperature)

        # simclr projector
        self.simclr_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.simclr_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.simclr_proj_hidden_dim, self.hparams.simclr_proj_output_dim),
        )


    def shared_forward(self, batch, batch_idx, mode="train"):
        images1, images2 = batch
        #print(f"images: {images1}\nshape: {len(images1)}")
        # only use first image for clip
        images1, images2 = torch.stack((images1)), torch.stack((images2))

        # simclr 
        feat1, feat2 = self.img_backbone(images1), self.img_backbone(images2)
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
        parser = parent_parser.add_argument_group("slip_simclr")

        parser.add_argument("--img_embedding_dim", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--temperature", type=float, default=0.1)

        # vicreg projector
        parser.add_argument("--simclr_proj_output_dim", type=int, default=2048)
        parser.add_argument("--simclr_proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--ssl_scale", type=float, default=1.0)

        return parent_parser

