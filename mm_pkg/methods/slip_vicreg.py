from ..methods.base import *
from ..methods.base import BASE
from ..losses.clip_loss import clip_loss
from ..losses.vicreg_loss import vicreg_loss


class SLIP_VICREG(BASE):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model(self.hparams.img_backbone, self.hparams.text_backbone, 
                    self.hparams.projection_dim, self.hparams.dropout)


    def _build_model(self, img_backbone, text_backbone, projection_dim, dropout):
        self.img_backbone = self.img_backbones[img_backbone]
        self.text_backbone = bert_model(self.hparams.text_backbone, self.hparams.pool)

        # clip projector
        self.img_projector = ProjectionHeadCLIP(self.hparams.img_embedding_dim,
                        self.hparams.projection_dim, self.hparams.dropout)
        self.text_projector = ProjectionHeadCLIP(self.hparams.text_embedding_dim,
                        self.hparams.projection_dim, self.hparams.dropout)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.text_backbone, use_fast=True)

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
        images1, images2, text_encodings = batch
        #print(f"images: {images1}\nshape: {len(images1)}")
        # only use first image for clip
        images1, images2 = torch.stack((images1)), torch.stack((images2))

        # clip
        # get embeddings
        image_features, text_features = self.img_backbone(images1), self.text_backbone(text_encodings)
        image_embeddings, text_embeddings = self.img_projector(image_features), self.text_projector(text_features)
        image_embeddings, text_embeddings = all_gather(image_embeddings), all_gather(text_embeddings)
        # compute loss
        c_loss = clip_loss(image_embeddings, text_embeddings, self.hparams.temperature).mean()

        # vicreg
        feat1, feat2 = self.img_backbone(images1), self.img_backbone(images2)
        z1, z2 = self.vicreg_projector(feat1), self.vicreg_projector(feat2)
        ssl_loss = vicreg_loss(z1, z2, invariance_lamb=self.hparams.invariance_lamb, 
                variance_mu=self.hparams.variance_mu, covairance_v=self.hparams.covariance_v)

        # slip final loss
        loss = c_loss + self.hparams.ssl_scale * ssl_loss
        return {"loss": loss, "clip_loss": c_loss, "ssl_loss": ssl_loss}


    def training_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx, "train")
        loss, clip_loss, ssl_loss = shared_out["loss"], shared_out["clip_loss"], shared_out["ssl_loss"]
        self.log("train_loss", loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_clip_loss", clip_loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_ssl_loss", ssl_loss, on_epoch=False, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx, "val")
        loss, clip_loss, ssl_loss = shared_out["loss"], shared_out["clip_loss"], shared_out["ssl_loss"]
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_clip_loss", clip_loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("val_ssl_loss", ssl_loss, on_epoch=False, on_step=True, prog_bar=True)


    @property
    def learnable_params(self):
        return [
            {"type": "backbone", "params": self.img_backbone.parameters()},
            {"type": "backbone", "params": self.text_backbone.parameters()},
            {"type": "projector", "params": self.img_projector.parameters()},
            {"type": "projector", "params": self.text_projector.parameters()},
            {"type": "projector", "params": self.vicreg_projector.parameters()},
        ]


    # collate_fn for tokenizing input
    def collate_fn_batch_encoding(self, batch):
        images1, images2, texts = zip(*batch)
        text_encodings = self.tokenizer.batch_encode_plus(
                    list(texts),
                    max_length=self.hparams.max_length,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt")
        return images1, images2, text_encodings


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
        parser = parent_parser.add_argument_group("slip_vicreg")

        # clip projector
        parser.add_argument("--img_embedding_dim", type=int, default=2048)
        parser.add_argument("--text_embedding_dim", type=int, default=768)
        parser.add_argument("--projection_dim", type=int, default=512)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--temperature", type=float, default=0.1)

        # vicreg projector
        parser.add_argument("--invariance_lamb", type=float, default=25)
        parser.add_argument("--variance_mu", type=float, default=25)
        parser.add_argument("--covariance_v", type=float, default=1.)
        parser.add_argument("--vicreg_proj_output_dim", type=int, default=8192)
        parser.add_argument("--vicreg_proj_hidden_dim", type=int, default=8192)
        parser.add_argument("--ssl_scale", type=float, default=1.0)

        return parent_parser


