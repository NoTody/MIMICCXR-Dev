from ..methods.base import *
from ..methods.base import BASE
from ..losses.clip_loss import clip_loss


class CLIP(BASE):

    def __init__(self, args):
        super().__init__(args)

        # Build models
        self._build_model()


    def _build_model(self):
        super()._build_model()
        self.text_backbone = bert_model(self.hparams.text_backbone, self.hparams.pool)
        self.img_projector = ProjectionHeadCLIP(self.hparams.img_embedding_dim, \
                        self.hparams.projection_dim, self.hparams.dropout)
        self.text_projector = ProjectionHeadCLIP(self.hparams.text_embedding_dim, \
                        self.hparams.projection_dim, self.hparams.dropout)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.text_backbone, use_fast=True)


    def shared_forward(self, batch, batch_idx):
        images_clip, text_encodings = batch
        images_clip = torch.stack((images_clip))

        # get embeddings
        image_features, text_features = self.img_backbone(images_clip), self.text_backbone(text_encodings)
        image_embeddings, text_embeddings = self.img_projector(image_features), self.text_projector(text_features)

        # compute loss
        loss = clip_loss(image_embeddings, text_embeddings, self.hparams.temperature_mm).mean()
        return {"loss": loss}


    @property
    def learnable_params(self):
        return [
            {"type": "backbone", "params": self.img_backbone.parameters()},
            {"type": "backbone", "params": self.text_backbone.parameters()},
            {"type": "projector", "params": self.img_projector.parameters()},
            {"type": "projector", "params": self.text_projector.parameters()},
        ]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("clip")

        return parent_parser


