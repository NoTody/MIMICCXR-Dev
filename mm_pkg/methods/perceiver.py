import pytorch_lightning as pl
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from module_utils import ProjectionHeadCLIP
import torch_optimizer as optim
from ..model_utils.misc_utils import *
from ..model_utils.misc_utils import WarmupCosineSchedule
from ..model_utils.module_utils import *
from ..data_utils.dataloader_utils import OaiDataSet_Unsupervised


class CLIP(BASE):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model(self.hparams.img_backbone, self.hparams.text_backbone, 
            self.hparams.embedding_dim, self.hparams.projection_dim, self.hparams.dropout)


    def _build_model(self, img_backbone, text_backbone, embedding_dim, projection_dim, dropout):
        self.img_backbone = self.img_backbones[img_backbone]
        self.text_backbone = self.text_backbones[text_backbone]

        self.img_projector = ProjectionHeadCLIP(self.hparams.img_embedding_dim, self.hparams.projection_dim, 
            self.hparams.dropout)
        self.text_projector = ProjectionHeadCLIP(self.hparams.text_embedding_dim, self.hparams.projection_dim, 
            self.hparams.dropout)


    def shared_forward(self, batch, batch_idx, mode="train"):
        images, texts = batch

        # get embeddings
        image_features, text_features = self.img_backbone(images), self.text_backbone(texts)
        image_embeddings, text_embeddings = self.img_projector(images), self.text_projector(texts)

        # compute loss
        logits = (text_embeddings @ image_embeddings.T) / self.hparams.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return {"loss": loss.mean()}


    def on_train_batch_start(self, batch, batch_idx):
        current_step = self.global_step - (len(self.ds_val) // (self.hparams.batch_size * max(1, self.trainer.num_devices))) \
                    * self.current_epoch
        #if self.hparams.method == "vicreg" or self.hparams.method == "barlowtwins":
        #    adjust_learning_rate_v2(self.optimizers().optimizer, self.hparams.lr_backbone, self.warmup_steps, \
        #                         self.total_steps, current_step)
        #else:
        adjust_learning_rate_v1(self.optimizers().optimizer, self.hparams.lr_predictor, current_step, \
                                self.lr_schedule, self.wd_schedule)


    def training_step(self, batch, batch_idx):
        shared_out = self.shared_step(batch, batch_idx, "train")
        return shared_out


    def validation_step(self, batch, batch_idx):
        shared_out = self.shared_step(batch, batch_idx, "val")
        #loss, outs = shared_out["loss"], shared_out["outs"]
        return shared_out
        #if self.hparams.multicrop:
        #    return {"val_loss": loss, "outs": outs, "multicrop_outs": shared_out["multicrop_outs"]}
        #else:
        #    return {"val_loss": loss, "outs": outs}
    

    @property
    def learnable_params(self):
        return [
            {"type": "backbone", "params": self.img_backbone.parameters()},
            {"type": "backbone", "params": self.text_backbone.parameters()},
            {"type": "projector", "params": self.img_projector.parameters()},
            {"type": "projector", "params": self.text_projector.parameters()},
        ]


    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_mem,
                          shuffle=True, drop_last=True)


    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_mem,
                          shuffle=True, drop_last=True)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(CLIP, CLIP).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("clip")

        # projector
        parser.add_argument("--img_embedding_dim", type=int, default=2048)
        parser.add_argument("--text_embedding_dim", type=int, default=768)
        parser.add_argument("--projection_dim", type=int, default=256)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--temperature", type=float, default=1.0)

        return parent_parser

