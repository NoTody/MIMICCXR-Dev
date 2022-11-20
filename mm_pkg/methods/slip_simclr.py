import pytorch_lightning as pl
import copy
import torch
import pickle
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from ..model_utils.misc_utils import *
from ..model_utils.misc_utils import WarmupCosineSchedule
from ..model_utils.module_utils import *
from ..model_utils.module_utils import ProjectionHeadCLIP
from ..data_utils.dataloader_utils import MIMIC_CXR_Unsupervised
from ..losses.clip_loss import clip_loss
from ..losses.simclr_loss import NT_Xent


class SLIP_SIMCLR(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # get args
        self.args = args
        self.hparams.update(vars(args))

        # get backbone
        self.img_backbones = {
            "resnet2d_18": resnet_model(size=18, features_dim=self.hparams.features_dim, pretrained=self.hparams.pretrained),
            "resnet2d_50": resnet_model(size=50, features_dim=self.hparams.features_dim, pretrained=self.hparams.pretrained),
            "resnet2d_101": resnet_model(size=101, features_dim=self.hparams.features_dim, pretrained=self.hparams.pretrained),
        }

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
        
        # simclr objective
        self.criterion = NT_Xent(self.hparams.batch_size, self.hparams.temperature)

        # simclr projector
        self.simclr_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.simclr_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.simclr_proj_hidden_dim, self.hparams.simclr_proj_output_dim),
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

        # simclr 
        feat1, feat2 = self.img_backbone(images1), self.img_backbone(images2)
        z1, z2 = self.simclr_projector(feat1), self.simclr_projector(feat2)
        ssl_loss = self.criterion(z1, z2)

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


    def on_after_backward(self):
        # clip gradients
        if self.hparams.clip_grad:
            clip_gradients(self.backbone, self.hparams.clip_grad)


    @property
    def learnable_params(self):
        return [
            {"type": "backbone", "params": self.img_backbone.parameters()},
            {"type": "backbone", "params": self.text_backbone.parameters()},
            {"type": "projector", "params": self.img_projector.parameters()},
            {"type": "projector", "params": self.text_projector.parameters()},
            {"type": "projector", "params": self.simclr_projector.parameters()},
        ]


    def setup(self, stage=None):
        mimic_cxr_path = Path('/gpfs/data/denizlab/Datasets/Public/physionet.org/files/mimic-cxr/2.0.0')
        # load all resized image mapping
        with open(mimic_cxr_path / 'mimic_cxr_imgs.pkl', 'rb') as handle:
            dict_image_mapping = dict(pickle.load(handle))
        print("Trainset Loading ...")
        self.ds_train = MIMIC_CXR_Unsupervised(args=self.args, dict_image_mapping=dict_image_mapping, 
                two_transform=self.hparams.two_transform, full_report=self.hparams.full_report, 
                data_df_path=self.hparams.train_df_path, train=True)

        print("Valset Loading ...")
        self.ds_val = MIMIC_CXR_Unsupervised(args=self.args, dict_image_mapping=dict_image_mapping, 
                two_transform=self.hparams.two_transform, full_report=self.hparams.full_report, 
                data_df_path=self.hparams.val_df_path, train=False)

        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.num_devices)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(self.ds_train.data_df) // tb_size) * ab_size
        print(f"total steps: {self.total_steps}")


    def configure_optimizers(self):
        learnable_params = self.learnable_params

        if self.hparams.optimizer == "adamw":
            optimizer = AdamW(learnable_params, lr=self.hparams.lr_backbone, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "lamb":
            optimizer = optim.Lamb(learnable_params, lr=self.hparams.lr_backbone, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(learnable_params, lr=self.hparams.lr_backbone, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        else:
            raise NotImplementedError(f"This {self.args.optimizer} optimizer is not implemented yet, \
                                    try one of adamw or lamb")
        # warmup and scheduler setup
        self.warmup_steps = self.hparams.per_warmup_steps * self.total_steps
        scheduler = WarmupCosineSchedule(optimizer, 0, self.hparams.max_epochs)

        return [optimizer], [scheduler]


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
        parser = parent_parser.add_argument_group("slip_simclr")

        # clip projector
        parser.add_argument("--img_embedding_dim", type=int, default=2048)
        parser.add_argument("--text_embedding_dim", type=int, default=768)
        parser.add_argument("--projection_dim", type=int, default=256)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--temperature", type=float, default=0.1)

        # vicreg projector
        parser.add_argument("--simclr_proj_output_dim", type=int, default=2048)
        parser.add_argument("--simclr_proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--ssl_scale", type=float, default=1.0)

        return parent_parser


