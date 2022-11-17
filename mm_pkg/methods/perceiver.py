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


class CLIP(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # get args
        self.args = args
        self.hparams.update(vars(args))

        # get backbone
        self.img_backbones = {
            "resnet3d_18": resnet_model(size=18, features_dim=self.hparams.features_dim),
            "resnet2d_50": resnet_model(size=50, features_dim=self.hparams.features_dim),
            "resnet3d_101": resnet_model(size=101, features_dim=self.hparams.features_dim),
            "resnet3d_tse": resnet_model(size="TSE", features_dim=self.hparams.features_dim),
        }

        self.text_backbones = {
            "bert_base": bert_model(self.hparams.text_backbone, self.hparams.max_length, self.hparams.pool),
        }

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
        ]


    def setup(self, stage=None):
        print("Trainset Loading ...")
        self.ds_train = OaiDataSet_Unsupervised(args=self.args, data_df_path=self.hparams.train_df_path, train=True)

        print("Valset Loading ...")
        #self.ds_val = self.ds_train
        self.ds_val = OaiDataSet_Unsupervised(args=self.args, data_df_path=self.hparams.val_df_path, train=False)

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
            raise NotImplementedError(f"This {self.args.optimizer} optimizer is not implemented yet,\
                                    try one of adamw or lamb")

        self.warmup_steps = (int)(self.hparams.per_warmup_steps * self.total_steps)

        #if self.hparams.method != "vicreg" and self.hparams.method != "barlowtwins": 
        lr_schedule_backbone = cosine_scheduler(
            self.hparams.lr_backbone * self.trainer.num_devices / self.hparams.linear_scale_factor,  # linear scaling rule
            self.hparams.min_lr_backbone,
            self.total_steps,
            self.warmup_steps,
        )
       
        lr_schedule_projector = cosine_scheduler(
            self.hparams.lr_projector * self.trainer.num_devices / self.hparams.linear_scale_factor,  # linear scaling rule
            self.hparams.min_lr_projector,
            self.total_steps,
            self.warmup_steps,
        )
       
        self.lr_schedule = {"backbone": lr_schedule_backbone, "projector": lr_schedule_projector}
       
        self.wd_schedule = cosine_scheduler(
            self.hparams.weight_decay,
            self.hparams.weight_decay_end,
            self.total_steps,
        )
        
        # momentum scheduling, set warmup step to 0
        self.momentum_schedule = cosine_scheduler(
            self.hparams.start_momentum,
            self.hparams.end_momentum,
            self.total_steps,
            0,
        )

        #warmup_steps = 100
        #scheduler = WarmupCosineSchedule(optimizer, warmup_steps, self.hparams.max_epochs)

        return [optimizer]
#, [scheduler]


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

