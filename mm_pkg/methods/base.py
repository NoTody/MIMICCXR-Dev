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


class BASE(pl.LightningModule):

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
            "densenet2d_121": densenet_model(size=121, features_dim=self.hparams.features_dim, pretrained=self.hparams.pretrained),
            "vit2d_b16": vit_model("base", self.hparams.pretrained, self.hparams.freeze_pos_embed)
        }


    # dummy forward. to be overrided
    def shared_forward(self, batch, batch_idx, mode):
        dummy_loss = 0
        return {"loss": dummy_loss}


    def training_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx, "train")
        loss = shared_out["loss"]
        self.log("train_loss", loss, on_epoch=False, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx, "val")
        loss = shared_out["loss"]
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)


    def on_after_backward(self):
        # clip gradients
        if self.hparams.clip_grad:
            clip_gradients(self.backbone, self.hparams.clip_grad)


    @property
    def learnable_params(self):
        return dict()


    def setup(self, stage=None):
        mimic_cxr_path = Path('/gpfs/data/denizlab/Datasets/Public/physionet.org/files/mimic-cxr/2.0.0')
        # load all resized image mapping
        with open(mimic_cxr_path / 'mimic_cxr_imgs_v3.pkl', 'rb') as handle:
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
        elif self.hparams.optimizer == "lars":
            optimizer = LARS(learnable_params, lr=self.hparams.lr_backbone, weight_decay=self.hparams.weight_decay, 
                    weight_decay_filter=exclude_bias_and_norm, lars_adaptation_filter=exclude_bias_and_norm)
        else:
            raise NotImplementedError(f"This {self.args.optimizer} optimizer is not implemented yet, \
                                    try one of adamw or lamb")
        # warmup and scheduler setup
        #self.warmup_steps = self.hparams.per_warmup_steps * self.total_steps
        scheduler = WarmupCosineSchedule(optimizer, 0, self.hparams.max_epochs)

        return [optimizer], [scheduler]


    def collate_fn_batch_encoding(self, batch):
        images, texts = zip(*batch)

        text_encodings = self.tokenizer.batch_encode_plus(
                list(texts),
                max_length=self.hparams.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")

        return images, text_encodings


    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_mem,
                          shuffle=True, drop_last=True, collate_fn=self.collate_fn_batch_encoding)


    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_mem,
                          shuffle=True, drop_last=True, collate_fn=self.collate_fn_batch_encoding)


