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
from torch.optim.lr_scheduler import MultiStepLR
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from ..backbones.SEED import SEED
from ..model_utils.misc_utils import *
from ..model_utils.misc_utils import WarmupCosineSchedule
from ..model_utils.module_utils import *
from ..model_utils.module_utils import ProjectionHeadCLIP, ProjectionHeadConVIRT
from ..losses.clip_loss import CLIP_Loss
from ..losses.convirt_loss import ConVIRT_Loss
from ..data_utils.dataloader_utils import MIMIC_CXR_Unsupervised


# base class for multi-modal
class BASE(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # get args
        self.args = args
        self.hparams.update(vars(args))

        # get backbone
        self.img_backbones = {
            "resnet2d_18": resnet_model(size=18, pretrained=self.hparams.pretrained),
            "resnet2d_50": resnet_model(size=50, pretrained=self.hparams.pretrained),
            "resnet2d_101": resnet_model(size=101, pretrained=self.hparams.pretrained),
            "densenet2d_121": densenet_model(size=121, pretrained=self.hparams.pretrained),
            "vit2d_b16": vit_model("base", self.hparams.pretrained, self.hparams.freeze_patch_embed),
        }


    def _build_model(self):
        self.img_backbone = self.img_backbones[self.hparams.img_backbone]
        self.img_projector = ProjectionHeadConVIRT(self.hparams.img_embedding_dim, \
                        self.hparams.projection_dim, self.hparams.dropout)

        if self.hparams.method == "CLIP":
            self.mm_criterion = CLIP_Loss(self.hparams.temperature_mm)
        elif self.hparams.method == "ConVIRT":
            self.mm_criterion = ConVIRT_Loss(self.hparams.batch_size, self.hparams.alpha, self.hparams.temperature_mm, \
                        self.hparams.gpus * self.hparams.num_nodes)
        else:
            pass


    def shared_forward(self, batch, batch_idx):
        images_mm, text_encodings = batch
        images_mm = torch.stack((images_mm))

        # get embeddings
        image_features, text_features = self.img_backbone(images_mm), self.text_backbone(text_encodings)
        image_embeddings, text_embeddings = self.img_projector(image_features), self.text_projector(text_features)

        # compute loss
        loss = self.mm_criterion(image_embeddings, text_embeddings)
        return {"loss": loss}


    def training_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx)
        loss = shared_out["loss"]
        self.log("train_loss", loss, on_epoch=False, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx)
        loss = shared_out["loss"]
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)


    def on_after_backward(self):
        # clip gradients
        if self.hparams.clip_grad:
            clip_gradients(self.img_backbone, self.hparams.clip_grad)
        # if multi-modal, clip text encoder
        matches = ["CLIP", "ConVIRT", "SLIP"]
        if any(x in self.hparams.method for x in matches):
            clip_gradients(self.text_backbone, self.hparams.clip_grad)


    @property
    def learnable_params(self):
        return [
                {"type": "backbone", "params": self.img_backbone.parameters(), "lr": self.hparams.lr_img_backbone},
                {"type": "backbone", "params": self.text_backbone.parameters(), "lr": self.hparams.lr_text_backbone},
                {"type": "projector", "params": self.img_projector.parameters(), "lr": self.hparams.lr_img_backbone},
                {"type": "projector", "params": self.text_projector.parameters(), "lr": self.hparams.lr_text_backbone},
            ]


    def setup(self, stage=None):
        mimic_cxr_path = Path('/gpfs/data/denizlab/Datasets/Public/physionet.org/files/mimic-cxr/2.0.0')
        # load all resized image mapping
        with open(mimic_cxr_path / 'mimic_cxr_imgs_v3.pkl', 'rb') as handle:
            dict_image_mapping = dict(pickle.load(handle))
        print("Trainset Loading ...")
        self.ds_train = MIMIC_CXR_Unsupervised(args=self.args, dict_image_mapping=dict_image_mapping, 
                ssl_transform=self.hparams.ssl_transform, full_report=self.hparams.full_report, 
                data_df_path=self.hparams.train_df_path, train=True)
        print("Valset Loading ...")
        self.ds_val = MIMIC_CXR_Unsupervised(args=self.args, dict_image_mapping=dict_image_mapping, 
                ssl_transform=self.hparams.ssl_transform, full_report=self.hparams.full_report, 
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
        self.warmup_steps = self.hparams.per_warmup_steps * self.total_steps

        if self.hparams.scheduler == "cosine":
            scheduler = WarmupCosineSchedule(optimizer, self.warmup_steps, self.total_steps)
        elif self.hparams.scheduler == "step":
            milestones = [(int)(0.5 * self.hparams.max_epochs), (int)(0.8 * self.hparams.max_epochs)]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        else:
            scheduler = None

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


# base class for self-supervision
class BASE_SSL(BASE):

    def __init__(self, args):
        super().__init__(args)

 
    @property
    def learnable_params(self):
        return [
                {"type": "backbone", "params": self.img_backbone.parameters(), "lr": self.hparams.lr_img_backbone},
            ]


    # collate_fn for tokenizing input
    def collate_fn_batch_encoding(self, batch):
        _, images_ssl1, images_ssl2, _ = zip(*batch)
        return images_ssl1, images_ssl2


# base class for joint multi-modal and self-supervision
class BASE_SLIP(BASE):

    def __init__(self, args):
        super().__init__(args)


    def _build_model(self):
        # image backbone
        super()._build_model()
        # text backbone
        self.text_backbone = bert_model(self.hparams.text_backbone, self.hparams.pool)

        # define multi-modal loss
        if self.hparams.multi_modal == "CLIP":
            # clip projector
            self.img_projector = ProjectionHeadCLIP(self.hparams.img_embedding_dim,
                            self.hparams.projection_dim, self.hparams.dropout)
            self.text_projector = ProjectionHeadCLIP(self.hparams.text_embedding_dim,
                            self.hparams.projection_dim, self.hparams.dropout)
            self.mm_criterion = CLIP_Loss(self.hparams.temperature_mm)
        elif self.hparams.multi_modal == "ConVIRT":
            # convirt projector
            self.img_projector = ProjectionHeadConVIRT(self.hparams.img_embedding_dim, \
                            self.hparams.projection_dim, self.hparams.dropout)
            self.text_projector = ProjectionHeadConVIRT(self.hparams.text_embedding_dim, \
                            self.hparams.projection_dim, self.hparams.dropout)
            self.mm_criterion = ConVIRT_Loss(self.hparams.batch_size, self.hparams.alpha, self.hparams.temperature_mm, \
                                        self.hparams.gpus * self.hparams.num_nodes)
        else:
            raise NotImplementedError("Multi-Modal Method Not Imeplmented!")

        # define text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.text_backbone, use_fast=True)


#    def text_image_matching(self, img_1, img_2, img_3, text_1, epsilon, total_batch_size):
        # calculate mean of three image embeddings
#        img_mean = torch.mean(torch.stack((img_1, img_2, img_3)), dim=0) 
        
        # similarity matrix
#        similarity_f = nn.CosineSimilarity(dim=2)
#        sim_mat = similarity_f(img_mean.unsqueeze(1), text_1.unsqueeze(0))
        # mean of off-diagonal matrix
#        sim_mat_offdiag = torch.mean(torch.stack((torch.triu(sim_mat, diagonal=1).t(), torch.tril(sim_mat, diagonal=-1))), dim=0)
        # mask to get hard negative samples less than epsilon
#        mask = (1 * (sim_mat_offdiag != 0) * 1 * (sim_mat_offdiag < epsilon))
#        sim_mat_filtered = mask * sim_mat_offdiag
        
        # get smallest top-k values in similarity off-diagonal matrix as hard negatives
#        vals, idx = torch.topk(sim_mat_filtered.flatten(), k=total_batch_size * 3, largest=False)
        # get image-text indices for smallest top-k values
#        idx_pairs = (np.array(np.unravel_index(idx.numpy(), sim_mat_filtered.shape)).T)


    def shared_forward(self, batch, batch_idx):
        images_mm, images_ssl1, images_ssl2, text_encodings = batch
        # only use first image for clip
        images_mm, images_ssl1, images_ssl2 = torch.stack((images_mm)), \
                    torch.stack((images_ssl1)), torch.stack((images_ssl2))

        # get embeddings
        img_feat_mm, text_feat = self.img_backbone(images_mm), self.text_backbone(text_encodings)
        img_feat_mm_proj, text_feat_proj = self.img_projector(img_feat_mm), self.text_projector(text_feat)

        # ssl embeddings
        img_feat1, img_feat2 = self.img_backbone(images_ssl1), self.img_backbone(images_ssl2)
        img_feat1_proj, img_feat2_proj = self.img_projector(img_feat1), self.img_projector(img_feat2)

        # compute loss
        mm_loss = self.mm_criterion(img_feat_mm_proj, text_feat_proj)
                self.mm_criterion(img_feat1_proj, text_feat_proj) + \
                self.mm_criterion(img_feat2_proj, text_feat_proj)) / 3

        return (img_feat1, img_feat2, img_feat_mm), (images_ssl1, images_ssl2, images_mm), mm_loss


    def training_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx)
        loss, mm_loss, ssl_loss = shared_out["loss"], shared_out["mm_loss"], shared_out["ssl_loss"]
        self.log("train_loss", loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_mm_loss", mm_loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_ssl_loss", ssl_loss, on_epoch=False, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx)
        loss, mm_loss, ssl_loss = shared_out["loss"], shared_out["mm_loss"], shared_out["ssl_loss"]
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_mm_loss", mm_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_ssl_loss", ssl_loss, on_epoch=True, on_step=False, prog_bar=True)


    # collate_fn for tokenizing input
    def collate_fn_batch_encoding(self, batch):
        images_mm, images_ssl1, images_ssl2, texts = zip(*batch)
        text_encodings = self.tokenizer.batch_encode_plus(
                        list(texts),
                        max_length=self.hparams.max_length,
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt")
        return images_mm, images_ssl1, images_ssl2, text_encodings


# base class for distilling multi-modal with self-supervised model
class BASE_DISTILL(BASE_SLIP):

    def __init__(self, args):
        super().__init__(args)


    def _build_model(self):
        # image backbone
        super()._build_model()
        # image backbone for trained model
        self.img_backbone_kd = copy.deepcopy(self.img_backbone)
        self.img_projector_kd = copy.deepcopy(self.img_projector)
        # load from checkpoint
        checkpoint = torch.load(self.hparams.teacher_load_path)
        state_dict = {k.replace("img_backbone.", ""): v for k, v in checkpoint['state_dict'].items()}
        load_status = self.img_backbone_kd.load_state_dict(state_dict, strict=False)
        for param in self.img_backbone_kd.parameters():
            param.requires_grad = False

        # load weight for projector
        checkpoint = torch.load(self.hparams.teacher_load_path)
        state_dict = {k.replace("img_projector.", ""): v for k, v in checkpoint['state_dict'].items()}
        load_status = self.img_projector_kd.load_state_dict(state_dict, strict=False)
        for param in self.img_projector_kd.parameters():
            param.requires_grad = False


    def shared_forward(self, batch, batch_idx):
        images_mm, images_ssl1, images_ssl2, text_encodings = batch
        images_mm, images_ssl1, images_ssl2 = torch.stack((images_mm)), \
                        torch.stack((images_ssl1)), torch.stack((images_ssl2))

        # mm embeddings
        img_feat_ssl1, img_feat_mm1 = self.img_backbone_kd(images_ssl1), self.img_backbone_kd(images_mm)
        img_feat_ssl1, img_feat_mm1_proj = self.img_projector_kd(img_feat_ssl1), self.img_projector_kd(img_feat_mm1)
        # ssl embeddings
        img_feat_ssl2, img_feat_mm2 = self.img_backbone(images_ssl1), self.img_backbone(images_mm)
        img_feat_mm2_proj = self.img_projector_kd(img_feat_mm2)
        # compute mm loss
        mm_loss = 0
        #mm_loss = self.mm_criterion(img_feat_mm1_proj, text_feat_proj)
        kd_temp = self.hparams.kd_temperature
        kd_loss = calc_kd_loss(img_feat_mm2_proj, img_feat_mm1_proj, kd_temp)
                #calc_kd_loss(img_feat_mm1_proj, img_feat_mm2_proj, kd_temp)

        #s_simi = F.cosine_similarity(img_feat_ssl2_proj, img_feat_mm2_proj, dim=1)
        #t_simi = F.cosine_similarity(img_feat_ssl1_proj, img_feat_mm1_proj, dim=1)
        #ss_loss = calc_kd_loss(s_simi, t_simi, 0.5)
        ss_loss = 0
        # compute kd loss
        #T = self.hparams.kd_temperature
        #student_out = F.log_softmax(img_feat_mm1_proj / T, dim=1)
        #teacher_out = F.softmax(img_feat_mm2_proj / T, dim=1)

        return (img_feat_ssl1, img_feat_ssl2, img_feat_mm1, img_feat_mm2), \
            (images_ssl1, images_ssl2, images_mm), mm_loss, kd_loss, ss_loss


    def training_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx)
        loss, ss_loss, kd_loss, mm_loss, ssl_loss = shared_out["loss"], shared_out["ss_loss"], \
                                        shared_out["kd_loss"], shared_out["mm_loss"], shared_out["ssl_loss"]
        self.log("train_loss", loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_ss_loss", ss_loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_kd_loss", kd_loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_mm_loss", mm_loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_ssl_loss", ssl_loss, on_epoch=False, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx)
        loss, ss_loss, kd_loss, mm_loss, ssl_loss = shared_out["loss"], shared_out["ss_loss"], \
                                        shared_out["kd_loss"], shared_out["mm_loss"], shared_out["ssl_loss"]
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_ss_loss", ss_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_kd_loss", kd_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_mm_loss", mm_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_ssl_loss", ssl_loss, on_epoch=True, on_step=False, prog_bar=True)


    #@property
    #def learnable_params(self):
    #    #extra_learnable_params = [{"type": "backbone", "params": self.img_backbone_kd.parameters(), \
        #                    "lr": self.hparams.lr_img_backbone}]
    #    return super().learnable_params + extra_learnable_params

# base class for distilling multi-modal with self-supervised model
class BASE_SEED(BASE_SLIP):

    def __init__(self, args):
        super().__init__(args)


    def _build_model(self):
        # image backbone
        super()._build_model()
        # image backbone for trained model
        self.img_backbone_kd = copy.deepcopy(self.img_backbone)
        self.img_projector_kd = copy.deepcopy(self.img_projector)
        # load from checkpoint
        checkpoint = torch.load(self.hparams.teacher_load_path)
        state_dict = {k.replace("img_backbone.", ""): v for k, v in checkpoint['state_dict'].items()}
        load_status = self.img_backbone_kd.load_state_dict(state_dict, strict=False)

        # load weight for projector
        checkpoint = torch.load(self.hparams.teacher_load_path)
        state_dict = {k.replace("img_projector.", ""): v for k, v in checkpoint['state_dict'].items()}
        load_status = self.img_projector_kd.load_state_dict(state_dict, strict=False)

        self.seed = SEED(self.img_backbone, self.img_backbone_kd, self.img_projector, self.img_projector_kd, \
                      65536, 0.2, 0.01, True)


    def shared_forward(self, batch, batch_idx):
        images_mm, images_ssl1, images_ssl2, text_encodings = batch
        images_mm, images_ssl1, images_ssl2 = torch.stack((images_mm)), \
                        torch.stack((images_ssl1)), torch.stack((images_ssl2))

        # mm embeddings
        img_feat_ssl1, img_feat_mm1 = self.img_backbone_kd(images_ssl1), self.img_backbone_kd(images_mm)
        img_feat_ssl1, img_feat_mm1_proj = self.img_projector_kd(img_feat_ssl1), self.img_projector_kd(img_feat_mm1)
        # ssl embeddings
        img_feat_ssl2, img_feat_mm2 = self.img_backbone(images_ssl1), self.img_backbone(images_ssl2)
        img_feat_mm2_proj = self.img_projector_kd(img_feat_mm2)
        # compute mm loss
        mm_loss = 0
        logit, label = self.seed(images_ssl2)
        kd_loss = soft_cross_entropy(logit, label)
        ss_loss = 0

        return (img_feat_ssl1, img_feat_ssl2, img_feat_mm1, img_feat_mm2), \
            (images_ssl1, images_ssl2, images_mm), mm_loss, kd_loss, ss_loss


    def training_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx)
        loss, ss_loss, kd_loss, mm_loss, ssl_loss = shared_out["loss"], shared_out["ss_loss"], \
                                        shared_out["kd_loss"], shared_out["mm_loss"], shared_out["ssl_loss"]
        self.log("train_loss", loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_ss_loss", ss_loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_kd_loss", kd_loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_mm_loss", mm_loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_ssl_loss", ssl_loss, on_epoch=False, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        shared_out = self.shared_forward(batch, batch_idx)
        loss, ss_loss, kd_loss, mm_loss, ssl_loss = shared_out["loss"], shared_out["ss_loss"], \
                                        shared_out["kd_loss"], shared_out["mm_loss"], shared_out["ssl_loss"]
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_ss_loss", ss_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_kd_loss", kd_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_mm_loss", mm_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_ssl_loss", ssl_loss, on_epoch=True, on_step=False, prog_bar=True)


