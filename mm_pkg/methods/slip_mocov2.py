from ..methods.base import *
from ..methods.base import BASE_SLIP
from ..losses.clip_loss import clip_loss
from ..losses.convirt_loss import ConVIRT_Loss
from ..losses.mocov2_loss import mocov2_loss
from ..model_utils.misc_utils import _batch_shuffle_ddp, _batch_unshuffle_ddp
from copy import deepcopy

class SLIP_MOCOV2(BASE_SLIP):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model(self.hparams.img_backbone, self.hparams.dropout)


    def _build_model(self, img_backbone, dropout):
        # model backbones
        self.img_backbone = self.img_backbones[img_backbone]
        self.img_backbone_ema = deepcopy(self.img_backbone)
        for param in self.img_backbone_ema.parameters():
            param.requires_grad = False

        self.text_backbone = bert_model(self.hparams.text_backbone, self.hparams.pool)

        if self.hparams.multi_modal == "CLIP":
            # clip projector
            self.img_projector = ProjectionHeadCLIP(self.hparams.img_embedding_dim,
                            self.hparams.projection_dim, self.hparams.dropout)
            self.text_projector = ProjectionHeadCLIP(self.hparams.text_embedding_dim,
                            self.hparams.projection_dim, self.hparams.dropout)
        elif self.hparams.multi_modal == "ConVIRT":
            # convirt projector
            self.img_projector = ProjectionHeadConVIRT(self.hparams.img_embedding_dim, \
                            self.hparams.projection_dim, self.hparams.dropout)
            self.text_projector = ProjectionHeadConVIRT(self.hparams.text_embedding_dim, \
                            self.hparams.projection_dim, self.hparams.dropout)
        else:
            raise NotImplementedError("Multi-Modal Method Not Imeplmented!")

        if self.hparams.multi_modal == "ConVIRT":
            self.criterion = ConVIRT_Loss(self.hparams.batch_size, self.hparams.alpha, self.hparams.temperature_mm, \
                                        self.hparams.gpus * self.hparams.num_nodes)

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.text_backbone, use_fast=True)

        # mocov2 projectors
        self.mocov2_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.mocov2_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.mocov2_proj_hidden_dim, self.hparams.mocov2_proj_output_dim),
        )
        self.mocov2_projector_ema = deepcopy(self.mocov2_projector) 
        for param in self.mocov2_projector_ema.parameters():
            param.requires_grad = False
       
        # create_queue 
        self.register_buffer("queue", torch.randn(2, self.hparams.mocov2_proj_output_dim, self.hparams.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Adds new samples and removes old samples from the queue in a fifo manner.
        Args:
            keys (torch.Tensor): output features of the momentum backbone.
        """

        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.hparams.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.hparams.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore


    def shared_forward(self, batch, batch_idx, mode="train"):
        images_clip, images_ssl1, images_ssl2, text_encodings = batch
        # only use first image for clip
        images_clip, images_ssl1, images_ssl2 = torch.stack((images_clip)), torch.stack((images_ssl1)), torch.stack((images_ssl2))

        # get embeddings
        image_features, text_features = self.img_backbone(images_clip), self.text_backbone(text_encodings)
        image_embeddings, text_embeddings = self.img_projector(image_features), self.text_projector(text_features)
        image_embeddings, text_embeddings = all_gather(image_embeddings), all_gather(text_embeddings)

        # compute loss
        if self.hparams.multi_modal == "CLIP":
            mm_loss = clip_loss(image_embeddings, text_embeddings, self.hparams.temperature_mm).mean()
        elif self.hparams.multi_modal == "ConVIRT":
            mm_loss = self.criterion(image_embeddings, text_embeddings)
        else:
            raise NotImplementedError("Multi-Modal Method Not Imeplmented!") 

        # mocov2
        # ema update
        ema(self.img_backbone, self.img_backbone_ema, self.hparams.ema_decay)
        ema(self.mocov2_projector, self.mocov2_projector_ema, self.hparams.ema_decay)

        # original encoder output
        feat1, feat2 = self.img_backbone(images_ssl1), self.img_backbone(images_ssl2)
        q1, q2 = self.mocov2_projector(feat1), self.mocov2_projector(feat2)
        # normalize
        q1, q2 = nn.functional.normalize(q1, dim=1), nn.functional.normalize(q1, dim=1)

        with torch.no_grad():
            # shuffle for making use of BN
            images1_k, idx_unshuffle1 = _batch_shuffle_ddp(images_ssl1)
            images2_k, idx_unshuffle2 = _batch_shuffle_ddp(images_ssl2)
            # ema encoder output
            feat1_ema, feat2_ema = self.img_backbone_ema(images1_k), self.img_backbone_ema(images2_k)
            k1, k2 = self.mocov2_projector_ema(feat1_ema), self.mocov2_projector_ema(feat2_ema)
            # normalize
            k1, k2 = nn.functional.normalize(k1, dim=1), nn.functional.normalize(k2, dim=1)
            # undo shuffle
            k1, k2 = _batch_unshuffle_ddp(k1, idx_unshuffle1), _batch_unshuffle_ddp(k2, idx_unshuffle2)

        # loss
        queue = self.queue.clone().detach()
        ssl_loss = (mocov2_loss(q1, k2, queue[1], self.hparams.temperature_mocov2)
                + mocov2_loss(q2, k1, queue[0], self.hparams.temperature_mocov2)) / 2

        # update queue
        keys = torch.stack((gather(k1), gather(k2)))
        self._dequeue_and_enqueue(keys)

        # slip final loss
        loss = mm_loss + self.hparams.ssl_scale * ssl_loss
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
            {"type": "projector", "params": self.mocov2_projector.parameters()},
        ]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("slip_mocov2")

        parser.add_argument("--img_embedding_dim", type=int, default=2048)
        parser.add_argument("--text_embedding_dim", type=int, default=768)
        parser.add_argument("--projection_dim", type=int, default=512)
        parser.add_argument("--dropout", type=int, default=0.1)

        parser.add_argument("--temperature_mm", type=float, default=0.1)
        parser.add_argument("--alpha", type=float, default=0.75)

        parser.add_argument("--temperature_mocov2", type=float, default=0.07)
        parser.add_argument("--ema_decay", type=float, default=0.999)

        # queue
        parser.add_argument("--queue_size", type=int, default=65536)

        # mocov2 projector
        parser.add_argument("--mocov2_proj_hidden_dim", type=int, default=512)
        parser.add_argument("--mocov2_proj_output_dim", type=int, default=128)
        parser.add_argument("--ssl_scale", type=float, default=1.0)

        return parent_parser


