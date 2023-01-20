from ..methods.base import *
from ..methods.base import BASE_DISTILL
from ..model_utils.misc_utils import calc_kd_loss
from ..losses.simclr_loss import SIMCLR_Loss


class DISTILL_SIMCLR(BASE_DISTILL):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model()


    def _build_model(self):
        super()._build_model()
        
        # simclr objective
        self.ssl_criterion = SIMCLR_Loss(self.hparams.batch_size, self.hparams.temperature_ssl, \
                            self.hparams.gpus * self.hparams.num_nodes) 

        # simclr projector
        self.simclr_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.simclr_proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.hparams.simclr_proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.simclr_proj_hidden_dim, self.hparams.simclr_proj_output_dim, bias=True),
        )
        #self.simclr_projector_student = copy.deepcopy(self.simclr_projector_teacher)

        # load weight for projector
        checkpoint = torch.load(self.hparams.teacher_load_path)
        state_dict = {k.replace("simclr_projector.", ""): v for k, v in checkpoint['state_dict'].items()}
        load_status = self.simclr_projector.load_state_dict(state_dict, strict=False)


    def shared_forward(self, batch, batch_idx):
        img_feats, imgs, mm_loss, kd_loss, ss_loss = super().shared_forward(batch, batch_idx)
        img_feat_ssl1, img_feat_ssl2, img_feat_mm1, img_feat_mm2 = img_feats

        # simclr 
        #z_ssl1, z_ssl2, z_mm1, z_mm2 = self.simclr_projector_teacher(img_feat_ssl1), self.simclr_projector_teacher(img_feat_ssl2), \
        #                        self.simclr_projector_teacher(img_feat_mm1), self.simclr_projector_teacher(img_feat_mm2)
        z_ssl2, z_mm2 = self.simclr_projector(img_feat_ssl2), self.simclr_projector(img_feat_mm2)
        ssl_loss = self.ssl_criterion(z_ssl2, z_mm2)

        # compute kd loss
        #kd_temp = self.hparams.kd_temperature
        #kd_loss = calc_kd_loss(z_mm1, z_mm2, kd_temp)

        # slip final loss
        loss = self.hparams.ssl_scale * ssl_loss + self.hparams.kd_scale * kd_loss + 10 * ss_loss
        return {"loss": loss, "kd_loss": kd_loss, "mm_loss": mm_loss, "ssl_loss": ssl_loss, "ss_loss": ss_loss}


    @property
    def learnable_params(self):
        extra_learnable_params = [{"type": "projector", "params": self.simclr_projector.parameters(), "lr": self.hparams.lr_img_backbone}]
        return super().learnable_params + extra_learnable_params


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("slip_simclr")

        # simclr projector
        parser.add_argument("--simclr_proj_hidden_dim", type=int, default=512)
        parser.add_argument("--simclr_proj_output_dim", type=int, default=128)

        return parent_parser


