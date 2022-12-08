# util libraries
import warnings

# user defined files
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoConfig, AutoTokenizer, AutoModel
from timm.models.vision_transformer import _create_vision_transformer
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer, _cfg


warnings.filterwarnings("ignore")


def get_embed(model, text_encodings, pool):
    #text_outputs = model(text_encodings['input_ids'], text_encodings['attention_mask'], return_dict=True)
    outputs = model(return_dict=True, **text_encodings)
    text_hidden_states = outputs.hidden_states

    if pool == 'cls':
        text_embed = text_hidden_states[-1][:, 0, :]
    elif pool == 'mean':
        text_embed = (text_hidden_states[-1] * text_encod['attention_mask'].unsqueeze(-1)).sum(1) \
                     / text_encod['attention_mask'].sum(-1).unsqueeze(-1)
    else:
        raise NotImplementedError("Wrong pool input!")

    return text_embed


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim)
            )
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        # https://pytorch.org/docs/stable/generated/torch.meshgrid.html
        # indexing –
        # (str, optional): the indexing mode, either “xy” or “ij”, defaults to “ij”.
        # If “xy” is selected, the first dimension corresponds to the cardinality of
        # the second input and the second dimension corresponds to the cardinality of the first input.
        # If “ij” is selected, the dimensions are in the same order as the cardinality of the inputs.
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
        )[None, :, :]

        assert self.num_prefix_tokens == 1, "Assuming one and only one token, [cls]"
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


# resnet model
class resnet_model(nn.Module):
    def __init__(self, size, features_dim, pretrained=False):
        super(resnet_model, self).__init__()

        if size==18:
            self.backbone = models.resnet18(pretrained=pretrained)
        elif size==50:
            self.backbone = models.resnet50(pretrained=pretrained)
        elif size==101:
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise NotImplementedError(f"ResNet with size {size} is not implemented!")

        #self.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_dim_in = self.backbone.fc.weight.shape[1]
        #self.backbone.fc = nn.Linear(in_features=self.feature_dim_in, out_features=features_dim, bias=True)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        #x = x.flatten()
        #x = self.fc(x)
        return x


# densenet model
class densenet_model(nn.Module):
    def __init__(self, size, features_dim, pretrained=False):
        super(densenet_model, self).__init__()

        if size == 121:
            self.backbone = models.densenet121(pretrained=pretrained)

        self.feature_dim_in = self.backbone.classifier.weight.shape[1]
        #self.backbone.classifier = nn.Linear(in_features=self.feature_dim_in, out_features=features_dim, bias=True)
        self.backbone.classifier = nn.Identity()        

    def forward(self, x):
        x = self.backbone(x)
        return x


# vit model
class vit_model(nn.Module):
    def __init__(self, size, pretrained=False, freeze_pos_embed=False, **kwargs):
        super(vit_model, self).__init__()

        if freeze_pos_embed:
            pass
        else:
            if size=="base":
                model_kwargs = dict(
                    patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0, **kwargs
                )
                self.backbone = _create_vision_transformer("vit_base_patch16_224", pretrained=pretrained, **model_kwargs)
            else:
                pass

    def forward(self, x):
        x = self.backbone(x)
        return x


# bert model
class bert_model(nn.Module):
    def __init__(self, model_name, pool):
        super(bert_model, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.pool = pool

    def forward(self, x):
        x = get_embed(self.model, x, self.pool)
        return x


# clip projection head
class ProjectionHeadCLIP(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


