# util libraries
import warnings

# user defined files
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoConfig, AutoTokenizer, AutoModel
#from ..backbones.vit_mocov3 import *
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
    def __init__(self, size, pretrained=False, freeze_patch_embed=False, **kwargs):
        super(vit_model, self).__init__()

        if size=="base":
            model_kwargs = dict(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0, **kwargs
            )
            self.backbone = _create_vision_transformer("vit_base_patch16_224", pretrained=pretrained, **model_kwargs)
        else:
            pass

        if freeze_patch_embed:
            self.backbone.patch_embed.requires_grad = False

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


