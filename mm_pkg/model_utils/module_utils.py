# util libraries
import warnings

# user defined files
import torch
import torch.nn as nn
import torchvision.models as models
from transfomers import AutoConfig, AutoTokenizer, AutoModel


warnings.filterwarnings("ignore")


def get_embed(model, text, pool, max_length, tokenizer):
    text_encod = tokenizer.batch_encode_plus(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
        )
    text_outputs = model(text_encod['input_ids'], text_encod['attention_mask'], return_dict=True)
    text_hidden_states = text_outputs.hidden_states

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

        self.feature_dim_in = self.backbone.fc.weight.shape[1]
        self.fc = nn.Linear(in_features=self.feature_dim_in, out_features=features_dim, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten()
        x = self.fc(x)
        return x


# bert model
class bert_model(nn.Module):
    def __init__(self, model_name, max_length, pool):
        super(bert_model, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, max_length, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length
        self.pool = pool


    def forward(self, x):
        x = get_embed(self.model, x, self.pool, self.max_length, self.tokenizer)
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

