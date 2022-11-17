import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from timm.models.layers import trunc_normal_
from ..backbones.vt_unet import SwinTransformerSys3D, PatchEmbed3D, BasicLayer, PatchMerging
from einops import rearrange


class Swin_Transformer_3D(nn.Module):
    def __init__(self, pretrained=None,
                 pretrained2d=True,
                 img_size=(36, 384, 384),
                 patch_size=(4, 4, 4),
                 in_chans=1,
                 num_classes=1,
                 embed_dim=96,
                 depths=[2, 2, 2, 2, 1],
                 num_heads=[3, 6, 12, 24, 48],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False, **kwargs):
        
        super().__init__()
        
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed3D(img_size=img_size,
                                        patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                        norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                depths=depths,
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.gap = nn.AdaptiveAvgPool3d(1)
        #self.fc = nn.Linear(in_features=self.num_features, out_features=self.num_classes, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x, v1, k1, q1, v2, k2, q2 = layer(x, i)
            #print('######')
            #print(x.size())
        
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        
        x = self.gap(x)
        x = x.view(x.size(0),x.size(1))
        #print(f"x.shape: {x.shape}")
        #x = self.fc(x)
        return x
        
        
