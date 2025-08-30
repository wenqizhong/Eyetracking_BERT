# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding负责将图像转换为一系列图像块patches
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # 使用卷积层将图像块转换为嵌入向量
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # 确保输入图像的大小符合预期的大小
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 使用卷积层将图像块转换为嵌入向量，并进行形状调整
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001,
                 latent_dim=768, max_length=15, att_layer=3, use_cls_token=True, vocab_size=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  #num_patches 196
        #创建了一个可训练的参数 cls_token，它是一个形状为 (1, 1, embed_dim) 的张量
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        #创建一个列表 dpr，其中包含了根据随机深度衰减规则生成的一组数值
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([    #self.blocks: 12层，这是一个包含多个 Transformer 块的 nn.ModuleList。每个块都是一个 Block 类的实例，参数设置和数量由模型的超参数（如 depth）决定。每个块都包含了一个 Transformer 层。
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.encoder = SimpleEncoder(
            latent_dim=latent_dim ,max_length=max_length, att_layer=att_layer,
            use_cls_token=use_cls_token, vocab_size=vocab_size)
        self.fusion = FeatureFusionNet(input_dim=768 * 2, hidden_dim=768)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()  #self.head: 一个线性层，用于将最终的 Transformer 输出映射到类别数（num_classes）
        # self.head = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()  #self.head: 一个线性层，用于将最终的 Transformer 输出映射到类别数（num_classes）

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)  #self.apply(self._init_weights): 这一行代码将一个自定义的初始化函数 _init_weights 应用到模型的所有参数上。该函数的实现可能包含一些特殊的初始化逻辑
        self.fix_init_weight()  #这是一个自定义函数，可能包含一些用于修正初始化权重的逻辑

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    #对不同类型的层（例如线性层和层归一化层）的权重进行特定的初始化操作
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 对线性层的权重进行截断正态分布初始化
            trunc_normal_(m.weight, std=.02)
            # 如果是线性层并且有偏置项，将偏置项初始化为常数0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 对层归一化层的偏置项初始化为常数0，权重初始化为常数1.0
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, patch_indices):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # x = self.pos_drop(x)


        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)
        if self.fc_norm is not None:
            t = x[:, 1:, :]
            batch_size = t.shape[0]
            num_indices = len(patch_indices)  # 这里是9
            features_dim = t.shape[2]
            # 初始化一个新的张量来存储提取的特征
            extracted_features = torch.zeros(batch_size, num_indices, features_dim, device=t.device)
            # 将 patch_indices 转换为 tensor，并确保是 long 类型
            patch_indices_tensor = torch.stack(patch_indices).long().to(t.device)  # 形状 [num_attention_points, batch_size]
            # 转置 patch_indices_tensor 以匹配 [batch_size, num_attention_points]
            # patch_indices_tensor = patch_indices_tensor.t()
        
            # 遍历每个注视点索引，提取特征
            for i in range(num_indices):
                # 将提取的特征放入结果张量
                extracted_features[:, i, :] = x[torch.arange(batch_size), patch_indices_tensor[i, :]]

            extracted_features_mean = self.fc_norm(extracted_features.mean(1))
            # 提取时序特征
            z = self.encoder(extracted_features)
            seq_feature = z[:,0,:]

            # features_fusion = torch.cat((extracted_features_mean, seq_feature), dim=1)

            features_fusion = self.fusion(extracted_features_mean, seq_feature)

            return features_fusion
        else:
            return x[:, 0]

    def forward(self, x, patch_indices):
        x = self.forward_features(x, patch_indices)
        x = self.head(x)
        return x

    def get_intermediate_layers(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            features.append(x)

        return features

class SelfAttention(nn.Module):
    def __init__(self,latent_dim=768,max_length=15):
      super().__init__()
      self.q=nn.Linear(latent_dim,latent_dim)
      self.k=nn.Linear(latent_dim,latent_dim)
      self.v=nn.Linear(latent_dim,latent_dim)
      self.norm1=nn.LayerNorm(latent_dim)
      self.drop=nn.Dropout()
      self.ffn=nn.Sequential(
         nn.Dropout(),
         nn.Linear(latent_dim,latent_dim),
      )
      self.norm2=nn.LayerNorm(latent_dim)
      self.max_length=max_length
      self.latent_dim=latent_dim
      # self.positionEmbed =nn.Parameter(self.init_pe())
    def forward(self,z):
      q=self.q(z) #
      k=self.k(z)
      v=self.v(z)
      w=F.softmax(k @ v.permute((0,2,1))/torch.sqrt(torch.tensor(float(self.latent_dim))),dim=2)
      y=w @ v
      y=self.drop(y)
      y=y + z
      y=self.norm1(y)
      y1=self.ffn(y)
      y1=y+y1
      y1=self.norm2(y1)
      return y1
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, latent_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = latent_dim // heads
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        B, T, C = x.shape
        queries = self.query(x).view(B, T, self.heads, self.head_dim)
        keys = self.key(x).view(B, T, self.heads, self.head_dim)
        values = self.value(x).view(B, T, self.heads, self.head_dim)
        
        # Attention mechanism (scaled dot-product attention)
        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)
        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(B, T, C)
        
        return self.fc_out(out)

    

    
class SimpleEncoder(nn.Module):
    def __init__(self, latent_dim=768,max_length=15,att_layer=3,use_cls_token=True,vocab_size=5):
      super().__init__()
      self.use_cls_token=use_cls_token
      self.max_length=max_length
      self.latent_dim=latent_dim
      self.vocab_size=vocab_size
      init_pe=self.init_pe()
      self.positionEmbed=nn.Parameter(init_pe)
      self.mask_token=nn.Parameter(torch.randn(latent_dim))
      self.enc=nn.ModuleList([
          SelfAttention(latent_dim,max_length)
          for i in range(att_layer)
      ])
      self.att_layer=att_layer
      self.lm_head=nn.Linear(latent_dim,vocab_size)
    def forward(self,z,mask_seq_bool=None):
      # z (B,9,768)
      # mask (B,9)
      batch_size, seq_len, _ = z.size()
      if mask_seq_bool is not None:
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        w = mask_seq_bool.unsqueeze(-1).type_as(mask_token)
        z = (1-w)*z + w*mask_token
      pe=self.positionEmbed.to(z.device)
      if self.use_cls_token:
        z=F.pad(z,(0,0,1,0,0,0),value=0)
      z1=z+pe
      for att in self.enc:
          z1=att(z1)
      return z1 #+z
    
    def token_prob(self,z):
       # z:(B,768)
       y = self.lm_head(z)
       y = F.softmax(y,dim=1)
       return y
    def init_pe(self):
      # use positional encoding as init embedding
      max_length=self.max_length+1 if self.use_cls_token else self.max_length
      latent_dim=self.latent_dim
      pe = torch.zeros(max_length, latent_dim)
      pos=torch.arange(0,max_length)
      for dim in range(latent_dim):
          pe[0::2,dim]=torch.sin(pos[0::2]/torch.exp(torch.log(torch.tensor(100.))*(dim)/latent_dim))
          pe[1::2,dim]=torch.cos(pos[1::2]/torch.exp(torch.log(torch.tensor(100.))*(dim)/latent_dim))
      return pe.unsqueeze(0)
    

class FeatureFusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureFusionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x1, x2):
        # x1, x2 分别是两组特征
        x = torch.cat((x1, x2), dim=-1)  # 拼接特征
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        # x = self.fc3(x)
        return x

@register_model
def beit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,   #嵌入维度指的是模型中表示每个输入 token（例如图像块或序列中的一个词）的向量的维度；深度指的是模型中的 Transformer 层的数量，每个 Transformer 层包含自注意力机制和前馈神经网络（MLP）
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)  
    model.default_cfg = _cfg()
    return model


@register_model
def beit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def beit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def beit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def beit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
