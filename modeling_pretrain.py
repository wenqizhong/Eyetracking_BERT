# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, RelativePositionBias
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

#定义截断正态分布初始化的函数：
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

#定义一个在导入模块时会被导出的模型名称列表
__all__ = [
    'beit_base_patch16_224_8k_vocab', 
    'beit_large_patch16_224_8k_vocab', 
]


class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)  #创建了一个 PatchEmbed 的实例，用于将输入图像划分为图像块并进行嵌入
        num_patches = self.patch_embed.num_patches      #获取了图像块的数量，这是通过 PatchEmbed 类计算得到的。

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  #创建了一个可学习的参数 cls_token，它是用来表示类别信息的令牌（class token）。这个令牌会在模型的前向传播中与图像块的嵌入进行拼接。
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  #创建了一个可学习的参数 mask_token，它是用来表示遮蔽信息的令牌（mask token）。在模型的前向传播中，模型将会用这个令牌替换图像中的遮蔽区域。
        if use_abs_pos_emb:   #是否使用位置嵌入
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)  #创建了一个 dropout 层 pos_drop，它将在位置嵌入上应用 dropout 操作，以防止过拟合。

        if use_shared_rel_pos_bias:  #是否使用共享的相对位置偏置
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  #创建了一个列表 dpr，dpr 中存储的是一个深度为 depth 的列表，其中每个元素表示一个渐变的 dropout 路径概率值。这个列表将在模型的构造函数中用于设置每个 Transformer 块的 dropout 路径概率。这样的渐变概率通常用于实现随机深度学习（stochastic depth）的训练策略。  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )   #创建了一个 nn.ModuleList，其中包含了多个 Block 类的实例，每个实例都是一个 Transformer 块。这些块被存储在 self.blocks 中，并由列表推导式创建，通过循环创建了深度为 depth 的多个块。每个块都使用了一些超参数，如嵌入维度 dim、注意力头的数量 num_heads、前馈网络的隐藏层与嵌入大小的比率 mlp_ratio 等
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)  #创建了一个层归一化层 norm，用于对嵌入维度进行归一化。embed_dim 是模型中嵌入的维度。

        self.init_std = init_std  #将 init_std 存储在 self.init_std 中，该值被用于初始化模型参数的标准差
        self.lm_head = nn.Linear(embed_dim, vocab_size)  #创建了一个线性层 lm_head，用于将模型的输出映射到词汇表中的词汇。这个线性层的输入维度是 embed_dim，输出维度是 vocab_size，即词汇表的大小。
        
        # 对各部分进行截断正态分布初始化
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()  #调用了模型的 fix_init_weight 方法，该方法用于重新缩放某些层的权重。在代码中，它被用来重新缩放注意力机制和前馈网络的权重。


    #通过对每个层的权重进行重新缩放，使得这些权重更好地适应模型的深度，有助于训练的稳定性
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))   #定义了一个内部函数 rescale，该函数用于重新缩放权重。接受两个参数，param 表示需要重新缩放的权重参数，layer_id 表示当前层的标识。

        for layer_id, layer in enumerate(self.blocks):    #使用 enumerate 函数遍历模型中的每个 Transformer 块，其中 layer_id 是当前层的标识，layer 是对应的 Transformer 块。
            rescale(layer.attn.proj.weight.data, layer_id + 1)   #对当前块的注意力机制（attn）的投影权重进行重新缩放。layer.attn.proj.weight.data 表示注意力机制中的投影权重，它会被传递给 rescale 函数进行重新缩放。layer_id + 1 表示当前层的标识，用于计算重新缩放的因子。
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)   #对当前块的前馈网络（mlp）的第二个全连接层（fc2）的权重进行重新缩放。与上一步类似，它将当前块的权重传递给 rescale 函数进行重新缩放。

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)  #x输入的特征，即图像的嵌入表示；bool_masked_pos: 一个布尔张量，指示哪些位置是遮蔽的。
        batch_size, seq_len, _ = x.size()   #通过 self.patch_embed 将输入图像嵌入为图像块。如果 bool_masked_pos 为 True，则表示对应位置需要进行遮蔽，将这些位置的图像块替换为遮蔽令牌（mask_token）。

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks #将类别令牌（cls_token）进行扩展，然后将其与图像块拼接在一起。
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None  #如果使用了绝对位置嵌入（self.pos_embed 不为 None），则将位置嵌入加到输入中。
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)   #通过循环将输入传递给每个 Transformer 块（self.blocks）。

        return self.norm(x)   #返回归一化后的输出

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        if return_all_tokens:   #return_all_tokens（默认为 False）: 如果为 True，则返回所有的令牌，否则仅返回遮蔽位置的令牌。
            return self.lm_head(x)
        else:
            # return the masked tokens
            return self.lm_head(x[bool_masked_pos])


@register_model
def beit_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()  #_cfg() 函数返回模型的默认配置，用于存储模型的一些参数
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"   #使用提供的检查点路径 kwargs["init_ckpt"]，加载预训练的权重到模型中
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def beit_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
