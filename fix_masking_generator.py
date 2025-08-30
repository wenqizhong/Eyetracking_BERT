"""
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman

Modified by Hangbo Bao, for generating the masked position for visual image transformer
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
import random
import math
import numpy as np
import re


class FixMaskingGenerator:
    def __init__(
            self, input_size):  #, num_masking_patches
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2    #14, 14
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        # self.num_masking_patches = num_masking_patches


   
    def __repr__(self):
        repr_str = "Generator(%d, %d)" % (  #, max = %d, %.3f ~ %.3f
            self.height, self.width) 
            # , self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, patch_indices):
        delta = 0
        # Offset positions to cover surrounding blocks
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 0), (0, 1),
                (1, -1), (1, 0), (1, 1)]

        for indice in patch_indices:
            row = int(indice // self.width)  # 整除得到行
            col = int(indice % self.width) - 1  # 取余得到列
            # Set the corresponding grid cell and surrounding cells to 1
            for offset in offsets:
                surrounding_col = col + offset[0]
                surrounding_row = row + offset[1]

                if 0 <= surrounding_col < self.width and 0 <= surrounding_row < self.height:
                    if mask[surrounding_row, surrounding_col] == 0:
                        mask[surrounding_row, surrounding_col] = 1
                        delta += 1  # 仅在新遮盖一个 patch 时增加 delta
        return delta



    def __call__(self, samples):
        mask = np.zeros(shape=self.get_shape(), dtype=int)
        self._mask(mask, samples[1])

        return mask
    



    


