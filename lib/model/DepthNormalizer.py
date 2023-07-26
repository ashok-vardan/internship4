# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthNormalizer(nn.Module):
    def __init__(self, opt):
        super(DepthNormalizer, self).__init__()
        self.opt = opt

    def forward(self, xyz, calibs=None, index_feat=None):
        '''
        Normalize depth value.
        args:
            xyz: [B, 3, N] depth value
        '''
        z_feat = xyz[:, 2:3, :] * (self.opt.loadSize // 2) / self.opt.z_size
        return z_feat

class Generator(nn.Module):
    # ... (other parts of the Generator)

    def forward(self, x):
        # ... (rest of the Generator code)

        normx = x
        z_feat = DepthNormalizer(opt)(xyz)  # Integrating the DepthNormalizer here

        # ... (rest of the Generator code)

        return outputs, normx
