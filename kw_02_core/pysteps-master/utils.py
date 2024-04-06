# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2023/9/21 17:16
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : utils
# @IDE     : PyCharm
# -----------------------------------------------------------------
import torch


def warp(input, flow, grid, mode="bilinear", padding_mode="zeros"):

    B, C, H, W = input.size()
    vgrid = grid + flow

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
    return output