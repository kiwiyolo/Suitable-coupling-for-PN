# -*- coding: UTF-8 -*-
#
# Copyright (C) 2023 - 2023 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2023/9/22 8:23
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : nowcasting_module
# @IDE     : PyCharm
# -----------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import json
import os
# TODO 还原nature-skilful文章中的方法

# 获取当前脚本文件所在的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 构建上一级文件夹的路径
parent_path = os.path.dirname(current_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configs = json.load(open(parent_path + '/config_NcN.json', 'r', encoding='utf-8', errors='ignore'))  # 加载模拟配置json
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=kernel//2)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding=kernel//2)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2))
        )

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut
        return x


class LSTMConv(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(LSTMConv, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels,
                              4 * self.hidden_channels,
                              self.kernel_size,
                              1,
                              self.padding)

        # Xavier Initialization
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, input, cur_state):
        hidden_state, cell_state = cur_state
        input_hidden_state = torch.cat((input, hidden_state), dim=0)
        conv_outputs = self.conv(input_hidden_state)

        f, i, c, o = torch.split(conv_outputs, self.hidden_channels, dim=0)
        # 避免就地操作（inplace）
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c = torch.tanh(c)

        new_cell_state = cell_state * f + i * c
        cell_state = new_cell_state
        hidden_state = o * torch.tanh(new_cell_state)

        return hidden_state, cell_state

class CDown(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class RDown(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.out_channels = out_channels
        self.maxpool = nn.MaxPool2d(2)
        self.convlstm =  LSTMConv(in_channels, out_channels, kernel)


    def forward(self, x):
        x = self.maxpool(x)
        hidden_state2 = cell_state2 = torch.zeros(self.out_channels, x.size(2), x.size(3)).to(device)
        convlstm2_outputs = []
        for t in range(x.size(0)):
            hidden_state2, cell_state2 = self.convlstm(x[t], [hidden_state2, cell_state2])
            convlstm2_outputs.append(hidden_state2)
        x = torch.stack(convlstm2_outputs, dim=0)
        return x


class CUp(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class RUp(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        self.out_channels = out_channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.convlstm = LSTMConv(in_channels, out_channels, kernel)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.convlstm = LSTMConv(in_channels, out_channels, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        hidden_state2 = cell_state2 = torch.zeros(self.out_channels, x.size(2), x.size(3)).to(device)
        convlstm2_outputs = []
        for t in range(x.size(0)):
            hidden_state2, cell_state2 = self.convlstm(x[t], [hidden_state2, cell_state2])
            convlstm2_outputs.append(hidden_state2)
        x = torch.stack(convlstm2_outputs, dim=0)
        return  x

class Up_S(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class Cropping3D(nn.Module):
    def __init__(self, cropping):
        super().__init__()  # 使用简化的super调用
        self.cropping = cropping

    def forward(self, x):
        return x[:, :,
               self.cropping[0][0]:-self.cropping[0][1] if self.cropping[0][1] != 0 else None,
               self.cropping[1][0]:-self.cropping[1][1] if self.cropping[1][1] != 0 else None,
               self.cropping[2][0]:-self.cropping[2][1] if self.cropping[2][1] != 0 else None]


class FNU(nn.Module):
    def __init__(self, input_size, hidden_size, num_ensemble):
        super(FNU, self).__init__()
        # TODO self.num_layers = num_layers

        # 输入到注意单元各组分的权重和偏置
        w_f = []
        for i in range(num_ensemble):
            f = nn.Parameter(torch.Tensor(input_size, hidden_size))
            w_f.append(f)
        self.w_f = nn.ParameterList(w_f)

        # 初始化权重和偏置
        self.init_weights()

    def init_weights(self):
        # 初始化权重矩阵为均匀分布，偏置向量为全零
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, f_ensemble):
        """
        :param f_ensemble: 用于验证集进行注意单元处理的集合体：张量
        :return:
        """
        f_ens = torch.zeros_like(f_ensemble[: ,0 ,: ,: , :])  # Initialize the result tensor with zeros

        for i in range(f_ensemble.size(1)):
            f_ens += self.w_f[i]@f_ensemble[: ,i ,: ,: , :]
        return f_ens.unsqueeze(1)


class Conv3D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size,padding=None):
        super(Conv3D, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        if padding == None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding

        self.conv = nn.Conv3d(self.input_channels, self.output_channels, self.kernel_size, 1, self.padding)
        # Xavier Initialization
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, input):
        return self.conv(input)




class ConvLSTM2D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM2D, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels,
                              4 * self.hidden_channels,
                              self.kernel_size,
                              1,
                              self.padding)

        # Xavier Initialization
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, input, cur_state):
        hidden_state, cell_state = cur_state
        input_hidden_state = torch.cat((input, hidden_state), dim=1)
        conv_outputs = self.conv(input_hidden_state)

        f, i, c, o = torch.split(conv_outputs, self.hidden_channels, dim=1)
        # 避免就地操作（inplace）
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c = torch.tanh(c)

        new_cell_state = cell_state * f + i * c
        cell_state = new_cell_state
        hidden_state = o * torch.tanh(new_cell_state)

        return hidden_state, cell_state

