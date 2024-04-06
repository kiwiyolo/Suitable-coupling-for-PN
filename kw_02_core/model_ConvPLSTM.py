import numpy as np
import torch
from pysteps import motion, nowcasts
from .nowcasting_module import *
from .utils import *
import os

# 获取当前脚本文件所在的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 构建上一级文件夹的路径
parent_path = os.path.dirname(current_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configs = json.load(open(parent_path + '/config_NcN.json', 'r', encoding='utf-8', errors='ignore'))  # 加载模拟配置json


class Evolution_Network(nn.Module):
    def __init__(self, n_channels, n_classes, base_c=64, bilinear=True):
        super(Evolution_Network, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c * 1, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c * 1, bilinear)
        self.outc = OutConv(base_c * 1, n_classes)
        self.gamma = nn.Parameter(torch.zeros(1, n_classes, 1, 1), requires_grad=True)

        self.up1_v = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2_v = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3_v = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4_v = Up(base_c * 2, base_c * 1, bilinear)
        self.outc_v = OutConv(base_c * 1, n_classes * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) * self.gamma

        v = self.up1_v(x5, x4)
        v = self.up2_v(v, x3)
        v = self.up3_v(v, x2)
        v = self.up4_v(v, x1)
        v = self.outc_v(v)
        return x, v


class ConvPLSTM(nn.Module):
    def __init__(self, convlstm_input_channels, convlstm_hidden_channels, convlstm_kernel_size,
                 conv3d_input_channels, conv3d_output_channels, conv3d_kernel_size, pres=False):
        super(ConvPLSTM, self).__init__()
        if pres.lower() == "true":
            self.pres = True
        else:
            self.pres = False
        print('---PSU调用状态：%s---' % pres)

        self.bn = nn.BatchNorm3d(convlstm_input_channels)

        self.downsampling1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.convlstm1 = ConvLSTM2D(convlstm_input_channels, convlstm_hidden_channels[0], convlstm_kernel_size)
        self.conv3d1 = Conv3D(conv3d_input_channels, conv3d_output_channels[0], conv3d_kernel_size)

        self.crop3d1 = Cropping3D(cropping=((6, 0), (0, 0), (0, 0)))
        self.downsampling2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.convlstm2 = ConvLSTM2D(conv3d_output_channels[0], convlstm_hidden_channels[1], convlstm_kernel_size)
        self.conv3d2 = Conv3D(convlstm_hidden_channels[1], conv3d_output_channels[1], conv3d_kernel_size)

        self.crop3d2 = Cropping3D(cropping=((4, 0), (0, 0), (0, 0)))
        self.downsampling3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.convlstm3 = ConvLSTM2D(conv3d_output_channels[1], convlstm_hidden_channels[2], convlstm_kernel_size)
        self.conv3d3 = Conv3D(convlstm_hidden_channels[2], conv3d_output_channels[2], conv3d_kernel_size)

        self.convlstm4 = ConvLSTM2D(conv3d_output_channels[2], convlstm_hidden_channels[3], convlstm_kernel_size)
        self.conv3d4 = Conv3D(convlstm_hidden_channels[3], conv3d_output_channels[3], conv3d_kernel_size, padding=(0, 1, 1))
        self.upsampling1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)

        self.conv3d5 = Conv3D(convlstm_hidden_channels[3], conv3d_output_channels[4], conv3d_kernel_size, padding=(0, 1, 1))
        self.convlstm5 = ConvLSTM2D(conv3d_output_channels[4]*2, convlstm_hidden_channels[4], convlstm_kernel_size)
        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)

        self.conv3d6 = Conv3D(convlstm_hidden_channels[4], conv3d_output_channels[5], conv3d_kernel_size, padding=(0, 1, 1))
        self.convlstm6 = ConvLSTM2D(conv3d_output_channels[5]*2, convlstm_hidden_channels[5], convlstm_kernel_size)
        self.upsampling3 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)



        self.conv3d7 = Conv3D(conv3d_output_channels[5], conv3d_output_channels[6], conv3d_kernel_size)
        if self.pres == True:
            self.psu = PSU_one(640, 640)

    def forward(self, input):
        batch_size, input_channels, seq_length, height, width = input.shape
        in_max = torch.max(input).item()
        print("in最大值：：%f" % in_max)
        # unit1
        input = self.bn(input)  # 进行批次归一化处理
        dsample1 = self.downsampling1(input)
        hidden_state1 = cell_state1 = torch.zeros(batch_size, self.convlstm1.hidden_channels,
                                                  dsample1.size(3), dsample1.size(4)).to(input.device)
        convlstm1_outputs = []
        for t in range(seq_length):
            hidden_state1, cell_state1 = self.convlstm1(dsample1[:, :, t], [hidden_state1, cell_state1])
            convlstm1_outputs.append(hidden_state1)
        del hidden_state1, cell_state1
        # 如果传递最后的hidden_state的话：
        # conv3d_input1 = convlstm1_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input1 = torch.stack(convlstm1_outputs, dim=2)
        del convlstm1_outputs  # 释放变量内存
        conv3d_output1 = self.conv3d1(conv3d_input1)
        # conv3d_output1 = self.upsampling1(conv3d_output1)
        del conv3d_input1
        c1 = self.crop3d1(conv3d_output1)

        # unit2
        dsample2 = self.downsampling2(conv3d_output1)
        hidden_state2 = cell_state2 = torch.zeros(batch_size, self.convlstm2.hidden_channels,
                                                  dsample2.size(3), dsample2.size(4)).to(input.device)
        del conv3d_output1
        convlstm2_outputs = []
        for t in range(seq_length):
            hidden_state2, cell_state2 = self.convlstm2(dsample2[:, :, t], [hidden_state2, cell_state2])
            convlstm2_outputs.append(hidden_state2)

        del hidden_state2, cell_state2  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input2 = torch.stack(convlstm2_outputs, dim=2)
        del convlstm2_outputs  # 释放变量内存

        conv3d_output2 = self.conv3d2(conv3d_input2)
        # conv3d_output2 = self.upsampling2(conv3d_output2)
        del conv3d_input2
        c2 = self.crop3d2(conv3d_output2)

        # unit3
        dsample3 = self.downsampling3(conv3d_output2)
        hidden_state3 = cell_state3 = torch.zeros(batch_size, self.convlstm3.hidden_channels,
                                                  dsample3.size(3), dsample3.size(4)).to(input.device)
        del conv3d_output2
        convlstm3_outputs = []
        for t in range(seq_length):
            hidden_state3, cell_state3 = self.convlstm3(dsample3[:, :, t], [hidden_state3, cell_state3])
            convlstm3_outputs.append(hidden_state3)

        del hidden_state3, cell_state3  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input3 = torch.stack(convlstm3_outputs, dim=2)
        del convlstm3_outputs  # 释放变量内存

        conv3d_output3 = self.conv3d3(conv3d_input3)
        # conv3d_output2 = self.upsampling2(conv3d_output2)
        del conv3d_input3

        # unit4
        hidden_state4 = cell_state4 = torch.zeros(batch_size, self.convlstm4.hidden_channels,
                                                  conv3d_output3.size(3), conv3d_output3.size(4)).to(input.device)
        convlstm4_outputs = []
        for t in range(seq_length):
            hidden_state4, cell_state4 = self.convlstm4(conv3d_output3[:, :, t], [hidden_state4, cell_state4])
            convlstm4_outputs.append(hidden_state4)

        del hidden_state4, cell_state4  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input4 = torch.stack(convlstm4_outputs, dim=2)
        del convlstm4_outputs  # 释放变量内存

        conv3d_output4 = self.conv3d4(conv3d_input4)

        del conv3d_input4
        usample1 = self.upsampling1(conv3d_output4)



        # unit5
        conv3d_output5 = self.conv3d5(usample1)
        conv3d_output5 = torch.cat((c2, conv3d_output5), dim=1)

        hidden_state5 = cell_state5 = torch.zeros(batch_size, self.convlstm5.hidden_channels,
                                                  conv3d_output5.size(3), conv3d_output5.size(4)).to(input.device)
        del conv3d_output4
        convlstm5_outputs = []
        for t in range(seq_length - 4):
            hidden_state5, cell_state5 = self.convlstm5(conv3d_output5[:, :, t], [hidden_state5, cell_state5])
            convlstm5_outputs.append(hidden_state5)

        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        convlstm5_outputs = torch.stack(convlstm5_outputs, dim=2)

        usample2 = self.upsampling2(convlstm5_outputs)

        # unit6
        conv3d_output6 = self.conv3d6(usample2)
        conv3d_output6 = torch.cat((c1, conv3d_output6), dim=1)
        hidden_state6 = cell_state6 = torch.zeros(batch_size, self.convlstm6.hidden_channels,
                                                  conv3d_output6.size(3), conv3d_output6.size(4)).to(input.device)
        del conv3d_output5
        convlstm6_outputs = []
        for t in range(seq_length - 6):
            hidden_state6, cell_state6 = self.convlstm6(conv3d_output6[:, :, t], [hidden_state6, cell_state6])
            convlstm6_outputs.append(hidden_state6)
        # 如果传递最后的hidden_state的话：
        # conv3d_input6 = convlstm6_outputs[-1][:, :, np.newaxis, :, :]
        # 如果传递序列（包括各层的hidden_state）的话：
        convlstm6_outputs = torch.stack(convlstm6_outputs, dim=2)
        usample3 = self.upsampling3(convlstm6_outputs)


        conv3d_output7 = self.conv3d7(usample3)
        if self.pres == True:
            # 调用光流法增加预感门
            conv3d_output7 = self.psu(conv3d_output7, input)

        max_value = torch.max(conv3d_output7).item()  # conv3d_output3最大值
        print("conv3d_output7最大值：%f" % max_value)
        return conv3d_output7


class ConvPLSTM_end(nn.Module):
    def __init__(self, convlstm_input_channels, convlstm_hidden_channels, convlstm_kernel_size,
                 conv3d_input_channels, conv3d_output_channels, conv3d_kernel_size, pres=False):
        super(ConvPLSTM_end, self).__init__()

        self.configs = configs
        if pres.lower() == "true":
            self.pres = True
        else:
            self.pres = False
        print('---PSU调用状态：%s---' % pres)
        sample_tensor = torch.zeros(1, 1, self.configs["model"]["img_height"], self.configs["model"]["img_width"])
        self.grid = make_grid(sample_tensor)

        self.bn = nn.BatchNorm3d(convlstm_input_channels)
        self.downsampling1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.convlstm1 = ConvLSTM2D(convlstm_input_channels, convlstm_hidden_channels[0], convlstm_kernel_size)
        self.conv3d1 = Conv3D(conv3d_input_channels, conv3d_output_channels[0], conv3d_kernel_size)
        self.bn1 = nn.BatchNorm3d(conv3d_output_channels[0])

        self.downsampling2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.convlstm2 = ConvLSTM2D(conv3d_output_channels[0], convlstm_hidden_channels[1], convlstm_kernel_size)
        self.conv3d2 = Conv3D(convlstm_hidden_channels[1], conv3d_output_channels[1], conv3d_kernel_size)
        self.bn2 = nn.BatchNorm3d(conv3d_output_channels[1])

        self.downsampling3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.convlstm3 = ConvLSTM2D(conv3d_output_channels[1], convlstm_hidden_channels[2], convlstm_kernel_size)
        self.conv3d3 = Conv3D(convlstm_hidden_channels[2], conv3d_output_channels[2], conv3d_kernel_size,
                              padding=(0, 1, 1))
        self.bn3 = nn.BatchNorm3d(conv3d_output_channels[2])

        self.convlstm4 = ConvLSTM2D(conv3d_output_channels[2], convlstm_hidden_channels[3], convlstm_kernel_size)
        self.conv3d4 = Conv3D(convlstm_hidden_channels[3], conv3d_output_channels[3], conv3d_kernel_size,
                              padding=(0, 1, 1))
        # self.downsampling2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.bn4 = nn.BatchNorm3d(conv3d_output_channels[3])

        self.upsampling1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.convlstm5 = ConvLSTM2D(conv3d_output_channels[3], convlstm_hidden_channels[4], convlstm_kernel_size)
        self.conv3d5 = Conv3D(convlstm_hidden_channels[4], conv3d_output_channels[4], conv3d_kernel_size,
                              padding=(0, 1, 1))
        # self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.bn5 = nn.BatchNorm3d(conv3d_output_channels[4])

        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.convlstm6 = ConvLSTM2D(conv3d_output_channels[4], convlstm_hidden_channels[5], convlstm_kernel_size)
        self.conv3d6 = Conv3D(convlstm_hidden_channels[5], conv3d_output_channels[5], conv3d_kernel_size)
        self.bn6 = nn.BatchNorm3d(conv3d_output_channels[5])

        self.upsampling3 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.conv3d7 = Conv3D(conv3d_output_channels[5], conv3d_output_channels[6], conv3d_kernel_size)

        if self.pres == True:
            self.motion = motion_net(self.configs["data"]["input_length"], self.configs["data"]["n_leadtimes"],
                                     base_c=8)

        self.conv3d8 = Conv3D(3, conv3d_output_channels[6], conv3d_kernel_size)

    def forward(self, input):
        batch_size, input_channels, seq_length, height, width = input.shape

        # unit1
        input = self.bn(input)  # 进行批次归一化处理
        dsample1 = self.downsampling1(input)
        hidden_state1 = cell_state1 = torch.zeros(batch_size, self.convlstm1.hidden_channels,
                                                  dsample1.size(3), dsample1.size(4)).to(input.device)
        convlstm1_outputs = []
        for t in range(seq_length):
            hidden_state1, cell_state1 = self.convlstm1(dsample1[:, :, t], [hidden_state1, cell_state1])
            convlstm1_outputs.append(hidden_state1)
        del hidden_state1, cell_state1
        # 如果传递最后的hidden_state的话：
        # conv3d_input1 = convlstm1_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input1 = torch.stack(convlstm1_outputs, dim=2)
        del convlstm1_outputs  # 释放变量内存
        conv3d_output1 = self.conv3d1(conv3d_input1)
        # conv3d_output1 = self.upsampling1(conv3d_output1)
        del conv3d_input1
        conv3d_output1 = self.bn1(conv3d_output1)

        # unit2
        dsample2 = self.downsampling2(conv3d_output1)
        hidden_state2 = cell_state2 = torch.zeros(batch_size, self.convlstm2.hidden_channels,
                                                  dsample2.size(3), dsample2.size(4)).to(input.device)
        del conv3d_output1
        convlstm2_outputs = []
        for t in range(seq_length):
            hidden_state2, cell_state2 = self.convlstm2(dsample2[:, :, t], [hidden_state2, cell_state2])
            convlstm2_outputs.append(hidden_state2)

        del hidden_state2, cell_state2  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input2 = torch.stack(convlstm2_outputs, dim=2)
        del convlstm2_outputs  # 释放变量内存

        conv3d_output2 = self.conv3d2(conv3d_input2)
        # conv3d_output2 = self.upsampling2(conv3d_output2)
        del conv3d_input2
        conv3d_output2 = self.bn2(conv3d_output2)

        # unit3
        dsample3 = self.downsampling3(conv3d_output2)
        hidden_state3 = cell_state3 = torch.zeros(batch_size, self.convlstm3.hidden_channels,
                                                  dsample3.size(3), dsample3.size(4)).to(input.device)
        del conv3d_output2
        convlstm3_outputs = []
        for t in range(seq_length):
            hidden_state3, cell_state3 = self.convlstm3(dsample3[:, :, t], [hidden_state3, cell_state3])
            convlstm3_outputs.append(hidden_state3)

        del hidden_state3, cell_state3  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input3 = torch.stack(convlstm3_outputs, dim=2)
        del convlstm3_outputs  # 释放变量内存

        conv3d_output3 = self.conv3d3(conv3d_input3)
        # conv3d_output2 = self.upsampling2(conv3d_output2)
        del conv3d_input3
        conv3d_output3 = self.bn3(conv3d_output3)

        # unit4
        hidden_state4 = cell_state4 = torch.zeros(batch_size, self.convlstm4.hidden_channels,
                                                  conv3d_output3.size(3), conv3d_output3.size(4)).to(input.device)
        convlstm4_outputs = []
        for t in range(seq_length - 2):
            hidden_state4, cell_state4 = self.convlstm4(conv3d_output3[:, :, t], [hidden_state4, cell_state4])
            convlstm4_outputs.append(hidden_state4)

        del hidden_state4, cell_state4  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input4 = torch.stack(convlstm4_outputs, dim=2)
        del convlstm4_outputs  # 释放变量内存

        conv3d_output4 = self.conv3d4(conv3d_input4)

        del conv3d_input4
        conv3d_output4 = self.bn4(conv3d_output4)

        # unit5
        usample1 = self.upsampling1(conv3d_output4)
        hidden_state5 = cell_state5 = torch.zeros(batch_size, self.convlstm5.hidden_channels,
                                                  usample1.size(3), usample1.size(4)).to(input.device)
        del conv3d_output4
        convlstm5_outputs = []
        for t in range(seq_length - 4):
            hidden_state5, cell_state5 = self.convlstm5(usample1[:, :, t], [hidden_state5, cell_state5])
            convlstm5_outputs.append(hidden_state5)

        del hidden_state5, cell_state5  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input5 = torch.stack(convlstm5_outputs, dim=2)
        del convlstm5_outputs  # 释放变量内存

        conv3d_output5 = self.conv3d5(conv3d_input5)
        # conv3d_output2 = self.upsampling2(conv3d_output2)
        del conv3d_input5
        conv3d_output5 = self.bn5(conv3d_output5)

        # unit6
        usample2 = self.upsampling2(conv3d_output5)
        hidden_state6 = cell_state6 = torch.zeros(batch_size, self.convlstm6.hidden_channels,
                                                  usample2.size(3), usample2.size(4)).to(input.device)
        del conv3d_output5
        convlstm6_outputs = []
        for t in range(seq_length - 6):
            hidden_state6, cell_state6 = self.convlstm6(usample2[:, :, t], [hidden_state6, cell_state6])
            convlstm6_outputs.append(hidden_state6)
        del hidden_state6, cell_state6  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input6 = convlstm6_outputs[-1][:, :, np.newaxis, :, :]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input6 = torch.stack(convlstm6_outputs, dim=2)

        del convlstm6_outputs  # 释放变量内存
        conv3d_output6 = self.conv3d6(conv3d_input6)
        conv3d_output6 = self.bn6(conv3d_output6)

        # unit7
        usample3 = self.upsampling3(conv3d_output6)
        del conv3d_input6, conv3d_output6
        conv3d_output7 = self.conv3d7(usample3)
        # TODO 构建端到端的预感门
        n_leadtimes = configs["data"]["n_leadtimes"]
        if self.pres == True:
            # 调用光流法增加预感门
            input_frames = input.reshape(batch_size, seq_length, height, width)
            motion = self.motion(input_frames)
            motion_ = motion.reshape(batch_size, n_leadtimes, 2, height, width)
            series = []
            last_frames = input[:, 0, (seq_length - 1):seq_length, :, :]
            grid = self.grid.repeat(batch_size, 1, 1, 1)
            for i in range(n_leadtimes):
                last_frames = warp(last_frames, motion_[:, i], grid.cuda(), mode="nearest", padding_mode="border")
                intensity_ = conv3d_output7[:, :, i]
                last_frames = last_frames + intensity_
                series.append(last_frames)
            evo_result = torch.cat(series, dim=1)

            evo_result = evo_result.reshape(batch_size, 1, n_leadtimes, height, width)

        max_value = torch.max(evo_result).item()  # conv3d_output3最大值
        print("evo_result最大值：%f" % max_value)
        return evo_result


class UNET_Like_3D(nn.Module):
    def __init__(self, input_shape, start_filter, conv_kernel_size, activation, dr_rate, deconv_kernel_size,
                 pres=False):
        super(UNET_Like_3D, self).__init__()
        if pres.lower() == "true":
            self.pres = True
        else:
            self.pres = False
        print('---PSU调用状态：%s---' % pres)

        # Extracting depth from input_shape for sequence handling
        self.in_channels = input_shape[1]
        self.depth = input_shape[2]

        # Encoding
        self.enc1 = self._make_enc_layer(self.in_channels, start_filter, 1, conv_kernel_size, activation)
        self.mp1 = self._make_maxp_layer(start_filter, dr_rate)
        self.enc2 = self._make_enc_layer(start_filter, start_filter * 2, 1, conv_kernel_size, activation)
        self.mp2 = self._make_maxp_layer(start_filter * 2, dr_rate)
        self.enc3 = self._make_enc_layer(start_filter * 2, start_filter * 4, 1, conv_kernel_size, activation)
        self.mp3 = self._make_maxp_layer(start_filter * 4, dr_rate)
        self.enc4 = self._make_enc_layer(start_filter * 4, start_filter * 8, 1, conv_kernel_size,
                                         activation)  # Depth becomes 1
        self.mp4 = self._make_maxp_layer(start_filter * 8, dr_rate)

        # Decoding
        self.midnet = self._make_dec_layer(start_filter * 8, start_filter * 8, 1, deconv_kernel_size, activation,
                                           dr_rate)
        self.crop3d = Cropping3D(cropping=((0, 0), (1, 0), (1, 0)))
        self.crop3d1 = Cropping3D(cropping=((2, 0), (0, 0), (0, 0)))
        self.dec1 = self._make_dec_layer(start_filter * 16, start_filter * 4, 1, deconv_kernel_size, activation,
                                         dr_rate)
        self.crop3d2 = Cropping3D(cropping=((4, 0), (0, 0), (0, 0)))
        self.dec2 = self._make_dec_layer(start_filter * 8, start_filter * 2, 1, deconv_kernel_size, activation, dr_rate)
        self.crop3d3 = Cropping3D(cropping=((6, 0), (0, 0), (0, 0)))
        self.dec3 = self._make_dec2_layer(start_filter * 4, start_filter, 1, deconv_kernel_size, activation, dr_rate)
        self.crop3d4 = Cropping3D(cropping=((6, 0), (0, 0), (0, 0)))
        self.dec4 = self._make_dec3_layer(start_filter * 2, start_filter // 2, activation, dr_rate)

        self.final_conv = nn.Conv3d(start_filter // 2, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        if self.pres == True:
            self.psu = PSU(640, 640)

    def forward(self, x):
        # Encoding
        x1 = self.enc1(x)
        m1 = self.mp1(x1)
        x2 = self.enc2(m1)
        m2 = self.mp2(x2)
        x3 = self.enc3(m2)
        m3 = self.mp3(x3)
        x4 = self.enc4(m3)
        m4 = self.mp4(x4)

        d = self.midnet(m4)

        cd1 = self.crop3d(d)
        cx4 = self.crop3d1(x4)
        c1 = self._make_conc_layer(cd1, cx4)
        d1 = self.dec1(c1)

        cd2 = self.crop3d(d1)
        cx3 = self.crop3d2(x3)
        c2 = self._make_conc_layer(cd2, cx3)
        d2 = self.dec2(c2)

        cd3 = self.crop3d(d2)
        cx2 = self.crop3d3(x2)
        c3 = self._make_conc_layer(cd3, cx2)
        d3 = self.dec3(c3)

        cd4 = self.crop3d(d3)
        cx1 = self.crop3d4(x1)
        c4 = self._make_conc_layer(cd4, cx1)
        d4 = self.dec4(c4)

        out = self.final_conv(d4)
        if self.pres == True:
            # 调用光流法增加预感门
            out = self.psu(out, input)
        max_value = torch.max(out).item()  # conv3d_output3最大值
        print("out最大值：%f" % max_value)
        return out

    def _make_enc_layer(self, in_channels, out_channels, depth, conv_kernel_size, activation):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(depth, conv_kernel_size, conv_kernel_size),
                      padding=(0, 1, 1)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(depth, conv_kernel_size, conv_kernel_size),
                      padding=(0, 1, 1)),
            nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        )

    def _make_maxp_layer(self, out_channels, dr_rate):
        return nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dr_rate)
        )

    def _make_conc_layer(self, d, e):
        # cropped_x_conv2_b=Cropping3D(cropping=((2, 0), (0, 0), (0, 0)))(d)

        concatenated = torch.cat((d, e), dim=1)
        return concatenated

    def _make_dec_layer(self, in_channels, out_channels, depth, deconv_kernel_size, activation, dr_rate):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(depth, deconv_kernel_size, deconv_kernel_size),
                               stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dr_rate)
        )

    def _make_dec2_layer(self, in_channels, out_channels, depth, deconv_kernel_size, activation, dr_rate):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(depth, deconv_kernel_size, deconv_kernel_size),
                               stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dr_rate)
        )

    def _make_dec3_layer(self, in_channels, out_channels, activation, dr_rate):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dr_rate)
        )


class UNET_mine(nn.Module):
    def __init__(self, input_shape, start_filter, conv_kernel_size, activation, dr_rate, deconv_kernel_size,
                 pres=False):
        super(UNET_mine, self).__init__()
        if pres.lower() == "true":
            self.pres = True
        else:
            self.pres = False
        print('---PSU调用状态：%s---' % pres)

        # Extracting depth from input_shape for sequence handling
        self.in_channels = input_shape[1]
        self.depth = input_shape[2]

        # Encoding
        self.enc1 = self._make_enc_layer(self.in_channels, start_filter, 1, conv_kernel_size, activation)
        self.mp1 = self._make_maxp_layer(start_filter, dr_rate)
        self.enc2 = self._make_enc_layer(start_filter, start_filter * 2, 1, conv_kernel_size, activation)
        self.mp2 = self._make_maxp_layer(start_filter * 2, dr_rate)
        self.enc3 = self._make_enc_layer(start_filter * 2, start_filter * 4, 1, conv_kernel_size, activation)
        self.mp3 = self._make_maxp_layer(start_filter * 4, dr_rate)
        self.enc4 = self._make_enc_layer(start_filter * 4, start_filter * 8, 1, conv_kernel_size,
                                         activation)  # Depth becomes 1
        self.mp4 = self._make_maxp_layer(start_filter * 8, dr_rate)

        # Decoding
        self.midnet = self._make_dec_layer(start_filter * 8, start_filter * 16, 1, deconv_kernel_size, activation,
                                           dr_rate)
        self.crop3d = Cropping3D(cropping=((0, 0), (1, 0), (1, 0)))
        self.dec1 = self._make_dec_layer(start_filter * 16, start_filter * 8, 1, deconv_kernel_size, activation,
                                         dr_rate)
        self.dec2 = self._make_dec_layer(start_filter * 8, start_filter * 4, 1, deconv_kernel_size, activation, dr_rate)
        self.dec3 = self._make_dec2_layer(start_filter * 4, start_filter * 2, 1, deconv_kernel_size, activation,
                                          dr_rate)
        self.dec4 = self._make_dec3_layer(start_filter * 2, start_filter // 2, activation, dr_rate)

        self.final_conv = nn.Conv3d(start_filter // 2, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        if self.pres == True:
            self.psu = PSU(640, 640)

    def forward(self, x):
        # Encoding
        x1 = self.enc1(x)
        m1 = self.mp1(x1)
        x2 = self.enc2(m1)
        m2 = self.mp2(x2)
        x3 = self.enc3(m2)
        m3 = self.mp3(x3)
        x4 = self.enc4(m3)
        m4 = self.mp4(x4)

        d = self.midnet(m4)

        cd1 = self.crop3d(d)
        d1 = self.dec1(cd1)

        cd2 = self.crop3d(d1)
        d2 = self.dec2(cd2)

        cd3 = self.crop3d(d2)
        d3 = self.dec3(cd3)

        cd4 = self.crop3d(d3)
        d4 = self.dec4(cd4)

        out = self.final_conv(d4)
        if self.pres == True:
            # 调用光流法增加预感门
            out = self.psu(out, x)
        max_value = torch.max(out).item()  # conv3d_output3最大值
        print("out最大值：%f" % max_value)
        return out

    def _make_enc_layer(self, in_channels, out_channels, depth, conv_kernel_size, activation):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(depth, conv_kernel_size, conv_kernel_size),
                      padding=(0, 1, 1)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(depth, conv_kernel_size, conv_kernel_size),
                      padding=(0, 1, 1)),
            nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        )

    def _make_maxp_layer(self, out_channels, dr_rate):
        return nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dr_rate)
        )

    def _make_dec_layer(self, in_channels, out_channels, depth, deconv_kernel_size, activation, dr_rate):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(depth, deconv_kernel_size, deconv_kernel_size),
                               stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dr_rate)
        )

    def _make_dec2_layer(self, in_channels, out_channels, depth, deconv_kernel_size, activation, dr_rate):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(depth, deconv_kernel_size, deconv_kernel_size),
                               stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dr_rate)
        )

    def _make_dec3_layer(self, in_channels, out_channels, activation, dr_rate):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dr_rate)
        )


class NET_ConvLSTM(nn.Module):
    def __init__(self, convlstm_input_channels, convlstm_hidden_channels, convlstm_kernel_size,
                 conv3d_input_channels, conv3d_output_channels, conv3d_kernel_size, pres=False):
        super(NET_ConvLSTM, self).__init__()
        if pres.lower() == "true":
            self.pres = True
        else:
            self.pres = False
        print('---PSU调用状态：%s---' % pres)

        self.downsampling1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.convlstm1 = ConvLSTM2D(convlstm_input_channels, convlstm_hidden_channels[0], convlstm_kernel_size)
        self.conv3d1 = Conv3D(conv3d_input_channels, conv3d_output_channels[0], conv3d_kernel_size, padding=(0, 1, 1))

        self.downsampling2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.convlstm2 = ConvLSTM2D(conv3d_output_channels[0], convlstm_hidden_channels[1], convlstm_kernel_size)
        self.conv3d2 = Conv3D(convlstm_hidden_channels[1], conv3d_output_channels[1], conv3d_kernel_size,
                              padding=(0, 1, 1))

        self.upsampling1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.convlstm3 = ConvLSTM2D(conv3d_output_channels[1], convlstm_hidden_channels[2], convlstm_kernel_size)
        self.conv3d3 = Conv3D(convlstm_hidden_channels[2], conv3d_output_channels[2], conv3d_kernel_size,
                              padding=(0, 1, 1))

        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.conv3d4 = Conv3D(conv3d_output_channels[2], conv3d_output_channels[3], conv3d_kernel_size)

        if self.pres == True:
            self.psu = PSU(640, 640)

    def forward(self, input):
        batch_size, input_channels, seq_length, height, width = input.shape

        # unit1
        dsample1 = self.downsampling1(input)
        hidden_state1 = cell_state1 = torch.zeros(batch_size, self.convlstm1.hidden_channels,
                                                  dsample1.size(3), dsample1.size(4)).to(input.device)
        convlstm1_outputs = []
        for t in range(seq_length):
            hidden_state1, cell_state1 = self.convlstm1(dsample1[:, :, t], [hidden_state1, cell_state1])
            convlstm1_outputs.append(hidden_state1)
        del hidden_state1, cell_state1
        # 如果传递最后的hidden_state的话：
        # conv3d_input1 = convlstm1_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input1 = torch.stack(convlstm1_outputs, dim=2)
        del convlstm1_outputs  # 释放变量内存
        conv3d_output1 = self.conv3d1(conv3d_input1)
        # conv3d_output1 = self.upsampling1(conv3d_output1)
        del conv3d_input1

        # unit2
        dsample2 = self.downsampling2(conv3d_output1)
        hidden_state2 = cell_state2 = torch.zeros(batch_size, self.convlstm2.hidden_channels,
                                                  dsample2.size(3), dsample2.size(4)).to(input.device)
        del conv3d_output1
        convlstm2_outputs = []
        for t in range(seq_length - 2):
            hidden_state2, cell_state2 = self.convlstm2(dsample2[:, :, t], [hidden_state2, cell_state2])
            convlstm2_outputs.append(hidden_state2)

        del hidden_state2, cell_state2  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input2 = torch.stack(convlstm2_outputs, dim=2)
        del convlstm2_outputs  # 释放变量内存

        conv3d_output2 = self.conv3d2(conv3d_input2)
        # conv3d_output2 = self.upsampling2(conv3d_output2)
        del conv3d_input2

        # unit3
        usample1 = self.upsampling1(conv3d_output2)
        hidden_state3 = cell_state3 = torch.zeros(batch_size, self.convlstm3.hidden_channels,
                                                  usample1.size(3), usample1.size(4)).to(input.device)
        del conv3d_output2
        convlstm3_outputs = []
        for t in range(

        ):
            hidden_state3, cell_state3 = self.convlstm3(usample1[:, :, t], [hidden_state3, cell_state3])
            convlstm3_outputs.append(hidden_state3)

        del hidden_state3, cell_state3  # 释放变量内存
        # 如果传递最后的hidden_state的话：
        # conv3d_input2 = convlstm2_outputs[-1]
        # 如果传递序列（包括各层的hidden_state）的话：
        conv3d_input3 = torch.stack(convlstm3_outputs, dim=2)
        del convlstm3_outputs  # 释放变量内存

        conv3d_output3 = self.conv3d3(conv3d_input3)
        # conv3d_output2 = self.upsampling2(conv3d_output2)
        del conv3d_input3

        # unit4
        usample2 = self.upsampling2(conv3d_output3)
        del conv3d_output3
        conv3d_output4 = self.conv3d4(usample2)
        if self.pres == True:
            # 调用光流法增加预感门
            conv3d_output4 = self.psu(conv3d_output4, input)

        max_value = torch.max(conv3d_output4).item()  # conv3d_output3最大值
        print("conv3d_output7最大值：%f" % max_value)

        return conv3d_output4


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
        nn.init.kaiming_normal_(self.conv.weight)
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
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, input):
        return self.conv(input)


class motion_net(nn.Module):
    def __init__(self, n_channels, n_classes, base_c=64, bilinear=True):
        super(motion_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c * 1, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1_v = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2_v = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3_v = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4_v = Up(base_c * 2, base_c * 1, bilinear)
        self.outc_v = OutConv(base_c * 1, n_classes * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        v = self.up1_v(x5, x4)
        v = self.up2_v(v, x3)
        v = self.up3_v(v, x2)
        v = self.up4_v(v, x1)
        v = self.outc_v(v)
        return v


class PSU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PSU, self).__init__()
        # TODO self.num_layers = num_layers

        # 需要更新信息（预感门交换）的权重和偏置p
        self.w_xp = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_ep = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_p = nn.Parameter(torch.Tensor(input_size))

        # 输入到预感门（预感信息）的权重和偏置i
        self.w_xm = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_em = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_m = nn.Parameter(torch.Tensor(input_size))

        # 初始化权重和偏置
        self.init_weights()

    def init_weights(self):
        # 初始化权重矩阵为xavier_uniform，偏置向量为全零
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.zeros_(param)
            elif 'w' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, input):
        """
        :param x: 前面模型训练得到的过去信息的记忆
        :param input: 输入的解释因子，过去的信息
        :return:
        """
        n_leadtimes = configs["data"]["n_leadtimes"]
        batch_size = configs["training"]["batch_size"]
        # Lucas-Kanade (LK)
        oflow_method = motion.get_method("LK")
        out = []
        for batch in range(len(input)):
            input_sample = input[batch].cpu().detach().numpy()[0]
            V = oflow_method(input_sample, verbose=False)

            # Extrapolate the last radar observation
            extrapolate = nowcasts.get_method("extrapolation")
            # R[~np.isfinite(R)] = metadata["zerovalue"]
            extra = extrapolate(input_sample[-1], V, n_leadtimes)
            extra = extra[np.newaxis, :, :, :]
            extra[np.isnan(extra)] = 0
            extra = torch.tensor(extra).to(device)
            batch_out = []
            for lt in range(n_leadtimes):
                x_2d = x[batch, :, lt, :, :].view(-1, x[batch].size(-1))
                extra_2d = extra[:, lt, :, :].view(-1, extra.size(-1))
                # 检测 NaN 值
                mask = torch.isnan(extra_2d)
                # 将 NaN 值替换为 0
                extra_2d = torch.where(mask, torch.tensor(0.0), extra_2d)
                p = torch.sigmoid(self.w_xp@x_2d + self.w_ep@extra_2d )
                m = torch.tanh(self.w_xm@x_2d + self.w_em@extra_2d)
                h = p*x_2d + p*m
                batch_out.append(h)
            batch_out = torch.stack(batch_out, 0)
            out.append(batch_out)
        out = torch.stack(out, 0)
        out = out.unsqueeze(1)
        return out


class PSU_one(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PSU_one, self).__init__()
        # TODO self.num_layers = num_layers

        # 需要更新信息（预感门交换）的权重和偏置p
        self.w_xp = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_ep = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_p = nn.Parameter(torch.Tensor(input_size))

        # 输入到预感门（预感信息）的权重和偏置i
        self.w_xm = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_em = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_m = nn.Parameter(torch.Tensor(input_size))

        # 初始化权重和偏置
        self.init_weights()

    def init_weights(self):
        # 初始化权重矩阵为xavier_uniform，偏置向量为全零
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.zeros_(param)
            elif 'w' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, input):
        """
        :param x: 前面模型训练得到的过去信息的记忆
        :param input: 输入的解释因子，过去的信息
        :return:
        """
        n_leadtimes = configs["data"]["n_leadtimes"]
        # Lucas-Kanade (LK)
        oflow_method = motion.get_method("LK")
        out = []
        for batch in range(len(input)):
            input_sample = input[batch].cpu().detach().numpy()[0]
            V = oflow_method(input_sample, verbose=False)

            # Extrapolate the last radar observation
            extrapolate = nowcasts.get_method("extrapolation")
            # R[~np.isfinite(R)] = metadata["zerovalue"]
            extra = extrapolate(input_sample[-1], V, n_leadtimes)
            extra = extra[np.newaxis, :, :, :]
            extra[np.isnan(extra)] = 0
            extra = torch.tensor(extra).to(device)
            batch_out = []
            for lt in range(n_leadtimes):
                x_2d = x[batch, :, lt, :, :].view(-1, x[batch].size(-1))
                extra_2d = extra[:, lt, :, :].view(-1, extra.size(-1))
                # 检测 NaN 值
                mask = torch.isnan(extra_2d)
                # 将 NaN 值替换为 0
                extra_2d = torch.where(mask, torch.tensor(0.0), extra_2d)
                p = torch.sigmoid(self.w_xp@x_2d + self.w_ep@extra_2d)
                h = p*x_2d + (1-p)*extra_2d
                batch_out.append(h)
            batch_out = torch.stack(batch_out, 0)
            out.append(batch_out)
        out = torch.stack(out, 0)
        out = out.unsqueeze(1)
        return out


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


class Cropping3D(nn.Module):
    def __init__(self, cropping):
        super().__init__()  # 使用简化的super调用
        self.cropping = cropping

    def forward(self, x):
        return x[:, :,
               self.cropping[0][0]:-self.cropping[0][1] if self.cropping[0][1] != 0 else None,
               self.cropping[1][0]:-self.cropping[1][1] if self.cropping[1][1] != 0 else None,
               self.cropping[2][0]:-self.cropping[2][1] if self.cropping[2][1] != 0 else None]









