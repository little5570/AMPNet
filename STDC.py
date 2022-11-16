import torch

torch.cuda.empty_cache()

torch.cuda.memory_cached()

import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

# class channelAttention_fusion(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(channelAttention_fusion,self).__init__()
#         self.conv1 = nn.Conv2d(in_channels,out_channels,1)
#         self.conv2 = nn.Conv2d(out_channels *2,out_channels,1,1,0)
#         self.avg = nn.AdaptiveAvgPool2d((1,1))
#         self.max = nn.AdaptiveMaxPool2d((1,1))     
#         self.drop = nn.Dropout(0.1)
#     def forward(self,inputs):
#         inputs = self.conv1(inputs)
#         avg = self.avg(inputs)
#         maxi = self.max(inputs)
#         im = torch.cat((avg,maxi),1)
#         im = self.drop(self.conv2(im))
# # #         x = torch.sigmoid(x)
# # # 这个时候im的大小是32*1*1
# #         im = im.reshape(inputs.size(0),1,-1)
# #         im_2 = inputs.reshape((inputs.size(0),inputs.size(1),-1))
# #         for i in range(im.size(0)):
# #             if i == 0:
# #                 out = torch.mm(im[i],im_2[i])
# #                 out = out.reshape(1,1,inputs.size(2),inputs.size(3))
# #             else:
# #                 out_im = torch.mm(im[i],im_2[i])
# #                 out_im = out_im.reshape(1,1,inputs.size(2),inputs.size(3))
# #                 out = torch.cat((out,out_im),0)
# #         这个sigmoid 是cbam里面的
#         im = torch.sigmoid(im)
#         out = im * inputs
#         return out

# caf = channelAttention_fusion(1024,512)
# x = torch.rand(4,1024,24,48)
# y = caf(x)
# print(y.shape)

# torch.save(caf.state_dict(),'/powerhope/wangzhuo/STDC_seg/test_weight/caf.pth')

# class atrous_conv(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size = 3,stride = 1,dilation = 1):
#         super(atrous_conv,self).__init__()
#         self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride,dilation,dilation)
# #         self.conv2 = nn.Conv2d(in_channels,out_channels,1,1)
#         self.bn = nn.BatchNorm2d(out_channels)
#     def forward(self,inputs):
#         x = self.conv1(inputs)
#         x = self.bn(x)
# #         x = self.conv2(x)
#         return x

# class ASPP_CAM(nn.Module):
#     def __init__(self,channels):
#         super(ASPP_CAM,self).__init__()
# #         self.astrous1 = atrous_conv(channels,channels // 4,3,1,1)
#         self.astrous1 = atrous_conv(channels,channels // 4,3,1,12)
#         self.astrous2 = atrous_conv(channels,channels // 4,3,1,24)
#         self.astrous3 = atrous_conv(channels,channels // 4,3,1,36)
#         self.last_conv = ConvX(channels*5 //4,channels,1,1)
# #         self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
# #                                     ConvX(channels,channels // 4))
#         self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
#             ConvX(channels,channels // 4, 1),)
#         self.cam = channelAttention_fusion(channels,channels // 4)
#         self.drop = nn.Dropout(0.1)
# #         self.shuffle = channal_shuffle(channels // 16)
#     def forward(self,inputs):
#         x1 = self.astrous1(inputs)
#         x2 = self.astrous2(inputs)
#         x3 = self.astrous3(inputs)
#         x4 = self.avg_pool(inputs)
#         x4 = F.interpolate(x4,size = (inputs.size(2),inputs.size(3)),mode = 'bilinear',align_corners = True)
#         x5 = self.cam(inputs)
#         x6 = torch.cat((x1,x2,x3,x4,x5),1)
#         x7 = self.drop(self.last_conv(x6))
        
        
# #         out = self.shuffle(out)
#         return x7

# x = torch.rand(4,1024,24,48)
# aspp_cam = ASPP_CAM(1024)
# y = aspp_cam(x)
# print(y.shape)

# class PyramidPoolingModule(nn.Module):
#     def __init__(self,channels,pyramids = [1,2,3,6]):
#         super(PyramidPoolingModule,self).__init__()
#         self.pyramids = pyramids
#         self.dropout = nn.Dropout(0.1)
#         self.conv = ConvX(channels,channels // 4,1,1)
#         self.last_conv = ConvX(channels,channels,1,1)
#     def forward(self,input):
#         feat = input
#         height,width = input.size()[2:]
#         need = []
#         for bin_size in self.pyramids:
#             x = F.adaptive_avg_pool2d(input,output_size = bin_size)
#             x = self.conv(x)
#             x = F.interpolate(x,size = (height,width))
#             need.append(x)
#         need = torch.cat(need,1)
#         need = self.last_conv(need)
#         feat = feat + need
#         feat = self.dropout(feat)
#         return feat

class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x

class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out

#STDC2Net
class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[4,5,3], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
        super(STDCNet1446, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base*16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):
        
        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict,strict = False)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out

# STDC1Net
device = torch.device('cpu')
class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
        super(STDCNet813, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base*16))
        self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)
        
#         self.aspp_cam = ASPP_CAM(1024)
#         self.ppm = PyramidPoolingModule(1024)
#         self.cbam = channelAttention_fusion(1024,1024)
        
        

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):
        
        state_dict = torch.load(pretrain_model,map_location = 'cpu')["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
#             if k != 'linear.weight':
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict,strict = False)
#         print('device',device)
#         print('state_dict_type',type(self_state_dict))

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
#         feat_32_cbam = self.cbam(feat32)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)
            feat_32_ppm = self.conv_last(feat_32_ppm)
            
        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out