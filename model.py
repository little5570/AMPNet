import torch

torch.cuda.empty_cache()

torch.cuda.memory_cached()

import io, os, sys, types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell
def find_notebook(fullname, path=None):
    """find a notebook, given its fully qualified name and an optional path
    
    This turns "foo.bar" into "foo/bar.ipynb"
    and tries turning "Foo_Bar" into "Foo Bar" if Foo_Bar
    does not exist.
    """
    name = fullname.rsplit('.', 1)[-1]
    if not path:
        path = ['']
    for d in path:
        nb_path = os.path.join(d, name + ".ipynb")
        if os.path.isfile(nb_path):
            return nb_path
        # let import Notebook_Name find "Notebook Name.ipynb"
        nb_path = nb_path.replace("_", " ")
        if os.path.isfile(nb_path):
            return nb_path
        
class NotebookLoader(object):
    """Module Loader for Jupyter Notebooks"""
    def __init__(self, path=None):
        self.shell = InteractiveShell.instance()
        self.path = path
    
    def load_module(self, fullname):
        """import a notebook as a module"""
        path = find_notebook(fullname, self.path)
        
        print ("importing Jupyter notebook from %s" % path)
                                       
        # load the notebook object
        with io.open(path, 'r', encoding='utf-8') as f:
            nb = read(f, 4)
        
        
        # create the module and add it to sys.modules
        # if name in sys.modules:
        #    return sys.modules[name]
        mod = types.ModuleType(fullname)
        mod.__file__ = path
        mod.__loader__ = self
        mod.__dict__['get_ipython'] = get_ipython
        sys.modules[fullname] = mod
        
        # extra work to ensure that magics that would affect the user_ns
        # actually affect the notebook module's ns
        save_user_ns = self.shell.user_ns
        self.shell.user_ns = mod.__dict__
        
        try:
          for cell in nb.cells:
            if cell.cell_type == 'code':
                # transform the input to executable Python
                code = self.shell.input_transformer_manager.transform_cell(cell.source)
                # run the code in themodule
                exec(code, mod.__dict__)
        finally:
            self.shell.user_ns = save_user_ns
        return mod
class NotebookFinder(object):
    """Module finder that locates Jupyter Notebooks"""
    def __init__(self):
        self.loaders = {}
    
    def find_module(self, fullname, path=None):
        nb_path = find_notebook(fullname, path)
        if not nb_path:
            return
        
        key = path
        if path:
            # lists aren't hashable
            key = os.path.sep.join(path)
        
        if key not in self.loaders:
            self.loaders[key] = NotebookLoader(path)
        return self.loaders[key]

sys.meta_path.append(NotebookFinder())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# from nets.stdcnet import STDCNet1446, STDCNet813
from STDC import STDCNet1446, STDCNet813

# import bn

# from bn import InPlaceABNSync

# # from modules.bn import InPlaceABNSync as BatchNorm2d
# from bn import InPlaceABNSync as BatchNorm2d
# # BatchNorm2d = nn.BatchNorm2d

BatchNorm2d = nn.BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
#         self.bn = BatchNorm2d(out_chan, activation='none')
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class PyramidPoolingModule(nn.Module):
    def __init__(self,channels,pyramids = [1,2,3,6]):
        super(PyramidPoolingModule,self).__init__()
        self.pyramids = pyramids
        self.dropout = nn.Dropout(0.1)
        self.conv = ConvBNReLU(channels,channels // 4,1,1,0)
        self.last_conv = ConvBNReLU(channels,channels,1,1,0)
    def forward(self,input):
        feat = input
        height,width = input.size()[2:]
        need = []
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input,output_size = bin_size)
            x = self.conv(x)
            x = F.interpolate(x,size = (height,width))
            need.append(x)
        need = torch.cat(need,1)
        need = self.last_conv(need)
        feat = feat + need
        feat = self.dropout(feat)
        return feat

class channelAttention_fusion(nn.Module):
    def __init__(self,channels):
        super(channelAttention_fusion,self).__init__()
        self.channels = channels
        self.conv2 = nn.Sequential(nn.Conv2d(channels *2,channels,1,1,0),
                                  nn.BatchNorm2d(channels))
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))     
        self.drop = nn.Dropout(0.1)
    def forward(self,inputs):
        avg = self.avg(inputs)
        maxi = self.max(inputs)
        im = torch.cat((avg,maxi),1)
        im = self.drop(self.conv2(im))
# #         x = torch.sigmoid(x)
# # 这个时候im的大小是32*1*1
        im = im.reshape(inputs.size(0),1,-1)
        im_2 = inputs.reshape((inputs.size(0),inputs.size(1),-1))
        for i in range(im.size(0)):
            if i == 0:
                out = torch.mm(im[i],im_2[i])
                out = out.reshape(1,1,inputs.size(2),inputs.size(3))
            else:
                out_im = torch.mm(im[i],im_2[i])
                out_im = out_im.reshape(1,1,inputs.size(2),inputs.size(3))
                out = torch.cat((out,out_im),0)
        out = out * inputs
        return out

# class nor_sam(nn.Module):
#     def __init__(self,in_channels):
#         super(nor_sam,self).__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.maxpool = nn.MaxPool2d((1,1))
#         self.conv = nn.Conv2d(2*in_channels,in_channels,1,1,0)
#     def forward(self,low):
#         low_avg = self.avgpool(low)
#         low_max = self.maxpool(low)
#         avg = torch.sigmoid(low_avg * low)
#         maxi = torch.sigmoid(low_max * low)
#         low = torch.cat((avg,maxi),1)
#         low = self.conv(low)


 

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

# class AttentionRefinementModule(nn.Module):
#     def __init__(self, in_chan, out_chan, *args, **kwargs):
#         super(AttentionRefinementModule, self).__init__()
#         self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
#         self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
#         # self.bn_atten = BatchNorm2d(out_chan)
#         self.bn_atten = BatchNorm2d(out_chan)

#         self.sigmoid_atten = nn.Sigmoid()
#         self.init_weight()

#     def forward(self, x):
#         feat = self.conv(x)
#         atten = F.avg_pool2d(feat, feat.size()[2:])
#         atten = self.conv_atten(atten)
#         atten = self.bn_atten(atten)
#         atten = self.sigmoid_atten(atten)
#         out = torch.mul(feat, atten)
#         return out

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ContextPath(nn.Module):
    def __init__(self, backbone='CatNetSmall', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()
        
        self.backbone_name = backbone
#         self.cam = channelAttention_fusion(1024,1024)
        self.ppm = PyramidPoolingModule(1024)
        
        self.upconv_32 = ConvBNReLU(1024,512, ks=1, stride=1, padding=0)
        self.upconv_16 = ConvBNReLU(512,256, ks=1, stride=1, padding=0)
        self.conv16 = ConvBNReLU(1024,512, ks=1, stride=1, padding=0)
        self.upconv_8 = ConvBNReLU(256,64, ks=1, stride=1, padding=0)
        self.conv8 = ConvBNReLU(512,256, ks=1, stride=1, padding=0)
        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
#             self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
#             self.arm32 = AttentionRefinementModule(inplanes, 128)
#             self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
#             self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
#             self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
#             self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
#             self.arm32 = AttentionRefinementModule(inplanes, 128)
#             self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
#             self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
#             self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        else:
            print("backbone is not in backbone lists")
            exit(0)
        self.cam = channelAttention_fusion(1024)
        self.cam16 = channelAttention_fusion(512)
#         self.sam = nor_sam(256)
        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]

        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        feat32_ppm = self.ppm(feat32)
# #         feat32_cam = self.cam(feat32)
        feat32_ffm = self.cam(feat32_ppm)
#         feat16_cam = self.cam16(feat16)
       
        H4, W4 = feat4.size()[2:]
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
#         H32, W32 = feat32.size()[2:]
        
#         avg = F.avg_pool2d(feat32, feat32.size()[2:])
        
        
#         avg = self.conv_avg(avg)
#         avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

#         feat32_arm = self.arm32(feat32)
#         feat32_sum = feat32_arm + avg_up
#         feat32_up = F.interpolate(feat32, (H16, W16), mode='bilinear')
        feat32_up = F.interpolate(feat32_ffm, (H16, W16), mode='bilinear')
        feat32_up = self.upconv_32(feat32_up)
#         print('feat32_up',feat32_up.shape)
#         print('feat16',feat16.shape)
#         feat32_ppm = feat32_up + feat16
#         feat32_up = self.conv_head32(feat32_up)

        feat16_cam = self.cam16(feat16)
#         feat16_sum = feat16 + feat32_up
      
        feat16_sum = torch.cat((feat16_cam,feat32_up),1)
        
        feat16_sum = self.conv16(feat16_sum)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        
        
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='bilinear')
        
        feat16_up = self.upconv_16(feat16_up)
        
#         feat_end = feat16_up + feat8
#         feat8_sam = self.sam(feat8)
        feat_end = torch.cat((feat16_up,feat8),1)
        
        feat_end = self.conv8(feat_end)
#         feat_end = self.sam(feat8,feat16_up)
    
        feat_end = F.interpolate(feat_end, (H4, W4), mode='bilinear')
        feat_end = self.upconv_8(feat_end)
#         feat_end = feat_end + feat4
        feat_end = torch.cat((feat_end,feat4),1)
        
        
#         return feat2, feat4, feat8, feat16, feat16_up, feat32_up # x8, x16
#         return feat2,feat4,feat8,feat16,feat32,feat32_ppm
#         return feat2,feat4,feat8,feat16,feat32,feat32_cam
        return feat2,feat4,feat8,feat16,feat32,feat_end

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

# class FeatureFusionModule(nn.Module):
#     def __init__(self, in_chan, out_chan, *args, **kwargs):
#         super(FeatureFusionModule, self).__init__()
#         self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(out_chan,
#                 out_chan//4,
#                 kernel_size = 1,
#                 stride = 1,
#                 padding = 0,
#                 bias = False)
#         self.conv2 = nn.Conv2d(out_chan//4,
#                 out_chan,
#                 kernel_size = 1,
#                 stride = 1,
#                 padding = 0,
#                 bias = False)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#         self.init_weight()

#     def forward(self, fsp, fcp):
#         fcat = torch.cat([fsp, fcp], dim=1)
#         feat = self.convblk(fcat)
#         atten = F.avg_pool2d(feat, feat.size()[2:])
#         atten = self.conv1(atten)
#         atten = self.relu(atten)
#         atten = self.conv2(atten)
#         atten = self.sigmoid(atten)
#         feat_atten = torch.mul(feat, atten)
#         feat_out = feat_atten + feat
#         return feat_out

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             if isinstance(module, (nn.Linear, nn.Conv2d)):
#                 wd_params.append(module.weight)
#                 if not module.bias is None:
#                     nowd_params.append(module.bias)
#             elif isinstance(module, BatchNorm2d):
#                 nowd_params += list(module.parameters())
#         return wd_params, nowd_params

class BiSeNet(nn.Module):
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(BiSeNet, self).__init__()
        
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        # self.heat_map = heat_map
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)
        
#         if use_conv_last:
#             self.last_conv = ConvBNReLU(1024,n_classes,3,1,0)
#         self.use_conv_last = use_conv_last
        
        if backbone == 'STDCNet1446':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDCNet813':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        else:
            print("backbone is not in backbone lists")
            exit(0)

#         self.ffm = FeatureFusionModule(inplane, 256)
#         self.conv_out = BiSeNetOutput(256, 256, n_classes)
#         self.conv_out = BiSeNetOutput(256, 64, n_classes)
        self.conv_out = BiSeNetOutput(128, 64, n_classes)
#         self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        self.conv_out16 = BiSeNetOutput(512, 64, n_classes)
#         self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(1024, 64, n_classes)

#         self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)
        
        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
#         self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)
#         self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)
        
        
        
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        
#         feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
#         feat_res2,feat_res4,feat_res8,feat_res16,feat_res32,feat_res32_cam = self.cp(x)
#         feat_res2,feat_res4,feat_res8,feat_res16,feat_res32 = self.cp(x)
        feat_res2,feat_res4,feat_res8,feat_res16,feat_res32,feat_end = self.cp(x)
        

#         feat_out_sp2 = self.conv_out_sp2(feat_res2)

#         feat_out_sp4 = self.conv_out_sp4(feat_res4)
  
        feat_out_sp8 = self.conv_out_sp8(feat_res8)

#         feat_out_sp16 = self.conv_out_sp16(feat_res16)

#         feat_fuse = self.ffm(feat_res8, feat_cp8)

#         feat_out = self.conv_out32(feat_res32)
        feat_out16 = self.conv_out16(feat_res16)
        feat_out32 = self.conv_out32(feat_res32)
#         feat_out32_cam = self.conv_out32(feat_res32_cam)
        feat_end = self.conv_out(feat_end)

        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
#         feat_out32_cam = F.interpolate(feat_out32_cam, (H, W), mode='bilinear', align_corners=True)
        feat_end = F.interpolate(feat_end, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
#         feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)

        
        
#         if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
#             return feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8
        
#         if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
#             return feat_out16, feat_out32,feat_out_sp4, feat_out_sp8

        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
#             return feat_out16, feat_out32, feat_out32_cam,feat_out_sp8
#             return feat_out16, feat_out32,feat_out_sp8
            return feat_out16, feat_out32, feat_end,feat_out_sp8
        
#         if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
#             return feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
