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

import logger

from logger import setup_logger

# from models.model_stages import BiSeNet
from model_stages_con_32pc_16c import BiSeNet

from cityscapes import CityScapes

from loss.loss import OhemCELoss

from loss.detail_loss import DetailAggregateLoss
from evaluation import MscEvalV0

from optimizer_loss import Optimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse

logger = logging.getLogger()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

import sys
sys.argv=['']
del sys



def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    parse.add_argument(
            '--n_workers_train',
            dest = 'n_workers_train',
            type = int,
            default = 8,
            )
    parse.add_argument(
            '--n_workers_val',
            dest = 'n_workers_val',
            type = int,
            default = 0,
            )
    parse.add_argument(
            '--n_img_per_gpu',
            dest = 'n_img_per_gpu',
            type = int,
            default = 8,
            )
    parse.add_argument(
            '--max_iter',
            dest = 'max_iter',
            type = int,
            default = 181000,
            )
    parse.add_argument(
            '--save_iter_sep',
            dest = 'save_iter_sep',
            type = int,
            default = 1000,
            )
    parse.add_argument(
            '--warmup_steps',
            dest = 'warmup_steps',
            type = int,
            default = 1000,
            )      
    parse.add_argument(
            '--mode',
            dest = 'mode',
            type = str,
            default = 'train',
            )
    parse.add_argument(
            '--ckpt',
            dest = 'ckpt',
            type = str,
            default = None,
            )
    parse.add_argument(
            '--respath',
            dest = 'respath',
            type = str,
            default = '/powerhope/wangzhuo/STDC_seg/checkpoint',
            )
    parse.add_argument(
            '--backbone',
            dest = 'backbone',
            type = str,
#             default = 'CatNetSmall',
            default = 'STDCNet1446'
            )
    parse.add_argument(
            '--pretrain_path',
            dest = 'pretrain_path',
            type = str,
            default = '',
            )
    parse.add_argument(
            '--use_conv_last',
            dest = 'use_conv_last',
            type = str2bool,
            default = False,
            )
    parse.add_argument(
            '--use_boundary_2',
            dest = 'use_boundary_2',
            type = str2bool,
            default = False,
            )
    parse.add_argument(
            '--use_boundary_4',
            dest = 'use_boundary_4',
            type = str2bool,
            default = False,
            )
    parse.add_argument(
            '--use_boundary_8',
            dest = 'use_boundary_8',
            type = str2bool,
            default = True,
            )
    parse.add_argument(
            '--use_boundary_16',
            dest = 'use_boundary_16',
            type = str2bool,
            default = False,
            )
    return parse.parse_args()

args = parse_args()

args

from numpy import *

import  matplotlib.pyplot as plt
loss_avg_me = []
ious = []
def train():
    args = parse_args()
    
    save_pth_path = os.path.join(args.respath, 'pths')
    dspth = '/powerhope/wangzhuo/cityscapes'
    
    # print(save_pth_path)
    # print(osp.exists(save_pth_path))
    # if not osp.exists(save_pth_path) and dist.get_rank()==0: 
    if not osp.exists(save_pth_path):
        os.makedirs(save_pth_path)
    
    
#     torch.cuda.set_device(args.local_rank)
#     dist.init_process_group(
#                 backend = 'nccl',
#                 init_method = 'tcp://127.0.0.1:33274',
#                 world_size = torch.cuda.device_count(),
#                 rank=args.local_rank
#                 )
    
    setup_logger(args.respath)
    ## dataset
    n_classes = 19
    n_img_per_gpu = args.n_img_per_gpu
    n_workers_train = args.n_workers_train
    n_workers_val = args.n_workers_val
    use_boundary_16 = args.use_boundary_16
    use_boundary_8 = args.use_boundary_8
    use_boundary_4 = args.use_boundary_4
    use_boundary_2 = args.use_boundary_2
    
    mode = args.mode
    cropsize = [1024, 512]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

#     if dist.get_rank()==0: 
#         logger.info('n_workers_train: {}'.format(n_workers_train))
#         logger.info('n_workers_val: {}'.format(n_workers_val))
#         logger.info('use_boundary_2: {}'.format(use_boundary_2))
#         logger.info('use_boundary_4: {}'.format(use_boundary_4))
#         logger.info('use_boundary_8: {}'.format(use_boundary_8))
#         logger.info('use_boundary_16: {}'.format(use_boundary_16))
#         logger.info('mode: {}'.format(args.mode))
    
    
    ds = CityScapes(dspth, cropsize=cropsize, mode=mode, randomscale=randomscale)
#     sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
#                     batch_size = n_img_per_gpu,
                    batch_size  =n_img_per_gpu,
                    shuffle = False,
#                     sampler = sampler,
                    num_workers = n_workers_train,
                    pin_memory = False,
                    drop_last = True)
    # exit(0)
    dsval = CityScapes(dspth, mode='val', randomscale=randomscale)
#     sampler_val = torch.utils.data.distributed.DistributedSampler(dsval)
    dlval = DataLoader(dsval,
                    batch_size = 2,
                    shuffle = False,
#                     sampler = sampler_val,
                    num_workers = n_workers_val,
                    drop_last = False)

    ## model
    ignore_idx = 255
    net = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, 
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8, 
    use_boundary_16=use_boundary_16, use_conv_last=args.use_conv_last)

    if not args.ckpt is None:
        net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        
    net.cuda()
    
    net.train()
#     net = nn.parallel.DistributedDataParallel(net,
#             device_ids = [args.local_rank, ],
#             output_device = args.local_rank,
#             find_unused_parameters=True
#             )

    score_thres = 0.7
    n_min = n_img_per_gpu*cropsize[0]*cropsize[1]//16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
#     criteria_32_aspp = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    
    boundary_loss_func = DetailAggregateLoss()
    ## optimizer
    maxmIOU50 = 0.
    maxmIOU75 = 0.
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = args.max_iter
    save_iter_sep = args.save_iter_sep
    power = 0.9
    warmup_steps = args.warmup_steps
    warmup_start_lr = 1e-5

#     if dist.get_rank()==0: 
#         print('max_iter: ', max_iter)
#         print('save_iter_sep: ', save_iter_sep)
#         print('warmup_steps: ', warmup_steps)
    optim = Optimizer(
#             model = net.module,
            model = net,
            loss = boundary_loss_func,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)
    
    ## train loop
    msg_iter = 50
    loss_avg = []
    loss_boundery_bce = []
    loss_boundery_dice = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        if it // 1000 == 0:
            loss_sum = []
        try:
            im, lb = next(diter)
            if not im.size()[0]==n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
#             sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()


        if use_boundary_2 and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail2, detail4, detail8 = net(im)
            
        
        if (not use_boundary_2) and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail4, detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
#             out16, out32, detail8 = net(im)
            out16,out32,out_end,detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and (not use_boundary_8):
#             out, out16, out32 = net(im)
            out16,out32,detail8 = net(im)

#         lossp = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss3 = criteria_32(out32, lb)
        loss4 = criteria_32(out_end, lb)
        
        boundery_bce_loss = 0.
        boundery_dice_loss = 0.
        
        
        if use_boundary_2: 
            # if dist.get_rank()==0:
            #     print('use_boundary_2')
            boundery_bce_loss2,  boundery_dice_loss2 = boundary_loss_func(detail2, lb)
            boundery_bce_loss += boundery_bce_loss2
            boundery_dice_loss += boundery_dice_loss2
        
        if use_boundary_4:
            # if dist.get_rank()==0:
            #     print('use_boundary_4')
            boundery_bce_loss4,  boundery_dice_loss4 = boundary_loss_func(detail4, lb)
            boundery_bce_loss += boundery_bce_loss4
            boundery_dice_loss += boundery_dice_loss4

        if use_boundary_8:
            # if dist.get_rank()==0:
            #     print('use_boundary_8')
            boundery_bce_loss8,  boundery_dice_loss8 = boundary_loss_func(detail8, lb)
            boundery_bce_loss += boundery_bce_loss8
            boundery_dice_loss += boundery_dice_loss8
        
        loss =  loss2 + loss3 + loss4 + boundery_bce_loss + boundery_dice_loss
#         loss =  loss2 + loss3 + boundery_bce_loss + boundery_dice_loss
        loss_sum.append(loss)
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        loss_boundery_bce.append(boundery_bce_loss.item())
        loss_boundery_dice.append(boundery_dice_loss.item())

        ## print training log message
        if (it+1)%msg_iter==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))

            loss_boundery_bce_avg = sum(loss_boundery_bce) / len(loss_boundery_bce)
            loss_boundery_dice_avg = sum(loss_boundery_dice) / len(loss_boundery_dice)
            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'boundery_bce_loss: {boundery_bce_loss:.4f}',
                'boundery_dice_loss: {boundery_dice_loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it = it+1,
                max_it = max_iter,
                lr = lr,
                loss = loss_avg,
                boundery_bce_loss = loss_boundery_bce_avg,
                boundery_dice_loss = loss_boundery_dice_avg,
                time = t_intv,
                eta = eta
            )
            
            logger.info(msg)
            loss_avg = []
            loss_boundery_bce = []
            loss_boundery_dice = []
            st = ed
            # print(boundary_loss_func.get_params())
        if (it+1)%save_iter_sep==0:# and it != 0:
            
            ## model
            logger.info('evaluating the model ...')
            logger.info('setup and restore model')
            
            net.eval()

            # ## evaluator
            logger.info('compute the mIOU')
            with torch.no_grad():
                net = net.cuda()
                single_scale1 = MscEvalV0()
                mIOU50 = single_scale1(net, dlval, n_classes)

                single_scale2= MscEvalV0(scale=0.75)
                mIOU75 = single_scale2(net, dlval, n_classes)
                
                ious.append(mIOU75)
                loss_avg_me.append(loss_sum)
                loss_sun = []
                
                
                
            save_pth = osp.join(save_pth_path, 'model_iter{}_mIOU50_{}_mIOU75_{}.pth'
            .format(it+1, str(round(mIOU50,4)), str(round(mIOU75,4))))
            
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
#             if dist.get_rank()==0: 
            torch.save(state, save_pth)

            logger.info('training iteration {}, model saved to: {}'.format(it+1, save_pth))

            if mIOU50 > maxmIOU50:
                maxmIOU50 = mIOU50
                save_pth = osp.join(save_pth_path, 'model_maxmIOU50.pth'.format(it+1))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
#                 if dist.get_rank()==0: 
                torch.save(state, save_pth)
                    
                logger.info('max mIOU model saved to: {}'.format(save_pth))
            
            if mIOU75 > maxmIOU75:
                maxmIOU75 = mIOU75
                save_pth = osp.join(save_pth_path, 'model_maxmIOU75.pth'.format(it+1))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
#                 if dist.get_rank()==0: 
                torch.save(state, save_pth)
                logger.info('max mIOU model saved to: {}'.format(save_pth))
#                 print('it',it+1)
#                 print('max_iter',max_iter)
#                 if max_iter - (it+1) < 10000:
#                     for i,data in enumerate(dlval):
#                         img,label = data 
#                         img = img.cuda()
#                         out_list = net(img)
#                         out_end = out_list[-2]
#                         for j in range(2):
#                             im_out = out_end[j]
#                             im_out = im_out.max(0)[1]
#                             x = i *2 + j
#                             plt.imshow(im_out.cpu().detach().numpy())
#                             plt.savefig('/powerhope/wangzhuo/STDC_seg/32ppm_camand16cam/output{}.jpg'.format(x))
            
            
            
            logger.info('mIOU50 is: {}, mIOU75 is: {}'.format(mIOU50, mIOU75))
            logger.info('maxmIOU50 is: {}, maxmIOU75 is: {}.'.format(maxmIOU50, maxmIOU75))

            net.train()
    
    ## dump the final model
    save_pth = osp.join(save_pth_path, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
#     if dist.get_rank()==0:
    torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))
    print('epoch: ', epoch)

train()
