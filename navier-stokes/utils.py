import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.functional import interpolate
from torchvision import transforms as T
from torchvision.utils import make_grid

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import seaborn as sns
import math
import wandb
import argparse
import datetime
from time import time
import os
import numpy as np

# local
from unet import Unet

def maybe_create_dir(f):
    if not os.path.exists(f):
        print("making", f)
        os.makedirs(f)

def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))                                                                                        

def is_type_for_logging(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return True
    elif isinstance(x, bool):
        return True
    elif isinstance(x, str):
        return True
    elif isinstance(x, list):
        return True
    elif isinstance(x, set):
        return True
    else:
        return False

## if you want to make a grid of images
def to_grid(x, grid_kwargs):
    nrow = int(np.floor(np.sqrt(x.shape[0])))
    return make_grid(x, nrow = nrow, **grid_kwargs)

def clip_grad_norm(model, max_norm):
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm = max_norm, 
        norm_type= 2.0, 
        error_if_nonfinite = False
    )

def get_cifar_dataloader(config):

    Flip = T.RandomHorizontalFlip()
    Tens = T.ToTensor()
    transform = T.Compose([Flip, Tens])
    ds = datasets.CIFAR10(
        config.data_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    batch_size = config.batch_size

    return DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = True, 
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = True, 
    )

def setup_wandb(config):
    if not config.use_wandb:
        return

    config.wandb_run = wandb.init(
        project = config.wandb_project,
        entity = config.wandb_entity,
        resume = None,
        id = None,
    )

    config.wandb_run_id = config.wandb_run.id

    for key in vars(config):
        item = getattr(config, key)
        if is_type_for_logging(item):
            setattr(wandb.config, key, item)
    print("finished wandb setup")


class DriftModel(nn.Module):
    def __init__(self, config):
        
        super(DriftModel, self).__init__()
        self.config = config
        c = config
        self._arch = Unet(
            num_classes = c.num_classes,
            in_channels = c.C * 2, # times two for conditioning
            out_channels= c.C,
            dim = c.unet_channels,
            dim_mults = c.unet_dim_mults ,
            resnet_block_groups = c.unet_resnet_block_groups,
            learned_sinusoidal_cond = c.unet_learned_sinusoidal_cond,
            random_fourier_features = c.unet_random_fourier_features,
            learned_sinusoidal_dim = c.unet_learned_sinusoidal_dim,
            attn_dim_head = c.unet_attn_dim_head,
            attn_heads = c.unet_attn_heads,
            use_classes = c.unet_use_classes,
        )
        num_params = np.sum([int(np.prod(p.shape)) for p in self._arch.parameters()])
        print("Num params in main arch for drift is", f"{num_params:,}")

    def forward(self, zt, t, y, cond=None):
        
        if not self.config.unet_use_classes:
            y = None


        if cond is not None:
            zt = torch.cat([zt, cond], dim = 1)

        out = self._arch(zt, t, y)

        return out

def maybe_subsample(x, subsampling_ratio):
    if subsampling_ratio:
        x = x[ : int(subsampling_ratio * x.shape[0]), ...]
    return x

def maybe_lag(data, time_lag):
    if time_lag > 0:
        inputs = data[:, :-time_lag, ...]
        outputs = data[:, time_lag:, ...]
    else:
        inputs, outputs = data, data
    return inputs, outputs

def maybe_downsample(inputs, outputs, lo_size, hi_size):    
    upsampler = nn.Upsample(scale_factor=int(hi_size/lo_size), mode='nearest')
    hi = interpolate(outputs, size=(hi_size,hi_size),mode='bilinear').reshape([-1,hi_size,hi_size])
    lo = upsampler(interpolate(inputs, size=(lo_size,lo_size),mode='bilinear'))
    return lo, hi

def flatten_time(lo, hi, hi_size):
    hi = hi.reshape([-1,hi_size,hi_size])
    lo = lo.reshape([-1,hi_size,hi_size])
    # make the data N C H W
    hi = hi[:,None,:,:] 
    lo = lo[:,None,:,:] 
    return lo, hi

def loader_from_tensor(lo, hi, batch_size, shuffle):
    return DataLoader(TensorDataset(lo, hi), batch_size = batch_size, shuffle = shuffle)

def get_forecasting_dataloader(config, shuffle = False):
    data_raw, time_raw = torch.load(config.data_fname)
    del time_raw
    

    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    
    
    #Ntj, Nts, Nx, Ny = data_raw.size() 
    #avg_pixel_norm = torch.norm(data_raw,dim=(2,3),p='fro').mean() / np.sqrt(Nx*Ny)
    avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori
    data_raw = data_raw/avg_pixel_norm
    new_avg_pixel_norm = 1.0

    # here "lo" will be the conditioning info (initial condition) and "hi" will be the target
    # lo is x_t and hi is x_{t+tau}, and lo might be lower res than hi

    lo, hi = maybe_lag(data_raw, config.time_lag)
    lo, hi = maybe_downsample(lo, hi, config.lo_size, config.hi_size)
    lo, hi = flatten_time(lo, hi, config.hi_size)

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)
    return loader, avg_pixel_norm, new_avg_pixel_norm

def make_one_redblue_plot(x, fname):
    plt.ioff()
    fig = plt.figure(figsize=(3,3))
    plt.imshow(x, cmap=sns.cm.icefire, vmin=-2, vmax=2.)
    plt.axis('off')
    plt.savefig(fname, bbox_inches = 'tight')
    plt.close("all")         

def open_redblue_plot_as_tensor(fname):
    return T.ToTensor()(Image.open(fname))

def make_redblue_plots(x, config):
    plt.ioff()
    x = x.cpu()
    bsz = x.size()[0]
    for i in range(bsz):
        make_one_redblue_plot(x[i,0,...], fname = config.home + f'tmp{i}.jpg')
    tensor_img = T.ToTensor()(Image.open(config.home + f'tmp1.jpg'))
    C, H, W = tensor_img.size()
    out = torch.zeros((bsz,C,H,W))
    for i in range(bsz):
        out[i,...] = open_redblue_plot_as_tensor(config.home + f'tmp{i}.jpg')
    return out


