from typing import Union
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
import math 
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class linear(nn.Module): 
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = nn.ParameterList()

        w = nn.Parameter(torch.ones(out_dim, in_dim))
        torch.nn.init.kaiming_normal_(w)
        self.params.append(w)
        self.params.append(nn.Parameter(torch.zeros(out_dim)))
        self.num_params = len(self.params)
    
    def forward(self, x, params):
        return F.linear(x, params[0], params[1])

class diffusion_step_encoder(nn.Module): 

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.params = nn.ParameterList()

        self.sinusoid_pos_embed = SinusoidalPosEmb(self.in_dim)
        self.linear_1 = linear(self.in_dim, self.out_dim)
        self.params.extend(self.linear_1.params)
        self.mish = nn.Mish()
        self.linear_2 = linear(self.out_dim, self.in_dim)
        self.params.extend(self.linear_2.params)
        self.num_params = len(self.params)
    
    def forward(self, x, params):
        x = self.sinusoid_pos_embed(x)
        x = self.linear_1(x, params[:self.linear_1.num_params])
        idx = self.linear_1.num_params
        x = self.mish(x)
        x = self.linear_2(x, params[idx:idx+self.linear_2.num_params])
        return x

class conv1d(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        self.params = nn.ParameterList()

        # [ch_out, ch_in, kernelsz]
        w = nn.Parameter(torch.ones(out_dim, in_dim, kernel_size))
        torch.nn.init.kaiming_normal_(w)
        self.params.append(w)
        self.params.append(nn.Parameter(torch.zeros(out_dim)))
        self.num_params = len(self.params)
    
    def forward(self, x, params):
        return F.conv1d(x, params[0], params[1], stride=self.stride, padding=self.padding)

class convt1d(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.params = nn.ParameterList()

        # [ch_out, ch_in, kernelsz]
        w = nn.Parameter(torch.ones(in_dim, out_dim, kernel_size))
        torch.nn.init.kaiming_normal_(w)
        self.params.append(w)
        self.params.append(nn.Parameter(torch.zeros(out_dim)))
        self.num_params = len(self.params)
    
    def forward(self, x, params):
        return F.conv_transpose1d(x, params[0], params[1], stride=self.stride, padding=self.padding)

class batch_norm(nn.Module): 
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

        self.params = nn.ParameterList()
        self.params_bn = nn.ParameterList()

        w = nn.Parameter(torch.ones(out_dim))
        self.params.append(w)
        self.params.append(nn.Parameter(torch.zeros(out_dim)))

        # must set requires_grad=False
        running_mean = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
        running_var = nn.Parameter(torch.ones(out_dim), requires_grad=False)
        self.params_bn.extend([running_mean, running_var])
        self.num_params = len(self.params)
        self.num_params_bn = len(self.params_bn)
    
    def forward(self, x, params, params_bn):
        return F.batch_norm(x, params_bn[0], params_bn[1], weight=params[0], bias=params[1], training=True)

    
class conv1d_block(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size

        self.params = nn.ParameterList()
        self.params_bn = nn.ParameterList()

        self.conv = conv1d(self.in_dim, self.out_dim, self.kernel_size, padding=self.kernel_size // 2)
        self.params.extend(self.conv.params)
        self.batch_norm = batch_norm(self.out_dim)
        self.params.extend(self.batch_norm.params)
        self.params_bn.extend(self.batch_norm.params_bn)
        self.mish = nn.Mish()
        self.num_params = len(self.params)
        self.num_params_bn = len(self.params_bn)
    
    def forward(self, x, params, params_bn):
        x = self.conv(x, params[:self.conv.num_params])
        idx = self.conv.num_params
        x = self.batch_norm(x, params[idx:idx+self.batch_norm.num_params], params_bn)
        x = self.mish(x)  
        return x
    
class cond_encoder(nn.Module):

    def __init__(self, cond_dim, cond_channels):
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_channels = cond_channels

        self.params = nn.ParameterList()

        self.linear = linear(self.cond_dim, self.cond_channels)
        self.params.extend(self.linear.params)
        self.mish = nn.Mish()
        self.num_params = len(self.params)
        self.rearr = Rearrange('batch t -> batch t 1')
    
    def forward(self, x, params):
        x = self.mish(self.linear(x, params))
        x = self.rearr(x)
        return x

class identity(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.params = nn.ParameterList()
        self.params.append(nn.Parameter(torch.eye(dim), requires_grad=False))
        self.num_params = len(self.params)
        self.identity = nn.Identity()
    
    def forward(self, x, params):
        return self.identity(x)


class conditional_residual_block1D(nn.Module):

    def __init__(self, 
                in_channels, 
                out_channels, 
                cond_dim,
                kernel_size=3,
                n_groups=8,):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.n_groups = n_groups

        self.params = nn.ParameterList()
        self.params_bn = nn.ParameterList()

        self.blocks = nn.ModuleList([
            conv1d_block(in_channels, out_channels, kernel_size),
            conv1d_block(out_channels, out_channels, kernel_size),
        ])
        self.params.extend(self.blocks[0].params)
        self.params_bn.extend(self.blocks[0].params_bn)
        cond_channels = out_channels
        self.cond_encoder = cond_encoder(cond_dim, cond_channels)
        self.params.extend(self.cond_encoder.params)

        self.params.extend(self.blocks[1].params)
        self.params_bn.extend(self.blocks[1].params_bn)

        if in_channels != out_channels:
            self.residual_conv = conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = identity(in_channels)
        self.params.extend(self.residual_conv.params)

        self.num_params  = len(self.params)
        self.num_params_bn = len(self.params_bn)

    def forward(self, x, cond, params, params_bn):
        out = self.blocks[0](x, params[:self.blocks[0].num_params], params_bn[:self.blocks[0].num_params_bn])
        idx = self.blocks[0].num_params
        idx_bn = self.blocks[0].num_params_bn
        embed = self.cond_encoder(cond, params[idx:idx + self.cond_encoder.num_params])
        idx += self.cond_encoder.num_params
        out = out + embed
        out = self.blocks[1](out, params[idx:idx+self.blocks[1].num_params], params_bn[idx_bn:idx_bn + self.blocks[1].num_params_bn])
        idx += self.blocks[1].num_params
        out = out + self.residual_conv(x, params[idx:idx+self.residual_conv.num_params])

        return out

class upsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params = nn.ParameterList()
        self.convt = convt1d(dim, dim, 4, 2, 1)
        self.params.extend(self.convt.params)
        self.num_params = len(self.params)

    def forward(self, x, params):
        return self.convt(x, params)

class downsample1d(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params = nn.ParameterList()
        self.conv = conv1d(dim, dim, 3, 2, 1)
        self.params.extend(self.conv.params)
        self.num_params = len(self.params)

    def forward(self, x, params):
        return self.conv(x, params)



        