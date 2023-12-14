from typing import Union
import logging
import torch
import torch.nn as nn
import einops
import math 
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from .param_definitions import diffusion_step_encoder, conditional_residual_block1D, conv1d, conv1d_block, downsample1d, upsample1d, identity



logger = logging.getLogger(__name__)

# class ConditionalResidualBlock1D(nn.Module):
#     def __init__(self, 
#             in_channels, 
#             out_channels, 
#             cond_dim,
#             kernel_size=3,
#             n_groups=8,
#             cond_predict_scale=False):
#         super().__init__()

#         self.blocks = nn.ModuleList([
#             Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
#             Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
#         ])

#         # FiLM modulation https://arxiv.org/abs/1709.07871
#         # predicts per-channel scale and bias
#         cond_channels = out_channels
#         if cond_predict_scale:
#             cond_channels = out_channels * 2
#         self.cond_predict_scale = cond_predict_scale
#         self.out_channels = out_channels
#         self.cond_encoder = nn.Sequential(
#             nn.Mish(),
#             nn.Linear(cond_dim, cond_channels),
#             Rearrange('batch t -> batch t 1'),
#         )

#         # make sure dimensions compatible
#         self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
#             if in_channels != out_channels else nn.Identity()

#     def forward(self, x, cond):
#         '''
#             x : [ batch_size x in_channels x horizon ]
#             cond : [ batch_size x cond_dim]

#             returns:
#             out : [ batch_size x out_channels x horizon ]
#         '''
#         out = self.blocks[0](x)
#         embed = self.cond_encoder(cond)
#         if self.cond_predict_scale:
#             embed = embed.reshape(
#                 embed.shape[0], 2, self.out_channels, 1)
#             scale = embed[:,0,...]
#             bias = embed[:,1,...]
#             out = scale * out + bias
#         else:
#             out = out + embed
#         out = self.blocks[1](out)
#         out = out + self.residual_conv(x)
#         return out


class MetaConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim=1,
        diffusion_step_embed_dim=16,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
        mid_dim = all_dims[-1]
        self.params = nn.ParameterList()
        self.params_bn = nn.ParameterList()

        self.diffusion_step_encoder = diffusion_step_encoder(dsed, dsed*4)
        self.params.extend(self.diffusion_step_encoder.params)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        down_modules = nn.ModuleList([])
        for ind , (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            resnet1 = conditional_residual_block1D(dim_in, dim_out, cond_dim=cond_dim, 
                                                                           kernel_size=kernel_size)
            resnet2 = conditional_residual_block1D(dim_out, dim_out, cond_dim=cond_dim, 
                                                                           kernel_size=kernel_size)
            downsample = downsample1d(dim_out) if not is_last else identity(dim_out)
            self.params.extend([*resnet1.params, *resnet2.params, *downsample.params])
            self.params_bn.extend([*resnet1.params_bn, *resnet2.params_bn])
            down_modules.append(nn.ModuleList([resnet1, resnet2, downsample]))
        self.down_modules = down_modules        

        resnet1 = conditional_residual_block1D(mid_dim, mid_dim, cond_dim=cond_dim, 
                                                                       kernel_size=kernel_size)
        resnet2 = conditional_residual_block1D(mid_dim, mid_dim, cond_dim=cond_dim, 
                                                                       kernel_size=kernel_size)
        
        self.params.extend([*resnet1.params, *resnet2.params])
        self.params_bn.extend([*resnet1.params_bn, *resnet2.params_bn])
        self.mid_modules = nn.ModuleList([resnet1, resnet2])

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            resnet1 = conditional_residual_block1D(dim_out*2, dim_in, cond_dim=cond_dim,
                                                kernel_size=kernel_size)
            resnet2 = conditional_residual_block1D(dim_in, dim_in, cond_dim=cond_dim,
                                                kernel_size=kernel_size)
            upsample = upsample1d(dim_in) if not is_last else identity(dim_in)
            self.params.extend([*resnet1.params, *resnet2.params, *upsample.params])
            self.params_bn.extend([*resnet1.params_bn, *resnet2.params_bn])
            up_modules.append(nn.ModuleList([resnet1, resnet2, upsample]))

        self.up_modules = up_modules

        final_conv_block = conv1d_block(start_dim, start_dim, kernel_size=kernel_size)
        final_conv_op = conv1d(start_dim, input_dim, 1)
        final_conv = nn.ModuleList([final_conv_block, 
                                    final_conv_op])
        self.params.extend([*final_conv_block.params, *final_conv_op.params])
        self.params_bn.extend(final_conv_block.params_bn)
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.params)
        )


    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, params=None, params_bn=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        if params is None: 
            params = self.params
        if params_bn is None:
            params_bn = self.params_bn
        sample = einops.rearrange(sample, 'b h t -> b t h')
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        global_feature = self.diffusion_step_encoder(timesteps, params[:self.diffusion_step_encoder.num_params])
        param_idx = self.diffusion_step_encoder.num_params

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = sample
        h = []
        param_idx_bn = 0
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature, params=params[param_idx:param_idx+resnet.num_params], params_bn=params_bn[param_idx_bn:param_idx_bn+resnet.num_params_bn])
            param_idx += resnet.num_params
            param_idx_bn += resnet.num_params_bn
            x = resnet2(x, global_feature, params=params[param_idx:param_idx+resnet2.num_params], params_bn=params_bn[param_idx_bn:param_idx_bn+resnet2.num_params_bn])
            param_idx += resnet2.num_params
            param_idx_bn += resnet2.num_params_bn
            h.append(x)
            x = downsample(x, params=params[param_idx:param_idx+downsample.num_params])
            param_idx += downsample.num_params

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature, params=params[param_idx:param_idx+mid_module.num_params], params_bn=params_bn[param_idx_bn:param_idx_bn+mid_module.num_params_bn])
            param_idx += mid_module.num_params
            param_idx_bn += mid_module.num_params_bn

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature, params=params[param_idx:param_idx+resnet.num_params], params_bn=params_bn[param_idx_bn:param_idx_bn+resnet.num_params_bn])
            param_idx += resnet.num_params
            param_idx_bn += resnet.num_params_bn
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            x = resnet2(x, global_feature, params=params[param_idx:param_idx+resnet2.num_params], params_bn=params_bn[param_idx_bn:param_idx_bn+resnet2.num_params_bn])
            param_idx += resnet2.num_params
            param_idx_bn += resnet2.num_params_bn
            x = upsample(x, params=params[param_idx:param_idx+upsample.num_params])
            param_idx += upsample.num_params

        x = self.final_conv[0](x, params=params[param_idx:param_idx+self.final_conv[0].num_params], params_bn=params_bn[param_idx_bn:param_idx_bn+self.final_conv[0].num_params_bn])
        param_idx += self.final_conv[0].num_params
        x = self.final_conv[1](x, params=params[param_idx:param_idx+self.final_conv[1].num_params])
        x = einops.rearrange(x, 'b t h -> b h t')
        return x

