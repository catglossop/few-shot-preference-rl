from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from mlp import MetaRewardMLPEnsemble
import torch
from torch import distributions, nn
from torch.nn import functional as F
        
class RewardDiffusionDPO(nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 seglen,
                 down_dims, 
                 cond_predict_scale,
                 net_type,):
        super().__init__()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        self.net_type = net_type
        if self.net_type == "cond_unet":
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=(observation_space.shape[0] + action_space.shape[0]), # which segment of th trajectory wins
                global_cond_dim=1, # check this to match to the conditional info 
                down_dims=down_dims,
                diffusion_step_embed_dim=seglen,
                cond_predict_scale=cond_predict_scale,
                
            )
        elif self.net_type == "mlp":
            self.noise_pred_net = MetaRewardMLPEnsemble(observation_space, 
                                                        action_space, 
                                                        ensemble_size=1,
                                                        )
        self.noise_pred_net.to('cuda')

        self.beta = 1
        parameters = self.noise_pred_net.parameters(recurse=True)
        params = {}
        idx = 0
        for param in parameters:
            params[f'layer_{idx}'] = param 
            idx += 1
        
        self.params = torch.nn.ParameterDict(params)


    def forward(self, x_w, x_l, cond_vec, params=None):
        if params is None:
            params = self.params
        if self.net_type == "cond_unet":
            noise = torch.randn(x_w.shape, device='cuda')
            cond_vec = cond_vec.unsqueeze(0).repeat(x_w.shape[0]).unsqueeze(1)
            B = x_w.shape[0]
            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device='cuda',
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_x_w = self.noise_scheduler.add_noise(
                x_w, noise, timesteps)
            
            noisy_x_l = self.noise_scheduler.add_noise(
                x_l, noise, timesteps)
            # Predict the noise residual
            noise_pred_x_w = self.noise_pred_net(sample=noisy_x_w, timestep=timesteps, global_cond=cond_vec)
            noise_pred_x_l = self.noise_pred_net(sample=noisy_x_l, timestep=timesteps, global_cond=cond_vec)
        elif self.net_type == "mlp":
            noise = torch.randn(x_w.shape, device='cuda')
            B = x_w.shape[0]
            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device='cuda',
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_x_w = self.noise_scheduler.add_noise(
                x_w, noise, timesteps)
            
            noisy_x_l = self.noise_scheduler.add_noise(
                x_l, noise, timesteps)
            
            # Predict the noise residual
            noise_pred_x_w = self.noise_pred_net(sample=noisy_x_w)
            noise_pred_x_l = self.noise_pred_net(sample=noisy_x_l)

        return noise_pred_x_w, noise_pred_x_l, noise
    
    def compute_loss(self, noise_pred_w, noise_pred_l, noise):

        # model errs
        model_err_w = noise_pred_w - noise
        model_err_w = model_err_w - model_err_w.min(2, keepdim=True).values
        model_err_w = model_err_w / model_err_w.max(2, keepdim=True).values
        model_err_w = model_err_w.norm().pow(2)
        
        model_err_l = noise_pred_l - noise
        model_err_l = model_err_l - model_err_l.min(2, keepdim=True).values
        model_err_l = model_err_l / model_err_l.max(2, keepdim=True).values
        model_err_l = model_err_l.norm().pow(2)

        inside_term = -1 * self.beta * (model_err_w - model_err_l)

        loss = -1 * torch.log(torch.sigmoid(inside_term))

        with torch.no_grad():
            reward_diff = self.beta*(model_err_w - model_err_l)
        return loss.mean(), reward_diff

    def get_reward(self, noise_pred_a, noise_pred_b, noise):
        
        with torch.no_grad():
            reward = self.beta*((noise - noise_pred_a).norm().pow(2) - (noise - noise_pred_b).norm().pow(2))

        return reward


