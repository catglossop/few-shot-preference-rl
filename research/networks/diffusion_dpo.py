from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from .conditional_unet1d import MetaConditionalUnet1D
from .mlp import MetaRewardMLPEnsemble
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
            num_train_timesteps=10,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        self.net_type = net_type
        if self.net_type == "cond_unet":
            self.noise_pred_net = MetaConditionalUnet1D(
                input_dim=(observation_space.shape[0] + action_space.shape[0]), # which segment of th trajectory wins
                global_cond_dim=1, # check this to match to the conditional info 
                down_dims=down_dims,
                diffusion_step_embed_dim=256,
                cond_predict_scale=cond_predict_scale,
                
            )
        elif self.net_type == "mlp":
            self.noise_pred_net = MetaRewardMLPEnsemble(observation_space, 
                                                        action_space, 
                                                        ensemble_size=1,
                                                        )

        self.beta = 1
        self.params = self.noise_pred_net.params
        self.params_bn = self.noise_pred_net.params_bn


    def forward(self, x_w, x_l, cond_vec, params=None, params_bn=None):
        if params is None:
            params = self.params
        if params_bn is None:
            params_bn = self.params_bn
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
            noise_pred_x_w = self.noise_pred_net(sample=noisy_x_w, timestep=timesteps, global_cond=cond_vec, params=params, params_bn=params_bn)
            noise_pred_x_l = self.noise_pred_net(sample=noisy_x_l, timestep=timesteps, global_cond=cond_vec, params=params, params_bn=params_bn)
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

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.params
    
    def compute_loss(self, noise_pred_w, noise_pred_l, noise):
        B = noise_pred_w.shape[0]  
        # model errs
        model_err_w = noise_pred_w - noise
        model_err_w = model_err_w - model_err_w.min(2, keepdim=True).values
        model_err_w = model_err_w / model_err_w.max(2, keepdim=True).values
        # model_err_w = model_err_w.flatten(start_dim=1).norm(dim=1).pow(2)
        model_err_w = model_err_w.flatten(start_dim=0, end_dim=1).norm(dim=1).pow(2)

        model_err_l = noise_pred_l - noise
        model_err_l = model_err_l - model_err_l.min(2, keepdim=True).values
        model_err_l = model_err_l / model_err_l.max(2, keepdim=True).values
        # model_err_l = model_err_l.flatten(start_dim=1).norm(dim=1).pow(2)
        model_err_l = model_err_l.flatten(start_dim=0, end_dim=1).norm(dim=1).pow(2)
        inside_term = -1 * self.beta * (model_err_w - model_err_l)
        log_term = torch.log(torch.sigmoid(inside_term))
        log_term = torch.where((log_term == -float('inf')), 0.0, log_term)
        log_term = torch.where(torch.isnan(log_term), 0.0, log_term)
        loss = -1 * log_term.mean()
        # print("Inner loss: ", loss) 
        if loss == float('inf'):
            breakpoint()
        with torch.no_grad():
            reward_diff = self.beta*(model_err_w - model_err_l).reshape(B, -1).sum(dim=1)
            reward_diff = torch.sigmoid(reward_diff)
        return loss, reward_diff

    def get_reward(self, noise_pred_a, noise_pred_b, noise):
        
        with torch.no_grad():
            reward = self.beta*((noise - noise_pred_a).norm().pow(2) - (noise - noise_pred_b).norm().pow(2))

        return reward

    def zero_grad_params(self, params=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if params is None:
                for p in self.params:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
    
    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            if g is not None:
                param_norm = g.data.norm(2)
                total_norm += param_norm.item() ** 2
                counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                if g is not None:
                    g.data.mul_(clip_coef)

        return total_norm/counter


