from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
import torch
        
class RewardDiffusionDPO(nn.Module):

    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=config["seglen"], # which segment of th trajectory wins
            global_cond_dim=config["encoding_size"], # check this to match to the conditional info 
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )

        self.device = device 
        self.beta = 2000

    def forward(self, x_w, x_l, cond_vec):
        noise = torch.randn(x_w.shape, device=self.device)
        B = x_w.shape[0]
        # Sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each diffusion iteration
        noisy_x_w = self.noise_scheduler.add_noise(
            x_w, noise, timesteps)
        
        noisy_x_l = self.noise_scheduler.add_noise(
            x_l, noise, timesteps)
        

        # Predict the noise residual
        noise_pred_x_w = self.noise_pred_net(sample=noisy_x_w, timestep=timesteps, global_cond=cond_vec)
        noise_pred_x_l = self.noise_pred_net(sample=noisy_x_w, timestep=timesteps, global_cond=cond_vec)

        return noise_pred_x_w, noise_pred_x_l, noise
    
    def compute_loss(self, noise_pred_w, noise_pred_l, noise):

        # model errs
        model_err_w = (noise - noise_pred_w).norm().pow(2)
        model_err_l = (noise - noise_pred_l).norm().pow(2)

        inside_term = -1 * self.beta * (model_err_w - model_err_l)

        loss = -1 * torch.log(torch.sigmoid(inside_term))

        return loss.mean()


