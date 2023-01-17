import torch
from tqdm import tqdm
from diffusers import PNDMPipeline, DDIMScheduler, UNet2DModel, PNDMScheduler
from diffusers import DiffusionPipeline
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler

import os
from os.path import join
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
#% Utility functions for analysis
import json
from core.diffusion_geometry_lib import proj2subspace, proj2orthospace, subspace_variance, \
    trajectory_geometry_pipeline, latent_PCA_analysis, latent_diff_PCA_analysis, diff_cosine_mat_analysis, PCA_data_visualize
from core.diffusion_traj_analysis_lib import compute_save_diff_imgs_diff, plot_diff_matrix
#%%
# model_id = "fusing/ddim-celeba-hq"

# model_id = "google/ddpm-cat-256"
# model_id = "google/ddpm-ema-cat-256"
# model_id = "google/ddpm-bedroom-256"
# model_id = "google/ddpm-ema-bedroom-256"
# model_id = "google/ddpm-church-256"
# model_id = "google/ddpm-ema-church-256"
# model_id = "google/ddpm-ema-celebahq-256"
model_id = "google/ddpm-celebahq-256" # most popular
# model_id = "google/ddpm-cifar10-32"
model_id = "dimpo/ddpm-mnist"  # most popular
model_id_short = model_id.split("/")[-1]
saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}"
# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.unet.requires_grad_(False).eval().to("cuda")#.half()

#%%
import matplotlib
matplotlib.use('Agg')
# use the interactive backend

# matplotlib.use('module://backend_interagg')
#%%
# save image
# image[0].save("ddpm_generated_image.png")
seed = 99

# for seed in range(200, 400):
latents_reservoir = []
t_traj = []

@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())
    t_traj.append(t)

tsteps = 51
out = pipe(callback=save_latents, num_inference_steps=tsteps,
           generator=torch.cuda.manual_seed(seed))
latents_reservoir = torch.cat(latents_reservoir, dim=0)
t_traj = torch.tensor(t_traj)

#%%
def sampling(unet, scheduler, batch_size=1, generator=None):
    noisy_sample = torch.randn(
        batch_size, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size,
        generator=generator, device=unet.device
    ).to(unet.device).to(unet.dtype)
    t_traj, sample_traj, residual_traj = [], [], []
    sample = noisy_sample
    sample_traj.append(sample.cpu().detach())
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        with torch.no_grad():
            residual = unet(sample, t).sample
        # 2. compute previous image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample
        residual_traj.append(residual.cpu().detach())
        sample_traj.append(sample.cpu().detach())
        t_traj.append(t)
    t_traj = torch.tensor(t_traj)
    sample_traj = torch.cat(sample_traj, dim=0)
    residual_traj = torch.cat(residual_traj, dim=0)
    return sample, sample_traj, residual_traj, t_traj


pipe.scheduler.set_timesteps(51)
sample, sample_traj, residual_traj, t_traj = sampling(pipe.unet, pipe.scheduler, batch_size=1)

#%%
alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
pred_x0 = (sample_traj[:-1] - residual_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / alphacum_traj.sqrt().view(-1, 1, 1, 1)
#%%
show_imgrid((pred_x0 + 1)/2, nrow=10, figsize=(10, 10))
#%%

def denorm_sample_mean_std(x):
    return ((x - x.mean(dim=(1,2,3), keepdims=True)) / x.std(dim=(1,2,3), keepdims=True) * 0.4 + 1) / 2


def denorm_sample_std(x):
    return ((x) / x.std(dim=(1,2,3), keepdims=True) * 0.4 + 1) / 2


#%%
show_imgrid(torch.clamp(denorm_sample_std(sample_traj[1:] - sample_traj[:-1]), 0, 1), nrow=10, figsize=(10, 10))
#%%
show_imgrid(torch.clamp(denorm_sample_mean_std(residual_traj[1:]-sample_traj[0:1]), 0, 1), nrow=10, figsize=(10, 10))
#%%
pipe.scheduler.set_timesteps(51)
#%%
pipe.scheduler.set_timesteps(51)
timesteps_orig = pipe.scheduler.timesteps
pipe.scheduler.timesteps = timesteps_orig # torch.cat((timesteps_orig[:25,], timesteps_orig[-1:]))
pipe.scheduler.num_inference_steps = len(pipe.scheduler.timesteps)
#%%

sample, sample_traj, residual_traj, t_traj = sampling(pipe.unet, pipe.scheduler, batch_size=1)
#%%

alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
pred_x0 = (sample_traj - residual_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / alphacum_traj.sqrt().view(-1, 1, 1, 1)
show_imgrid((pred_x0 + 1)/2, nrow=10, figsize=(10, 10))
#%%
#%%
plt.figure(figsize=(8, 3))
plt.subplot(1,3,1)
plt.plot(pipe.scheduler.betas[pipe.scheduler.timesteps])
plt.title("beta")
plt.subplot(1,3,2)
plt.plot(pipe.scheduler.alphas[pipe.scheduler.timesteps])
plt.title("alpha")
plt.subplot(1,3,3)
plt.plot(pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps])
plt.title("alpha_cumprod")
plt.suptitle("betas, alphas, alphas_cumprod")
plt.tight_layout()
plt.show()