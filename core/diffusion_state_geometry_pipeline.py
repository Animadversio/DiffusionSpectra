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
#%
# computation code
def sampling(unet, scheduler, batch_size=1):
    noisy_sample = torch.randn(
        batch_size, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size
    ).to(unet.device).to(unet.dtype)
    t_traj, sample_traj, residual_traj = [], [], []
    sample = noisy_sample

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        with torch.no_grad():
            residual = unet(sample, t).sample

        # 2. compute previous image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample

        residual_traj.append(residual.cpu().detach())
        sample_traj.append(sample.cpu().detach())
        t_traj.append(t)
    return sample, sample_traj, t_traj
#%%
repo_id = "google/ddpm-cifar10-32"  # Note this model has self-attention in it.
# repo_id = "nbonaker/ddpm-celeb-face-32"
model = UNet2DModel.from_pretrained(repo_id)
model.requires_grad_(False).eval().to("cuda")#.half()
scheduler = DDIMScheduler.from_pretrained(repo_id)
scheduler.set_timesteps(num_inference_steps=100, )
#%%
sample, sample_traj, t_traj = sampling(model, scheduler, batch_size=64)
show_imgrid((sample + 1) / 2)
sample_traj = torch.stack(sample_traj)
#%%
# model_id = "fusing/ddim-celeba-hq"

model_id = "google/ddpm-celebahq-256"
savedir = r"F:\insilico_exps\Diffusion_traj\face_ffhq"

# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.unet.requires_grad_(False).eval().to("cuda")#.half()
#%%
# run pipeline in inference (sample random noise and denoise)
image = pipe()
image.images[0].show()
#%%
# save image
# image[0].save("ddpm_generated_image.png")
latents_reservoir = []
@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())


seed = 45
tsteps = 51
out = pipe(callback=save_latents, num_inference_steps=tsteps,
           generator=torch.cuda.manual_seed(seed))
out.images[0].show()
latents_reservoir = torch.cat(latents_reservoir, dim=0)
#%% Utility functions for analysis
#%%
from core.diffusion_geometry_lib import proj2subspace, proj2orthospace, subspace_variance, \
    trajectory_geometry_pipeline, latent_PCA_analysis, latent_diff_PCA_analysis
#%%
savedir = r"F:\insilico_exps\Diffusion_traj\face_ffhq"
os.makedirs(savedir, exist_ok=True)
# trajectory_geometry_pipeline(latents_reservoir, savedir, )
import math
from torchmetrics.functional import pairwise_cosine_similarity
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms

init_latent = latents_reservoir[:1].flatten(1).float()
end_latent = latents_reservoir[-1:].flatten(1).float()
init_end_cosine = pairwise_cosine_similarity(init_latent, end_latent).item()
init_end_angle = math.acos(init_end_cosine)
init_end_ang_deg = init_end_angle / math.pi * 180
unitbasis1 = end_latent / end_latent.norm()  # end state
unitbasis2 = proj2orthospace(end_latent, init_latent)  # init noise that is ortho to the end state
unitbasis2 = unitbasis2 / unitbasis2.norm()  # unit normalize
proj_coef1 = torch.matmul(latents_reservoir.flatten(1).float(), unitbasis1.T)
proj_coef2 = torch.matmul(latents_reservoir.flatten(1).float(), unitbasis2.T)
residue = latents_reservoir.flatten(1).float() - (proj_coef1 @ unitbasis1 + proj_coef2 @ unitbasis2)
residue_frac = residue.norm(dim=1) ** 2 / latents_reservoir.flatten(1).float().norm(dim=1) ** 2
#%%
show_imgrid((latents_reservoir[5]-latents_reservoir[0])*20 +0.5)
#%%
expvar_vec, U, D, V = latent_PCA_analysis(latents_reservoir, savedir, )
expvar_diff, U_diff, D_diff, V_diff = latent_diff_PCA_analysis(latents_reservoir, savedir, )
#%%

trajectory_geometry_pipeline(latents_reservoir, savedir, )
