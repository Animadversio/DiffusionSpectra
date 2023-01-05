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
savedir = r"F:\insilico_exps\Diffusion_traj\cifar10-32"

# computation code
def sampling(model, scheduler, batch_size=1):
    noisy_sample = torch.randn(
        batch_size, model.config.in_channels, model.config.sample_size, model.config.sample_size
    ).to(model.device).to(model.dtype)
    t_traj, sample_traj, = [], []
    sample = noisy_sample

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        with torch.no_grad():
            residual = model(sample, t).sample

        # 2. compute previous image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample

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

#%%
def denorm(img):
    return (img + 1) / 2


def denorm_std(img):
    """a hacky way to make the image with std=0.2 mean=0.5 similar to ImageNet"""
    return (img / img.std() * 0.4 + 1) / 2
sumdir = r"F:\insilico_exps\Diffusion_traj\cifar10-32\summary"
os.makedirs(sumdir, exist_ok=True)
#%%

show_imgrid((sample_traj[99] + 1) / 2)
#%%
import umap
#%%
repo_id = "nbonaker/ddpm-celeb-face-32"
pipe = DiffusionPipeline.from_pretrained(repo_id)
#%%
pipe.unet.cuda()
#%%
image = pipe(num_inference_steps=1000)
image.images[0].show()
#%%
