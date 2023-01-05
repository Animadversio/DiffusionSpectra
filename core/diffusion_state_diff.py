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
show_imgrid((sample_traj[50] - sample_traj[0] + 1) / 2)
#%%
show_imgrid((sample_traj[65] - sample_traj[45] + 1) / 2)
#%%
show_imgrid((sample_traj[75] - sample_traj[50] + 1) / 2)
#%%
show_imgrid((sample_traj[99] - sample_traj[90] + 1) / 2)
#%%
def denorm(img):
    return (img + 1) / 2


def denorm_std(img):
    """a hacky way to make the image with std=0.2 mean=0.5 similar to ImageNet"""
    return (img / img.std() * 0.4 + 1) / 2
sumdir = r"F:\insilico_exps\Diffusion_traj\cifar10-32\summary"
os.makedirs(sumdir, exist_ok=True)
#%%
savedir = r"F:\insilico_exps\Diffusion_traj\cifar10-32\exp1"
os.makedirs(savedir, exist_ok=True)
for i in [*range(0, 100, 10), 99]:
    save_imgrid(denorm(sample_traj[i]),
                    join(savedir, f"diffusion_step_{i:02d}.png"))
    for j in [*range(i+10, 100, 10), 99]:
        save_imgrid(denorm(sample_traj[j] - sample_traj[i]),
                    join(savedir, f"diffusion_traj_{j:02d}-{i:02d}.png"))
        save_imgrid(denorm_std(sample_traj[j] - sample_traj[i]),
                    join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_std.png"))
    # show_imgrid((sample_traj[i] + 1) / 2)
#%%
import matplotlib.pyplot as plt
figh, axs = plt.subplots(11, 11, figsize=(20, 21))
for i, ax in enumerate(axs.flatten()):
    ax.axis("off")

for i in [*range(0, 100, 10), ]:
    for j in [*range(i+10, 100, 10), 99]:
        axs[(i+1)//10 + 1, (j+1)//10].imshow(plt.imread(join(savedir, f"diffusion_traj_{j:02d}-{i:02d}.png"))) # make_grid(denorm(sample_traj[j] - sample_traj[i]),).permute(1, 2, 0)
        axs[(i+1)//10 + 1, (j+1)//10].set_title(f"{j}-{i}")

for i in [*range(0, 100, 10), 99]:
    axs[0, (i+1)//10,].imshow(plt.imread(join(savedir, f"diffusion_step_{i:02d}.png")))
    axs[0, (i+1)//10,].set_title(f"t={i}")

plt.suptitle("x difference along Trajectories", fontsize=16)
plt.tight_layout()
saveallforms(savedir, "diffusion_traj_diff_mtg.png", figh)
plt.show()
#%%
figh, axs = plt.subplots(11, 11, figsize=(20, 21))
for i, ax in enumerate(axs.flatten()):
    ax.axis("off")

for i in [*range(0, 100, 10), ]:
    for j in [*range(i+10, 100, 10), 99]:
        axs[(i+1)//10 + 1, (j+1)//10].imshow(plt.imread(join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_std.png"))) # make_grid(denorm(sample_traj[j] - sample_traj[i]),).permute(1, 2, 0)
        axs[(i+1)//10 + 1, (j+1)//10].set_title(f"{j}-{i}")

for i in [*range(0, 100, 10), 99]:
    axs[0, (i+1)//10,].imshow(plt.imread(join(savedir, f"diffusion_step_{i:02d}.png")))
    axs[0, (i+1)//10,].set_title(f"t={i}")

plt.suptitle("Normalized x (std=0.2) difference along Trajectories", fontsize=16)
plt.tight_layout()
saveallforms(savedir, "diffusion_traj_diff_mtg_stdnorm.png", figh)
plt.show()
#%%
show_imgrid((sample_traj[99] + 1) / 2)