import os
from os.path import join
import json
import torch
from tqdm import tqdm, trange
from diffusers import DiffusionPipeline
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler
from diffusers import PNDMPipeline, DDIMScheduler, UNet2DModel, PNDMScheduler
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
#% Utility functions for analysis
from core.diffusion_geometry_lib import proj2subspace, proj2orthospace, subspace_variance, \
    trajectory_geometry_pipeline, latent_PCA_analysis, latent_diff_PCA_analysis, diff_cosine_mat_analysis, PCA_data_visualize
from core.diffusion_traj_analysis_lib import compute_save_diff_imgs_diff, plot_diff_matrix
#%%
from diffusers import UNet2DModel
model_id = "google/ddpm-cifar10-32"
# model_id = "dimpo/ddpm-mnist"  # most popular
model_id_short = model_id.split("/")[-1]
# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.unet.requires_grad_(False).eval().to("cuda")#.half()
#%%
def sampling(unet, scheduler, batch_size=1, generator=None, noisy_sample=None):
    if noisy_sample is None:
        noisy_sample = torch.randn(
            batch_size, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size,
            generator=generator, device=unet.device
        ).to(unet.device).to(unet.dtype)
    else:
        noisy_sample = noisy_sample.to(unet.device).to(unet.dtype)
        assert noisy_sample.shape[0] == batch_size
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
    t_traj = t_traj.long().cpu()
    sample_traj = torch.cat(sample_traj, dim=0)
    residual_traj = torch.cat(residual_traj, dim=0)
    return sample, sample_traj, residual_traj, t_traj
#%%
imgshape = [3, 32, 32]
ndim = np.prod(imgshape)
saveroot = r"F:\insilico_exps\Diffusion_traj\ddpm-cifar10-32_scheduler"
outdir = r"F:\insilico_exps\Diffusion_traj\cifar_uncond_gmm_exact"
nameCls = "DDIM"
#%%
tsteps = 51
pipe.scheduler.set_timesteps(tsteps)
t_eval = np.linspace(1, 0, 51)
for seed in trange(200, 500):
    np.random.seed(seed)
    xT = np.random.randn(ndim).reshape(imgshape)[None, ...]
    sample, sample_traj, residual_traj, t_traj = sampling(pipe.unet, pipe.scheduler, batch_size=1,
                                      noisy_sample=torch.from_numpy(xT).float().cuda())
    # %%
    savedir = join(saveroot, nameCls, f"seed{seed}_np")
    os.makedirs(savedir, exist_ok=True)
    torch.save({"latents_traj": sample_traj,
                "residue_traj": residual_traj,
                "t_traj": t_traj
                }, join(savedir, "state_reservoir.pt"))
    json.dump({"tsteps": tsteps, "seed": seed}, open(join(savedir, "prompt.json"), "w"))

    alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
    pred_x0 = (sample_traj[:-1] - residual_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / \
              alphacum_traj.sqrt().view(-1, 1, 1, 1)
    pred_x0_imgs = (pred_x0 + 1) / 2

    save_imgrid(pred_x0_imgs, join(savedir, "proj_z0_vae_decode.jpg"), nrow=10, )

    images = (sample_traj[-1] / 2 + 0.5).clamp(0, 1)
    save_imgrid(images, join(savedir, f"samples_all.png"))
    save_imgrid(images, join(outdir, f"uncond_RND{seed:03d}_DDIM.png"))