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

# model_id = "google/ddpm-cat-256"
# model_id = "google/ddpm-ema-cat-256"
# model_id = "google/ddpm-bedroom-256"
# model_id = "google/ddpm-ema-bedroom-256"
# model_id = "google/ddpm-church-256"
# model_id = "google/ddpm-ema-church-256"
# model_id = "google/ddpm-ema-celebahq-256"
model_id = "google/ddpm-celebahq-256" # most popular
model_id_short = model_id.split("/")[-1]
saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}"
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


seed = 1520
tsteps = 51
out = pipe(callback=save_latents, num_inference_steps=tsteps,
           generator=torch.cuda.manual_seed(seed))
out.images[0].show()
latents_reservoir = torch.cat(latents_reservoir, dim=0)
#%% Utility functions for analysis
import json
from core.diffusion_geometry_lib import proj2subspace, proj2orthospace, subspace_variance, \
    trajectory_geometry_pipeline, latent_PCA_analysis, latent_diff_PCA_analysis, diff_cosine_mat_analysis
from core.diffusion_traj_analysis_lib import compute_save_diff_imgs_diff, plot_diff_matrix
#%%
savedir = join(saveroot, f"seed{seed}")
os.makedirs(savedir, exist_ok=True)
torch.save(latents_reservoir, join(savedir, "state_reservoir.pt"))
json.dump({"tsteps": tsteps, "seed": seed}, open(join(savedir, "prompt.json"), "w"))
#%%
trajectory_geometry_pipeline(latents_reservoir, savedir, )
diff_cosine_mat_analysis(latents_reservoir, savedir, )
expvar_vec, U, D, V = latent_PCA_analysis(latents_reservoir, savedir, )
expvar_diff, U_diff, D_diff, V_diff = latent_diff_PCA_analysis(latents_reservoir, savedir, )
#%%
compute_save_diff_imgs_diff(savedir, range(0, 51, 5), latents_reservoir)
plot_diff_matrix(savedir, range(0, 51, 5), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
                 save_sfx="_img_stdnorm", tril=True)
#%%
compute_save_diff_imgs_diff(savedir, range(0, 16, 1), latents_reservoir)
plot_diff_matrix(savedir, range(0, 16, 1), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
                 save_sfx="_img_stdnorm_early0-15", tril=True)
#%%
compute_save_diff_imgs_diff(savedir, range(0, 101, 10), latents_reservoir)
plot_diff_matrix(savedir, range(0, 101, 10), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
                 save_sfx="_img_stdnorm_101", tril=True)



#%%
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
# model_id = "CompVis/ldm-text2im-large-256"
model_id = "CompVis/ldm-celebahq-256"
# load model and scheduler
pipeline = DiffusionPipeline.from_pretrained(model_id)
# run pipeline in inference (sample random noise and denoise)
image = pipeline(num_inference_steps=200)["sample"]