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
# model_id = "google/ddpm-cifar10-32"
model_id = "dimpo/ddpm-mnist"  # most popular
# model_id = "google/ddpm-celebahq-256" # most popular
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


for seed in range(200, 400):
    latents_reservoir = []

    @torch.no_grad()
    def save_latents(i, t, latents):
        latents_reservoir.append(latents.detach().cpu())

    tsteps = 51
    out = pipe(callback=save_latents, num_inference_steps=tsteps,
               generator=torch.cuda.manual_seed(seed))
    latents_reservoir = torch.cat(latents_reservoir, dim=0)
    #%%
    savedir = join(saveroot, f"seed{seed}")
    os.makedirs(savedir, exist_ok=True)
    torch.save(latents_reservoir, join(savedir, "state_reservoir.pt"))
    json.dump({"tsteps": tsteps, "seed": seed}, open(join(savedir, "prompt.json"), "w"))
    out.images[0].save(join(savedir, "sample0.png"))
    images = (latents_reservoir[50] / 2 + 0.5).clamp(0, 1)
    save_imgrid(images, join(savedir, f"samples_all.png"))
    #%%
    trajectory_geometry_pipeline(latents_reservoir, savedir, )
    diff_cosine_mat_analysis(latents_reservoir, savedir, )
    expvar_vec, U, D, V = latent_PCA_analysis(latents_reservoir, savedir, )
    expvar_diff, U_diff, D_diff, V_diff = latent_diff_PCA_analysis(latents_reservoir, savedir, )
    PCA_data_visualize(latents_reservoir, U, D, V, savedir, topcurv_num=8, topImg_num=16, prefix="latent_traj")
    PCA_data_visualize(latents_reservoir, U_diff, D_diff, V_diff, savedir, topcurv_num=8, topImg_num=16, prefix="latent_diff")
    #%%
    compute_save_diff_imgs_diff(savedir, range(0, 51, 5), latents_reservoir)
    plot_diff_matrix(savedir, range(0, 51, 5), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
                     save_sfx="_img_stdnorm", tril=True)
    #%%
    compute_save_diff_imgs_diff(savedir, range(0, 16, 1), latents_reservoir)
    plot_diff_matrix(savedir, range(0, 16, 1), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
                     save_sfx="_img_stdnorm_early0-15", tril=True)

