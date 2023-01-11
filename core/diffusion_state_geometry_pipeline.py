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
#%%
compute_save_diff_imgs_diff(savedir, range(0, 101, 10), latents_reservoir)
plot_diff_matrix(savedir, range(0, 101, 10), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
                 save_sfx="_img_stdnorm_101", tril=True)


#%%
"""Most simple model"""
# model_id = "google/ddpm-cifar10-32"
model_id = "dimpo/ddpm-mnist"  # most popular
model_id_short = model_id.split("/")[-1]
saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}"
# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.unet.requires_grad_(False).eval().to("cuda")#.half()
#%%
# image = pipe(batch_size=64)
# show_imgrid(latents_reservoir[50] * 0.5 + 0.5)
#%%
latents_reservoir = []
@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())


seed = 120
tsteps = 51
out = pipe(callback=save_latents, num_inference_steps=tsteps, batch_size=1,
           generator=torch.cuda.manual_seed(seed))
out.images[0].show()
latents_reservoir = torch.stack(latents_reservoir, dim=0)
show_imgrid(latents_reservoir[50] * 0.5 + 0.5)
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
PCA_data_visualize(latents_reservoir, U, D, V, savedir, topcurv_num=8, topImg_num=16, prefix="latent_traj")
PCA_data_visualize(latents_reservoir, U_diff, D_diff, V_diff, savedir, topcurv_num=8, topImg_num=16, prefix="latent_diff")


#%%
compute_save_diff_imgs_diff(savedir, range(0, 16, 1), latents_reservoir)
plot_diff_matrix(savedir, range(0, 16, 1), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
                 save_sfx="_img_stdnorm_early0-15", tril=True)
#%%









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
#%%
#%% Dev zone
expvar_diff, U_diff, D_diff, V_diff = latent_diff_PCA_analysis(latents_reservoir, savedir, )
topPC_num = 8
plt.figure()
plt.plot(U_diff[:,:topPC_num] * D_diff[:topPC_num], lw=2.5, alpha=0.7)
plt.legend([f"PC{i+1}" for i in range(topPC_num)])
plt.title("PCs of the latent state difference")
plt.ylabel("PC projection")
plt.xlabel("Time step")
saveallforms(savedir, "latent_diff_PCA_projcurve")
plt.show()
plt.figure()
plt.plot(U_diff[:,:topPC_num], lw=2.5, alpha=0.7)
plt.legend([f"PC{i+1}" for i in range(topPC_num)])
plt.title("PCs of the latent state difference")
plt.ylabel("PC projection (norm 1)")
plt.xlabel("Time step")
saveallforms(savedir, "latent_diff_PCA_projcurve_norm1")
plt.show()
topPC_num = 16
PC_imgs = V_diff[:, :topPC_num].T
PC_imgs = PC_imgs.reshape(topPC_num, *latents_reservoir.shape[-3:])
PC_imgs_norm = (PC_imgs) / PC_imgs.std(dim=(1, 2, 3), keepdim=True) * 0.2 + 0.5
save_imgrid(PC_imgs_norm, join(savedir, "latent_diff_topPC_imgs_vis.png"), nrow=4, )
#%%
compute_save_diff_imgs_diff(savedir, range(0, 51, 5), latents_reservoir)
plot_diff_matrix(savedir, range(0, 51, 5), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
                 save_sfx="_img_stdnorm", tril=True)