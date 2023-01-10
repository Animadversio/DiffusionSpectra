import json
import math
import os
from os.path import join
from tqdm import tqdm
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from diffusers import pipelines, StableDiffusionPipeline
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
# exproot = r"/home/binxuwang/insilico_exp/Diffusion_Hessian/StableDiffusion"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
# pipeline.to(torch.half)
def dummy_checker(images, **kwargs): return images, False

pipe.safety_checker = dummy_checker
#%%
# with torch.autocast("cuda"):
latents_reservoir = []
@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())


seed = 45
tsteps = 51
# prompt = "a cute and classy mice wearing dress and heels"
# prompt = "a beautiful ballerina in yellow dress under the starry night in Van Gogh style"
# prompt = "a classy ballet flat with a bow on the toe on a wooden floor"
prompt = "a cat riding a motor cycle in a desert in a bright sunny day"
prompt = "a bowl of soup looks like a portal to another dimension"
out = pipe(prompt, callback=save_latents,
           num_inference_steps=tsteps, generator=torch.cuda.manual_seed(seed))
out.images[0].show()
latents_reservoir = torch.cat(latents_reservoir, dim=0)
#%% Utility functions for analysis
from core.diffusion_geometry_lib import proj2subspace, proj2orthospace, subspace_variance, \
        trajectory_geometry_pipeline, diff_cosine_mat_analysis, \
        latent_PCA_analysis, latent_diff_PCA_analysis
from core.diffusion_traj_analysis_lib import \
    denorm_std, denorm_var, denorm_sample_std, \
    latents_to_image, latentvecs_to_image, \
    compute_save_diff_imgs_diff, compute_save_diff_imgs_ldm, plot_diff_matrix

#%%
# savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion\mice_dress_heels1"
savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion\ballerina_van_gogh"
savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion\ballet_flats"
savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion\cat_motorcycle"
savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion\bowl_portal"
os.makedirs(savedir, exist_ok=True)

torch.save(latents_reservoir, join(savedir, "latents_reservoir.pt"))
json.dump({"prompt": prompt, "tsteps": tsteps, "seed": seed}, open(join(savedir, "prompt.json"), "w"))
#%%
def visualize_traj_2d_cycle(latents_reservoir, pipe, savedir, ticks=range(0,360,10)):
    """Plot the 2d cycle of the latent states plane of trajectory"""
    init_latent = latents_reservoir[:1].flatten(1).float()
    end_latent = latents_reservoir[-1:].flatten(1).float()
    unitbasis1 = end_latent / end_latent.norm()  # end state
    unitbasis2 = proj2orthospace(end_latent, init_latent)  # init noise that is ortho to the end state
    unitbasis2 = unitbasis2 / unitbasis2.norm()  # unit normalize
    imgtsrs = []
    for phi in tqdm(ticks):
        phi = phi * np.pi / 180
        imgtsr = latentvecs_to_image((unitbasis2 * math.sin(phi) +
                                      unitbasis1 * math.cos(phi)) * end_latent.norm(), pipe)
        imgtsrs.append(imgtsr)
    imgtsrs = torch.cat(imgtsrs, dim=0)
    show_imgrid(imgtsrs, nrow=9)
    save_imgrid(imgtsrs, join(savedir, "latent_2d_cycle_visualization.png"), nrow=9)
#%%
compute_save_diff_imgs_ldm(savedir, range(0, 51, 5), latents_reservoir, pipe, )
plot_diff_matrix(savedir, range(0, 51, 5), diff_x_sfx="_vae_decode",  step_x_sfx="_vae_decode",
                                            save_sfx="_vae_decode")
plot_diff_matrix(savedir, range(0, 51, 5), diff_x_sfx="_vae_decode_stdfinal",  step_x_sfx="_vae_decode",
                                            save_sfx="_vae_decode_stdfinal")
plot_diff_matrix(savedir, range(0, 51, 5), diff_x_sfx="_latent_stdnorm",  step_x_sfx="_latent_stdnorm",
                                            save_sfx="_latent_stdnorm")
#%%
compute_save_diff_imgs_ldm(savedir, range(0, 16, 1), latents_reservoir, pipe, )
plot_diff_matrix(savedir, range(0, 16, 1), diff_x_sfx="_vae_decode", step_x_sfx="_vae_decode",
                                            save_sfx="_vae_decode_early0-15")
plot_diff_matrix(savedir, range(0, 16, 1), diff_x_sfx="_vae_decode_stdfinal", step_x_sfx="_vae_decode",
                                            save_sfx="_vae_decode_stdfinal_early0-15")
plot_diff_matrix(savedir, range(0, 16, 1), diff_x_sfx="_latent_stdnorm", step_x_sfx="_latent_stdnorm",
                                            save_sfx="_latent_stdnorm_early0-15")

#%%
compute_save_diff_imgs_ldm(savedir, range(0, 21, 2), latents_reservoir, pipe, )
plot_diff_matrix(savedir, range(0, 21, 2),
                 diff_x_sfx="_vae_decode", step_x_sfx="_vae_decode", save_sfx="_vae_decode_early0-20")
plot_diff_matrix(savedir, range(0, 21, 2),
                 diff_x_sfx="_vae_decode_stdfinal", step_x_sfx="_vae_decode", save_sfx="_vae_decode_stdfinal_early0-20")
plot_diff_matrix(savedir, range(0, 21, 2),
                 diff_x_sfx="_latent_stdnorm", step_x_sfx="_latent_stdnorm", save_sfx="_latent_stdnorm_early0-20")

#%%
""" Correlogram of the latent state difference """

#%%
"""Geometry of the trajectory in 2d projection"""
diff_cosine_mat_analysis(latents_reservoir, savedir, lags=(1,2,3,4,5,10))
"""Geometry of the trajectory in 2d projection"""
trajectory_geometry_pipeline(latents_reservoir, savedir)
visualize_traj_2d_cycle(latents_reservoir, pipe, savedir)
#%%
"""PCA analysis of the latent state / difference"""
latent_PCA_analysis(latents_reservoir, savedir,)
latent_diff_PCA_analysis(latents_reservoir, savedir,)


#%%
latents_norm = latents_reservoir.flatten(1).double().norm(dim=1)
# latent_residue01 = proj2orthospace(latents_reservoir[[0,2,4],:].flatten(1).double(), latents_reservoir.flatten(1).double())
latent_proj01 = proj2subspace(latents_reservoir[range(10),:].flatten(1).double(), latents_reservoir.flatten(1).double())
latent_proj01 = proj2subspace(torch.stack((latents_reservoir[0],
                                           100 * latents_reservoir[2] - latents_reservoir[0],) ).flatten(1).double(), latents_reservoir.flatten(1).double())
# latent_residue02 = proj2orthospace(latents_reservoir[:3,:].flatten(1).float(), latents_reservoir.flatten(1).float())
# plt.plot(latent_residue01.norm(dim=1)**2 / latents_norm**2, label="subspace 01")
plt.plot(latent_proj01.norm(dim=1)**2 / latents_norm**2, label="subspace 01")
# plt.plot(latent_residue02.norm(dim=1) / latents_norm, label="subspace 012")
plt.legend()
plt.title("residue of the latent state in the subspace spanned by the first 2 or 3 states")
plt.xlabel("t")
plt.ylabel("residue norm / latent norm")
# saveallforms(savedir, "residue_norm", plt.gcf())
plt.show()
#%%
"""show the images corresponding to projections to the latent space"""
show_imgrid(latentvecs_to_image(proj2subspace(latents_reservoir[[0,-1]].flatten(1).double(),
                                    latents_reservoir[[2]].flatten(1).double()), pipe))
#%%
subspace_variance(latents_reservoir[[1,-1]].double(), latents_reservoir[[0,1]].double())
#%%
subspace_variance(latents_reservoir[[1,2,-2]].double(), latents_reservoir[[0,-1]].double())
#%%
subspace_variance(latents_reservoir[[1,2,-2]].double(), latents_reservoir[[0]].double())
#%%



#%%
"""test if correlation and coef signal the same thing?"""
sns.heatmap(torch.corrcoef(latents_reservoir.flatten(1).float(), ))
plt.show()
#%%
sns.heatmap(torch.corrcoef((latents_reservoir[1:]-latents_reservoir[:-1]).flatten(1).float(), ))
plt.show()
