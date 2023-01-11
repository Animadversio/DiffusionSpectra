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

from core.diffusion_geometry_lib import proj2subspace, proj2orthospace, subspace_variance, \
        trajectory_geometry_pipeline, diff_cosine_mat_analysis, \
        latent_PCA_analysis, latent_diff_PCA_analysis, PCA_data_visualize, ldm_PCA_data_visualize
from core.diffusion_traj_analysis_lib import \
    denorm_std, denorm_var, denorm_sample_std, \
    latents_to_image, latentvecs_to_image, \
    compute_save_diff_imgs_diff, compute_save_diff_imgs_ldm, plot_diff_matrix, visualize_traj_2d_cycle

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
import matplotlib
matplotlib.use('Agg')
# use the interactive backend

# matplotlib.use('module://backend_interagg')
#%%
import platform
if platform.system() == "Windows":
    saveroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion"
elif platform.system() == "Linux":
    saveroot = r"/home/binxuwang/insilico_exp/Diffusion_traj/StableDiffusion"
# prompt = "a cute and classy mice wearing dress and heels"
# prompt = "a beautiful ballerina in yellow dress under the starry night in Van Gogh style"
# prompt = "a classy ballet flat with a bow on the toe on a wooden floor"
# prompt = "a cat riding a motor cycle in a desert in a bright sunny day"
# prompt = "a bowl of soup looks like a portal to another dimension"
# prompt = "a box containing an apple and a toy teddy bear"
# prompt = "a photo of a photo of a photo of a photo of a cute dog"
prompt = "a large box containing an apple and a toy teddy bear"
tsteps = 51
for seed in range(102, 150):
    latents_reservoir = []
    @torch.no_grad()
    def save_latents(i, t, latents):
        latents_reservoir.append(latents.detach().cpu())

    # seed = 45
    out = pipe(prompt, callback=save_latents,
               num_inference_steps=tsteps, generator=torch.cuda.manual_seed(seed))
    # out.images[0].show()
    latents_reservoir = torch.cat(latents_reservoir, dim=0)
    # savedir = rf"F:\insilico_exps\Diffusion_traj\StableDiffusion\mice_dress_heels-seed{seed}"
    savedir = join(saveroot, f"box_apple_bear-seed{seed}")
    os.makedirs(savedir, exist_ok=True)

    torch.save(latents_reservoir, join(savedir, "latents_reservoir.pt"))
    json.dump({"prompt": prompt, "tsteps": tsteps, "seed": seed}, open(join(savedir, "prompt.json"), "w"))


    """ Correlogram of the latent state difference """
    diff_cosine_mat_analysis(latents_reservoir, savedir, lags=(1,2,3,4,5,10))
    """Geometry of the trajectory in 2d projection"""
    trajectory_geometry_pipeline(latents_reservoir, savedir)
    visualize_traj_2d_cycle(latents_reservoir, pipe, savedir)
    """PCA analysis of the latent state / difference"""
    expvar, U, D, V = latent_PCA_analysis(latents_reservoir, savedir,)
    expvar_diff, U_diff, D_diff, V_diff = latent_diff_PCA_analysis(latents_reservoir, savedir,
                               proj_planes=[(i, j) for i in range(8) for j in range(i+1, 8)])
    ldm_PCA_data_visualize(latents_reservoir, pipe, U, D, V, savedir, topcurv_num=8, topImg_num=8, prefix="latent_traj")
    ldm_PCA_data_visualize(latents_reservoir, pipe, U_diff, D_diff, V_diff, savedir, topcurv_num=8, topImg_num=8, prefix="latent_diff")
    torch.save({"expvar": expvar, "U": U, "D": D, "V": V}, join(savedir, "latent_PCA.pt"))
    torch.save({"expvar_diff": expvar_diff, "U_diff": U_diff, "D_diff": D_diff, "V_diff": V_diff}, join(savedir, "latent_diff_PCA.pt"))
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
    plt.close('all')
