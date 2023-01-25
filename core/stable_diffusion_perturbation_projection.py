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
#%%
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
import platform
if platform.system() == "Windows":
    saveroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_perturb"
    outroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_perturb_proj"
elif platform.system() == "Linux":
    saveroot = r"/home/binxuwang/insilico_exp/Diffusion_traj/StableDiffusion_perturb"
    outroot = r"/home/binxuwang/insilico_exp/Diffusion_traj/StableDiffusion_perturb_proj"
else:
    raise RuntimeError("Unknown system")
#%%
prompt_dir_pair = [
    ("a portrait of an aristocrat", "portrait_aristocrat"),
    ("a portrait of an light bulb", "portrait_lightbulb"),
    ("a large box containing an apple and a toy teddy bear", "box_apple_bear"),
    ("a photo of a cat sitting with a dog on a cozy couch", "cat_dog_couch"),
    ("a CG art of a brain composed of eletronic wires and circuits", "brain_wire_circuits"),
    ("a handsome cat dancing Tango with a female dancer in Monet style", "cat_tango_dancer"),
    ("a bug crawling on a textbook under a bright light, photo", "bug_book_photo"),
]
#%%
tsteps = 51
prompt, dirname = ("a portrait of an aristocrat", "portrait_aristocrat")
seed = 101
prompt, dirname = ("a large box containing an apple and a toy teddy bear", "box_apple_bear")
seed = 130

savedir = join(saveroot, f"{dirname}-seed{seed}")
outdir = join(outroot, f"{dirname}-seed{seed}")
os.makedirs(outdir, exist_ok=True)
pipe.scheduler.set_timesteps(tsteps)
# image, latents_traj, residue_traj, noise_uncond_traj, noise_text_traj = SD_sampler_perturb(pipe, prompt,
#            num_inference_steps=tsteps, generator=torch.cuda.manual_seed(seed),
#            perturb_latent=None, perturb_step=10, pert_scale=1.0, )
# for iPC in [0, 1, 2, 10, 45, 49]:#[*range(0, 16), *range(45, 51), ]: # *range(45, 50)
#     for inject_step in range(0, 51, 5):
#         for pert_scale in [-10.0, -5.0, 5.0, 10.0]:
#             perturb_data = torch.load(join(savedir, f"latent_PC{iPC:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.pt"))
#             latents_traj_perturb = perturb_data["latents_traj"]
#             residue_traj_perturb = perturb_data["residue_traj"]
#             t_traj = pipe.scheduler.timesteps.cpu()
#             alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
#             pred_z0 = (latents_traj_perturb[:-1] -
#                        residue_traj_perturb * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1, 1)) / \
#                       alphacum_traj.sqrt().view(-1, 1, 1, 1, 1)
#             pred_z0_traj = latents_to_image(pred_z0[:, 0].half().to('cuda'), pipe, batch_size=11)
#             save_imgrid(pred_z0_traj, join(outdir, f"proj_z0_decode_PC{iPC:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.jpg"), nrow=10, )


for RNDseed in range(5,15):#range(0, 5):
    for inject_step in [50]:#range(0, 51, 5):
        for pert_scale in [-10.0, -5.0, 5.0, 10.0]:
            perturb_data = torch.load(join(savedir, f"latent_RND{RNDseed:03d}_T{inject_step:02d}_scale{pert_scale:.1f}.pt"))
            latents_traj_perturb = perturb_data["latents_traj"]
            residue_traj_perturb = perturb_data["residue_traj"]
            t_traj = pipe.scheduler.timesteps.cpu()
            alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
            pred_z0 = (latents_traj_perturb[:-1] -
                       residue_traj_perturb * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1, 1)) / \
                      alphacum_traj.sqrt().view(-1, 1, 1, 1, 1)
            pred_z0_traj = latents_to_image(pred_z0[:, 0].half().to('cuda'), pipe, batch_size=11)
            save_imgrid(pred_z0_traj,
                        join(outdir, f"proj_z0_decode_RND{RNDseed:03d}_T{inject_step:02d}_scale{pert_scale:.1f}.jpg"),
                        nrow=10, )

# #%
# show_img(image[0])
# image[0].save(join(savedir, "sample_orig.png"))