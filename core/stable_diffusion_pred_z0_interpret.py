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
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
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
figdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\StableDiffusion_application"

# savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_projection\portrait_lightbulb-seed101"
# savedir = r"/home/binxu/insilico_exps/Diffusion_traj/StableDiffusion_projection/portrait_lightbulb-seed101"
# savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_projection\portrait_aristocrat-seed124"
#%%
from pathlib import Path
import platform
if platform.system() == "Windows":
    saveroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_projection"
elif platform.system() == "Linux":
    saveroot = r"/home/binxu/insilico_exps/Diffusion_traj/StableDiffusion_projection"
else:
    raise RuntimeError("Unknown system")

#%%
savedir = Path(saveroot) / "box_apple_bear-seed130"
shortname = savedir.name
data = torch.load(join(savedir, "latents_noise_trajs.pt"))
latents_traj = data["latents_traj"]
residue_traj = data["residue_traj"]
noise_uncond_traj = data["noise_uncond_traj"]
noise_text_traj = data["noise_text_traj"]
pred_z0_PCA = torch.load(join(savedir, "pred_z0_PCA.pt"))
D_pca = pred_z0_PCA["D"]
# #%%
# pred_z0_PCA["U"].shape
# pred_z0_PCA["D"].shape
# pred_z0_PCA["V"].shape
#%%
latents_z0 = latents_traj[-1]
RND_vec = torch.randn(latents_z0.shape)
RND_vec = RND_vec / RND_vec.flatten().norm()
perturbs_linear = latents_z0 + torch.linspace(-100, 100, 9)[:,None,None,None] * \
                    RND_vec
#%%
from tqdm import trange, tqdm
for PCi in trange(16):
    ticks = torch.linspace(-100, 100, 7)
    latents_z0 = latents_traj[-1]
    perturbs_linear = latents_z0 + ticks[:,None,None,None] * \
                      pred_z0_PCA["V"][:, PCi].view(latents_z0.shape)
    perturb_imgs = latents_to_image(perturbs_linear.half().to('cuda'), pipe, batch_size=4)
    #%%
    plt.figure(figsize=[20, 4])
    plt.imshow(to_imgrid(perturb_imgs, nrow=11))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    to_imgrid(perturb_imgs, nrow=11).save(join(figdir, f"perturb_imgs_{shortname}_PC{PCi}.jpg"))
#%%
from tqdm import trange, tqdm
for PCi in trange(8):
    ticks = torch.linspace(-3*D_pca[PCi].sqrt(), 3*D_pca[PCi].sqrt(), 7)
    latents_z0 = latents_traj[-1]
    perturbs_linear = latents_z0 + ticks[:,None,None,None] * \
                      pred_z0_PCA["V"][:, PCi].view(latents_z0.shape)
    perturb_imgs = latents_to_image(perturbs_linear.half().to('cuda'), pipe, batch_size=4)
    #%%
    plt.figure(figsize=[20, 4])
    plt.imshow(to_imgrid(perturb_imgs, nrow=11))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    to_imgrid(perturb_imgs, nrow=11).save(join(figdir, f"perturb_imgs_{shortname}_PC{PCi}_std.jpg"))

#%%
latents_z0 = latents_traj[-1]
# 2d perturbations
perturbs_2d = []
PCi, PCj = 2, 3
# iticks = torch.linspace(-100, 100, 7)
# jticks = torch.linspace(-100, 100, 7)
iticks = torch.linspace(-5*D_pca[PCi].sqrt(), 5*D_pca[PCi].sqrt(), 7)
jticks = torch.linspace(-5*D_pca[PCi].sqrt(), 5*D_pca[PCi].sqrt(), 7)
for i in range(len(iticks)):
    perturbs_2d.append(latents_z0 + iticks[i] * pred_z0_PCA["V"][:, PCi].view(latents_z0.shape) +
                           jticks[:,None,None,None] * pred_z0_PCA["V"][:, PCj].view(latents_z0.shape))
perturbs_2d = torch.concatenate(perturbs_2d, dim=0)
perturb_imgs_all2d = latents_to_image(perturbs_2d.half().to('cuda'), pipe, batch_size=4)
#%%
plt.figure(figsize=[20, 20])
plt.imshow(to_imgrid(perturb_imgs_all2d, nrow=len(iticks)))
plt.axis('off')
plt.tight_layout()
plt.show()
to_imgrid(perturb_imgs_all2d, nrow=7).save(join(figdir, f"perturb_imgs_all2d_{shortname}_PC{PCi}-{PCj}_std.jpg"))
#%%
# to_imgrid(perturb_imgs_all2d, nrow=7).save(join(outdir, "perturb_imgs_all2d_PC12_portrait2bulb.jpg"))


#%%
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