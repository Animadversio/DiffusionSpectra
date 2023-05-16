import json
import math
import os
from os.path import join
from tqdm import tqdm
import torch
import numpy as np
import seaborn as sns
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from diffusers import pipelines, StableDiffusionPipeline, PNDMScheduler, DDIMScheduler
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid

from core.diffusion_geometry_lib import proj2subspace, proj2orthospace, subspace_variance, \
        trajectory_geometry_pipeline, diff_cosine_mat_analysis, \
        latent_PCA_analysis, latent_diff_PCA_analysis, PCA_data_visualize, ldm_PCA_data_visualize
#%%
import platform
if platform.system() == "Windows":
    saveroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_projection"
elif platform.system() == "Linux":
    saveroot = r"/home/binxuwang/insilico_exp/Diffusion_traj/StableDiffusion_projection"
else:
    raise RuntimeError("Unknown system")

prompt_dir_pair = [
    ("a portrait of an aristocrat", "portrait_aristocrat"),
    ("a portrait of an light bulb", "portrait_lightbulb"),
    ("a large box containing an apple and a toy teddy bear", "box_apple_bear"),
    ("a photo of a cat sitting with a dog on a cozy couch", "cat_dog_couch"),
    ("a CG art of a brain composed of eletronic wires and circuits", "brain_wire_circuits"),
    ("a handsome cat dancing Tango with a female dancer in Monet style", "cat_tango_dancer"),
    ("a bug crawling on a textbook under a bright light, photo", "bug_book_photo"),
]
from core.diffusion_geometry_lib import state_PCA_compute
tsteps = 51
PCA_col = {}
# pipe.scheduler.set_timesteps(tsteps, )
for prompt, dirname in prompt_dir_pair[:]:
    D_col = []
    U_col = []
    for seed in trange(100, 150):
        # image, latents_traj, residue_traj, noise_uncond_traj, noise_text_traj = SD_sampler(pipe, prompt,
        #            num_inference_steps=tsteps, generator=torch.cuda.manual_seed(seed))
        savedir = join(saveroot, f"{dirname}-seed{seed}")
        # pred_z0_PCA = state_PCA_compute(pred_z0, savedir, "pred_z0")
        pred_z0_PCA = torch.load(join(savedir, f"pred_z0_PCA.pt"))
        D_col.append(pred_z0_PCA['D'])
        U_col.append(pred_z0_PCA['U'])
        # raise Exception("Not finished")
    D_col = torch.stack(D_col)
    U_col = torch.stack(U_col)
    PCA_col[dirname] = {'D': D_col, 'U': U_col}
#%%
plt.figure(figsize=(5, 6))
for prompt, dirname in prompt_dir_pair[:]:
    Dmat = PCA_col[dirname]['D']
    expvarmat = (Dmat ** 2) / (Dmat ** 2).sum(dim=1, keepdim=True)
    # plt.plot(PCA_col[dirname]['D'].mean(0).numpy(), label=dirname)
    # # plot quantile range 25 75
    # plt.fill_between(np.arange(52),
    #                     PCA_col[dirname]['D'].quantile(0.25, 0).numpy(),
    #                     PCA_col[dirname]['D'].quantile(0.75, 0).numpy(),
    #                     alpha=0.2)
    plt.plot(expvarmat.mean(0).numpy(), label=dirname, alpha=0.7, lw=2)
    # plot quantile range 25 75
    plt.fill_between(np.arange(52),
                     expvarmat.quantile(0.25, 0).numpy(),
                     expvarmat.quantile(0.75, 0).numpy(),
                     alpha=0.25)
plt.xlim(0, 8)
plt.ylabel("Explained variance")
plt.xlabel("PCs")
plt.legend()
plt.show()
#%%
for prompt, dirname in prompt_dir_pair[:]:
    plt.figure(figsize=(6, 5))
    sns.heatmap(PCA_col[dirname]['U'].mean(dim=0), cmap="YlGnBu")
    plt.title(dirname)
    plt.axis("image")
    plt.show()
#%%
for prompt, dirname in prompt_dir_pair[:]:
    plt.figure(figsize=(6, 5))
    sns.heatmap(PCA_col[dirname]['U'][0], cmap="YlGnBu")
    plt.title(dirname)
    plt.axis("image")
    plt.show()
#%%
for prompt, dirname in prompt_dir_pair[:]:
    plt.figure(figsize=(6, 5))
    sns.heatmap((PCA_col[dirname]['D'][:,None]*PCA_col[dirname]['U']).abs().mean(dim=0), cmap="YlGnBu")
    plt.title(dirname)
    plt.axis("image")
    plt.show()
#%%
plt.figure()
plt.plot(pred_z0_PCA['U'][40:50, :].T)
plt.show()
#%%
plt.figure()
plt.plot(pred_z0_PCA['D'])
plt.show()
#%%
import seaborn as sns

plt.figure()
sns.heatmap(pred_z0_PCA['U'], cmap="YlGnBu")
plt.ylabel("time steps")
plt.xlabel("PCs")
plt.show()
#%%
plt.figure()
sns.heatmap(pred_z0_PCA['U']*pred_z0_PCA['D'][None], cmap="YlGnBu")
plt.ylabel("time steps")
plt.xlabel("PCs")
plt.show()
