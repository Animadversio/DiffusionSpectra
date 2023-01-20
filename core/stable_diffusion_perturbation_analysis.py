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
from core.utils.plot_utils import saveallforms
import platform
if platform.system() == "Windows":
    saveroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_perturb"
elif platform.system() == "Linux":
    saveroot = r"/home/binxuwang/insilico_exp/Diffusion_traj/StableDiffusion_perturb"
else:
    raise RuntimeError("Unknown system")
prompt, dirname = ("a portrait of an aristocrat", "portrait_aristocrat")
seed = 100
savedir = join(saveroot, f"{dirname}-seed{seed}")
#%%
data_orig = torch.load(join(savedir, "latents_noise_trajs.pt"))
traj_orig = data_orig["latents_traj"]
residue_orig = data_orig["residue_traj"]
#%%
prefix = "PC"
# prefix = "RND"
perturb_scales = [-20.0, -15.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
# perturb_scales = [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0,]
# for iPC in [*range(0, 16), *range(45, 51), ]: # *range(45, 50)
for iPC in range(15): # *range(45, 50)
    # iPC = 1
    traj_col = {}
    res_col = {}
    img_col = {}
    for inject_step in range(0, 51, 5):
        for pert_scale in perturb_scales:
            if prefix == "PC":
                img = plt.imread(join(savedir, f"sample_{prefix}{iPC:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.png"))
                data = torch.load(join(savedir, f"latent_{prefix}{iPC:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.pt"))
            elif prefix == "RND":
                img = plt.imread(join(savedir, f"sample_{prefix}{iPC:03d}_T{inject_step:02d}_scale{pert_scale:.1f}.png"))
                data = torch.load(join(savedir, f"latent_{prefix}{iPC:03d}_T{inject_step:02d}_scale{pert_scale:.1f}.pt"))
            traj_col[(inject_step, pert_scale)] = data["latents_traj"]
            res_col[(inject_step, pert_scale)] = data["residue_traj"]
            img_col[(inject_step, pert_scale)] = img
    #%%
    plt.figure(figsize=(8, 6))
    for T in range(0, 51, 5):
        for pert_scale in perturb_scales:
            traj_diff = traj_col[(T, pert_scale)] - traj_orig
            diff_norm_curv = traj_diff.flatten(1).double().norm(dim=1)
            plt.plot(diff_norm_curv, label=f"T{T} Scale {pert_scale:.1f}", alpha=0.7)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Norm of deviation")
    plt.xlabel("Time step")
    plt.title(f"Norm of deviation of latent space trajectory for {prefix}{iPC}", fontsize=16)
    plt.tight_layout()
    saveallforms(join(savedir, "summary"), f"traj_deviation_norm_{prefix}{iPC:02d}")
    # plt.savefig(join(savedir, "summary", f"traj_deviation_norm_{prefix}{iPC:02d}.png"))
    plt.show()
    #%%
    perturb_latent = ((traj_col[(5, 1.0)] - traj_col[(5, -1.0)])[6] / 2).double()
    perturb_vec = perturb_latent.flatten(1)
    #%%
    plt.figure(figsize=(8, 6))
    for T in range(0, 51, 5):
        for pert_scale in perturb_scales:
            traj_diff = traj_col[(T, pert_scale)] - traj_orig
            diff_proj_curv = traj_diff.flatten(1).double() @ perturb_vec.T
            plt.plot(diff_proj_curv, label=f"T{T} Scale {pert_scale:.1f}", alpha=0.7)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Deviation projection on perturbation vector")
    plt.xlabel("Time step")
    plt.title(f"Deviation projection onto perturb vector \nlatent space trajectory for {prefix}{iPC}", fontsize=16)
    plt.tight_layout()
    saveallforms(join(savedir, "summary"), f"traj_deviation_projcoef_{prefix}{iPC:02d}")
    plt.show()

#%%
torch.allclose((traj_col[(5, 1.0)] - traj_orig)[6],
               (traj_orig - traj_col[(5, -1.0)])[6], rtol=1E-3, atol=1E-3)


