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
from core.utils.plot_utils import saveallforms, show_imgrid, save_imgrid
from lpips import LPIPS
import platform
if platform.system() == "Windows":
    saveroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_perturb"
elif platform.system() == "Linux":
    saveroot = r"/home/binxuwang/insilico_exp/Diffusion_traj/StableDiffusion_perturb"
else:
    raise RuntimeError("Unknown system")


Dist = LPIPS(net='squeeze').cuda().eval()
Dist.requires_grad_(False)
#%%

#%%
def diffusion_orig_data_load(savedir):
    data_orig = torch.load(join(savedir, "latents_noise_trajs.pt"))
    traj_orig = data_orig["latents_traj"]
    residue_orig = data_orig["residue_traj"]
    img_orig = plt.imread(join(savedir, "sample_orig.png"))
    return traj_orig, residue_orig, img_orig


def diffusion_perturb_data_load(savedir, prefix, iPC, perturb_scales, inject_steps=range(0, 51, 5)):
    traj_col = {}
    res_col = {}
    img_col = {}
    for inject_step in tqdm(inject_steps):
        for pert_scale in tqdm(perturb_scales):
            try:
                if prefix == "PC":
                    img = plt.imread(
                        join(savedir, f"sample_{prefix}{iPC:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.png"))
                    data = torch.load(
                        join(savedir, f"latent_{prefix}{iPC:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.pt"))
                elif prefix == "RND":
                    img = plt.imread(
                        join(savedir, f"sample_{prefix}{iPC:03d}_T{inject_step:02d}_scale{pert_scale:.1f}.png"))
                    data = torch.load(
                        join(savedir, f"latent_{prefix}{iPC:03d}_T{inject_step:02d}_scale{pert_scale:.1f}.pt"))
                else:
                    raise RuntimeError("Unknown prefix")
                traj_col[(inject_step, pert_scale)] = data["latents_traj"]
                res_col[(inject_step, pert_scale)] = data["residue_traj"]
                img_col[(inject_step, pert_scale)] = img
            except FileNotFoundError as e:
                print("File not found", repr(e))
                pass
    return traj_col, res_col, img_col


def trajectory_deviation_analysis(savedir, traj_col, traj_orig,
          prefix, iPC, perturb_scales, inject_steps=range(0, 51, 5), savesfx=""):
    plt.figure(figsize=(8, 6))
    for T in inject_steps:
        for pert_scale in perturb_scales:
            traj_diff = traj_col[(T, pert_scale)] - traj_orig
            diff_norm_curv = traj_diff.flatten(1).double().norm(dim=1)
            plt.plot(diff_norm_curv, label=f"T{T} Scale {pert_scale:.1f}", alpha=0.7)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Norm of deviation")
    plt.xlabel("Time step")
    plt.title(f"Norm of deviation of latent space trajectory for {prefix}{iPC:02d}", fontsize=16)
    plt.tight_layout()
    saveallforms(join(savedir, "summary"), f"traj_deviation_norm_{prefix}{iPC:02d}{savesfx}")
    plt.show()
    #
    perturb_latent = ((traj_col[(5, 1.0)] - traj_col[(5, -1.0)])[6] / 2).double()
    perturb_vec = perturb_latent.flatten(1)
    plt.figure(figsize=(8, 6))
    for T in inject_steps:
        for pert_scale in perturb_scales:
            traj_diff = traj_col[(T, pert_scale)] - traj_orig
            diff_proj_curv = traj_diff.flatten(1).double() @ perturb_vec.T
            plt.plot(diff_proj_curv, label=f"T{T} Scale {pert_scale:.1f}", alpha=0.7)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Deviation projection on perturbation vector")
    plt.xlabel("Time step")
    plt.title(f"Deviation projection onto perturb vector \nlatent space trajectory for {prefix}{iPC:02d}", fontsize=16)
    plt.tight_layout()
    saveallforms(join(savedir, "summary"), f"traj_deviation_projcoef_{prefix}{iPC:02d}{savesfx}")
    plt.show()
    plt.close('all')
    return  # TODO: return the dist


def _insert_column2distmat(distmat, perturb_scales):
    nrow, ncol = distmat.shape
    nmid = ncol // 2
    distmat_insert = torch.cat((distmat[:, :nmid],
                                torch.zeros((nrow, 1), dtype=distmat.dtype),
                                distmat[:, nmid:]), dim=1)
    perturb_scales_insert = perturb_scales.copy()
    perturb_scales_insert.insert(nmid, 0.0)
    return distmat_insert, perturb_scales_insert


def image_deviation_analysis(savedir, Dist, img_col, img_orig,
    prefix, iPC, perturb_scales, inject_steps=range(0, 51, 5), savesfx="", batch=11):
    img_orig_tsr = torch.from_numpy(img_orig).permute(2, 0, 1)[None, :].float()
    img_tsrs = []
    for T in inject_steps:
        for pert_scale in perturb_scales:
            img = img_col[(T, pert_scale)]
            img_tsrs.append(img)
    img_tsrs = torch.from_numpy(np.stack(img_tsrs)).permute(0, 3, 1, 2).float()
    Dist.spatial = False
    with torch.no_grad():
        # compute distance in batch
        dist = []
        for i in range(0, len(img_tsrs), batch):
            dist.append(Dist(img_orig_tsr.cuda(), img_tsrs[i:i + batch].cuda()).cpu())
        dist = torch.cat(dist)
    distmat = dist.reshape(len(inject_steps), len(perturb_scales))

    Dist.spatial = True
    with torch.no_grad():
        distmaps = []
        for i in range(0, len(img_tsrs), batch):
            distmaps.append(Dist(img_orig_tsr.cuda(), img_tsrs[i:i + batch].cuda()).cpu())
        distmaps = torch.cat(distmaps)
    save_imgrid(distmaps, join(savedir, "summary", f"perturb_distmap_mtg_{prefix}{iPC:02d}{savesfx}.jpg"),
                nrow=len(perturb_scales), )

    distmat_insert, perturb_scales_insert = _insert_column2distmat(distmat, perturb_scales)

    plt.figure(figsize=(7, 7))
    sns.heatmap(distmat_insert, )
    plt.axis("image")
    plt.xlabel("Perturbation scale")
    plt.ylabel("Time step")
    plt.gca().set_xticklabels(perturb_scales_insert)
    plt.gca().set_yticklabels(range(0, 51, 5))
    plt.title(f"Distance between perturbed image original image", fontsize=16)
    saveallforms(join(savedir, "summary"), f"perturb_distmat_{prefix}{iPC:02d}{savesfx}")
    plt.show()

    torch.save({"distmat": distmat, "perturb_scales": perturb_scales},
               join(savedir, "summary", f"perturb_distmat_{prefix}{iPC:02d}{savesfx}.pt"))
    return distmat, dist, distmaps


#%% Mass compute
# prompt, dirname = ("a portrait of an aristocrat", "portrait_aristocrat")
# seed = 101
# prompt, dirname = ("a portrait of an aristocrat", "portrait_aristocrat")
# seed = 100
prompt, dirname = ("a large box containing an apple and a toy teddy bear", "box_apple_bear")
seed = 130
savedir = join(saveroot, f"{dirname}-seed{seed}")
#%%
# for iPC in range(0, 16):
for iPC in [*range(0, 16), *range(45, 51)]:
    traj_orig, residue_orig, img_orig = diffusion_orig_data_load(savedir)
    traj_col, res_col, img_col = diffusion_perturb_data_load(savedir, "PC", iPC, [-20.0, -15.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
    #%%
    trajectory_deviation_analysis(savedir, traj_col, traj_orig,
              "PC", iPC, [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0], inject_steps=range(0, 51, 5))
    trajectory_deviation_analysis(savedir, traj_col, traj_orig,
              "PC", iPC, [-20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0], inject_steps=range(0, 51, 5), savesfx="_wide5")
    distmat, dist, distmaps = image_deviation_analysis(savedir, Dist, img_col, img_orig,
               "PC", iPC, [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0], inject_steps=range(0, 51, 5), savesfx="")
    distmat, dist, distmaps = image_deviation_analysis(savedir, Dist, img_col, img_orig,
               "PC", iPC, [-20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0], inject_steps=range(0, 51, 5), savesfx="_wide5")


#%%
for iRND in range(0, 15):
    traj_orig, residue_orig, img_orig = diffusion_orig_data_load(savedir)
    traj_col, res_col, img_col = diffusion_perturb_data_load(savedir, "RND", iRND,
                             [-20.0, -15.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
    #%%
    trajectory_deviation_analysis(savedir, traj_col, traj_orig,
              "RND", iRND, [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0], inject_steps=range(0, 51, 5))
    trajectory_deviation_analysis(savedir, traj_col, traj_orig,
              "RND", iRND, [-20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0], inject_steps=range(0, 51, 5), savesfx="_wide5")
    distmat, dist, distmaps = image_deviation_analysis(savedir, Dist, img_col, img_orig,
               "RND", iRND, [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0], inject_steps=range(0, 51, 5), savesfx="")
    distmat, dist, distmaps = image_deviation_analysis(savedir, Dist, img_col, img_orig,
               "RND", iRND, [-20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0], inject_steps=range(0, 51, 5), savesfx="_wide5")





#%% Dev scratch zone
data_orig = torch.load(join(savedir, "latents_noise_trajs.pt"))
traj_orig = data_orig["latents_traj"]
residue_orig = data_orig["residue_traj"]
# prefix = "PC"
prefix = "RND"
# perturb_scales = [-20.0, -15.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
perturb_scales = [-20.0, -15.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
# perturb_scales = [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0,]
# for iPC in [*range(0, 16), *range(45, 51), ]: # *range(45, 50)
for iPC in range(15):  # *range(45, 50)
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
    plt.title(f"Norm of deviation of latent space trajectory for {prefix}{iPC:02d}", fontsize=16)
    plt.tight_layout()
    saveallforms(join(savedir, "summary"), f"traj_deviation_norm_{prefix}{iPC:02d}")
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
    plt.title(f"Deviation projection onto perturb vector \nlatent space trajectory for {prefix}{iPC:02d}", fontsize=16)
    plt.tight_layout()
    saveallforms(join(savedir, "summary"), f"traj_deviation_projcoef_{prefix}{iPC:02d}")
    plt.show()

#%%

#%%
torch.allclose((traj_col[(5, 1.0)] - traj_orig)[6],
               (traj_orig - traj_col[(5, -1.0)])[6], rtol=1E-3, atol=1E-3)

#%% Dev zone
perturb_scales = [-20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0]
plt.figure(figsize=(7, 7))
sns.heatmap(distmat, )
plt.axis("image")
plt.xlabel("Perturbation scale")
plt.ylabel("Time step")
plt.gca().set_xticklabels(perturb_scales)
plt.gca().set_yticklabels(range(0, 51, 5))
plt.title(f"Distance between perturbed image original image", fontsize=16)
plt.show()
#%%
distmat_insert, perturb_scales_insert = _insert_column2distmat(distmat, perturb_scales)
plt.figure(figsize=(7, 7))
sns.heatmap(distmat_insert, )
plt.axis("image")
plt.xlabel("Perturbation scale")
plt.ylabel("Time step")
plt.gca().set_xticklabels(perturb_scales_insert)
plt.gca().set_yticklabels(range(0, 51, 5))
plt.title(f"Distance between perturbed image original image", fontsize=16)
saveallforms(join(savedir, "summary"), f"perturb_distmat_PC00{savesfx}")
plt.show()
