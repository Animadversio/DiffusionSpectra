#%%
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
exportdir = rf"F:\insilico_exps\Diffusion_traj\StableDiffusion_perturb_proj\box_apple_bear-seed130"
outdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\Perturb_geometry\box_apple_bear-seed130"

from core.utils.montage_utils import crop_from_montage, crop_all_from_montage, make_grid_np
#%% Export the perturbed trajectory
# for iPC in [0, 1, 2, 10, 45, 49]:#[*range(0, 16), *range(45, 51), ]: # *range(45, 50)
iPC = 2
ticks = [0, 5, 10, 15, 20, 25, 35, 50]
for inject_step in range(0, 51, 5):
    for pert_scale in [-10.0, -5.0, 5.0, 10.0]:
        mtg = plt.imread(join(exportdir, f"proj_z0_decode_PC{iPC:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.jpg"))
        img_col = crop_all_from_montage(mtg, 52, imgsize=512, pad=2)
        img_sel = [img_col[i] for i in ticks]
        mtg = make_grid_np(img_sel, nrow=len(img_sel))
        plt.imsave(join(outdir, f"proj_z0_decode_PC{iPC:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.jpg"),
                   mtg, )
#%% Export the original trajectory
mtg = plt.imread(join(exportdir, "proj_z0_vae_decode_new.jpg"))
img_col = crop_all_from_montage(mtg, 52, imgsize=512, pad=2)
img_sel = [img_col[i] for i in ticks]
mtg = make_grid_np(img_sel, nrow=len(img_sel))
plt.imsave(join(outdir, "proj_z0_vae_decode_new_original_seq.jpg"), mtg, )

#%% Quantify the effect of perturbation
from lpips import LPIPS
Dist = LPIPS(net='alex').cuda().eval()
Dist.requires_grad_(False)
#%%
mtg = plt.imread(join(exportdir, "proj_z0_vae_decode_new.jpg"))
img_col = crop_all_from_montage(mtg, 52, imgsize=512, pad=2)
orig_img_traj = torch.from_numpy(np.stack(img_col))\
                    .float().permute([0, 3, 1, 2]) / 255.0
#%%
prefix = "RND"
prefix = "PC"
for iPC in [0, 1, 2, 10, 45, 49]: #range(4,15):
    dist_trace = {}
    for inject_step in range(0, 51, 5):
        for pert_scale in [-10.0, -5.0, 5.0, 10.0]:
            if prefix == "PC":
                mtg = plt.imread(join(exportdir, f"proj_z0_decode_PC{iPC:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.jpg"))
            elif prefix == "RND":
                mtg = plt.imread(join(exportdir, f"proj_z0_decode_RND{iPC:03d}_T{inject_step:02d}_scale{pert_scale:.1f}.jpg"))
            img_col = crop_all_from_montage(mtg, 52, imgsize=512, pad=2)
            pert_img_traj = torch.from_numpy(np.stack(img_col))\
                        .float().permute([0, 3, 1, 2]) / 255.0
            with torch.no_grad():
                dists = Dist(orig_img_traj.cuda(), pert_img_traj.cuda()).cpu()
            dist_trace[(inject_step, pert_scale)] = dists.detach()
    torch.save(dist_trace, join(outdir, f"{prefix}{iPC:02d}_dist_trace.pt"))
    #%%
    for perturb_scale in [-10.0, -5.0, 5.0, 10.0]:
        perturb_scale = 10.0
        plt.figure()
        for inject_step in range(5, 51, 5):
            plt.plot(dist_trace[(inject_step, perturb_scale)].squeeze().numpy(),
                     lw=1.0) # label=f"inject_step={inject_step}",
        plt.xlabel("Time step")
        plt.ylabel("LPIPS distance")
        plt.title(f"Effect of perturbation on projected outcome \n{prefix}{iPC:02d} scale{perturb_scale:.1f}")
        saveallforms(outdir, f"perturb_effect_{prefix}{iPC:02d}_scale{perturb_scale:.1f}")
        plt.show()
    #%%
    plt.figure()
    for i, perturb_scale in enumerate([-10.0, -5.0, 5.0, 10.0]):
        for inject_step in range(5, 51, 5):
            plt.plot(dist_trace[(inject_step, perturb_scale)].squeeze().numpy(),
                     lw=1.0, linestyle=["-",":","-.","--"][i],
                     label=f"scale={perturb_scale:.1f}" if inject_step==5 else None) # label=f"inject_step={inject_step}",
    plt.xlabel("Time step")
    plt.ylabel("LPIPS distance")
    plt.title(f"Effect of perturbation on projected outcome \n{prefix}{iPC:02d}")
    plt.legend()
    saveallforms(outdir, f"perturb_effect_{prefix}{iPC:02d}")
    plt.show()
#%%
for inject_step in range(5, 51, 5):
    plt.figure()
    for i, perturb_scale in enumerate([-10.0, -5.0, 5.0, 10.0]):
        plt.plot(dist_trace[(inject_step, perturb_scale)].squeeze().numpy(),
                 lw=1.5, linestyle=["-",":","-.","--"][i],
                 label=f"scale={perturb_scale:.1f}")
        # label=f"inject_step={inject_step}",
    plt.xlabel("Time step")
    plt.ylabel("LPIPS distance")
    plt.title(f"Effect of perturbation on projected outcome \n{prefix}{iPC:02d}")
    plt.legend()
    # saveallforms(outdir, f"perturb_effect_{prefix}{iPC:02d}")
    plt.show()
    break

