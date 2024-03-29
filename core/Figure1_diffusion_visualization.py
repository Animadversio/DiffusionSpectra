import os
import shutil
import matplotlib.pyplot as plt
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage, make_grid_np
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid, \
    make_grid, make_grid_T
from os.path import join
figout = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\GeneralObservation"
#%%
src_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\GeneralObservation\Portrait_aristocrat-seed100_cfg7.5_PNDM"
dirname, seed = "box_apple_bear", 146


mtg_Xt = plt.imread(join(src_dir, "latent_traj_vae_decode.jpg"))
mtg_X0hat = plt.imread(join(src_dir, "proj_z0_vae_decode_new.jpg"))
mtg_dXt = plt.imread(join(src_dir, "latent_diff_lag1_stdnorm_vae_decode_PNDM.png"))
Xt_col = crop_all_from_montage(mtg_Xt, 53, imgsize=512, pad=2)
X0hat_col = crop_all_from_montage(mtg_X0hat, 52, imgsize=512, pad=2)
dXt_col = crop_all_from_montage(mtg_dXt, 52, imgsize=512, pad=2)
ticks = [0, 5, 10, 15, 20, 25, 35, 51]
Xt_mtg_sel = make_grid_np([Xt_col[t] for t in ticks], nrow=len(ticks), )
X0hat_mtg_sel = make_grid_np([X0hat_col[t] for t in ticks], nrow=len(ticks), )
dXt_mtg_sel = make_grid_np([dXt_col[t] for t in ticks], nrow=len(ticks), )
plt.imsave(join(src_dir, "Xt_mtg_sel.jpg"), Xt_mtg_sel)
plt.imsave(join(src_dir, "X0hat_mtg_sel.jpg"), X0hat_mtg_sel)
plt.imsave(join(src_dir, "dXt_mtg_sel.jpg"), dXt_mtg_sel)

#%% MNIST
orig_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\ddpm-mnist"
geom_dir = r"F:\insilico_exps\Diffusion_traj\ddpm-mnist_scheduler\DDIM"
seed = 286  # 316
mtg_Xt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_traj.jpg"))
mtg_X0hat = plt.imread(join(orig_dir, f"DDIM_seed{seed}_proj_z0_vae_decode.jpg"))
mtg_dXt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_diff_lag1_stdnorm_vae_decode.jpg"))
Xt_col = crop_all_from_montage(mtg_Xt, 53, imgsize=32, pad=2)
X0hat_col = crop_all_from_montage(mtg_X0hat, 52, imgsize=32, pad=2)
dXt_col = crop_all_from_montage(mtg_dXt, 52, imgsize=32, pad=2)
src_dir = join(figout, rf"MNIST_DDIM_seed{seed}")
os.makedirs(src_dir, exist_ok=True)
ticks = range(0, 52, 3)
Xt_mtg_sel = make_grid_np([Xt_col[t] for t in ticks], nrow=len(ticks), )
X0hat_mtg_sel = make_grid_np([X0hat_col[t] for t in ticks], nrow=len(ticks), )
dXt_mtg_sel = make_grid_np([dXt_col[t] for t in ticks], nrow=len(ticks), )
plt.imsave(join(src_dir, "Xt_mtg_sel.jpg"), Xt_mtg_sel)
plt.imsave(join(src_dir, "X0hat_mtg_sel.jpg"), X0hat_mtg_sel)
plt.imsave(join(src_dir, "dXt_mtg_sel.jpg"), dXt_mtg_sel)
# copy file
shutil.copy2(join(geom_dir, f"seed{seed}", f"latent_trajectory_2d_proj.pdf"), src_dir)#join(src_dir, f"latent_trajectory_2d_proj.pdf"))
shutil.copy2(join(geom_dir, f"seed{seed}", f"latent_trajectory_2d_proj.png"), src_dir)#join(src_dir, f"latent_trajectory_2d_proj.png"))


#%% CIFAR10
orig_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\ddpm-cifar10-32"
geom_dir = r"F:\insilico_exps\Diffusion_traj\ddpm-cifar10-32_scheduler\DDIM"
seed = 246 # 266 #
src_dir = join(figout, rf"CIFAR_DDIM_seed{seed}")
os.makedirs(src_dir, exist_ok=True)
mtg_Xt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_traj.jpg"))
mtg_X0hat = plt.imread(join(orig_dir, f"DDIM_seed{seed}_proj_z0_vae_decode.jpg"))
mtg_dXt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_diff_lag1_stdnorm_vae_decode.jpg"))
Xt_col = crop_all_from_montage(mtg_Xt, 53, imgsize=32, pad=2)
X0hat_col = crop_all_from_montage(mtg_X0hat, 52, imgsize=32, pad=2)
dXt_col = crop_all_from_montage(mtg_dXt, 52, imgsize=32, pad=2)
ticks = range(0, 52, 3)#[0, 5, 10, 15, 20, 25, 35, 50]
Xt_mtg_sel = make_grid_np([Xt_col[t] for t in ticks], nrow=len(ticks), )
X0hat_mtg_sel = make_grid_np([X0hat_col[t] for t in ticks], nrow=len(ticks), )
dXt_mtg_sel = make_grid_np([dXt_col[t] for t in ticks], nrow=len(ticks), )
plt.imsave(join(src_dir, "Xt_mtg_sel.jpg"), Xt_mtg_sel)
plt.imsave(join(src_dir, "X0hat_mtg_sel.jpg"), X0hat_mtg_sel)
plt.imsave(join(src_dir, "dXt_mtg_sel.jpg"), dXt_mtg_sel)
# copy file
shutil.copy2(join(geom_dir, f"seed{seed}", f"latent_trajectory_2d_proj.pdf"), src_dir)#join(src_dir, f"latent_trajectory_2d_proj.pdf"))
shutil.copy2(join(geom_dir, f"seed{seed}", f"latent_trajectory_2d_proj.png"), src_dir)#join(src_dir, f"latent_trajectory_2d_proj.png"))

#%% CelebA
orig_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\ddpm-celebahq-256"

seed = 129
mtg_Xt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_traj.jpg"))
mtg_X0hat = plt.imread(join(orig_dir, f"DDIM_seed{seed}_proj_z0_vae_decode.jpg"))
mtg_dXt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_diff_lag1_stdnorm_vae_decode.jpg"))
Xt_col = crop_all_from_montage(mtg_Xt, 53, imgsize=256, pad=2)
X0hat_col = crop_all_from_montage(mtg_X0hat, 52, imgsize=256, pad=2)
dXt_col = crop_all_from_montage(mtg_dXt, 52, imgsize=256, pad=2)
src_dir = join(figout, rf"CelebA_DDIM_seed{seed}")
os.makedirs(src_dir, exist_ok=True)
ticks = [0, 5, 10, 15, 20, 25, 35, 50]
Xt_mtg_sel = make_grid_np([Xt_col[t] for t in ticks], nrow=len(ticks), )
X0hat_mtg_sel = make_grid_np([X0hat_col[t] for t in ticks], nrow=len(ticks), )
dXt_mtg_sel = make_grid_np([dXt_col[t] for t in ticks], nrow=len(ticks), )
plt.imsave(join(src_dir, "Xt_mtg_sel.jpg"), Xt_mtg_sel)
plt.imsave(join(src_dir, "X0hat_mtg_sel.jpg"), X0hat_mtg_sel)
plt.imsave(join(src_dir, "dXt_mtg_sel.jpg"), dXt_mtg_sel)

#%% Church
orig_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\ddpm-church-256"
seed = 129  #  128
mtg_Xt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_traj.jpg"))
mtg_X0hat = plt.imread(join(orig_dir, f"DDIM_seed{seed}_proj_z0_vae_decode.jpg"))
mtg_dXt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_diff_lag1_stdnorm_vae_decode.jpg"))
Xt_col = crop_all_from_montage(mtg_Xt, 53, imgsize=256, pad=2)
X0hat_col = crop_all_from_montage(mtg_X0hat, 52, imgsize=256, pad=2)
dXt_col = crop_all_from_montage(mtg_dXt, 52, imgsize=256, pad=2)
src_dir = join(figout, rf"Church_DDIM_seed{seed}")
os.makedirs(src_dir, exist_ok=True)
ticks = [0, 5, 10, 15, 20, 25, 35, 50]
Xt_mtg_sel = make_grid_np([Xt_col[t] for t in ticks], nrow=len(ticks), )
X0hat_mtg_sel = make_grid_np([X0hat_col[t] for t in ticks], nrow=len(ticks), )
dXt_mtg_sel = make_grid_np([dXt_col[t] for t in ticks], nrow=len(ticks), )
plt.imsave(join(src_dir, "Xt_mtg_sel.jpg"), Xt_mtg_sel)
plt.imsave(join(src_dir, "X0hat_mtg_sel.jpg"), X0hat_mtg_sel)
plt.imsave(join(src_dir, "dXt_mtg_sel.jpg"), dXt_mtg_sel)

#%%


orig_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\ddpm-celebahq-256"
seed = 129 # 191
mtg_Xt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_traj.jpg"))
mtg_X0hat = plt.imread(join(orig_dir, f"DDIM_seed{seed}_proj_z0_vae_decode.jpg"))
mtg_dXt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_diff_lag1_stdnorm_vae_decode.jpg"))
Xt_col = crop_all_from_montage(mtg_Xt, 53, imgsize=256, pad=2)
X0hat_col = crop_all_from_montage(mtg_X0hat, 52, imgsize=256, pad=2)
dXt_col = crop_all_from_montage(mtg_dXt, 52, imgsize=256, pad=2)
src_dir = join(figout, rf"CelebA_DDIM_seed{seed}")
os.makedirs(src_dir, exist_ok=True)
ticks = [0, 5, 10, 15, 20, 25, 35, 40, 45, 50]
Xt_mtg_sel = make_grid_np([Xt_col[t] for t in ticks], nrow=len(ticks), )
X0hat_mtg_sel = make_grid_np([X0hat_col[t] for t in ticks], nrow=len(ticks), )
dXt_mtg_sel = make_grid_np([dXt_col[t] for t in ticks], nrow=len(ticks), )
plt.imsave(join(src_dir, "Xt_mtg_sel_full5.jpg"), Xt_mtg_sel)
plt.imsave(join(src_dir, "X0hat_mtg_sel_full5.jpg"), X0hat_mtg_sel)
plt.imsave(join(src_dir, "dXt_mtg_sel_full5.jpg"), dXt_mtg_sel)
ticks = range(0, 6)
Xt_mtg_sel = make_grid_np([Xt_col[t] for t in ticks], nrow=len(ticks), )
X0hat_mtg_sel = make_grid_np([X0hat_col[t] for t in ticks], nrow=len(ticks), )
dXt_mtg_sel = make_grid_np([dXt_col[t] for t in ticks], nrow=len(ticks), )
plt.imsave(join(src_dir, "Xt_mtg_sel_init5.jpg"), Xt_mtg_sel)
plt.imsave(join(src_dir, "X0hat_mtg_sel_init5.jpg"), X0hat_mtg_sel)
plt.imsave(join(src_dir, "dXt_mtg_sel_init5.jpg"), dXt_mtg_sel)
#%%
src_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\GeneralObservation\Portrait_aristocrat-seed100_cfg7.5_PNDM"
mtg_Xt = plt.imread(join(src_dir, "latent_traj_vae_decode.jpg"))
mtg_X0hat = plt.imread(join(src_dir, "proj_z0_vae_decode_new.jpg"))
mtg_dXt = plt.imread(join(src_dir, "latent_diff_lag1_stdnorm_vae_decode_PNDM.png"))
Xt_col = crop_all_from_montage(mtg_Xt, 53, imgsize=512, pad=2)
X0hat_col = crop_all_from_montage(mtg_X0hat, 52, imgsize=512, pad=2)
dXt_col = crop_all_from_montage(mtg_dXt, 52, imgsize=512, pad=2)
ticks = [0, 5, 10, 15, 20, 25, 35, 40, 45, 51]
Xt_mtg_sel = make_grid_np([Xt_col[t] for t in ticks], nrow=len(ticks), )
X0hat_mtg_sel = make_grid_np([X0hat_col[t] for t in ticks], nrow=len(ticks), )
dXt_mtg_sel = make_grid_np([dXt_col[t] for t in ticks], nrow=len(ticks), )
plt.imsave(join(src_dir, "Xt_mtg_sel_full5.jpg"), Xt_mtg_sel)
plt.imsave(join(src_dir, "X0hat_mtg_sel_full5.jpg"), X0hat_mtg_sel)
plt.imsave(join(src_dir, "dXt_mtg_sel_full5.jpg"), dXt_mtg_sel)
#%% Church
orig_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\ddpm-church-256"
seed = 128  # 129
mtg_Xt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_traj.jpg"))
mtg_X0hat = plt.imread(join(orig_dir, f"DDIM_seed{seed}_proj_z0_vae_decode.jpg"))
mtg_dXt = plt.imread(join(orig_dir, f"DDIM_seed{seed}_sample_diff_lag1_stdnorm_vae_decode.jpg"))
Xt_col = crop_all_from_montage(mtg_Xt, 53, imgsize=256, pad=2)
X0hat_col = crop_all_from_montage(mtg_X0hat, 52, imgsize=256, pad=2)
dXt_col = crop_all_from_montage(mtg_dXt, 52, imgsize=256, pad=2)
src_dir = join(figout, rf"Church_DDIM_seed{seed}")
os.makedirs(src_dir, exist_ok=True)
ticks = [0, 5, 10, 15, 20, 25, 35, 40, 45, 50]
Xt_mtg_sel = make_grid_np([Xt_col[t] for t in ticks], nrow=len(ticks), )
X0hat_mtg_sel = make_grid_np([X0hat_col[t] for t in ticks], nrow=len(ticks), )
dXt_mtg_sel = make_grid_np([dXt_col[t] for t in ticks], nrow=len(ticks), )
plt.imsave(join(src_dir, "Xt_mtg_sel_full5.jpg"), Xt_mtg_sel)
plt.imsave(join(src_dir, "X0hat_mtg_sel_full5.jpg"), X0hat_mtg_sel)
plt.imsave(join(src_dir, "dXt_mtg_sel_full5.jpg"), dXt_mtg_sel)

