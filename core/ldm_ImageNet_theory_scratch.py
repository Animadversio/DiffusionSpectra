import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from os.path import join
from core.utils.plot_utils import saveallforms, save_imgrid, to_imgrid
from core.ODE_analytical_lib import *
#%%
PCAdir = r"/home/binxu/DL_Projects/ldm-imagenet/latents_save"
traj_dir = r"/home/binxu/insilico_exps/Diffusion_traj/ldm_imagenet/DDIM"
outdir = r"/home/binxu/insilico_exps/Diffusion_traj/ldm_imagenet_analytical"
saveroot = r"/home/binxu/insilico_exps/Diffusion_traj/ldm_imagenet"
os.makedirs(outdir, exist_ok=True)
#%%
# PCAdir = r"F:\insilico_exps\Diffusion_traj\ldm-imagenet-pca\latents_save"
# traj_dir = r"F:\insilico_exps\Diffusion_traj\ldm_imagenet\DDIM"
# outdir = r"F:\insilico_exps\Diffusion_traj\ldm_imagenet_analytical"
# saveroot = r"F:\insilico_exps\Diffusion_traj\ldm_imagenet"
# os.makedirs(outdir, exist_ok=True)
#%%
ddim_ab = torch.load(join(saveroot, "ddim_alphas_betas.pt"))
alphas_cumprod = ddim_ab["alphas_cumprod"]
alphas_cumprod_prev = ddim_ab["alphas_cumprod_prev"]
betas = ddim_ab["betas"]
#%% Temporal trial
class_id = 1
PCA_data = torch.load(join(PCAdir, f"class{class_id}_z_pca.pt"))
U = PCA_data['U']
V = PCA_data['V']
S = PCA_data['S']
imgmean = PCA_data['mean']
cov_eigs = S**2 / (U.shape[0] - 1)
#%%
RNDseed = 1
traj_data = torch.load(join(traj_dir, f"class{class_id:03d}_seed{RNDseed:03d}", "state_traj.pt"))
z_traj = traj_data['z_traj'].cpu()
pred_z0_traj = traj_data['pred_z0_traj'].cpu()
t_traj = traj_data['t_traj']
idx_traj = 1000 - t_traj - 1
# Analytical prediction
alphacum_traj = alphas_cumprod[idx_traj].cpu()
zT_vec = z_traj[0:1].flatten(1)
mu_vec = imgmean[None, :] #.flatten(1) #  * 2 - 1
#%% predict xt
print("Solving ODE for xt...")
xt_traj, _, _, _ = \
    xt_ode_solution(zT_vec, mu_vec, V, cov_eigs, alphacum_traj)
xt_traj_4, _, _, _ = \
    xt_ode_solution(zT_vec, mu_vec, V, cov_eigs / 4, alphacum_traj)
xt_traj_9, _, _, _ = \
    xt_ode_solution(zT_vec, mu_vec, V, cov_eigs / 9, alphacum_traj)
xt_traj_16, _, _, _ = \
    xt_ode_solution(zT_vec, mu_vec, V, cov_eigs / 16, alphacum_traj)
# predict x0hat
print("Solving ODE for x0hat...")
x0hatxt_traj, xttraj_coef, xttraj_coef_modulated = x0hat_ode_solution( \
    zT_vec, mu_vec, V, cov_eigs / 9, alphacum_traj)
#%%
cov_eigs_flat = torch.ones_like(cov_eigs)
Vperm = V[:, torch.randperm(V.shape[1], generator=torch.Generator().manual_seed(42))]
Vrand = torch.randn_like(V)
print("Solving ODE for xt...")
print("Solving ODE for x0hat...")
xt_traj_perm, _, _, _ = xt_ode_solution(zT_vec, mu_vec, Vperm, cov_eigs / 9, alphacum_traj)
x0hatxt_traj_perm, _, _ = x0hat_ode_solution(zT_vec, mu_vec, Vperm, cov_eigs / 9, alphacum_traj)
xt_traj_perm2, _, _, _ = xt_ode_solution(zT_vec, mu_vec, Vrand, cov_eigs / 9, alphacum_traj)
x0hatxt_traj_perm2, _, _ = x0hat_ode_solution(zT_vec, mu_vec, Vrand, cov_eigs / 9, alphacum_traj)
xt_traj_perm3, _, _, _ = xt_ode_solution(zT_vec, mu_vec, Vperm, cov_eigs_flat, alphacum_traj)
x0hatxt_traj_perm3, _, _ = x0hat_ode_solution(zT_vec, mu_vec, Vperm, cov_eigs_flat, alphacum_traj)
#%%
xt_pred_mse = (xt_traj - z_traj[1:].flatten(1)).pow(2).mean(dim=-1)
xt_pred_mse_perm = (xt_traj_perm.flatten(1) - z_traj[1:].flatten(1)).pow(2).mean(dim=-1)
xt_pred_mse_perm2 = (xt_traj_perm2.flatten(1) - z_traj[1:].flatten(1)).pow(2).mean(dim=-1)
xt_pred_mse_perm3 = (xt_traj_perm3.flatten(1) - z_traj[1:].flatten(1)).pow(2).mean(dim=-1)
x0hat_pred_mse = (x0hatxt_traj.flatten(1) - pred_z0_traj[1:].flatten(1)).pow(2).mean(dim=-1)
x0hat_pred_mse_perm = (x0hatxt_traj_perm.flatten(1) - pred_z0_traj[1:].flatten(1)).pow(2).mean(dim=-1)
x0hat_pred_mse_perm2 = (x0hatxt_traj_perm2.flatten(1) - pred_z0_traj[1:].flatten(1)).pow(2).mean(dim=-1)
x0hat_pred_mse_perm3 = (x0hatxt_traj_perm3.flatten(1) - pred_z0_traj[1:].flatten(1)).pow(2).mean(dim=-1)
#%%
xt_pred_mse = (xt_traj - z_traj[1:].flatten(1)).pow(2).mean(dim=-1)
xt_pred_mse_4 = (xt_traj_4 - z_traj[1:].flatten(1)).pow(2).mean(dim=-1)
xt_pred_mse_9 = (xt_traj_9 - z_traj[1:].flatten(1)).pow(2).mean(dim=-1)
xt_pred_mse_16 = (xt_traj_16 - z_traj[1:].flatten(1)).pow(2).mean(dim=-1)
#%%
plt.figure()
plt.plot(xt_pred_mse, label="original")
plt.plot(xt_pred_mse_4, label="original, cov/4")
plt.plot(xt_pred_mse_9, label="original, cov/9")
plt.plot(xt_pred_mse_16, label="original, cov/16")
plt.ylabel("MSE of deviation")
plt.xlabel("timestep")
plt.title("L2 norm of deviation between empirical and analytical prediction of xt")
plt.legend()
plt.show()
#%%
plt.figure()
plt.plot(xt_pred_mse)
plt.plot(xt_pred_mse_perm)
# plt.plot(xt_pred_mse_perm2)
plt.plot(xt_pred_mse_perm3)
plt.ylabel("MSE of deviation")
plt.xlabel("timestep")
plt.title("L2 norm of deviation between empirical and analytical prediction of xt")
plt.legend(["original", "permuted", "permuted2"])
plt.show()

#%%
plt.figure()
plt.plot(x0hat_pred_mse, label="correct")
plt.plot(x0hat_pred_mse_perm, label="PC permuted")
# plt.plot(x0hat_pred_mse_perm2)
plt.plot(x0hat_pred_mse_perm3, label="PC permuted, flat cov")
plt.ylabel("MSE of deviation")
plt.xlabel("timestep")
plt.title("L2 norm of deviation between empirical and analytical prediction of x0hat")
plt.legend()
plt.show()


#%%
plt.figure()
plt.imshow(to_imgrid((mu_vec.reshape(1, 3, 64, 64) * 3 + 0.5).clamp(0,1)))
plt.axis("off")
plt.show()
#%%
plt.figure()
plt.semilogy(cov_eigs[:-1])
plt.show()
#%%
zs_all = torch.load(join(PCAdir, f"class{class_id}_zs.pt"))["zs"]
#%%

#%%
betas = betas.cpu()
tticks = torch.linspace(0, 1, 1000)
plt.figure()
plt.plot(tticks, alphas_cumprod.cpu())
plt.plot(tticks, torch.exp(- 1000 / 3 / (betas[-1].sqrt() - betas[0].sqrt()) *
                   (((betas[-1].sqrt() - betas[0].sqrt()) * tticks + betas[0].sqrt()) ** 3 - betas[0] ** 1.5)))
# plt.plot(tticks, alphas_cumprod_4)
plt.show()
#%%
plt.figure()
plt.plot(tticks, betas.cpu())
plt.plot(tticks, (tticks * betas[-1].sqrt() + (1 - tticks) * betas[0].sqrt())**2)
plt.show()
#%%
import math
def ldm_beta_fun(t, beta0=0.0015, beta1=0.0195):
    return (t * math.sqrt(beta1) + (1 - t) * math.sqrt(beta0)) ** 2


def ldm_alpha_fun(t, beta0=0.0015, beta1=0.0195):
    return torch.exp(- 1000 / 3 / (math.sqrt(beta1) - math.sqrt(beta0)) *
                   (((math.sqrt(beta1) - math.sqrt(beta0)) * t + math.sqrt(beta0)) ** 3 - beta0 ** 1.5))


tticks = torch.linspace(0, 1, 1000)
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.plot(tticks, betas.cpu(), lw=2.5, alpha=0.5)
plt.plot(tticks, ldm_beta_fun(tticks), lw=2.5, alpha=0.5)
plt.subplot(1,2,2)
plt.plot(tticks, alphas_cumprod.cpu(), lw=2.5, alpha=0.5)
plt.plot(tticks, ldm_alpha_fun(tticks), lw=2.5, alpha=0.5)
plt.show()

