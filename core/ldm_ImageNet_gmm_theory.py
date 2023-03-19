import math
import torch
import matplotlib.pyplot as plt
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
ddim_ab = torch.load(join(saveroot, "ddim_alphas_betas.pt"))
alphas_cumprod = ddim_ab["alphas_cumprod"]
alphas_cumprod_prev = ddim_ab["alphas_cumprod_prev"]
betas = ddim_ab["betas"]
#%%

def ldm_beta_fun(t, beta0=0.0015, beta1=0.0195):
    return (t * math.sqrt(beta1) + (1 - t) * math.sqrt(beta0)) ** 2


def ldm_alphacumprod_fun(t, beta0=0.0015, beta1=0.0195):
    return np.exp(- 1000 / 3 / (math.sqrt(beta1) - math.sqrt(beta0)) *
                   (((math.sqrt(beta1) - math.sqrt(beta0)) * t + math.sqrt(beta0)) ** 3 - beta0 ** 1.5)) * 0.9985

def ldm_alpha_fun(t, beta0=0.0015, beta1=0.0195):
    return np.exp(- 500 / 3 / (math.sqrt(beta1) - math.sqrt(beta0)) *
                   (((math.sqrt(beta1) - math.sqrt(beta0)) * t + math.sqrt(beta0)) ** 3 - beta0 ** 1.5)) * 0.9992


tticks = torch.linspace(0, 1, 1000)
plt.figure(figsize=(8, 4))
plt.subplot(1,3,1)
plt.plot(tticks, betas.cpu(), lw=2.5, alpha=0.5, label="ldm")
plt.plot(tticks, ldm_beta_fun(tticks), lw=2.5, alpha=0.5, label="analytical")
plt.title("beta")
plt.subplot(1,3,2)
plt.plot(tticks, alphas_cumprod.cpu(), lw=2.5, alpha=0.5, label="ldm")
plt.plot(tticks, ldm_alphacumprod_fun(tticks), lw=2.5, alpha=0.5, label="analytical")
plt.legend()
plt.title("alpha cumprod")
plt.subplot(1,3,3)
plt.plot(tticks, alphas_cumprod.cpu().sqrt(), lw=2.5, alpha=0.5, label="ldm")
plt.plot(tticks, ldm_alpha_fun(tticks), lw=2.5, alpha=0.5, label="analytical")
plt.legend()
plt.title("alpha t ")
plt.show()

# U = PCA_data['U']
# V = PCA_data['V']
# S = PCA_data['S']
# imgmean = PCA_data['mean']
# cov_eigs = S**2 / (U.shape[0] - 1)
#%%
class_id = 0
PCA_data = torch.load(join(PCAdir, f"class{class_id}_z_pca.pt"))
zs_all = torch.load(join(PCAdir, f"class{class_id}_zs.pt"))["zs"]
imgmean = PCA_data['mean']
mus = zs_all.flatten(1)
#%%
RNDseed = 2
traj_data = torch.load(join(traj_dir, f"class{class_id:03d}_seed{RNDseed:03d}", "state_traj.pt"))
z_traj = traj_data['z_traj'].cpu()
pred_z0_traj = traj_data['pred_z0_traj'].cpu()
t_traj = traj_data['t_traj']
idx_traj = 1000 - t_traj - 1
# Analytical prediction
alphacum_traj = alphas_cumprod[idx_traj].cpu()
# mu_vec = imgmean[None, :] #.flatten(1) #  * 2 - 1
#%%
import sys
from core.gmm_special_diffusion_lib import *
sys.path.append(r"/home/binxu/GitHub/DiffusionSpectra")
#%%
tevals = np.linspace(1, 0, 51)
xT = z_traj[0].flatten()  # torch.randn(mus.shape[1])
x0_exact, sol_exact = exact_delta_gmm_reverse_diff(mus.double().numpy(), 1E-3, xT, t_eval=tevals,
                           alpha_fun=ldm_alpha_fun, beta_fun=ldm_beta_fun)
#%%
plt.imshow(x0_exact.reshape(3, 64, 64).transpose(1, 2, 0) / 5 + 0.5)
plt.show()
#%%
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(z_traj[-1].reshape(3, 64, 64).permute(1, 2, 0) / 5 + 0.5)
plt.subplot(1, 2, 2)
plt.imshow(x0_exact.reshape(3, 64, 64).transpose(1, 2, 0) / 5 + 0.5)
plt.show()
#%%
L2dist = (mus - x0_exact[None, :]).norm(dim=1)
print(L2dist.min(), L2dist.max())
