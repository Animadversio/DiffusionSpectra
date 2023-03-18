from os.path import join
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from core.utils.plot_utils import saveallforms
from core.utils.montage_utils import make_grid_T
from core.utils.plot_utils import to_imgrid
import pickle as pkl
from pathlib import Path
from glob import glob
from tqdm import tqdm
saveroot = r"F:\insilico_exps\Diffusion_traj"
figoutdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\MNIST_CIFAR_gmm_simul_plot"  #join(saveroot, "figs")
#%%
#%%
def plot_mean_with_quantile(data_arr, quantile, label, color=None, ax=None):
    if ax is None:
        ax = plt.gca()
    mean_vec = data_arr.mean(axis=0)
    ax.plot(mean_vec, label=label, color=color, lw=2)  # ttraj,
    ax.fill_between(range(len(mean_vec)),
                    np.quantile(data_arr, quantile[0], axis=0),
                    np.quantile(data_arr, quantile[1], axis=0),
                    alpha=0.3, color=color)
    return ax


def sweep_RNDseed_dist(savedir, RNDrange=range(400)):
    dist_gmm2uni_arr = []
    dist_exact2uni_arr = []
    dist_exact2gmm_arr = []
    for RNDseed in tqdm(RNDrange):
        data = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_all.pkl"), "rb"))
        sol_uni = data["sol_uni"]
        sol_gmm = data["sol_gmm"]
        sol_exact = data["sol_exact"]
        ttraj = sol_uni.t
        xtraj_uni = sol_uni.y
        xtraj_gmm = sol_gmm.y
        xtraj_exact = sol_exact.y
        dist_gmm2uni = ((xtraj_gmm - xtraj_uni)**2).mean(axis=0)
        dist_exact2uni = ((xtraj_uni - xtraj_exact)**2).mean(axis=0)
        dist_exact2gmm = ((xtraj_gmm - xtraj_exact)**2).mean(axis=0)
        dist_gmm2uni_arr.append(dist_gmm2uni)
        dist_exact2uni_arr.append(dist_exact2uni)
        dist_exact2gmm_arr.append(dist_exact2gmm)
    dist_gmm2uni_arr = np.array(dist_gmm2uni_arr)
    dist_exact2uni_arr = np.array(dist_exact2uni_arr)
    dist_exact2gmm_arr = np.array(dist_exact2gmm_arr)
    return dist_gmm2uni_arr, dist_exact2uni_arr, dist_exact2gmm_arr

#%%  UnConditional MNIST
model_cond = "mnist_uncond"
savedir = join(saveroot, "mnist_uncond_gmm_exact")
dist_gmm2uni_arr, dist_exact2uni_arr, dist_exact2gmm_arr = sweep_RNDseed_dist(savedir, RNDrange=range(400))

#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
ax.plot(dist_exact2uni_arr.T, color="C0", alpha=0.02) # ttraj,
ax.plot(dist_gmm2uni_arr.T, color="C1", alpha=0.02) # ttraj,
ax.plot(dist_exact2uni_arr.mean(axis=0), label="Exact to Unimodal", lw=2, alpha=0.8) # ttraj,
ax.plot(dist_gmm2uni_arr.mean(axis=0), label="Gaussian Mixture to Unimodal", lw=2, alpha=0.8) # ttraj,
ax.set_xlabel("Time steps")
ax.set_ylabel("Mean squared error")
ax.set_title("Deviation from Gaussian solution")
ax.legend()
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_gmm_exact2uni_traj_deviation", figh=fig)
plt.show()


#%% plot the quantile of the mean
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
ax.plot(dist_exact2uni_arr.T, color="C0", alpha=0.02)
ax.plot(dist_exact2gmm_arr.T, color="C1", alpha=0.02)
ax.plot(dist_exact2uni_arr.mean(axis=0), label="Unimodal to Exact", lw=2, alpha=0.8) # ttraj,
ax.plot(dist_exact2gmm_arr.mean(axis=0), label="Gaussian Mixture to Exact", lw=2, alpha=0.8)  # ttraj,
ax.set_xlabel("Time steps")
ax.set_ylabel("Mean squared error")
ax.set_title("Deviation from full data exact solution")
ax.legend()
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_gmm_uni2exact_traj_deviation", figh=fig)
plt.show()

#%% plot the quantile of the mean
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
plot_mean_with_quantile(dist_exact2uni_arr, (0.25, 0.75), "Unimodal to Exact", color="C0", ax=ax)
plot_mean_with_quantile(dist_exact2gmm_arr, (0.25, 0.75), "Gaussian Mixture to Exact", color="C1", ax=ax)
ax.set_xlabel("Time steps")
ax.set_ylabel("Mean squared error")
ax.legend()
ax.set_title("Deviation from full data exact solution")
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_gmm_uni2exact_traj_deviation_quartile", figh=fig)
plt.show()

#%% same for to the uni modal
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
plot_mean_with_quantile(dist_gmm2uni_arr, (0.25, 0.75), "Gaussian Mixture to Unimodal", color="C0", ax=ax)
plot_mean_with_quantile(dist_exact2uni_arr, (0.25, 0.75), "Exact to Unimodal", color="C1", ax=ax)
ax.set_xlabel("Time steps")
ax.set_ylabel("Mean squared error")
ax.legend()
ax.set_title("Deviation from Gaussian solution")
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_gmm_exact2uni_traj_deviation_quartile", figh=fig)
plt.show()

#%% Un Conditional CIFAR
model_cond = "cifar_uncond"
savedir = join(saveroot, r"cifar_uncond_gmm_exact")
dist_gmm2uni_arr, dist_exact2uni_arr, dist_exact2gmm_arr = sweep_RNDseed_dist(savedir, RNDrange=range(100))
#%%
def sweep_cond_RNDseed_dist(savedir, class_id, RNDrange=range(100)):
    dist_exact2uni_arr = []
    for RNDseed in tqdm(RNDrange):
        data = pkl.load(open(join(savedir, f"class{class_id}_RND{RNDseed:03d}_all.pkl"), "rb"))
        sol_uni = data["sol_gauss"]
        sol_exact = data["sol_exact"]
        ttraj = sol_uni.t
        xtraj_uni = sol_uni.y
        xtraj_exact = sol_exact.y
        dist_exact2uni = ((xtraj_uni - xtraj_exact)**2).mean(axis=0)
        dist_exact2uni_arr.append(dist_exact2uni)
    dist_exact2uni_arr = np.array(dist_exact2uni_arr)
    return dist_exact2uni_arr

#%% Conditional MNIST
model_cond = "mnist_cond"
savedir = join(saveroot, r"mnist_cond_gmm_exact")
dist_exact2uni_dict = {}
for class_id in range(10):
    dist_exact2uni_cond = sweep_cond_RNDseed_dist(savedir, class_id, RNDrange=range(100))
    dist_exact2uni_dict[class_id] = dist_exact2uni_cond
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
for class_id in range(10):
    plot_mean_with_quantile(dist_exact2uni_dict[class_id],
                    (0.25, 0.75), f"Class {class_id}", color=f"C{class_id}", ax=ax)
ax.set_xlabel("Time steps")
ax.set_ylabel("Mean squared error")
ax.legend()
ax.set_title("Deviation from Gaussian solution")
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_exact2uni_traj_deviation_quatile_allclass", figh=fig)
plt.show()
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
for class_id in range(10):
    ax.plot(dist_exact2uni_dict[class_id].mean(axis=0),
            label=f"Digit {class_id}", lw=2.5, alpha=0.6)  # ttraj,
ax.set_xlabel("Time steps")
ax.set_ylabel("Mean squared error")
ax.set_title("Deviation from Gaussian solution")
ax.legend()
plt.tight_layout()
saveallforms(figoutdir, f"mnist_cond_exact2uni_traj_deviation_allclass", figh=fig)
plt.show()


#%% Conditional CIFAR
model_cond = "cifar_cond"
savedir = join(saveroot, r"cifar_cond_gmm_exact")
dist_exact2uni_dict = {}
for class_id in range(10):
    dist_exact2uni_cond = sweep_cond_RNDseed_dist(savedir, class_id, RNDrange=range(100))
    dist_exact2uni_dict[class_id] = dist_exact2uni_cond
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
for class_id in range(10):
    plot_mean_with_quantile(dist_exact2uni_dict[class_id],
                    (0.25, 0.75), f"Class {class_id}", color=f"C{class_id}", ax=ax)
ax.set_xlabel("Time steps")
ax.set_ylabel("Mean squared error")
ax.legend()
ax.set_title("Deviation from Gaussian solution")
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_exact2uni_traj_deviation_quatile_allclass", figh=fig)
plt.show()
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
for class_id in range(10):
    ax.plot(dist_exact2uni_dict[class_id].mean(axis=0),
            label=f"Class {class_id}", lw=2.5, alpha=0.6)  # ttraj,
ax.set_xlabel("Time steps")
ax.set_ylabel("Mean squared error")
ax.set_title("Deviation from Gaussian solution")
ax.legend()
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_cond_exact2uni_traj_deviation_allclass", figh=fig)
plt.show()
#%%

#%% Dev zone
RNDseed = 11
# load data
data = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_all.pkl"), "rb"))
#%%
sol_uni = data["sol_uni"]
sol_gmm = data["sol_gmm"]
sol_exact = data["sol_exact"]

ttraj = sol_uni.t
xtraj_uni = sol_uni.y
xtraj_gmm = sol_gmm.y
xtraj_exact = sol_exact.y
#%%
dist_gmm2uni = ((xtraj_gmm - xtraj_uni)**2).mean(axis=0)
dist_exact2uni = ((xtraj_uni - xtraj_exact)**2).mean(axis=0)
dist_exact2gmm = ((xtraj_gmm - xtraj_exact)**2).mean(axis=0)
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(dist_gmm2uni, label="GMM to Uni modal") # ttraj,
ax.plot(dist_exact2uni, label="Exact to Uni modal") # ttraj,
ax.set_xlabel("Time")
ax.set_ylabel("Mean squared error")
ax.legend()
plt.show()
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(dist_exact2uni, label="Uni modal to Exact") # ttraj,
ax.plot(dist_exact2gmm, label="GMM to Exact") # ttraj,
ax.set_xlabel("Time")
ax.set_ylabel("Mean squared error")
ax.legend()
plt.show()

#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
ax.plot(dist_gmm2uni_arr.mean(axis=0), label="GMM to Uni modal")  # ttraj,
ax.plot(dist_exact2uni_arr.mean(axis=0), label="Exact to Uni modal")  # ttraj,
ax.set_xlabel("Time steps")
ax.set_ylabel("Mean squared error")
ax.legend()
plt.tight_layout()
plt.show()