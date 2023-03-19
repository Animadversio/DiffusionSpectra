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
from collections import defaultdict
saveroot = r"F:\insilico_exps\Diffusion_traj"
figoutdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\MNIST_CIFAR_gmm_simul_plot"  #join(saveroot, "figs")
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
#%%
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


#%%
# linear lut, interpolate the alpha cum prod
def alpha(t, beta0=0.02, beta1=0.0001, nT=1000):
    # return np.exp(- 1000 * (0.01 * t**2 + 0.0001 * t))
    # return np.exp(- 10 * t**2 - 0.1 * t) * 0.9999
    return np.exp(- nT * (0.5 * (beta0 - beta1) * t**2 + beta1 * t)) * (1 - beta1)


ttraj = np.linspace(0, 1, 51)
alphacumprod_ddim = alpha(ttraj, nT=400)
alphacumprod_gmm = alpha(ttraj, nT=1000)
#%%

def _dist_from_keys(xt_col, key1, key2):
    dist_arr = ((xt_col[key1] - xt_col[key2])**2).mean(axis=-1)
    return dist_arr

def sweep_ddim_traj_remap_traj(savedir, RNDrange=range(400)):
    ttraj = np.linspace(1, 0, 51)
    alphacumprod_ddim = alpha(ttraj, nT=400)
    alpha_t_ddim = np.sqrt(alphacumprod_ddim)  # sqrt to make alphacumprod in ddim, ddpm paper our definetion of alpha
    alpha_t_gmm = alpha(ttraj, nT=1000)
    xt_col = defaultdict(list)
    for RNDseed in tqdm(RNDrange):
        analydata = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_all.pkl"), "rb"))
        ddimdata = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_ddim_traj.pkl"), "rb"))
        xt_ddim = ddimdata["x_t"]
        xt_ddim = xt_ddim.reshape(xt_ddim.shape[0], -1)
        xt_col["ddim"].append(xt_ddim)
        sol_uni = analydata["sol_uni"]
        sol_gmm = analydata["sol_gmm"]
        sol_exact = analydata["sol_exact"]
        ttraj = sol_uni.t
        xt_uni = sol_uni.y.T
        xt_gmm = sol_gmm.y.T
        xt_exact = sol_exact.y.T
        interp_alpha_val = alpha_t_ddim  # 0.999 to avoid extrapolation
        xt_uni_remap = interp1d(alpha_t_gmm, xt_uni, axis=0, kind="linear", fill_value="extrapolate")(interp_alpha_val)
        xt_gmm_remap = interp1d(alpha_t_gmm, xt_gmm, axis=0, kind="linear", fill_value="extrapolate")(interp_alpha_val)
        xt_exact_remap = interp1d(alpha_t_gmm, xt_exact, axis=0, kind="linear", fill_value="extrapolate")(
            interp_alpha_val)
        xt_col["uni"].append(xt_uni_remap[1:])
        xt_col["gmm"].append(xt_gmm_remap[1:])
        xt_col["exact"].append(xt_exact_remap[1:])

    for key in xt_col:
        xt_col[key] = np.stack(xt_col[key], axis=0)
    return xt_col


def sweep_ddim_traj_remap_dist(savedir, RNDrange=range(400)):
    ttraj = np.linspace(1, 0, 51)
    alphacumprod_ddim = alpha(ttraj, nT=400)
    alpha_t_ddim = np.sqrt(alphacumprod_ddim)  # sqrt to make alphacumprod in ddim, ddpm paper our definetion of alpha
    alpha_t_gmm = alpha(ttraj, nT=1000)
    dist_uni2ddim_arr = []
    dist_gmm2ddim_arr = []
    dist_exact2ddim_arr = []
    for RNDseed in tqdm(RNDrange):
        analydata = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_all.pkl"), "rb"))
        ddimdata = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_ddim_traj.pkl"), "rb"))
        xt_ddim = ddimdata["x_t"]
        xt_ddim = xt_ddim.reshape(xt_ddim.shape[0], -1)
        sol_uni = analydata["sol_uni"]
        sol_gmm = analydata["sol_gmm"]
        sol_exact = analydata["sol_exact"]
        ttraj = sol_uni.t
        xt_uni = sol_uni.y.T
        xt_gmm = sol_gmm.y.T
        xt_exact = sol_exact.y.T
        # interp_alpha_val = np.minimum(np.sqrt(alphacumprod_ddim[::-1]), 0.9999)  # 0.999 to avoid extrapolation
        interp_alpha_val = alpha_t_ddim  # 0.999 to avoid extrapolation
        xt_uni_remap = interp1d(alpha_t_gmm, xt_uni, axis=0, kind="linear", fill_value="extrapolate")(interp_alpha_val)
        xt_gmm_remap = interp1d(alpha_t_gmm, xt_gmm, axis=0, kind="linear", fill_value="extrapolate")(interp_alpha_val)
        xt_exact_remap = interp1d(alpha_t_gmm, xt_exact, axis=0, kind="linear", fill_value="extrapolate")(interp_alpha_val)
        dist_uni2ddim = ((xt_uni_remap[1:] - xt_ddim)**2).mean(axis=1)
        dist_gmm2ddim = ((xt_gmm_remap[1:] - xt_ddim)**2).mean(axis=1)
        dist_exact2ddim = ((xt_exact_remap[1:] - xt_ddim)**2).mean(axis=1)
        dist_uni2ddim_arr.append(dist_uni2ddim)
        dist_gmm2ddim_arr.append(dist_gmm2ddim)
        dist_exact2ddim_arr.append(dist_exact2ddim)
    dist_uni2ddim_arr = np.array(dist_uni2ddim_arr)
    dist_gmm2ddim_arr = np.array(dist_gmm2ddim_arr)
    dist_exact2ddim_arr = np.array(dist_exact2ddim_arr)
    return dist_uni2ddim_arr, dist_gmm2ddim_arr, dist_exact2ddim_arr


model_cond = "mnist_uncond"
savedir = join(saveroot, "mnist_uncond_gmm_exact")
# dist_uni2ddim_arr, dist_gmm2ddim_arr, dist_exact2ddim_arr = sweep_ddim_traj_remap_dist(savedir, RNDrange=range(400))
uncond_xt_traj = sweep_ddim_traj_remap_traj(savedir, RNDrange=range(400))

#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
# plot_mean_with_quantile(dist_uni2ddim_arr, (0.25, 0.75), "Unimodal to DDIM", color="C0", ax=ax)
# plot_mean_with_quantile(dist_gmm2ddim_arr, (0.25, 0.75), "Gaussian Mixture to DDIM", color="C1", ax=ax)
# plot_mean_with_quantile(dist_exact2ddim_arr, (0.25, 0.75), "DeltaGMM to DDIM", color="C2", ax=ax)
plot_mean_with_quantile(_dist_from_keys(uncond_xt_traj,"uni","ddim"), (0.25, 0.75), "Unimodal to DDIM", color="C0", ax=ax)
plot_mean_with_quantile(_dist_from_keys(uncond_xt_traj,"gmm","ddim"), (0.25, 0.75), "Gaussian Mixture to DDIM", color="C1", ax=ax)
plot_mean_with_quantile(_dist_from_keys(uncond_xt_traj,"exact","ddim"), (0.25, 0.75), "DeltaGMM to DDIM", color="C2", ax=ax)
ax.set_xlabel("Time steps ")
ax.set_ylabel("Mean squared error")
ax.legend()
ax.set_title("Deviation of GMM solution trajectories from DDIM sampling")
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_uni_gmm_exact2ddim_traj_deviation_DDIMremap_quartile", figh=fig)
plt.show()
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
plot_mean_with_quantile(_dist_from_keys(uncond_xt_traj,"uni","exact"), (0.25, 0.75), "Unimodal to Exact", color="C0", ax=ax)
plot_mean_with_quantile(_dist_from_keys(uncond_xt_traj,"gmm","exact"), (0.25, 0.75), "Gaussian Mixture to Exact", color="C1", ax=ax)
ax.set_xlabel("Time steps ")
ax.set_ylabel("Mean squared error")
ax.legend()
ax.set_title("Deviation of Uni and GMM solution trajectories from DeltaGMM sampling")
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_uni_gmm2exact_traj_deviation_DDIMremap_quartile", figh=fig)
plt.show()
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
plot_mean_with_quantile(_dist_from_keys(uncond_xt_traj,"uni","gmm"), (0.25, 0.75), "Unimodal to Gaussian Mixture", color="C0", ax=ax)
plot_mean_with_quantile(_dist_from_keys(uncond_xt_traj,"uni","exact"), (0.25, 0.75), "Unimodal to Exact", color="C1", ax=ax)
ax.set_xlabel("Time steps ")
ax.set_ylabel("Mean squared error")
ax.legend()
ax.set_title("Deviation of Uni and GMM solution trajectories from DeltaGMM sampling")
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_exact_gmm2uni_traj_deviation_DDIMremap_quartile", figh=fig)
plt.show()
#%%
def sweep_cond_ddim_traj_remap_trajs(savedir, class_id, RNDrange=range(100)):
    ttraj = np.linspace(1, 0, 51)
    alphacumprod_ddim = alpha(ttraj, nT=400)
    alpha_t_ddim = np.sqrt(alphacumprod_ddim)  # sqrt to make alphacumprod in ddim, ddpm paper our definetion of alpha
    alpha_t_gmm = alpha(ttraj, nT=1000)
    xt_col = defaultdict(list)
    for RNDseed in tqdm(RNDrange):
        data = pkl.load(open(join(savedir, f"class{class_id}_RND{RNDseed:03d}_all.pkl"), "rb"))
        ddimdata = pkl.load(open(join(savedir, f"class{class_id}_RND{RNDseed:03d}_ddim_traj.pkl"), "rb"))
        # print(list(ddimdata))
        for w in [0.0, 0.5, 2.0, 5.0]:
            xt_ddim = ddimdata[w]["x_t"]
            xt_ddim = xt_ddim.reshape(xt_ddim.shape[0], -1)
            xt_col[f"w{w:.1f}"].append(xt_ddim)

        sol_uni = data["sol_gauss"]
        sol_exact = data["sol_exact"]
        ttraj = sol_uni.t
        xt_uni = sol_uni.y.T
        xt_exact = sol_exact.y.T
        # interp_alpha_val = np.minimum(np.sqrt(alpha_t_ddim), 0.9999)  # 0.999 to avoid extrapolation
        interp_alpha_val = alpha_t_ddim  # 0.999 to avoid extrapolation
        xt_uni_remap = interp1d(alpha_t_gmm, xt_uni, axis=0, kind="linear", fill_value="extrapolate")(interp_alpha_val)
        xt_exact_remap = interp1d(alpha_t_gmm, xt_exact, axis=0, kind="linear", fill_value="extrapolate")(interp_alpha_val)
        xt_col["uni"].append(xt_uni_remap[1:])
        xt_col["exact"].append(xt_exact_remap[1:])

    for k in xt_col:
        xt_col[k] = np.stack(xt_col[k])

    return xt_col


def sweep_cond_ddim_traj_remap_dist(savedir, class_id, RNDrange=range(100)):
    ttraj = np.linspace(1, 0, 51)
    alphacumprod_ddim = alpha(ttraj, nT=400)
    alpha_t_ddim = np.sqrt(alphacumprod_ddim)  # sqrt to make alphacumprod in ddim, ddpm paper our definetion of alpha
    alpha_t_gmm = alpha(ttraj, nT=1000)
    dist_uni2ddim_arr = []
    dist_exact2ddim_arr = []
    dist_exact2unit_arr = []
    for RNDseed in tqdm(RNDrange):
        data = pkl.load(open(join(savedir, f"class{class_id}_RND{RNDseed:03d}_all.pkl"), "rb"))
        ddimdata = pkl.load(open(join(savedir, f"class{class_id}_RND{RNDseed:03d}_ddim_traj.pkl"), "rb"))
        # print(list(ddimdata))
        xt_ddim = ddimdata[0.0]["x_t"]
        xt_ddim = xt_ddim.reshape(xt_ddim.shape[0], -1)
        # xt_ddim = ddimdata[0.5]["x_t"]
        # xt_ddim = xt_ddim.reshape(xt_ddim.shape[0], -1)
        # xt_ddim = ddimdata[2.0]["x_t"]
        # xt_ddim = xt_ddim.reshape(xt_ddim.shape[0], -1)
        # xt_ddim = ddimdata[5.0]["x_t"]
        # xt_ddim = xt_ddim.reshape(xt_ddim.shape[0], -1)

        sol_uni = data["sol_gauss"]
        sol_exact = data["sol_exact"]
        ttraj = sol_uni.t
        xt_uni = sol_uni.y.T
        xt_exact = sol_exact.y.T
        # interp_alpha_val = np.minimum(np.sqrt(alpha_t_ddim), 0.9999)  # 0.999 to avoid extrapolation
        interp_alpha_val = alpha_t_ddim  # 0.999 to avoid extrapolation
        xt_uni_remap = interp1d(alpha_t_gmm, xt_uni, axis=0, kind="linear", fill_value="extrapolate")(interp_alpha_val)
        xt_exact_remap = interp1d(alpha_t_gmm, xt_exact, axis=0, kind="linear", fill_value="extrapolate")(interp_alpha_val)
        dist_uni2ddim = ((xt_uni_remap[1:] - xt_ddim)**2).mean(axis=1)
        dist_exact2ddim = ((xt_exact_remap[1:] - xt_ddim)**2).mean(axis=1)
        dist_exact2uni = ((xt_exact_remap[1:] - xt_uni_remap[1:])**2).mean(axis=1)
        dist_uni2ddim_arr.append(dist_uni2ddim)
        dist_exact2ddim_arr.append(dist_exact2ddim)
        dist_exact2unit_arr.append(dist_exact2uni)

    dist_uni2ddim_arr = np.array(dist_uni2ddim_arr)
    dist_exact2ddim_arr = np.array(dist_exact2ddim_arr)
    dist_exact2unit_arr = np.array(dist_exact2unit_arr)
    return dist_uni2ddim_arr, dist_exact2ddim_arr, dist_exact2unit_arr
#%%
model_cond = "mnist_cond"
savedir = join(saveroot, "mnist_cond_gmm_exact")
for class_id in range(10):
    # dist_uni2ddim_remap, dist_exact2ddim_remap, dist_exact2unit_remap = \
    #     sweep_cond_ddim_traj_remap_dist(savedir, class_id, RNDrange=range(100))
    cond_xt_traj = sweep_cond_ddim_traj_remap_trajs(savedir, class_id, RNDrange=range(100))
    # dist_arr = _dist_from_keys(cond_xt_traj, "uni", "w0.0")
    np.savez(join(savedir, f"{model_cond}_class{class_id}_xt_traj.npz"), **cond_xt_traj)
    #%%
    fig, ax = plt.subplots(figsize=(4, 3.5))
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "uni", "w0.0"), (0.25, 0.75), "Uni Gaussian to DDIM", color="C0", ax=ax)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "exact", "w0.0"), (0.25, 0.75), "Exact to DDIM", color="C1", ax=ax)
    # plot_mean_with_quantile(dist_exact2unit_remap, (0.25, 0.75), "Exact to Gaussian Mixture", color="C2", ax=ax)
    plt.title(f"Deviation of GMM solution trajectories from DDIM sampling\nConditional MNIST digit {class_id}")
    plt.legend()
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Time step (DDIM)")
    plt.tight_layout()
    saveallforms(figoutdir, f"{model_cond}_class{class_id}_uni_exact2ddim_w0_traj_deviation_DDIMremap_quartile", )
    plt.show()
    #%%
    fig, ax = plt.subplots(figsize=(4, 3.5))
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "uni", "exact"), (0.25, 0.75), "Exact to Uni Gaussian", color="C0", ax=ax)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "w0.0", "exact"), (0.25, 0.75), "Exact to DDIM w0.0", color="C1", ax=ax)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "w0.5", "exact"), (0.25, 0.75), "Exact to DDIM w0.5", color="C2", ax=ax)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "w2.0", "exact"), (0.25, 0.75), "Exact to DDIM w2.0", color="C3", ax=ax)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "w5.0", "exact"), (0.25, 0.75), "Exact to DDIM w5.0", color="C4", ax=ax)
    plt.title(f"Deviation of DDIM solution trajectories from DeltaGMM sampling\nConditional MNIST digit {class_id}")
    plt.legend()
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Time step (DDIM)")
    plt.tight_layout()
    saveallforms(figoutdir, f"{model_cond}_class{class_id}_uni_cfgall2exact_traj_deviation_DDIMremap_quartile", )
    plt.show()
    #%%
    fig, ax = plt.subplots(figsize=(4, 3.5))
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "uni", "exact"), (0.25, 0.75), "Uni Gaussian to Exact Gaussian", color="C0", ax=ax)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "w0.0", "uni"), (0.25, 0.75), "Uni to DDIM w0.0", color="C1", ax=ax)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "w0.5", "uni"), (0.25, 0.75), "Uni to DDIM w0.5", color="C2", ax=ax)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "w2.0", "uni"), (0.25, 0.75), "Uni to DDIM w2.0", color="C3", ax=ax)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "w5.0", "uni"), (0.25, 0.75), "Uni to DDIM w5.0", color="C4", ax=ax)
    plt.title(f"Deviation of DDIM solution trajectories from Uni Gaussian sampling\nConditional MNIST digit {class_id}")
    plt.legend()
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Time step (DDIM)")
    plt.tight_layout()
    saveallforms(figoutdir, f"{model_cond}_class{class_id}_exact_cfgall2uni_traj_deviation_DDIMremap_quartile", )
    plt.show()
#%%
model_cond = "mnist_cond"
savedir = join(saveroot, "mnist_cond_gmm_exact")
plt.figure(figsize=(4, 3.5))
for class_id in range(10):
    cond_xt_traj = np.load(join(savedir, f"{model_cond}_class{class_id}_xt_traj.npz"), allow_pickle=True)
    plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, "uni", "exact"), (0.25, 0.75),
                            f"Digit{class_id}", color=f"C{class_id}", ax=None)
plt.title(f"Deviation of exact and unimodal samplingnConditional MNIST digit {class_id}")
plt.legend()
plt.ylabel("Mean Squared Error")
plt.xlabel("Time step (DDIM)")
plt.tight_layout()
saveallforms(figoutdir, f"{model_cond}_allclass_uni2exact_traj_deviation_DDIMremap_quartile", )
plt.show()
#%%
for w in [0.0, 0.5, 2.0, 5.0]:
    plt.figure(figsize=(4, 3.5))
    for class_id in range(10):
        cond_xt_traj = np.load(join(savedir, f"{model_cond}_class{class_id}_xt_traj.npz"), allow_pickle=True)
        plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, f"w{w:.1f}", "exact"), (0.25, 0.75),
                                f"Digit{class_id}", color=f"C{class_id}", ax=None)
    plt.title(f"Deviation of DeltaGMM from\nDDIM sampling with w={w:.1f}")
    plt.legend()
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Time step (DDIM)")
    plt.tight_layout()
    saveallforms(figoutdir, f"{model_cond}_allclass_exact2ddim{w:.1f}_traj_deviation_DDIMremap_quartile", )
    plt.show()
#%%
for w in [0.0, 0.5, 2.0, 5.0]:
    plt.figure(figsize=(4, 3.5))
    for class_id in range(10):
        cond_xt_traj = np.load(join(savedir, f"{model_cond}_class{class_id}_xt_traj.npz"), allow_pickle=True)
        plot_mean_with_quantile(_dist_from_keys(cond_xt_traj, f"w{w:.1f}", "uni"), (0.25, 0.75),
                                f"Digit{class_id}", color=f"C{class_id}", ax=None)
    plt.title(f"Deviation of uniGaussian from\nDDIM sampling with w={w:.1f}")
    plt.legend()
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Time step (DDIM)")
    plt.tight_layout()
    saveallforms(figoutdir, f"{model_cond}_allclass_uni2ddim{w:.1f}_traj_deviation_DDIMremap_quartile", )
    plt.show()

#%% Dev zone for interpolation
plt.plot(ttraj, alphacumprod_ddim, label="DDIM")
plt.plot(ttraj, alphacumprod_gmm, label="GMM")
plt.legend()
plt.show()

#%% Dev zone for interpolation

#%% scratch space
RNDseed = 0
analydata = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_all.pkl"), "rb"))
ddimdata = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_ddim_traj.pkl"), "rb"))
# list(ddimdata)  # ['x_t', 'eps1', 'eps2', 'eps', 'pred_x0', 't']
# list(analydata)  # ['x0_uni', 'x0_gmm', 'x0_exact', 'sol_uni', 'sol_gmm', 'sol_exact']
ddimdata["x_t"].shape  # (50, 1, 1, 28, 28)
analydata['sol_uni'].y.shape  # (784, 51)
#%%
# use the entries of alphacumprod_ddim to find the index in alphacumprod_gmm and interpolate the value
from scipy.interpolate import interp1d
alphacumprod_ddim_interp = interp1d(alphacumprod_gmm, ttraj)
RNDseed = 0
analydata = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_all.pkl"), "rb"))
ddimdata = pkl.load(open(join(savedir, f"uncond_RND{RNDseed:03d}_ddim_traj.pkl"), "rb"))
# list(ddimdata)  # ['x_t', 'eps1', 'eps2', 'eps', 'pred_x0', 't']
# list(analydata)  # ['x0_uni', 'x0_gmm', 'x0_exact', 'sol_uni', 'sol_gmm', 'sol_exact']
ddimdata["x_t"].shape  # (50, 1, 1, 28, 28)
analydata['sol_uni'].y.shape  # (784, 51)
xt_ddim = ddimdata["x_t"]
xt_ddim = xt_ddim.reshape(xt_ddim.shape[0], -1)
sol_uni = analydata["sol_uni"]
sol_gmm = analydata["sol_gmm"]
sol_exact = analydata["sol_exact"]
ttraj = sol_uni.t
xt_uni = sol_uni.y.T
xt_gmm = sol_gmm.y.T
xt_exact = sol_exact.y.T
xt_ddim_aug = np.concatenate([xt_uni[:1], xt_ddim], axis=0)
#%%
# xt_ddim_remap = interp1d(alphacumprod_ddim, xt_ddim_aug, axis=0, kind="linear")(alphacumprod_gmm)
xt_uni_remap = interp1d(alphacumprod_gmm[::-1], xt_uni, axis=0, kind="linear")(alphacumprod_ddim[::-1])
xt_gmm_remap = interp1d(alphacumprod_gmm[::-1], xt_gmm, axis=0, kind="linear")(alphacumprod_ddim[::-1])
xt_exact_remap = interp1d(alphacumprod_gmm[::-1], xt_exact, axis=0, kind="linear")(alphacumprod_ddim[::-1])
#%%




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