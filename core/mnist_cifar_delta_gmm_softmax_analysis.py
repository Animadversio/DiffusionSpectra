"""This demo shows if the exact score given by the delta function
GMM distribution is used and sampled using the reverse ODE, it will yield similar results.

"""
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from tqdm import trange, tqdm
from pathlib import Path
from easydict import EasyDict as edict
import pandas as pd
import pickle as pkl
from core.utils.plot_utils import saveallforms
import numpy as np
import matplotlib.pyplot as plt
from core.gmm_special_diffusion_lib import f_VP_vec, alpha, beta, \
    GMM_scores, GMM_density, exact_delta_gmm_reverse_diff
from scipy.integrate import solve_ivp
from scipy.special import softmax, log_softmax

#%% test on MNIST
dataset = MNIST(r'E:\Datasets', download=True)
# dataset = MNIST(r'~/Datasets', download=True)
# load whole MNIST into a single tensor
xtsr = torch.cat([ToTensor()(img) for img, _ in dataset], dim=0)
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
#%%
yonehot = torch.zeros(ytsr.shape[0], 10)
yonehot.scatter_(1, ytsr.unsqueeze(1), 1)
#%%
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
#%%
# mus = xtsr[ytsr == 4, :, :].flatten(1).numpy()
mus = xtsr[:, :, :].flatten(1).numpy()
#%%
savedir = Path(r"")

#%%
# set random seed
for SEED in range(250):
    np.random.seed(SEED)
    xT = np.random.randn(ndim)
    sol = solve_ivp(lambda t, x: f_VP_vec(t, x, mus, sigma=1E-5),
                    (1, 0), xT, method="RK45",
                    vectorized=True)
    # t_eval=np.linspace(1, 0, 100)
    x0 = sol.y[:, -1:]  # [space dim, traj T]
    savedict = {"x0": x0, "xT":xT, "sol": sol, "SEED": SEED}
    pkl.dump(savedict, open(savedir/f"mnist_cifar_delta_gmm_exact_{SEED}.pkl", "wb"))
    #%%
    # tticks = np.linspace(0, 1, 101)
    # alphaseq = alpha(tticks)
    alpha_ts = alpha(sol.t)
    sigma_ts = np.sqrt(1 - alpha_ts**2)
    alphay = sol.y / (alpha_ts[None, :])
    # L2 distance matrix between mus and sol.y
    L2distmat = np.sum((mus[:, None, :] - alphay.T[None, :, :])**2, axis=2)
    # is there a low mem way to do this?
    # L2distmat = np.zeros((len(mus), len(sol.t)))
    # for i in range(len(sol.t)):
    #     L2distmat[:, i] = np.sum((mus - alphay[:, i][None,:])**2, axis=-1)

    L2distmat_normed = L2distmat / sigma_ts[None, :]**2 * alpha_ts[None, :]**2
    L2distmat_normed_softmax = softmax(-L2distmat_normed, axis=0)
    L2distmat_normed_logsoftmax = log_softmax(-L2distmat_normed, axis=0)
    yonehot_dist_mat = yonehot.T@L2distmat_normed_softmax
    #%%
    for step in range(len(sol.t)):
        w_dist = L2distmat_normed_softmax[:, step]
        w_dist_log = L2distmat_normed_logsoftmax[:, step]
        yonehot_dist = yonehot_dist_mat[:, step].numpy()
        w_max = np.max(w_dist)
        w_min = np.min(w_dist)
        w_max0_1_cnt = np.sum(w_dist > w_max * 0.1)
        w_max0_01_cnt = np.sum(w_dist > w_max * 0.01)
        entropy = -np.sum(w_dist * w_dist_log)
        y_entropy = -np.sum(yonehot_dist * np.log(yonehot_dist))
        y_max = np.argmax(yonehot_dist)
        print(f"step {step}, t {sol.t[step]:.2f} w max {w_max:.3f}, # of w > 0.1max {w_max0_1_cnt}, "
              f"# of w > 0.01max  {w_max0_01_cnt} entropy {entropy:.3f}"
              f"y entropy {y_entropy:.3f} y max {y_max}")
        #%%
        img_w_center = w_dist @ mus
        img_w_center = img_w_center.reshape(imgshape)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(sol.y[:,step].reshape(imgshape), cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(img_w_center, cmap="gray")
        plt.suptitle(f"step {step}, t {sol.t[step]:.2f}\nw max {w_max:.3f}, # of w > 0.1max {w_max0_1_cnt}, # of w > 0.01max  {w_max0_01_cnt}\nentropy {entropy:.3f}")
        plt.show()
#%%

trajdir = Path("F:\insilico_exps\Diffusion_traj\mnist_uncond_gmm_exact")
#%%
plt.get_backend() # 'module://backend_interagg'
# use agg backend to avoid crash
plt.switch_backend("agg")

#%%
# plt.figure()
# plt.imshow(x0.reshape(imgshape), cmap="gray")# .transpose([1,2,0])
# plt.show()
for SEED in trange(400):
    data = pkl.load(open(trajdir/rf"uncond_RND{SEED:03d}_all.pkl", "rb"))
    sol = data["sol_exact"]
    x0 = data["x0_exact"]
    #%%
    alpha_ts = alpha(sol.t)
    sigma_ts = np.sqrt(1 - alpha_ts**2)
    alphay = sol.y / (alpha_ts[None, :])
    # L2 distance matrix between mus and sol.y
    # L2distmat = np.sum((mus[:, None, :] - alphay.T[None, :, :]) ** 2, axis=2)
    # is there a low mem way to do this?
    L2distmat = np.zeros((len(mus), len(sol.t)))
    for i in range(len(sol.t)):
        L2distmat[:, i] = np.sum((mus - alphay[:, i][None,:])**2, axis=-1)
    L2distmat_normed = L2distmat / sigma_ts[None, :] ** 2 * alpha_ts[None, :] ** 2 / 2
    L2distmat_normed_softmax = softmax(-L2distmat_normed, axis=0)
    L2distmat_normed_logsoftmax = log_softmax(-L2distmat_normed, axis=0)
    yonehot_dist_mat = yonehot.T@L2distmat_normed_softmax
    # %%
    stat_col = []
    for step in range(len(sol.t)):
        w_dist = L2distmat_normed_softmax[:, step]
        w_dist_log = L2distmat_normed_logsoftmax[:, step]
        yonehot_dist = yonehot_dist_mat[:, step].numpy()
        w_max = np.max(w_dist)
        w_min = np.min(w_dist)
        w_max0_1_cnt = np.sum(w_dist > w_max * 0.1)
        w_max0_01_cnt = np.sum(w_dist > w_max * 0.01)
        entropy = -np.sum(w_dist * w_dist_log)
        y_entropy = -np.sum(yonehot_dist * np.log(yonehot_dist))
        y_max = np.argmax(yonehot_dist)
        print(f"step {step}, t {sol.t[step]:.2f} alpha {alpha_ts[step]:.3f}, sigma {sigma_ts[step]:.3f}  "
              f" w max {w_max:.3f}, # of w > 0.1max {w_max0_1_cnt}, "
              f"# of w > 0.01max  {w_max0_01_cnt} entropy {entropy:.3f}"
              f"y entropy {y_entropy:.3f} y max {y_max}")
                # %%
        edict_stat = edict(t=sol.t[step], alpha=alpha_ts[step], sigma=sigma_ts[step],
                           w_max=w_max, w_max0_1_cnt=w_max0_1_cnt, w_max0_01_cnt=w_max0_01_cnt,
                           entropy=entropy, y_entropy=y_entropy, y_max=y_max)

        stat_col.append(edict_stat)
    #%%
    # stat_col = edict(stat_col)
    df = pd.DataFrame(stat_col)
    df.to_csv(trajdir/rf"uncond_RND{SEED:03d}_softmax_stat.csv")
    #%%
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    df.plot(x="t", y=["w_max0_1_cnt", "w_max0_01_cnt"],
            ax=plt.gca(), logy=True)
    plt.subplot(1, 2, 2)
    df.plot(x="t", y=["entropy", "y_entropy"], ax=plt.gca())
    plt.suptitle(f"MNIST Unconditional RND{SEED:03d}")
    saveallforms(str(trajdir), rf"uncond_RND{SEED:03d}_softmax_stat")
    plt.show()
#%%

