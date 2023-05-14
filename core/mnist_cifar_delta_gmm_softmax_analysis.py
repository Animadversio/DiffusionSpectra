"""This demo shows if the exact score given by the delta function
GMM distribution is used and sampled using the reverse ODE, it will yield similar results.

"""

import numpy as np
import matplotlib.pyplot as plt
from core.gmm_special_diffusion_lib import f_VP_vec, alpha, beta, GMM_scores, GMM_density, exact_delta_gmm_reverse_diff
from scipy.integrate import solve_ivp
from scipy.special import softmax, log_softmax
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor

#%% test on MNIST
dataset = MNIST(r'E:\Datasets', download=True)
# dataset = MNIST(r'~/Datasets', download=True)
# load whole MNIST into a single tensor
xtsr = torch.cat([ToTensor()(img) for img, _ in dataset], dim=0)
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
#%%
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
#%%
# mus = xtsr[ytsr == 4, :, :].flatten(1).numpy()
mus = xtsr[:, :, :].flatten(1).numpy()
#%%
xT = np.random.randn(ndim)
sol = solve_ivp(lambda t, x: f_VP_vec(t, x, mus, sigma=1E-5),
                (1, 0), xT, method="RK45",
                vectorized=True)
# t_eval=np.linspace(1, 0, 100)
x0 = sol.y[:, -1:]  # [space dim, traj T]
#%%
# plt.figure()
# plt.imshow(x0.reshape(imgshape), cmap="gray")# .transpose([1,2,0])
# plt.show()
#%%
# tticks = np.linspace(0, 1, 101)
# alphaseq = alpha(tticks)
alpha_ts = alpha(sol.t)
sigma_ts = 1 - alpha_ts
alphay = sol.y / (alpha_ts[None, :])
# L2 distance matrix between mus and sol.y
# L2distmat = np.sum((mus[:, None, :] - alphay.T[None, :, :])**2, axis=2)
# is there a low mem way to do this?
L2distmat = np.zeros((len(mus), len(sol.t)))
for i in range(len(sol.t)):
    L2distmat[:, i] = np.sum((mus - alphay[:, i][None,:])**2, axis=-1)

L2distmat_normed = L2distmat / sigma_ts[None, :]**2 * alpha_ts[None, :]**2
L2distmat_normed_softmax = softmax(-L2distmat_normed, axis=0)
L2distmat_normed_logsoftmax = log_softmax(-L2distmat_normed, axis=0)
#%%
for step in range(len(sol.t)):
    w_dist = L2distmat_normed_softmax[:, step]
    w_dist_log = L2distmat_normed_logsoftmax[:, step]
    w_max = np.max(w_dist)
    w_min = np.min(w_dist)
    w_max0_1_cnt = np.sum(w_dist > w_max * 0.1)
    w_max0_01_cnt = np.sum(w_dist > w_max * 0.01)
    entropy = -np.sum(w_dist * w_dist_log)
    print(f"step {step}, t {sol.t[step]:.2f} w max {w_max:.3f}, # of w > 0.1max {w_max0_1_cnt}, # of w > 0.01max  {w_max0_01_cnt} entropy {entropy:.3f}")
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

