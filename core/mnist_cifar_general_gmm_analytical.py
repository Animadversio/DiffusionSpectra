"""This demo shows if the exact score given by the delta function
GMM distribution is used and sampled using the reverse ODE, it will yield similar results.

"""
import os

import numpy as np
import matplotlib.pyplot as plt
from core.gmm_special_diffusion_lib import f_VP_vec, alpha, beta, GMM_scores, GMM_density, exact_delta_gmm_reverse_diff
from scipy.integrate import solve_ivp
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from core.gmm_general_diffusion_lib import f_VP_gmm_vec, exact_general_gmm_reverse_diff, \
    demo_gaussian_mixture_diffusion
from tqdm import tqdm, trange


def mean_cov_from_xarr(xarr):
    # xarr shape [n sample, ndim]
    # get mean and covariance of the xarr
    mu = xarr.mean(axis=0, )
    cov = np.cov(xarr.T)
    # eigen decomposition
    Lambda, U = np.linalg.eigh(cov)
    # assert np.allclose(cov, U @ np.diag(Lambda) @ U.T)
    return mu, cov, Lambda, U
#%% test on MNIST
dataset = MNIST(r'E:\Datasets', download=True)
# load whole MNIST into a single tensor
xtsr = torch.stack([ToTensor()(img) for img, _ in dataset], dim=0)
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)

imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
#%%
#%%
xarr = xtsr[ytsr == 4, :, :].flatten(1).numpy()
# get mean and covariance of the xarr
mus = xarr.mean(axis=0, keepdims=True)
cov = np.cov(xarr.T)
# eigen decomposition
Lambda, U = np.linalg.eigh(cov)
assert np.allclose(cov, U @ np.diag(Lambda) @ U.T)

#%%
xT = np.random.randn(ndim)
sol = solve_ivp(lambda t, x: f_VP_gmm_vec(t, x, mus, U[None,], Lambda[None,], sigma=1E-5),
                (1, 0), xT, method="RK45", vectorized=True)
# t_eval=np.linspace(1, 0, 100)
x0 = sol.y[:, -1]  # [space dim, traj T]

x0img = x0.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
plt.figure()
plt.imshow(x0img, cmap="gray")
plt.show()

#%% Compute mean and covariance of each class
mu_cls = []
cov_cls = []
Lambda_cls = []
U_cls = []
weights = []
for label in range(10):
    xarr = xtsr[ytsr == label, :, :].flatten(1).numpy()
    mu, cov, Lambda, U = mean_cov_from_xarr(xarr)
    assert np.allclose(cov, U @ np.diag(Lambda) @ U.T)
    mu_cls.append(mu)
    cov_cls.append(cov)
    Lambda_cls.append(Lambda)
    U_cls.append(U)
    weights.append(xarr.shape[0])

mu_cls = np.stack(mu_cls, axis=0)
cov_cls = np.stack(cov_cls, axis=0)
Lambda_cls = np.stack(Lambda_cls, axis=0)
U_cls = np.stack(U_cls, axis=0)
# single Gaussian approximation
xarr_all = xtsr.flatten(1).numpy()
mu_all, cov_all, Lambda_all, U_all = mean_cov_from_xarr(xarr_all)

#%%
import pickle as pkl
from os.path import join
from core.utils.montage_utils import make_grid_np
#%% Unconditional generation using the exact delta function GMM, GMM, and single Gaussian
outdir = r"F:\insilico_exps\Diffusion_traj\mnist_uncond_gmm_exact"
os.makedirs(outdir, exist_ok=True)
t_eval = np.linspace(1, 0, 51)
for RNDseed in trange(100):
    np.random.seed(RNDseed)
    xT = np.random.randn(ndim)
    x0_uni, sol_uni = exact_general_gmm_reverse_diff(mu_all[None], U_all[None], Lambda_all[None], xT, sigma=1E-4, t_eval=t_eval)
    x0_gmm, sol_gmm = exact_general_gmm_reverse_diff(mu_cls, U_cls, Lambda_cls, xT, sigma=1E-4, t_eval=t_eval)
    x0_exact, sol_exact = exact_delta_gmm_reverse_diff(xarr_all, sigma=1E-4, xT=xT, t_eval=t_eval)
    x0img_uni = x0_uni.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
    x0img_gmm = x0_gmm.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
    x0img_exact = x0_exact.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
    # save
    plt.imsave(join(outdir, f"uncond_RND{RNDseed:03d}_unigauss.png"), x0img_uni.repeat(3, axis=2))
    plt.imsave(join(outdir, f"uncond_RND{RNDseed:03d}_gmm.png"), x0img_gmm.repeat(3, axis=2))
    plt.imsave(join(outdir, f"uncond_RND{RNDseed:03d}_exact.png"), x0img_exact.repeat(3, axis=2))
    plt.imsave(join(outdir, f"uncond_RND{RNDseed:03d}_cmb.png"),
               make_grid_np([x0img_uni, x0img_gmm, x0img_exact], nrow=3, padding=4))
    pkl.dump({"x0_uni": x0_uni, "x0_gmm": x0_gmm, "x0_exact": x0_exact,
              "sol_uni": sol_uni, "sol_gmm": sol_gmm, "sol_exact": sol_exact},
             open(join(outdir, f"uncond_RND{RNDseed:03d}_all.pkl"), "wb"))

    # plt.figure(figsize=(11, 4))
    # plt.subplot(131)
    # plt.imshow(x0img_gmm, cmap="gray")
    # plt.title("GMM")
    # plt.subplot(132)
    # plt.imshow(x0img_uni, cmap="gray")
    # plt.title("Single Gaussian")
    # plt.subplot(133)
    # plt.imshow(x0img_exact, cmap="gray")
    # plt.title("all data delta gmm")
    # plt.tight_layout()
    # plt.show()


#%% Conditional generation using the exact delta function GMM, and single Gaussian
outdir = r"F:\insilico_exps\Diffusion_traj\mnist_cond_gmm_exact"
os.makedirs(outdir, exist_ok=True)
t_eval = np.linspace(1, 0, 51)
for class_id in trange(10):
    mus = xtsr[ytsr == class_id, :, :, :].flatten(1).numpy()
    mu_sing, cov_sing, Lambda_sing, U_sing = mean_cov_from_xarr(mus)
    # set manual seed
    for RNDseed in trange(100):
        np.random.seed(RNDseed)
        xT = np.random.randn(ndim)
        x0_gauss, sol_gauss = exact_general_gmm_reverse_diff(mu_sing[None], U_sing[None], Lambda_sing[None], xT, sigma=1E-4, t_eval=t_eval)
        x0_exact, sol_exact = exact_delta_gmm_reverse_diff(mus, sigma=1E-4, xT=xT, t_eval=t_eval)
        x0img_gauss = x0_gauss.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
        x0img_exact = x0_exact.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
        plt.imsave(join(outdir, f"class{class_id}_RND{RNDseed:03d}_gauss.png"), x0img_gauss.repeat(3, axis=2))
        plt.imsave(join(outdir, f"class{class_id}_RND{RNDseed:03d}_exact.png"), x0img_exact.repeat(3, axis=2))
        plt.imsave(join(outdir, f"class{class_id}_RND{RNDseed:03d}_cmb.png"),
                   make_grid_np([x0img_gauss, x0img_exact], nrow=2, padding=4))
        pkl.dump({"x0_gauss": x0_gauss, "x0_exact": x0_exact,
                  "sol_gauss": sol_gauss, "sol_exact": sol_exact},
                 open(join(outdir, f"class{class_id}_RND{RNDseed:03d}_all.pkl"), "wb"))
        # pkl.dump(sol_gauss, open(join(outdir, f"class{class_id}_RND{RNDseed:03d}_gauss.pkl"), "wb"))
        # pkl.dump(sol_exact, open(join(outdir, f"class{class_id}_RND{RNDseed:03d}_exact.pkl"), "wb"))
        # plt.figure(figsize=(11, 4))
        # plt.subplot(121)
        # plt.imshow(x0img_gauss, cmap="gray")
        # plt.title("Single Gaussian")
        # plt.subplot(122)
        # plt.imshow(x0img_exact, cmap="gray")
        # plt.title("all data delta gmm")
        # plt.tight_layout()
        # plt.show()
        # raise ValueError



#%%
# conduct PCA of sol.y
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(sol.y.T)
#%%
plt.figure()
plt.scatter(pca.transform(sol.y.T)[:, 0], pca.transform(sol.y.T)[:, 1], s=32)
plt.show()







#%% test on CIFAR10
dataset = CIFAR10(r'E:\Datasets', download=True)
# load whole MNIST into a single tensor
xtsr = torch.stack([ToTensor()(img) for img, _ in dataset], dim=0)
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
#%%
mus = xtsr[ytsr == 9, :, :, :].flatten(1).numpy()
# set manual seed
np.random.seed(0)
xT = np.random.randn(ndim)
sol = solve_ivp(lambda t, x: f_VP_vec(t, x, mus, sigma=1E-5),
                (1, 0), xT, method="RK45",
                vectorized=True, t_eval=np.linspace(1, 0, 51))
#
x0 = sol.y[:, -1]  # [space dim, traj T]
#%%
x0_img = x0.clip(0, 1).reshape(imgshape).transpose((1, 2, 0))
plt.figure()
plt.imshow(x0_img, )
plt.show()
#%%
from os.path import join
import pickle as pkl
outdir = r"F:\insilico_exps\Diffusion_traj\cifar_cond_gmm_exact"
os.makedirs(outdir, exist_ok=True)
t_eval = np.linspace(1, 0, 51)
for class_id in trange(9, 10):
    mus = xtsr[ytsr == class_id, :, :, :].flatten(1).numpy()
    mu_sing, cov_sing, Lambda_sing, U_sing = mean_cov_from_xarr(mus)
    # set manual seed
    for RNDseed in trange(100):
        np.random.seed(RNDseed)
        xT = np.random.randn(ndim)
        x0_gauss, sol_gauss = exact_general_gmm_reverse_diff(mu_sing[None], U_sing[None], Lambda_sing[None], xT, sigma=1E-4, t_eval=t_eval)
        x0_exact, sol_exact = exact_delta_gmm_reverse_diff(mus, sigma=1E-4, xT=xT, t_eval=t_eval)
        x0img_gauss = x0_gauss.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
        x0img_exact = x0_exact.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
        plt.imsave(join(outdir, f"class{class_id}_RND{RNDseed:03d}_gauss.png"), x0img_gauss)
        plt.imsave(join(outdir, f"class{class_id}_RND{RNDseed:03d}_exact.png"), x0img_exact)
        plt.imsave(join(outdir, f"class{class_id}_RND{RNDseed:03d}_cmb.png"),
                   make_grid_np([x0img_gauss, x0img_exact], nrow=2, padding=4))
        pkl.dump({"x0_gauss": x0_gauss, "x0_exact": x0_exact,
                  "sol_gauss": sol_gauss, "sol_exact": sol_exact},
                 open(join(outdir, f"class{class_id}_RND{RNDseed:03d}_all.pkl"), "wb"))

#%%
outdir = r"F:\insilico_exps\Diffusion_traj\cifar_uncond_gmm_exact"
os.makedirs(outdir, exist_ok=True)
t_eval = np.linspace(1, 0, 51)
for RNDseed in trange(100):
    np.random.seed(RNDseed)
    xT = np.random.randn(ndim)
    x0_uni, sol_uni = exact_general_gmm_reverse_diff(mu_all[None], U_all[None], Lambda_all[None], xT, sigma=1E-4, t_eval=t_eval)
    x0_gmm, sol_gmm = exact_general_gmm_reverse_diff(mu_cls, U_cls, Lambda_cls, xT, sigma=1E-4, t_eval=t_eval)
    x0_exact, sol_exact = exact_delta_gmm_reverse_diff(xarr_all, sigma=1E-4, xT=xT, t_eval=t_eval)
    x0img_uni = x0_uni.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
    x0img_gmm = x0_gmm.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
    x0img_exact = x0_exact.reshape(imgshape).transpose((1, 2, 0)).clip(0, 1)
    # save
    plt.imsave(join(outdir, f"uncond_RND{RNDseed:03d}_unigauss.png"), x0img_uni)
    plt.imsave(join(outdir, f"uncond_RND{RNDseed:03d}_gmm.png"), x0img_gmm)
    plt.imsave(join(outdir, f"uncond_RND{RNDseed:03d}_exact.png"), x0img_exact)
    plt.imsave(join(outdir, f"uncond_RND{RNDseed:03d}_cmb.png"),
               make_grid_np([x0img_uni, x0img_gmm, x0img_exact], nrow=3, padding=4))
    pkl.dump({"x0_uni": x0_uni, "x0_gmm": x0_gmm, "x0_exact": x0_exact,
              "sol_uni": sol_uni, "sol_gmm": sol_gmm, "sol_exact": sol_exact},
             open(join(outdir, f"uncond_RND{RNDseed:03d}_all.pkl"), "wb"))

