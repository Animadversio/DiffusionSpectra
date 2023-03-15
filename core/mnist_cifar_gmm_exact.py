"""This demo shows if the exact score given by the delta function
GMM distribution is used and sampled using the reverse ODE, it will yield similar results.

"""

import numpy as np
import matplotlib.pyplot as plt
from core.gmm_special_diffusion_lib import f_VP_vec, alpha, beta, GMM_scores, GMM_density, exact_delta_gmm_reverse_diff
from scipy.integrate import solve_ivp
#%%
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
#%% test on MNIST
dataset = MNIST(r'E:\Datasets', download=True)
# load whole MNIST into a single tensor
xtsr = torch.cat([ToTensor()(img) for img, _ in dataset], dim=0)
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
#%%
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
#%%
mus = xtsr[ytsr == 4, :, :].flatten(1).numpy()
#%%
xT = np.random.randn(ndim)
sol = solve_ivp(lambda t, x: f_VP_vec(t, x, mus, sigma=1E-3),
                (1, 0), xT, method="RK45",
                vectorized=True)
# t_eval=np.linspace(1, 0, 100)
x0 = sol.y[:, -1]  # [space dim, traj T]
#%%
plt.figure()
plt.imshow(x0.reshape(imgshape).transpose([1,2,0]), cmap="gray")
plt.show()
#%%
# conduct PCA of sol.y
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(sol.y.T)
#%%
plt.figure()
plt.scatter(pca.transform(sol.y.T)[:, 0], pca.transform(sol.y.T)[:, 1], s=32)
plt.show()
#%%


#%% test on CIFAR10
dataset = CIFAR10(r'E:\Datasets', download=True)
# load whole MNIST into a single tensor
xtsr = torch.stack([ToTensor()(img) for img, _ in dataset], dim=0)
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
#%%
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
from tqdm import tqdm, trange
outdir = r"F:\insilico_exps\Diffusion_traj\cifar10_gmm_exact"
for class_id in trange(10):
    mus = xtsr[ytsr == class_id, :, :, :].flatten(1).numpy()
    # set manual seed
    for RNDseed in trange(100):
        np.random.seed(RNDseed)
        xT = np.random.randn(ndim)
        sol = solve_ivp(lambda t, x: f_VP_vec(t, x, mus, sigma=1E-5),
                        (1, 0), xT, method="RK45",
                        vectorized=True, t_eval=np.linspace(1, 0, 51))

        x0 = sol.y[:, -1]  # [space dim, traj T]
        x0_img = x0.clip(0, 1).reshape(imgshape).transpose((1, 2, 0))
        plt.imsave(join(outdir, f"{class_id}_{RNDseed}.png"), x0_img)
        pkl.dump(sol, open(join(outdir, f"{class_id}_{RNDseed}_sol.pkl"), "wb"))


