# gaussian mixture density
import torch
import numpy as np
from core.gaussian_mixture_lib import GaussianMixture_torch, GaussianMixture, quiver_plot
ndim = 2
mus = [np.random.randn(ndim),
       # np.random.rand(ndim),
       # np.random.rand(ndim),
       np.random.randn(ndim)]
basis1 = np.random.randn(ndim, ndim) / 5
basis2 = np.random.randn(ndim, ndim) / 5
basis3 = np.random.randn(ndim, ndim) / 5
basis4 = np.random.randn(ndim, ndim) / 5
covs = [basis1@np.eye(ndim)@basis1.T,
        # basis2@np.eye(ndim)@basis2.T,
        # basis3@np.eye(ndim)@basis3.T,
        basis4@np.eye(ndim)@basis4.T, ]
gmm = GaussianMixture(mus, covs, np.ones(len(covs)))
# mu_norm = np.linalg.norm(np.array(mus),axis=1)
xx, yy = np.mgrid[-3:3:.05, -3:3:.05]
#%
density = gmm.pdf(np.stack((xx.flatten(), yy.flatten())).T)
UV = gmm.score(np.stack((xx.flatten(), yy.flatten())).T)
U = UV[:, 0].reshape(xx.shape)
V = UV[:, 1].reshape(xx.shape)
density = density.reshape(xx.shape)
#%%
slc = slice(None, None, 6)
from matplotlib import pyplot as plt
plt.quiver(xx[slc,slc], yy[slc,slc], U[slc,slc], V[slc,slc], scale=5000)
plt.axis('image')
# quiver_plot(xx, yy, U, V)
plt.show()
#%%
plt.contourf(xx, yy, density)
plt.axis('image')
plt.show()

