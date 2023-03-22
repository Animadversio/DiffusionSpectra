import matplotlib.pyplot as plt
import numpy as np

from core.gaussian_mixture_lib import GaussianMixture, GaussianMixture_torch
from core.gmm_special_diffusion_lib import demo_delta_gmm_diffusion
from core.gmm_general_diffusion_lib import demo_gaussian_mixture_diffusion, _random_orthogonal_matrix
#%%
mus = np.linspace(-1, 1, 101)[:, None]
mus = np.concatenate([mus, mus], axis=1)
demo_delta_gmm_diffusion(nreps=500, mus=mus, sigma=1E-5)
#%%
mus = np.array([[0, -1],
                [-.8, 0.5],
                [1, 1], ])  # [N comp, N
Lambdas = np.array([[.8, .2],
                        [.5, .2],
                        [.2, .8], ])
# Us = np.stack([_random_orthogonal_matrix(2) for i in range(3)], axis=0)
U1 = np.array([[-0.93850443,  0.3452672 ],
               [ 0.3452672 ,  0.93850443]])
U2 = np.array([[-0.91488226, -0.40372076],
               [-0.40372076,  0.91488226]])
U3 = np.array([[-0.8923019 ,  0.45143916],
               [ 0.45143916,  0.8923019 ]])
Us = np.stack([U1, U2, U3], axis=0)
demo_gaussian_mixture_diffusion(nreps=500, mus=mus, Us=Us, Lambdas=Lambdas, weights=None)
#%%
covs = [Us[i] @ np.diag(Lambdas[i]) @ Us[i].T for i in range(3)]
gmm = GaussianMixture(mus, covs, weights=[1,1,1])
#%%
gmm_samples,_,_ = gmm.sample(500)
mean_gmm = gmm_samples.mean(axis=0, keepdims=True)
cov_gmm = np.cov(gmm_samples.T)
Lambda_gmm, U_gmm = np.linalg.eigh(cov_gmm)

figh = demo_gaussian_mixture_diffusion(nreps=500, mus=mus, Us=Us, Lambdas=Lambdas, weights=None)
figh.gca().scatter(gmm_samples[:, 0], gmm_samples[:, 1], s=100, c='k')
figh.show()
#%%
figh2 = demo_delta_gmm_diffusion(nreps=500, mus=gmm_samples, sigma=1E-4)
#%%
figh3 = demo_gaussian_mixture_diffusion(nreps=500, mus=mean_gmm, Us=U_gmm[None], Lambdas=Lambda_gmm[None], weights=None)
