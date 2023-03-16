"""Playground for Gaussian mixture model diffusion dynamics
wit general covariance matrix.
Reverse diffusion
"""
import numpy as np
import matplotlib.pyplot as plt
from core.gmm_general_diffusion_lib import exact_general_gmm_reverse_diff, \
    demo_gaussian_mixture_diffusion, _random_orthogonal_matrix, \
    gaussian_mixture_logprob_score
#%%
mus = np.array([[0, 0],
               [1, 1],
               [.5, .5], ])  # [N comp, N dim]
Lambdas = np.array([[.8, .2],
                   [.5, .2],
                   [.2, .8],]) # [N comp, N dim]
Us = np.stack([_random_orthogonal_matrix(2) for i in range(3)], axis=0)
#%%
demo_gaussian_mixture_diffusion(500, mus, Us, Lambdas, )
#%%
xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
pnts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
logprob_pnts, score_pnts = gaussian_mixture_logprob_score(pnts, mus=mus, Us=Us, Lambdas=Lambdas, weights=None)
#%%
slc = slice(None, None, 5)
plt.figure(figsize=[8, 8])
plt.contour(xx, yy, logprob_pnts.reshape(xx.shape), 30)
plt.colorbar()
plt.quiver(xx[slc,slc], yy[slc,slc],
           score_pnts[:, 0].reshape(xx.shape)[slc,slc],
           score_pnts[:, 1].reshape(xx.shape)[slc,slc])
plt.axis("image")
plt.tight_layout()
plt.show()
#%%
#%% Visualize the reverse diffusion of a single point
xT = np.random.randn(2)  # np.array([1.5, 0.5])
t_eval = np.linspace(1, 0, 100)
x0, sol = exact_general_gmm_reverse_diff(mus, Us, Lambdas, xT, t_eval=t_eval)
#%%
plt.figure()
plt.contour(xx, yy, logprob_pnts.reshape(xx.shape), 30)
plt.plot(sol.y[0, :], sol.y[1, :], 'r')
plt.plot(xT[0], xT[1], 'rx')
plt.plot(x0[0], x0[1], 'bo')
plt.show()
#%%
sol_col = []
for i in range(500):
    xT = np.random.randn(2)
    x0, sol = exact_general_gmm_reverse_diff(mus, Us, Lambdas, xT, t_eval=t_eval)
    sol_col.append(sol)

x0_col = [sol.y[:, -1] for sol in sol_col]
xT_col = [sol.y[:, 0] for sol in sol_col]
x0_col = np.stack(x0_col, axis=0)
xT_col = np.stack(xT_col, axis=0)
#%%
plt.figure(figsize=(8, 8))
plt.contour(xx, yy, logprob_pnts.reshape(xx.shape), 30)
for i, sol in enumerate(sol_col):
    plt.plot(sol.y[0, :], sol.y[1, :], c="k", alpha=0.1, lw=0.75,label=None if i > 0 else "trajectories")
plt.scatter(x0_col[:, 0], x0_col[:, 1], s=40, c="b", alpha=0.3, label="final x0")
plt.scatter(xT_col[:, 0], xT_col[:, 1], s=40, c="k", alpha=0.1, label="initial xT")
plt.scatter(mus[:, 0], mus[:, 1], s=64, c="r", alpha=0.3, label="GMM centers")
plt.axis("image")
plt.legend()
plt.tight_layout()
plt.show()
#%% test unimodal Gaussian
# ndim = 2
# x = np.random.randn(10, ndim)  # [N batch, N dim]
# mu = np.random.randn(ndim)
# U = _random_orthogonal_matrix(ndim)
# Lambda = np.exp(5 * np.random.rand(ndim))  # np.array([5, 1])
# logdetSigma = np.sum(np.log(Lambda))
# residual = (x - mu[None, :])  # [N batch, N dim]
# rot_residual = residual @ U   # [N batch, N dim]
# MHdist = np.sum(rot_residual ** 2 / Lambda[None, :], axis=-1)  # [N batch,]
# logprob = - 0.5 * (logdetSigma + MHdist) - 0.5 * ndim * np.log(2 * np.pi)  # [N batch,]
# score_vec = - (rot_residual / Lambda[None, :]) @ U.T  # [N batch, N dim]
# cov = U @ np.diag(Lambda) @ U.T
# logprob_scipy = multivariate_normal.logpdf(x, mean=mu, cov=cov)
# assert np.allclose(logprob, logprob_scipy)
# #%%
# # gaussian mixture density from scipy

#%% Test mulit modal GMM mixture
# # a function of Multivariate Gaussian density
# mus = np.array([[0, 0],
#                [1, 1],
#                [.5, .5],]) # [N comp, N dim]
# Lambdas = np.array([[5, 1],
#                    [1, 5],
#                    [1, 1],]) # [N comp, N dim]
# Us = np.stack([_random_orthogonal_matrix(2) for i in range(3)], axis=0)
# # cov = U @ np.diag(Lambda) @ U.T
# weights = np.array([0.3, 0.3, 0.4])  # [N comp,]
# #%%
# x = np.random.randn(10, 2)  # [N batch, N dim]
#
#
# ndim = x.shape[-1]
# alpha_t_sq = alpha(0.5)
# sigma_t_sq = (1 - alpha_t_sq)
# Lambdas_t = Lambdas * alpha_t_sq + sigma_t_sq  # [N comp, N dim]
#
#
# logdetSigmas = np.sum(np.log(Lambdas_t), axis=-1)  # [N comp,]
# residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
# # `Us` has shape [N comp, N dim, N dim]
# rot_residuals = np.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
# MHdists = np.sum(rot_residuals ** 2 / Lambdas_t[None, :, :], axis=-1)  # [N batch, N comp]
# if weights is not None:
#     logprobs = - 0.5 * (logdetSigmas[None, :] + MHdists) + np.log(weights) # - 0.5 * ndim * np.log(2 * np.pi)  # [N batch, N comp]
# else:
#     logprobs = - 0.5 * (logdetSigmas[None, :] + MHdists)
# participance = softmax(logprobs, axis=-1)  # [N batch, N comp]
# compo_score_vecs = np.einsum("BCD,CED->BCE", - (rot_residuals / Lambdas_t[None, :, :]), Us)  # [N batch, N comp, N dim]
# score_vecs = np.einsum("BC,BCE->BE", participance, compo_score_vecs)  # [N batch, N dim]
# prob = np.exp(logprobs).sum(axis=-1) / (2 * np.pi) ** (ndim / 2)  # [N batch,]

