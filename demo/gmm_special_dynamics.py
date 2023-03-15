import numpy as np
from scipy.special import softmax

class DeltaGMM:
    def __init__(self, mus, sigma):
        self.mus = np.array(mus)
        self.sigma = sigma
        self.n_component = len(self.mus)
        self.dim = self.mus.shape[1]

    def score(self, x):
        """
        Compute the score $\nabla_x \log p(x)$ for the given $x$.
        """
        scores = np.zeros_like(x)
        for i in range(self.n_component):
            scores += (x - self.mus[i])
        return scores
#%%
mus = np.array([[0, 0],
                [1, 1],
                [2, 2]])

x = np.array([[0.5, 0.5]])
sigma = 1
sigma2 = sigma**2
res = x[:, None, :] - mus[None, :, :]  # [x batch, mu, space dim]
dist2 = np.sum(res ** 2, axis=-1)  # [x batch, mu]
participance = softmax(- dist2 / sigma2 / 2, axis=1)  # [x batch, mu]
scores = np.einsum("ij,ijk->ik", participance, res)   # [x batch, space dim]

#%%
def GMM_density(mus, sigma, x):
    Nbranch = mus.shape[0]
    Ndim = mus.shape[1]
    sigma2 = sigma**2
    normfactor = np.sqrt((2 * np.pi * sigma)**Ndim)
    res = x[:, None, :] - mus[None, :, :]  # [x batch, mu, space dim]
    dist2 = np.sum(res ** 2, axis=-1)  # [x batch, mu]
    prob = np.exp(- dist2 / sigma2 / 2, )  # [x batch, mu]
    prob_all = np.sum(prob, axis=1) / Nbranch / normfactor # [x batch,]
    return prob_all


def GMM_scores(mus, sigma_sq, x):
    # sigma2 = sigma**2
    res = x[:, None, :] - mus[None, :, :]  # [x batch, mu, space dim]
    dist2 = np.sum(res ** 2, axis=-1)  # [x batch, mu]
    participance = softmax(- dist2 / sigma_sq / 2, axis=1)  # [x batch, mu]
    scores = - np.einsum("ij,ijk->ik", participance, res) / sigma_sq   # [x batch, space dim]
    return scores
#%%
mus = np.random.randn(500, 2)
sigma = 0.7
xx, yy = np.meshgrid(np.linspace(-3, 3, 51), np.linspace(-3, 3, 51))
pnts = np.stack([xx, yy], axis=-1).reshape(-1, 2)
scores_vecs = GMM_scores(mus, sigma**2, pnts)
scores_vecs = scores_vecs.reshape((*xx.shape, -1))
density = GMM_density(mus, sigma, pnts)
density = density.reshape(xx.shape)
U = scores_vecs[:, :, 0]
V = scores_vecs[:, :, 1]

#%%
import matplotlib.pyplot as plt
sub_slice = slice(0, None, 2)
plt.figure(figsize=(6, 6))
plt.quiver(xx[sub_slice,sub_slice], yy[sub_slice,sub_slice], U[sub_slice,sub_slice], V[sub_slice,sub_slice])
plt.contour(xx, yy, density, levels=10)
plt.scatter(mus[:, 0], mus[:, 1], c="r", alpha=0.7)
plt.axis("image")
plt.show()

#%%
from scipy.integrate import solve_ivp
def beta(t):
    return (0.02 * t + 0.0001 * (1 - t)) * 1000


def alpha(t):
    # return np.exp(- 1000 * (0.01 * t**2 + 0.0001 * t))
    return np.exp(- 10 * t**2 - 0.1 * t) * 0.9999

#%%
def score_t(t, x):
    alpha_t = alpha(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return GMM_scores(alpha_t * mus, sigma_t_sq, x[None, :])[0, :]

def score_t_vec(t, x):
    alpha_t = alpha(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return GMM_scores(alpha_t * mus, sigma_t_sq, x.T).T


def f_VP(t, x):
    alpha_t = alpha(t)
    beta_t = beta(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + GMM_scores(alpha_t * mus, sigma_t_sq, x[None, :])[0, :])


def f_VP_vec(t, x):
    alpha_t = alpha(t)
    beta_t = beta(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    # sigma_t_sq = (1 - alpha_t) + sigma**2
    return - beta_t * (x + GMM_scores(alpha_t * mus, sigma_t_sq, x.T).T) # np.sqrt(alpha_t)


def f_VP_noise_vec(t, x, noise_std=0.01):
    alpha_t = alpha(t)
    beta_t = beta(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + GMM_scores(alpha_t * mus, sigma_t_sq, x.T).T + noise_std * np.random.randn(*x.shape))
#%%
# mus = np.random.randn(500, 2)
# # mus = np.array([[1.0, 0.0]])#np.random.randn(5, 2)
# sigma = 0.001 #1.0#0.1
mus = np.array([[0.0, 0.0]])#np.random.randn(5, 2)
sigma = 0.5 #0.001 #1.0#0.1
# x0 = np.array([0.9, -0.1])
sol_col = []
for i in range(500):
    xT = np.random.randn(2)
    sol = solve_ivp(f_VP_vec, (1, 0), xT, method="RK45",
                    t_eval=np.linspace(1, 0, 1000), vectorized=True)
    sol_col.append(sol)
# sol = solve_ivp(f_VP_vec, (1, 0), x0, method="RK45",
#                 t_eval=np.linspace(1, 0, 1000),
#                 vectorized=True)
#%
x0_col = [sol.y[:, -1] for sol in sol_col]
xT_col = [sol.y[:, 0] for sol in sol_col]
x0_col = np.stack(x0_col, axis=0)
xT_col = np.stack(xT_col, axis=0)
#%%
plt.figure(figsize=(8, 8))
for i, sol in enumerate(sol_col):
    plt.plot(sol.y[0, :], sol.y[1, :], c="k", alpha=0.1, lw=0.75,label=None if i > 0 else "trajectories")
plt.scatter(x0_col[:, 0], x0_col[:, 1], s=40, c="b", alpha=0.3, label="final x0")
plt.scatter(xT_col[:, 0], xT_col[:, 1], s=40, c="k", alpha=0.1, label="initial xT")
plt.scatter(mus[:, 0], mus[:, 1], s=64, c="r", alpha=0.3, label="GMM centers")
plt.axis("image")
plt.legend()
plt.tight_layout()
plt.show()
#%%
# plot the final position of the trajectories
plt.figure(figsize=(8, 8))
plt.scatter(x0_col[:, 0], x0_col[:, 1], s=40, c="b", alpha=0.3, label="final x0")
plt.scatter(xT_col[:, 0], xT_col[:, 1], s=40, c="k", alpha=0.1, label="initial xT")
plt.scatter(mus[:, 0], mus[:, 1], s=64, c="r", alpha=0.3, label="GMM centers")
plt.axis("image")
plt.legend()
plt.tight_layout()
plt.show()
#%%
from diffusers import DDIMPipeline, DDPMPipeline
# model_id = "google/ddpm-cifar10-32"
model_id = "google/ddpm-celebahq-256" # most popular
# model_id = "dimpo/ddpm-mnist"  # most popular
model_id_short = model_id.split("/")[-1]
pipe = DDIMPipeline.from_pretrained(model_id)   # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.scheduler.set_timesteps(101)
t_seq = pipe.scheduler.timesteps
alphas = pipe.scheduler.alphas
betas = pipe.scheduler.betas
alphas_cumprod = pipe.scheduler.alphas_cumprod
#%%
t_ticks = np.linspace(0, 1, 1000)
alpha_ts = alpha(t_ticks)
beta_ts = beta(t_ticks)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t_ticks, alpha_ts, alpha=0.6, label="analytical")
plt.plot(t_ticks, alphas_cumprod, alpha=0.6, label="DDPM")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t_ticks, beta_ts, alpha=0.6, label="analytical")
plt.plot(t_ticks, betas, alpha=0.6, label="DDPM")
plt.legend()
plt.show()
#%%
