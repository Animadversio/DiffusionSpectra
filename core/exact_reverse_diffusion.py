import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class GaussianMixture:
    def __init__(self, mus, covs, weights):
        """
        mus: a list of K 1d np arrays (D,)
        covs: a list of K 2d np arrays (D, D)
        weights: a list or array of K unnormalized non-negative weights, signifying the possibility of sampling from each branch.
          They will be normalized to sum to 1. If they sum to zero, it will err.
        """
        self.n_component = len(mus)
        self.mus = mus
        self.covs = covs
        self.precs = [np.linalg.inv(cov) for cov in covs]
        self.weights = np.array(weights)
        self.norm_weights = self.weights / self.weights.sum()
        self.RVs = []
        for i in range(len(mus)):
            self.RVs.append(multivariate_normal(mus[i], covs[i]))
        self.dim = len(mus[0])

    def add_component(self, mu, cov, weight=1):
        self.mus.append(mu)
        self.covs.append(cov)
        self.precs.append(np.linalg.inv(cov))
        self.RVs.append(multivariate_normal(mu, cov))
        self.weights.append(weight)
        self.norm_weights = self.weights / self.weights.sum()
        self.n_component += 1

    def pdf(self, x):
        """
          probability density (PDF) at $x$.
        """
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        prob = np.dot(component_pdf, self.norm_weights)
        return prob

    def score(self, x):
        """
        Compute the score $\nabla_x \log p(x)$ for the given $x$.
        """
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        scores = np.zeros_like(x)
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ self.precs[i]
            scores += participance[:, i:i + 1] * gradvec

        return scores

    def score_decompose(self, x):
        """
        Compute the grad to each branch for the score $\nabla_x \log p(x)$ for the given $x$.
        """
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        gradvec_list = []
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ self.precs[i]
            gradvec_list.append(gradvec)
            # scores += participance[:, i:i+1] * gradvec

        return gradvec_list, participance

    def sample(self, N):
        """ Draw N samples from Gaussian mixture
        Procedure:
          Draw N samples from each Gaussian
          Draw N indices, according to the weights.
          Choose sample between the branches according to the indices.
        """
        rand_component = np.random.choice(self.n_component, size=N, p=self.norm_weights)
        all_samples = np.array([rv.rvs(N) for rv in self.RVs])
        gmm_samps = all_samples[rand_component, np.arange(N), :]
        return gmm_samps, rand_component, all_samples


from torch.distributions import MultivariateNormal
class GaussianMixture_torch:
    def __init__(self, mus, covs, weights):
        """
        mus: a list of K 1d np arrays (D,)
        covs: a list of K 2d np arrays (D, D)
        weights: a list or array of K unnormalized non-negative weights, signifying the possibility of sampling from each branch.
          They will be normalized to sum to 1. If they sum to zero, it will err.
        """
        self.n_component = len(mus)
        self.mus = mus
        self.covs = covs
        self.precs = [torch.linalg.inv(cov) for cov in covs]
        if weights is None:
            self.weights = torch.ones(self.n_component)
        else:
            self.weights = torch.tensor(weights)
        self.norm_weights = self.weights / self.weights.sum()
        self.RVs = []
        for i in range(len(mus)):
            self.RVs.append(MultivariateNormal(mus[i], covs[i]))
        self.dim = len(mus[0])

    def add_component(self, mu, cov, weight=1):
        self.mus.append(mu)
        self.covs.append(cov)
        self.precs.append(torch.linalg.inv(cov))
        self.RVs.append(MultivariateNormal(mu, cov))
        self.weights.append(weight)
        self.norm_weights = self.weights / self.weights.sum()
        self.n_component += 1

    def pdf(self, x):
        """
          probability density (PDF) at $x$.
        """
        component_pdf = torch.stack([rv.log_prob(x) for rv in self.RVs]).exp().T
        prob = torch.dot(component_pdf, self.norm_weights)
        return prob

    def score(self, x):
        """
        Compute the score $\nabla_x \log p(x)$ for the given $x$.
        """
        component_logpdf = torch.stack([rv.log_prob(x) for rv in self.RVs]).T
        component_pdf_norm = torch.softmax(component_logpdf, dim=1)
        weighted_compon_pdf = component_pdf_norm * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        scores = torch.zeros_like(x)
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ self.precs[i]
            scores += participance[:, i:i + 1] * gradvec

        return scores

    def score_decompose(self, x):
        """
        Compute the grad to each branch for the score $\nabla_x \log p(x)$ for the given $x$.
        """
        component_logpdf = torch.stack([rv.log_prob(x) for rv in self.RVs]).T
        component_pdf_norm = torch.softmax(component_logpdf, dim=1)
        weighted_compon_pdf = component_pdf_norm * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        gradvec_list = []
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ self.precs[i]
            gradvec_list.append(gradvec)
            # scores += participance[:, i:i+1] * gradvec

        return gradvec_list, participance

    def sample(self, N):
        """ Draw N samples from Gaussian mixture
        Procedure:
          Draw N samples from each Gaussian
          Draw N indices, according to the weights.
          Choose sample between the branches according to the indices.
        """
        rand_component = torch.multinomial(self.norm_weights, N, replacement=True)
        # rand_component = np.random.choice(self.n_component, size=N, p=self.norm_weights)
        all_samples = torch.stack([rv.sample((N,)) for rv in self.RVs])
        gmm_samps = all_samples[rand_component, torch.arange(N), :]
        return gmm_samps, rand_component, all_samples


def quiver_plot(pnts, vecs, *args, **kwargs):
    plt.quiver(pnts[:, 0], pnts[:, 1], vecs[:, 0], vecs[:, 1], *args, **kwargs)


def marginal_prob_std(t, sigma):
  """Note that this std -> 0, when t->0
  So it's not numerically stable to sample t=0 in the dataset
  Note an earlier version missed the sqrt...
  """
  return torch.sqrt( (sigma**(2*t) - 1) / 2 / torch.log(torch.tensor(sigma)) ) # sqrt fixed Jun.19


def marginal_prob_std_np(t, sigma):
  return np.sqrt( (sigma**(2*t) - 1) / 2 / np.log(sigma) )


def diffuse_gmm(gmm, t, sigma):
  lambda_t = marginal_prob_std_np(t, sigma)**2 # variance
  noise_cov = np.eye(gmm.dim) * lambda_t
  covs_dif = [cov + noise_cov for cov in gmm.covs]
  return GaussianMixture(gmm.mus, covs_dif, gmm.weights)


def diffuse_gmm_torch(gmm, t, sigma):
  lambda_t = marginal_prob_std(t, sigma)**2 # variance
  noise_cov = torch.eye(gmm.dim) * lambda_t
  covs_dif = [cov + noise_cov for cov in gmm.covs]
  return GaussianMixture_torch(gmm.mus, covs_dif, gmm.weights)
#%%
ndim = 1000
mus = [np.random.rand(ndim),
       np.random.rand(ndim),
       np.random.rand(ndim),
       np.random.rand(ndim)]
covs = [np.eye(ndim), np.eye(ndim), np.eye(ndim), np.eye(ndim)]
gmm = GaussianMixture(mus, covs, np.ones(len(covs)))
mu_norm = np.linalg.norm(np.array(mus),axis=1)
#%%
# x0, _, _ = gmm.sample(1000)
from tqdm import tqdm
import math
sampN = 10
sigma = 10
nsteps = 500
x0, _, _ = gmm.sample(sampN)

lambdaT = (sigma**2 - 1) / (2 * np.log(sigma)) # marginal_prob_std_np(1.0, sigma)
xT = np.sqrt(lambdaT) * np.random.randn(sampN, ndim)
x_traj_rev = np.zeros((*x0.shape, nsteps, ))
x_traj_rev[:,:,0] = xT
dt = 1 / nsteps
for i in tqdm(range(1, nsteps)):
  t = (nsteps - i) * dt # note the time fly back
  # transport the gmm to that at time $t$ and compute score at that time $\nabla \log p_t(x)$
  gmm_t = diffuse_gmm(gmm, t, sigma) # note the time fly back! start from the largest noise scale
  score_xt = gmm_t.score(x_traj_rev[:,:,i-1])
  eps_z = torch.randn(*x0.shape)
  x_traj_rev[:,:,i] = x_traj_rev[:,:,i-1] + eps_z * (sigma ** t) * math.sqrt(dt) + score_xt * dt * sigma**(2*t)
#%%
x_traj1 = x_traj_rev[-1]
x_traj1 = x_traj1 - x_traj1.mean(axis=0)
U, S, V = np.linalg.svd(x_traj1, full_matrices=False)
#%%
plt.plot((S*V.T)[:, :10])
plt.show()
#%%
plt.plot((S**2).cumsum() / (S**2).sum())
plt.show()

#%%
ndim = 2500
mus = [torch.randn(ndim),
       torch.randn(ndim),
       torch.randn(ndim),
       torch.randn(ndim)]
covs = [torch.eye(ndim) for _ in range(len(mus))]
gmm = GaussianMixture_torch(mus, covs, torch.ones(len(covs)))
mu_norm = torch.stack(mus).norm(dim=1)
#%%
# x0, _, _ = gmm.sample(1000)
from tqdm import tqdm
import math
sampN = 10
sigma = 30
nsteps = 51
x0, _, _ = gmm.sample(sampN)

lambdaT = (sigma**2 - 1) / (2 * np.log(sigma)) # marginal_prob_std_np(1.0, sigma)
xT = np.sqrt(lambdaT) * torch.randn(sampN, ndim)
x_traj_rev = torch.zeros((*x0.shape, nsteps, ))
x_traj_rev[:,:,0] = xT
dt = 1 / nsteps
for i in tqdm(range(1, nsteps)):
  t = (nsteps - i) * dt # note the time fly back
  # transport the gmm to that at time $t$ and compute score at that time $\nabla \log p_t(x)$
  gmm_t = diffuse_gmm_torch(gmm, t, sigma) # note the time fly back! start from the largest noise scale
  score_xt = gmm_t.score(x_traj_rev[:,:,i-1])
  eps_z = torch.randn(*x0.shape)
  x_traj_rev[:,:,i] = x_traj_rev[:,:,i-1] + eps_z * (sigma ** t) * math.sqrt(dt) + score_xt * dt * sigma**(2*t)
#%%
x_traj1 = x_traj_rev[-1]
x_traj1 = x_traj1 - x_traj1.mean(axis=0)
U, S, V = torch.svd(x_traj1, some=True)

#%%
plt.plot((S**2).cumsum(0) / (S**2).sum())
plt.show()
#%%
plt.plot((S*V.T)[:, :10])
plt.show()
