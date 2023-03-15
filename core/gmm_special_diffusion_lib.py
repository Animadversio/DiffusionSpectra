import numpy as np
from scipy.special import softmax


def GMM_density(mus, sigma, x):
    Nbranch = mus.shape[0]
    Ndim = mus.shape[1]
    sigma2 = sigma**2
    normfactor = np.sqrt((2 * np.pi * sigma)**Ndim)
    res = x[:, None, :] - mus[None, :, :]  # [x batch, mu, space dim]
    dist2 = np.sum(res ** 2, axis=-1)  # [x batch, mu]
    prob = np.exp(- dist2 / sigma2 / 2, )  # [x batch, mu]
    prob_all = np.sum(prob, axis=1) / Nbranch / normfactor  # [x batch,]
    return prob_all


def GMM_scores(mus, sigma, x):
    # for both input x and mus, the shape is [batch, space dim]
    sigma2 = sigma**2
    res = x[:, None, :] - mus[None, :, :]  # [x batch, mu, space dim]
    dist2 = np.sum(res ** 2, axis=-1)  # [x batch, mu]
    participance = softmax(- dist2 / sigma2 / 2, axis=1)  # [x batch, mu]
    scores = - np.einsum("ij,ijk->ik", participance, res) / sigma2   # [x batch, space dim]
    return scores


def beta(t):
    return (0.02 * t + 0.0001 * (1 - t)) * 1000


def alpha(t):
    # return np.exp(- 1000 * (0.01 * t**2 + 0.0001 * t))
    return np.exp(- 10 * t**2 - 0.1 * t) * 0.9999


def score_t(t, x, mus, sigma=1E-6):
    """Score function of p(x,t) according to VP SDE probability flow"""
    alpha_t = alpha(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return GMM_scores(alpha_t * mus, sigma_t_sq, x[None, :])[0, :]


def score_t_vec(t, x, mus, sigma=1E-6):
    """Vectorized version of score_t"""
    alpha_t = alpha(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return GMM_scores(alpha_t * mus, sigma_t_sq, x.T).T


def f_VP(t, x, mus, sigma=1E-6):
    """Right hand side of the VP SDE probability flow ODE"""
    alpha_t = alpha(t)
    beta_t = beta(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + GMM_scores(alpha_t * mus, np.sqrt(sigma_t_sq), x[None, :])[0, :])


def f_VP_vec(t, x, mus, sigma=1E-6):
    """Right hand side of the VP SDE probability flow ODE, vectorized version"""
    alpha_t = alpha(t)
    beta_t = beta(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + GMM_scores(alpha_t * mus, np.sqrt(sigma_t_sq), x.T).T)


def f_VP_noise_vec(t, x, mus, sigma=1E-6, noise_std=0.01):
    alpha_t = alpha(t)
    beta_t = beta(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + GMM_scores(alpha_t * mus, np.sqrt(sigma_t_sq), x.T).T + noise_std * np.random.randn(*x.shape))


from scipy.integrate import solve_ivp
def exact_delta_gmm_reverse_diff(mus, sigma, xT, t_eval=None):
    sol = solve_ivp(lambda t, x: f_VP_vec(t, x, mus, sigma=sigma),
                    (1, 0), xT, method="RK45",
                    vectorized=True, t_eval=t_eval)
    return sol.y[:, -1]