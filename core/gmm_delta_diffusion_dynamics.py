import numpy as np
from scipy.special import softmax
from core.gmm_special_diffusion_lib import GMM_density, GMM_scores, beta, alpha, \
    GMM_logprob, demo_delta_gmm_diffusion, exact_delta_gmm_reverse_diff, \
    f_VP_vec, f_VP_noise_vec, score_t_vec

figh = demo_delta_gmm_diffusion(nreps=500, mus=np.random.randn(100, 2),
                         sigma=1E-6)


