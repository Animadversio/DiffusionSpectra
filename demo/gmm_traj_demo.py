import numpy as np

from core.gmm_special_diffusion_lib import demo_delta_gmm_diffusion
from core.gmm_general_diffusion_lib import demo_gaussian_mixture_diffusion


mus = np.linspace(-1.5, 1.5, 201)
mus = np.stack([mus, -mus], axis=1)
figh = demo_delta_gmm_diffusion(nreps=500, mus=mus, sigma=1E-4, )
