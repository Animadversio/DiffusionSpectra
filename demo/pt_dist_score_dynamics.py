from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage, make_grid_np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
#%%
# plot the iso-contours of the Gaussian distribution in 2d space
# (for the 2d case, the iso-contours are ellipses)
def plot_gaussian_contours(mu, cov, ax, nstd=2, **kwargs):
    """
    Plot iso-contours of a Gaussian distribution in 2d space.
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=mu, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip

#%%
#%%
from diffusers import DDIMPipeline, DDPMPipeline
# model_id = "google/ddpm-cifar10-32"
model_id = "google/ddpm-celebahq-256" # most popular
# model_id = "dimpo/ddpm-mnist"  # most popular
model_id_short = model_id.split("/")[-1]
# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)   # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.unet.requires_grad_(False).eval().to("cuda")  # .half()

#%%
pipe.scheduler.set_timesteps(101)
t_seq = pipe.scheduler.timesteps
alphas = pipe.scheduler.alphas
alphas_cumprod = pipe.scheduler.alphas_cumprod
alphacumprod_seq = alphas_cumprod[t_seq]
alpha_seq = alphas[t_seq]
#%%
# plot the iso-contours of the Gaussian distribution in 2d space
# (for the 2d case, the iso-contours are ellipses)
# vector field of the score function for Gaussian

mu = torch.tensor([0., 0.])
cov = torch.tensor([[5., 0.], [0., 0.5]])
Identity = torch.eye(2)
#%%
xx, yy = torch.meshgrid(torch.linspace(-3, 3, 100), torch.linspace(-3, 3, 100))
pos = torch.stack([xx, yy], dim=-1)
density = torch.distributions.MultivariateNormal(mu, cov).log_prob(pos)
score_vec = - (pos - mu[None, None, ]) @ cov.inverse()
#%%
def get_score_logprob(mu, cov, pos):
    score_vec = - (pos - mu[None, None, ]) @ cov.inverse()
    logprob = torch.distributions.MultivariateNormal(mu, cov).log_prob(pos)
    return score_vec, logprob


#%%
slc = slice(0, 100, 9)
plt.figure(figsize=(6, 5))
plt.contour(xx, yy, density.numpy(), levels=10)
plt.quiver(xx[slc, slc], yy[slc, slc], score_vec[slc, slc, 0], score_vec[slc, slc, 1])
plt.axis('image')
plt.colorbar()
plt.show()
#%%
import os
from os.path import join
anim_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Talk\Score_demo"
os.makedirs(join(anim_dir, "src"), exist_ok=True)
for t_cur in t_seq:
    alpha_t2 = alphas_cumprod[t_cur]
    score_vec, logprob = get_score_logprob(mu, alpha_t2 * cov + (1 - alpha_t2) * Identity, pos)
    slc = slice(0, 100, 9)
    plt.subplots(2, 1, figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.contour(xx, yy, logprob.numpy(), levels=np.arange(-15, 0, 1))
    plt.quiver(xx[slc, slc], yy[slc, slc], score_vec[slc, slc, 0], score_vec[slc, slc, 1],
               scale_units='xy', scale=6)
    plt.axis('image')
    # plt.colorbar()
    plt.subplot(1, 2, 2)
    # plt.plot(t_seq / 1000, alphacumprod_seq)
    plt.plot(np.arange(1000) / 1000, alphas_cumprod)
    plt.axvline(t_cur / 1000, color='r', linestyle='--', alpha=0.5)
    plt.hlines(alpha_t2, 0, t_cur / 1000, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('t')
    plt.ylabel('alpha_t')
    plt.tight_layout()
    plt.savefig(join(anim_dir, "src", f"pt_dist_score_dynamics_{t_cur:03d}_longer.png"))
    plt.show()
    # break

#%%

# make gif from pngs
import imageio
from os.path import join
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib import rc
rc('animation', html='jshtml')
#%%
images = []
for t_cur in t_seq:
    filename = join(anim_dir, "src", f"pt_dist_score_dynamics_{t_cur:03d}.png")
    images.append(imageio.imread(filename))
#%%
images_longer = []
for t_cur in t_seq:
    filename = join(anim_dir, "src", f"pt_dist_score_dynamics_{t_cur:03d}_longer.png")
    images_longer.append(imageio.imread(filename))
#%%
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics.gif"), images, fps=10)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_fast.gif"), images, fps=20)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_slow.gif"), images, fps=5)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_reverse.gif"),  images[::-1], fps=10)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_reverse_fast.gif"), images[::-1], fps=20)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_reverse_slow.gif"), images[::-1], fps=5)
#%% save them as mp4
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics.mp4"), images, fps=10)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_fast.mp4"), images, fps=20)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_slow.mp4"), images, fps=5)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_reverse.mp4"), images[::-1], fps=10)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_reverse_fast.mp4"), images[::-1], fps=20)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_reverse_slow.mp4"), images[::-1], fps=5)
#%% save them as mp4
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_longer.mp4"), images_longer, fps=10)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_longer_fast.mp4"), images_longer, fps=20)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_longer_slow.mp4"), images_longer, fps=5)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_longer_reverse.mp4"), images_longer[::-1], fps=10)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_longer_reverse_fast.mp4"), images_longer[::-1], fps=20)
imageio.mimsave(join(anim_dir, f"pt_dist_score_dynamics_longer_reverse_slow.mp4"), images_longer[::-1], fps=5)


