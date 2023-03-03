
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from diffusers import DDIMPipeline, DDPMPipeline
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage, make_grid_np
#%%
def gauss_function(XX, YY, center, precision):
    XXc = XX - center[0]
    YYc = YY - center[1]
    ZZ = np.exp(- 0.5*(XXc**2 * precision[0, 0] +
                       YYc**2 * precision[1, 1] +
                2 * XXc * YYc * precision[0, 1]))
    return ZZ


XX, YY = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
fun1 = gauss_function(XX, YY, center=[0.5, 0.8],
                  precision=np.array([[2.5, 1], [1, 6]]))
fun2 = gauss_function(XX, YY, center=[-0.5, -0.3],
                  precision=np.array([[3, 1.5], [1.5, 2]]))
fun3 = gauss_function(XX, YY, center=[0.5, -0.3],
                  precision=np.array([[0.2, 0], [0, 0.5]]))
ZZ = 0.6 * fun1 + fun2 + 0.04405 * fun3
#%%
from core.gaussian_mixture_lib import diffuse_gmm_torch, diffuse_gmm, \
    GaussianMixture, GaussianMixture_torch
mus = [np.array([0.6, 0.8]),
       np.array([-0.2, -0.3]),
       # np.array([0.5, -0.3])
       ]
covs = [np.linalg.inv(np.array([[2.5, 1], [1, 6]])),
        np.linalg.inv(np.array([[2, 1.5], [1.5, 3]])),
        # np.linalg.inv(np.array([[0.2, 0], [0, 0.5]])),
        ]
weights = [0.5, 1.0, ]
XX, YY = np.meshgrid(np.linspace(-2.5, 2.5, 100), np.linspace(-2.5, 2.5, 100))
gmm = GaussianMixture(mus, covs, weights=weights)
prob, logprob, score_vecs = gmm.score_grid([XX, YY])
# pnts = np.stack([XX.flatten(), YY.flatten()], axis=1)
# score_vecs = gmm.score(pnts)
# prob = gmm.pdf(pnts)
# logprob = np.log(prob)
# prob = prob.reshape(XX.shape)
# logprob = logprob.reshape(XX.shape)
# score_vecs = score_vecs.reshape((*XX.shape,-1))
#%
slc = slice(None, None, 5)
plt.contour(XX, YY, prob, levels=np.arange(0, 0.3, 0.025))
# plt.contour(XX, YY, logprob, levels=np.arange(-15, -1, 1))
plt.quiver(XX[slc, slc], YY[slc, slc],
           score_vecs[slc, slc, 0],
           score_vecs[slc, slc, 1])
plt.axis('image')
plt.show()

#%%
def morph_gmm(gmm, alpha_t):
    noise_cov = np.eye(gmm.dim) * (1 - alpha_t**2)
    mus_dif = [mu * alpha_t for mu in gmm.mus]
    covs_dif = [alpha_t**2 * cov + noise_cov for cov in gmm.covs]
    return GaussianMixture(mus_dif, covs_dif, gmm.weights)


#%%
from diffusers import DDIMPipeline, DDPMPipeline
# model_id = "google/ddpm-cifar10-32"
model_id = "google/ddpm-celebahq-256" # most popular
# model_id = "dimpo/ddpm-mnist"  # most popular
model_id_short = model_id.split("/")[-1]
# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)   # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
# pipe.unet.requires_grad_(False).eval().to("cuda")  # .half()
pipe.scheduler.set_timesteps(101)
t_seq = pipe.scheduler.timesteps
alphas = pipe.scheduler.alphas
alphas_cumprod = pipe.scheduler.alphas_cumprod
alphacumprod_seq = alphas_cumprod[t_seq]
alpha_seq = alphas[t_seq]
betas = pipe.scheduler.betas

#%%
import os
from os.path import join
anim_dir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Talk\Score_gmm_demo"
os.makedirs(join(anim_dir, "src"), exist_ok=True)
gmm = GaussianMixture(mus, covs, weights=weights)
for t, alpha_t2 in zip(t_seq, alphacumprod_seq):
    gmm_t = morph_gmm(gmm, np.sqrt(alpha_t2.numpy()))
    prob, logprob, score_vecs = gmm_t.score_grid([XX, YY])
    slc = slice(None, None, 5)
    plt.figure(figsize=[5, 5])
    plt.contour(XX, YY, prob, levels=np.arange(0, 0.3, 0.025))
    # plt.contour(XX, YY, logprob, levels=np.arange(-15, -1, 1))
    plt.quiver(XX[slc, slc], YY[slc, slc],
               score_vecs[slc, slc, 0],
               score_vecs[slc, slc, 1],
               scale_units='xy', scale=12)
    plt.axis('image')
    plt.tight_layout()
    plt.savefig(join(anim_dir, "src", f"gmm_prob_score_dynamics_{t:03d}.png"))
    plt.show()
    plt.figure(figsize=[5, 5])
    # plt.imshow(prob, )
    # sns.heatmap(XX, YY, prob)
    plt.contour(XX, YY, prob, levels=np.arange(0, 0.3, 0.025))
    plt.axis('image')
    plt.tight_layout()
    plt.savefig(join(anim_dir, "src", f"gmm_prob_heatmap_dynamics_{t:03d}.png"))
    plt.show()
    # raise Exception
#%%
import imageio
images = []
for t_cur in t_seq:
    filename = join(anim_dir, "src", f"gmm_prob_score_dynamics_{t_cur:03d}.png")
    images.append(imageio.imread(filename))

images_hm = []
for t_cur in t_seq:
    filename = join(anim_dir, "src", f"gmm_prob_heatmap_dynamics_{t_cur:03d}.png")
    images_hm.append(imageio.imread(filename))
#%%
for fps, fps_lab in zip([10, 20, 5], ["", "_fast", "_slow"]):
    for stepdir, dir_lab in zip([1, -1], ["", "_reverse"]):
    # for stepdir, dir_lab in zip([-1], ["_reverse"]):
        for sfx in ["gif", "mp4"]:
            imageio.mimsave(join(anim_dir, f"gmm_prob_score_dynamics{dir_lab}{fps_lab}.{sfx}"), images[::stepdir], fps=fps)
#%%
for fps, fps_lab in zip([10, 20, 5], ["", "_fast", "_slow"]):
    for stepdir, dir_lab in zip([1, -1], ["", "_reverse"]):
    # for stepdir, dir_lab in zip([-1], ["_reverse"]):
        for sfx in ["gif", "mp4"]:
            imageio.mimsave(join(anim_dir, f"gmm_prob_heatmap_dynamics{dir_lab}{fps_lab}.{sfx}"), images_hm[::stepdir], fps=fps)