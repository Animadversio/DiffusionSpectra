import json
import os
from os.path import join
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from diffusers import pipelines, StableDiffusionPipeline
# exproot = r"/home/binxuwang/insilico_exp/Diffusion_Hessian/StableDiffusion"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
#%%
# pipe = pipeline
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
# pipeline.to(torch.half)
#%%
# with torch.autocast("cuda"):
latents_reservoir = []
@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())


out = pipe("a cute and classy mice wearing dress and heels", callback=save_latents,
           num_inference_steps=51)
out.images[0].show()
latents_reservoir = torch.cat(latents_reservoir, dim=0)
#%%
def denorm_std(x):
    return ((x - x.mean()) / x.std() * 0.4 + 1) / 2


def denorm_sample_std(x):
    return ((x - x.mean(dim=0, keepdims=True)) / x.std(dim=0, keepdims=True) * 0.4 + 1) / 2


def denorm_var(x, mu, std):
    return (x - x.mean()) / x.std() * std + mu


def latents_to_image(latents, pipe):
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents.to(pipe.vae.device)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image.cpu()
#%%
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid

show_imgrid(denorm_std((latents_reservoir[0] - latents_reservoir[10]).unsqueeze(1)), nrow=2,)
#%%
show_imgrid(denorm_std((latents_reservoir[50] - latents_reservoir[25]).unsqueeze(1)), nrow=2,)
#%%
show_imgrid(denorm_std((latents_reservoir[7] - latents_reservoir[2]).unsqueeze(1)), nrow=2,)

#%%
# img_interim = pipe.vae.decode((latents_reservoir[7] - latents_reservoir[2]).unsqueeze(0).cuda()).sample.cpu()
img_interim = latents_to_image(2*(latents_reservoir[7] - latents_reservoir[2])[None, :], pipe)
show_imgrid(img_interim, nrow=1,)
#%%
img_interim = latents_to_image(20.0*(latents_reservoir[5] - latents_reservoir[0])[None, :], pipe)
show_imgrid(img_interim, nrow=1, )
#%%
img_interim = latents_to_image(10.0*(latents_reservoir[50] - latents_reservoir[20])[None, :], pipe)
show_imgrid(img_interim, nrow=1, )
#%%
savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion\mice_dress_heels1"
os.makedirs(savedir, exist_ok=True)
#%% record the inter mediate images
mean_fin = latents_reservoir[-1].mean(dim=0)
std_fin = latents_reservoir[-1].std(dim=0)
for i in range(0, 51, 5):
    latent = latents_reservoir[i]
    save_imgrid(denorm_std(latent[:, None, :, :]),
                    join(savedir, f"diffusion_step_{i:02d}_latent_stdnorm.png"), nrow=2)
    img = latents_to_image(latent[None, :], pipe)
    save_imgrid(img, join(savedir, f"diffusion_step_{i:02d}_vae_decode.png"))
    for j in range(0, 51, 5):
        latent_diff = latents_reservoir[j] - latents_reservoir[i]
        save_imgrid(denorm_std(latent_diff[:, None, :, :]),
                    join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_latent_stdnorm.png"), nrow=2, )
        img_interim = latents_to_image(latent_diff[None, :], pipe)
        save_imgrid(img_interim, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode.png"), nrow=1, )
        img_interim_denorm = latents_to_image(denorm_var(latent_diff[None, :], mean_fin, std_fin), pipe)
        save_imgrid(img_interim_denorm, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode_stdfinal.png"), nrow=1, )

#%%
def compute_save_diff_imgs(savedir, step_list, latents_reservoir):
    for i in step_list:
        latent = latents_reservoir[i]
        save_imgrid(denorm_std(latent[:, None, :, :]),
                        join(savedir, f"diffusion_step_{i:02d}_latent_stdnorm.png"), nrow=2)
        img = latents_to_image(latent[None, :], pipe)
        save_imgrid(img, join(savedir, f"diffusion_step_{i:02d}_vae_decode.png"))
        for j in step_list:
            latent_diff = latents_reservoir[j] - latents_reservoir[i]
            save_imgrid(denorm_std(latent_diff[:, None, :, :]),
                        join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_latent_stdnorm.png"), nrow=2, )
            img_interim = latents_to_image(latent_diff[None, :], pipe)
            save_imgrid(img_interim, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode.png"), nrow=1, )
            img_interim_denorm = latents_to_image(denorm_var(latent_diff[None, :], mean_fin, std_fin), pipe)
            save_imgrid(img_interim_denorm, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode_stdfinal.png"), nrow=1, )


compute_save_diff_imgs(savedir, range(0, 21, 2), latents_reservoir)
#%%
compute_save_diff_imgs(savedir, range(0, 16, 1), latents_reservoir)
#%%
torch.save(latents_reservoir, join(savedir, "latents_reservoir.pt"))
# json.dump({"prompt": text})
#%%
def plot_diff_matrix(savedir, step_list, diff_x_sfx="", step_x_sfx="", save_sfx="", tril=True):
    """

    :param savedir: directory to load the components and to save the figures
    :param step_list: the time steps to fetch the existing data
    :param diff_x_sfx: the suffix of the difference image file name
    :param step_x_sfx: the suffix of the step image file name
    :param save_sfx:    the suffix of the saved figure file name
    :param tril:   whether to plot the lower triangle of the matrix
    :return:
    """
    nrow = len(step_list)
    figh, axs = plt.subplots(nrow, nrow, figsize=(2 * nrow, 2 * nrow + 2))
    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")

    for i, stepi in enumerate(step_list):
        for j, stepj in enumerate(step_list):
            if i + 1 == nrow:
                continue
            if i >= j and tril:
                continue
            axs[i + 1, j].imshow(plt.imread(join(savedir,
                                                 f"diffusion_traj_{stepj:02d}-{stepi:02d}{diff_x_sfx}.png")))  # make_grid(denorm(sample_traj[j] - sample_traj[i]),).permute(1, 2, 0)
            axs[i + 1, j].set_title(f"{stepj}-{stepi}")

    for i, stepi in enumerate(step_list):
        axs[0, i, ].imshow(plt.imread(join(savedir, f"diffusion_step_{stepi:02d}{step_x_sfx}.png")))
        axs[0, i, ].set_title(f"t={stepi}")

    plt.suptitle(f"x difference along Trajectories diff={diff_x_sfx}, step={step_x_sfx}", fontsize=16)
    plt.tight_layout()
    saveallforms(savedir, f"diffusion_traj_diff_mtg{save_sfx}", figh)
    plt.show()
    return figh


plot_diff_matrix(savedir, range(0, 51, 5),
                 diff_x_sfx="_vae_decode", step_x_sfx="_vae_decode", save_sfx="_vae_decode")
plot_diff_matrix(savedir, range(0, 51, 5),
                 diff_x_sfx="_vae_decode_stdfinal", step_x_sfx="_vae_decode", save_sfx="_vae_decode_stdfinal")
plot_diff_matrix(savedir, range(0, 51, 5),
                 diff_x_sfx="_latent_stdnorm", step_x_sfx="_latent_stdnorm", save_sfx="_latent_stdnorm")


#%%
plot_diff_matrix(savedir, range(0, 21, 2),
                 diff_x_sfx="_vae_decode", step_x_sfx="_vae_decode", save_sfx="_vae_decode_early0-20")
plot_diff_matrix(savedir, range(0, 21, 2),
                 diff_x_sfx="_vae_decode_stdfinal", step_x_sfx="_vae_decode", save_sfx="_vae_decode_stdfinal_early0-20")
plot_diff_matrix(savedir, range(0, 21, 2),
                 diff_x_sfx="_latent_stdnorm", step_x_sfx="_latent_stdnorm", save_sfx="_latent_stdnorm_early0-20")
#%%
plot_diff_matrix(savedir, range(0, 16, 1),
                 diff_x_sfx="_vae_decode", step_x_sfx="_vae_decode", save_sfx="_vae_decode_early0-15")
plot_diff_matrix(savedir, range(0, 16, 1),
                 diff_x_sfx="_vae_decode_stdfinal", step_x_sfx="_vae_decode", save_sfx="_vae_decode_stdfinal_early0-15")
plot_diff_matrix(savedir, range(0, 16, 1),
                 diff_x_sfx="_latent_stdnorm", step_x_sfx="_latent_stdnorm", save_sfx="_latent_stdnorm_early0-15")

#%%
""" Correlogram of the latent state difference """
import seaborn as sns
from torchmetrics.functional import pairwise_cosine_similarity
def diff_lag(x, lag=1, ):
    assert lag >= 1
    return x[lag:] - x[:-lag]


def avg_cosine_sim_mat(X):
    cosmat = pairwise_cosine_similarity(X,)
    idxs = torch.tril_indices(cosmat.shape[0], cosmat.shape[1], offset=-1)
    cosmat_vec = cosmat[idxs[0], idxs[1]]
    return cosmat, cosmat_vec.mean()


cosmat, cosmat_avg = avg_cosine_sim_mat(latents_reservoir.flatten(1).float())
plt.figure()
sns.heatmap(cosmat, cmap="coolwarm", vmin=-1, vmax=1)
plt.title(f"cosine similarity matrix of latent states, avg={cosmat_avg:.3f}")
plt.show()
#%%
cosmat, cosmat_avg = avg_cosine_sim_mat(latents_reservoir.flatten(1).float())
#%%
for lag in [1, 2, 3, 4, 5, 10]:
    cosmat, cosmat_avg = avg_cosine_sim_mat(diff_lag(latents_reservoir, lag).flatten(1).float())
    figh = plt.figure(figsize=(7, 6))
    sns.heatmap(cosmat, cmap="coolwarm", vmin=-1, vmax=1)
    plt.axis("image")
    plt.title(f"cosine similarity matrix of latent states diff z_t+{lag} - z_t\n avg cosine={cosmat_avg:.3f} lag={lag}")
    plt.xlabel("t1")
    plt.ylabel("t2")
    saveallforms(savedir, f"cosine_mat_latent_diff_lag{lag}", figh)
    plt.show()
#%%
for lag in [1, 2, 3, 4, 5, 10]:
    cosvec_end = pairwise_cosine_similarity(diff_lag(latents_reservoir, lag).flatten(1).float(),
                                        latents_reservoir[-1:].flatten(1).float())
    cosvec_init = pairwise_cosine_similarity(diff_lag(latents_reservoir, lag).flatten(1).float(),
                                        latents_reservoir[:1].flatten(1).float())
    figh = plt.figure()
    plt.plot(cosvec_end, label="with end z_T")
    plt.plot(cosvec_init, label="with init z_0")
    plt.axhline(0, color="k", linestyle="--")
    plt.title(f"cosine similarity of latent states diff z_t+{lag} - z_t with z_0, z_T")
    plt.xlabel("t")
    plt.ylabel("cosine similarity")
    plt.legend()
    saveallforms(savedir, f"cosine_trace_w_init_end_latent_diff_lag{lag}", figh)
    plt.show()
#%%
""" Geometry of latent space evolution in the subspace spanned by the initial state and final states """
init_latent = latents_reservoir[:1].flatten(1).float()
end_latent = latents_reservoir[-1:].flatten(1).float()
proj_coef_init = torch.matmul(latents_reservoir.flatten(1).float(), init_latent.T) / torch.norm(init_latent)**2
proj_coef_end = torch.matmul(latents_reservoir.flatten(1).float(), end_latent.T) / torch.norm(end_latent)**2
plt.figure()
plt.plot(proj_coef_init, label="with init z_0")
plt.plot(proj_coef_end, label="with end z_T")
plt.axhline(0, color="k", linestyle="--")
plt.title(f"projection of latent states diff z_t+{lag} - z_t with z_T")
plt.xlabel("t")
plt.ylabel("projection")
plt.legend()
plt.show()
#%
plt.figure()
plt.scatter(proj_coef_init, proj_coef_end, label="latent trajectory")
plt.axhline(0, color="k", linestyle="--")
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("projection with z_0")
plt.ylabel("projection with z_T")
plt.title("latent trajectory in 2d proejction space")
plt.show()
#%%
(proj_coef_init @ init_latent + proj_coef_end @ end_latent).norm(dim=1, keepdim=True) ** 2 / \
    latents_reservoir.flatten(1).float().norm(dim=1, keepdim=True)**2
#%%
def proj2subspace(A,b):
    return (A.T@torch.linalg.inv(A@A.T)@A@b.T).T


def proj2orthospace(A, b):
    return b - (A.T@torch.linalg.inv(A@A.T)@A@b.T).T


# proj2subspace(init_latent, latents_reservoir.flatten(1).float())
latents_proj = proj2subspace(torch.cat((init_latent, end_latent)),
              latents_reservoir.flatten(1).float())
latents_proj_out = proj2orthospace(torch.cat((init_latent, end_latent)),
              latents_reservoir.flatten(1).float())
latents_proj_out_noise = proj2orthospace(init_latent,
              latents_reservoir.flatten(1).float())
#%%
latents_proj.norm(dim=1) ** 2 / latents_reservoir.flatten(1).float().norm(dim=1) ** 2
#%%
latents_proj_out.norm(dim=1) ** 2 / latents_reservoir.flatten(1).float().norm(dim=1) ** 2
#%%
proj_img = latents_to_image(latents_proj.reshape(latents_reservoir.shape).half()[5:6], pipe)
show_imgrid(proj_img)
#%%
proj_img = latents_to_image(30*latents_proj_out_noise.reshape(latents_reservoir.shape).half()[2:3], pipe)
show_imgrid(proj_img)
#%%
"""plot latent space trajectory on the 2d plane spanned by the initial and final states"""

unitbasis1 = end_latent / end_latent.norm()  # end state
unitbasis2 = proj2orthospace(end_latent, init_latent)  # init noise that is ortho to the end state
unitbasis2 = unitbasis2 / unitbasis2.norm()  # unit normalize
proj_coef1 = torch.matmul(latents_reservoir.flatten(1).float(), unitbasis1.T)
proj_coef2 = torch.matmul(latents_reservoir.flatten(1).float(), unitbasis2.T)
residue = latents_reservoir.flatten(1).float() - (proj_coef1 @ unitbasis1 + proj_coef2 @ unitbasis2)
residue_frac = residue.norm(dim=1) ** 2 / latents_reservoir.flatten(1).float().norm(dim=1) ** 2

plt.figure()
plt.plot([0, proj_coef1[0].item()], [0, proj_coef2[0].item()], label="noise init", color="r")
plt.plot([0, proj_coef1[-1].item()], [0, proj_coef2[-1].item()], label="final latent", color="g")
plt.scatter(proj_coef1, proj_coef2, label="latent trajectory")
plt.axhline(0, color="k", linestyle="--", lw=0.5)
plt.axvline(0, color="k", linestyle="--", lw=0.5)
plt.legend()
plt.xlabel("projection with z_T")
plt.ylabel("projection with ortho part of z_0")
plt.title("latent trajectory in 2d projection space (z0,zT)")
saveallforms(savedir, f"latent_trajectory_2d_proj", plt.gcf())
plt.show()
#%%
"""There is little variance outside the subspace spanned by the initial and final states"""
plt.figure()
plt.plot(residue_frac)
plt.title("fraction of residue ortho to the 2d subspace spanned by z_0 and z_T")
plt.xlabel("t")
plt.ylabel("fraction of var")
saveallforms(savedir, f"latent_trajectory_2d_proj_residue_trace", plt.gcf())
plt.show()
#%%
"""The geometry of the differences"""
plt.figure()
plt.scatter(proj_coef1[1:]-proj_coef1[:-1], proj_coef2[1:]-proj_coef2[:-1], c=range(50), label="latent diff")
plt.plot(proj_coef1[1:]-proj_coef1[:-1], proj_coef2[1:]-proj_coef2[:-1], color="k", alpha=0.5)
plt.axhline(0, color="k", linestyle="--", lw=0.5)
plt.axvline(0, color="k", linestyle="--", lw=0.5)
plt.axline((0, 0), slope=proj_coef2[0].item() / proj_coef1[0].item(),
           color="r", linestyle="--", label="init noise direction")
plt.legend()
plt.axis("equal")
plt.xlabel("projection with z_T")
plt.ylabel("projection with ortho part of z_0")
plt.title("latent diff (z_t+1 - z_t) in 2d projection space (z0,zT)")
saveallforms(savedir, f"latent_diff_2d_proj", plt.gcf())
plt.show()

#%% dev zone
diff_x_sfx = "_latent_stdnorm"
# diff_x_sfx = "_vae_decode"
step_x_sfx = "_latent_stdnorm"

figh, axs = plt.subplots(11, 11, figsize=(20, 21))
for i, ax in enumerate(axs.flatten()):
    ax.axis("off")

for i, stepi in enumerate([*range(0, 51, 5)]):
    for j, stepj in enumerate([*range(0, 51, 5)]):
        if i >= j:
            continue
        axs[i + 1, j].imshow(plt.imread(join(savedir, f"diffusion_traj_{stepj:02d}-{stepi:02d}{diff_x_sfx}.png"))) # make_grid(denorm(sample_traj[j] - sample_traj[i]),).permute(1, 2, 0)
        axs[i + 1, j].set_title(f"{stepj}-{stepi}")

for i, stepi in enumerate([*range(0, 51, 5)]):
    axs[0, i, ].imshow(plt.imread(join(savedir, f"diffusion_step_{stepi:02d}{step_x_sfx}.png")))
    axs[0, i, ].set_title(f"t={stepi}")

plt.suptitle("x difference along Trajectories", fontsize=16)
plt.tight_layout()
saveallforms(savedir, f"diffusion_traj_diff_mtg{diff_x_sfx}", figh)
plt.show()



