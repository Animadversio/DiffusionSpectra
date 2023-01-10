import json
import math
import os
from os.path import join
from tqdm import tqdm
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from diffusers import pipelines, StableDiffusionPipeline
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from torchmetrics.functional import pairwise_cosine_similarity
from core.diffusion_traj_analysis_lib import latents_to_image, latentvecs_to_image,\
    denorm_std, denorm_var, denorm_sample_std
from core.diffusion_geometry_lib import avg_cosine_sim_mat, diff_lag
#%%
# exproot = r"/home/binxuwang/insilico_exp/Diffusion_Hessian/StableDiffusion"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
# pipeline.to(torch.half)
def dummy_checker(images, **kwargs): return images, False

pipe.safety_checker = dummy_checker
#%%
latents_reservoir = []
@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())


seed = 45
tsteps = 51
prompt = "a cute and classy mice wearing dress and heels"
# prompt = "a beautiful ballerina in yellow dress under the starry night in Van Gogh style"
# prompt = "a classy ballet flat with a bow on the toe on a wooden floor"
# prompt = "a cat riding a motor cycle in a desert in a bright sunny day"
prompt = "a bowl of soup looks like a portal to another dimension"
out = pipe(prompt, callback=save_latents,
           num_inference_steps=tsteps, generator=torch.cuda.manual_seed(seed))
out.images[0].show()
latents_reservoir = torch.cat(latents_reservoir, dim=0)
#%%

savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion\mice_dress_heels1"
#%%
"""PCA of the trajectory """
latents_mat = latents_reservoir.flatten(1).double()
latents_mat = latents_mat - latents_mat.mean(dim=0)
U, D, V = torch.svd(latents_mat, )
torch.cumsum(D**2 / (D**2).sum(), dim=0)
# (D**2 / (D**2).sum(), )
#%%
projvar = (D[:2]**2).sum() / (D**2).sum()
plt.figure(figsize=(6,6.5))
plt.plot(U[:,0] * D[0], U[:,1] * D[1], "-")
plt.scatter(U[:,0] * D[0], U[:,1] * D[1], c=range(len(U)))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axis("equal")
plt.title(f"Latent state projection onto the first 2 PCs\n{projvar:.2%} of the variance is explained")
saveallforms(savedir, "latent_traj_PC1_PC2_proj", plt.gcf())
plt.show()
#%%
"""PCA of the trajectory """
latents_mat = latents_reservoir.flatten(1).double()
latents_diff_mat = latents_mat[1:] - latents_mat[:-1]
U_diff, D_diff, V_diff = torch.svd(latents_diff_mat, )
torch.cumsum(D_diff**2 / (D_diff**2).sum(), dim=0)
# (D**2 / (D**2).sum(), )
#%%
"""Project the latent steps to different PC planes."""
PCi = 0
PCj = 1
for PCi, PCj  in [(0,1), (0,2), (0,3), (1,2), (1,3), (2, 3)]:
    projvar = (D_diff[[PCi,PCj]]**2).sum() / (D_diff**2).sum()
    plt.figure(figsize=(6,6.5))
    plt.plot(U_diff[:, PCi] * D_diff[PCi], U_diff[:,PCj] * D_diff[PCj], ":k", lw=1)
    plt.scatter(U_diff[:,PCi] * D_diff[PCi], U_diff[:,PCj] * D_diff[PCj], c=range(len(U_diff)))
    plt.xlabel(f"PC{PCi+1}")
    plt.ylabel(f"PC{PCj+1}")
    plt.axis("equal")
    plt.title(f"Latent Step projection onto the 2 PCs (PC{PCi+1} vs PC{PCj+1})"
              f"\n{projvar:.2%} of the variance is explained")
    saveallforms(savedir, f"latent_diff_PC{PCi+1}_PC{PCj+1}_proj", plt.gcf())
    plt.show()
#%%
latents_mat = latents_reservoir.flatten(1).double()
latents_diff_mat = latents_mat[1:] - latents_mat[:-1]
latent_diff_norm = latents_diff_mat.norm(dim=1)
plt.plot(latent_diff_norm)
plt.title("Norm of the latent step")
plt.xlabel("t")
plt.ylabel("norm")
saveallforms(savedir, "latent_diff_norm", plt.gcf())
plt.show()
plt.plot(latents_mat.norm(dim=1))
plt.title("Norm of the latent state")
plt.xlabel("t")
plt.ylabel("norm")
saveallforms(savedir, "latent_state_norm", plt.gcf())
plt.show()
#%%
show_imgrid(latentvecs_to_image(100*V[:,0:4].T,pipe))
#%%
""" Geometry of latent space evolution in the subspace spanned by the initial state and final states 
un orthogonal basis
"""
init_latent = latents_reservoir[:1].flatten(1).float()
end_latent = latents_reservoir[-1:].flatten(1).float()
init_end_cosine = pairwise_cosine_similarity(init_latent, end_latent).item()
proj_coef_init = torch.matmul(latents_reservoir.flatten(1).float(), init_latent.T) / torch.norm(init_latent)**2
proj_coef_end = torch.matmul(latents_reservoir.flatten(1).float(), end_latent.T) / torch.norm(end_latent)**2
plt.figure()
plt.plot(proj_coef_init, label="with init z_0")
plt.plot(proj_coef_end, label="with end z_T")
plt.axhline(0, color="k", linestyle="--")
plt.title(f"projection of latent states diff z_t with z_T\n init end cosine={init_end_cosine:.3f}")
plt.xlabel("t")
plt.ylabel("projection")
plt.legend()
saveallforms(savedir, "latent_trajectory_projcoef", plt.gcf())
plt.show()
#%
plt.figure()
plt.scatter(proj_coef_init, proj_coef_end, label="latent trajectory")
plt.axhline(0, color="k", linestyle="--")
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("projection with z_0")
plt.ylabel("projection with z_T")
plt.title(f"latent trajectory in 2d proejction space\n(in a non orthogonal basis. init end cosine={init_end_cosine:.3f})")
saveallforms(savedir, "latent_trajectory_2d_projection_nonortho", plt.gcf())
plt.show()
#%%

#%% dev zone
"""cosine similarity matrix of the latent state diffs"""
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
"""cosine similarity of the latent state difference and init / end"""
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
"""cosine similarity of the latent state and init / end"""
cosvec_end = pairwise_cosine_similarity(latents_reservoir.flatten(1).float(),
                                        latents_reservoir[-1:].flatten(1).float())
cosvec_init = pairwise_cosine_similarity(latents_reservoir.flatten(1).float(),
                                    latents_reservoir[:1].flatten(1).float())
figh = plt.figure()
plt.plot(cosvec_end, label="with end z_T")
plt.plot(cosvec_init, label="with init z_0")
plt.axhline(0, color="k", linestyle="--")
plt.title(f"cosine similarity of latent states z_t with z_0, z_T")
plt.xlabel("t")
plt.ylabel("cosine similarity")
plt.legend()
saveallforms(savedir, f"cosine_trace_w_init_end_latent", figh)
plt.show()

#%% dev zone
cosmat, cosmat_avg = avg_cosine_sim_mat(latents_reservoir.flatten(1).float())
plt.figure()
sns.heatmap(cosmat, cmap="coolwarm", vmin=-1, vmax=1)
plt.title(f"cosine similarity matrix of latent states, avg={cosmat_avg:.3f}")
plt.show()


#%% dev zone
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




#%% dev zone
"""plot latent space trajectory on the 2d plane spanned by the initial and final states"""

init_end_cosine = pairwise_cosine_similarity(init_latent, end_latent).item()
init_end_angle = math.acos(init_end_cosine)
init_end_ang_deg = init_end_angle / math.pi * 180
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
plt.title(f"latent trajectory in 2d projection space (z0,zT)\ninit end cosine={init_end_cosine:.3f} angle={init_end_ang_deg:.1f} deg")
saveallforms(savedir, f"latent_trajectory_2d_proj", plt.gcf())
plt.show()
#%
"""The geometry of the differences"""
plt.figure()
plt.scatter(proj_coef1[1:]-proj_coef1[:-1], proj_coef2[1:]-proj_coef2[:-1], c=range(50), label="latent diff")
plt.plot(proj_coef1[1:]-proj_coef1[:-1], proj_coef2[1:]-proj_coef2[:-1], color="k", alpha=0.5)
plt.axhline(0, color="k", linestyle="--", lw=0.5)
plt.axvline(0, color="k", linestyle="--", lw=0.5)
plt.axline((0, 0), slope=proj_coef2[0].item() / proj_coef1[0].item(),
           color="r", linestyle="--", label="init noise direction")
plt.axline((0, 0), slope=0,
           color="g", linestyle="--", label="final latent direction")
plt.legend()
plt.axis("equal")
plt.xlabel("projection with z_T")
plt.ylabel("projection with ortho part of z_0")
plt.title(f"latent diff (z_t+1 - z_t) in 2d projection space (z0,zT)\ninit end cosine={init_end_cosine:.3f} angle={init_end_ang_deg:.1f} deg")
saveallforms(savedir, f"latent_diff_2d_proj", plt.gcf())
plt.show()
#%
"""There is little variance outside the subspace spanned by the initial and final states"""
plt.figure()
plt.plot(residue_frac)
plt.title("fraction of residue ortho to the 2d subspace spanned by z_0 and z_T")
plt.xlabel("t")
plt.ylabel("fraction of var")
saveallforms(savedir, f"latent_trajectory_2d_proj_residue_trace", plt.gcf())
plt.show()
#%
"""There is little variance of vector norm"""
plt.figure()
plt.plot(latents_reservoir.flatten(1).float().norm(dim=1))
plt.title("Norm of latent states")
plt.xlabel("t")
plt.ylabel("L2 norm")
saveallforms(savedir, f"latent_trajectory_norm_trace", plt.gcf())
plt.show()
#%%
"""Plot the 2d cycle of the latent states plane of trajectory"""
imgtsrs = []
for phi in np.linspace(0, 360, 36, endpoint=False):
    phi = phi * np.pi / 180
    imgtsr = latentvecs_to_image((unitbasis2 * math.sin(phi) + unitbasis1 * math.cos(phi)) * end_latent.norm(), pipe)
    imgtsrs.append(imgtsr)
imgtsrs = torch.cat(imgtsrs, dim=0)
show_imgrid(imgtsrs, nrow=9)
save_imgrid(imgtsrs, join(savedir, "latent_2d_cycle_visualization.png"), nrow=9)



#%% dev zone
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
phi = 180 * np.pi / 180
show_imgrid(latents_to_image(((unitbasis2 * math.sin(phi) + unitbasis1 * math.cos(phi)) * end_latent.norm())\
                             .reshape(latents_reservoir[-1:].shape).half(), pipe))

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




