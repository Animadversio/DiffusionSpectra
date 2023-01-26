
import os
from os.path import join
import json
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler, StableDiffusionPipeline
from diffusers import PNDMPipeline, DDIMScheduler, UNet2DModel, PNDMScheduler
import PIL.Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
#% Utility functions for analysis
from core.diffusion_geometry_lib import proj2subspace, proj2orthospace, subspace_variance, \
    trajectory_geometry_pipeline, latent_PCA_analysis, latent_diff_PCA_analysis, diff_cosine_mat_analysis, PCA_data_visualize
from core.diffusion_traj_analysis_lib import compute_save_diff_imgs_diff, plot_diff_matrix
figdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\TrajGeometryStats"
#%%
def PCA_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir):
    """ Plot the PCA data for a sweep of seeds for a given model
    """
    xt_PC_expvars = []
    deltaxt_PC_expvars = []
    eps_PC_expvars = []
    for seed in tqdm(seed_range):
        savedir = join(exproot, f"seed{seed}")
        # {"expvar": expvar_vec, "U": U, "D": D, "V": V}
        data = torch.load(join(savedir, "latent_PCA.pt"))
        xt_PC_expvars.append(data["expvar"].detach().clone())
        # {"expvar_diff": expvar_diff, "U_diff": U_diff, "D_diff": D_diff, "V_diff": V_diff}
        data = torch.load(join(savedir, "latent_diff_PCA.pt"))
        deltaxt_PC_expvars.append(data["expvar_diff"].detach().clone())
        # {"expvar": expvar_noise, "U": U_noise, "D": D_noise, "V": V_noise}
        data = torch.load(join(savedir, "noise_pred_PCA.pt"))
        eps_PC_expvars.append(data["expvar"].detach().clone())

    xt_PC_expvars = torch.stack(xt_PC_expvars)
    deltaxt_PC_expvars = torch.stack(deltaxt_PC_expvars)
    eps_PC_expvars = torch.stack(eps_PC_expvars)

    outdict = {"xt_PC_expvars": xt_PC_expvars, "deltaxt_PC_expvars": deltaxt_PC_expvars, "eps_PC_expvars": eps_PC_expvars}
    torch.save(
        {"xt_PC_expvars": xt_PC_expvars, "deltaxt_PC_expvars": deltaxt_PC_expvars, "eps_PC_expvars": eps_PC_expvars},
        join(figdir, f"{model_id_short}_traj_PCA_expvars.pt"))

    expvars_df = pd.DataFrame({"xt": xt_PC_expvars.mean(dim=0)[:-1],
                               "deltaxt": deltaxt_PC_expvars.mean(dim=0),
                               "eps": eps_PC_expvars.mean(dim=0), })
    expvars_df.to_csv(join(figdir, f"{model_id_short}_traj_PCA_expvars_mean.csv"))
    plt.figure(figsize=(5, 4))
    plt.plot(xt_PC_expvars.mean(dim=0), label="Latent")
    plt.plot(deltaxt_PC_expvars.mean(dim=0), label="Diff")
    plt.plot(eps_PC_expvars.mean(dim=0), label="Noise")
    plt.legend()
    plt.xlabel("PC index")
    plt.ylabel("Explained variance")
    plt.title(f"Latent, diff, and noise explained variance for\n{model_id_short} seed {seed_range[0]}-{seed_range[-1]}")
    plt.tight_layout()
    saveallforms(figdir, f"{model_id_short}_PC_expvar")
    plt.show()
    return expvars_df, outdict


def Proj2d_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir, alpha_cumprods):
    proj2d_varfrac_tsr = []
    rotate_varfrac_tsr = []
    for seed in tqdm(seed_range):
        savedir = join(exproot, f"seed{seed}")
        data = torch.load(join(savedir, "state_reservoir.pt"))
        latents_traj = data['latents_traj']
        latents_mat = latents_traj.flatten(1)
        latents_traj_proj2d = proj2subspace(latents_mat[[0, -1], :], latents_mat)
        deviation = latents_traj_proj2d - latents_mat
        proj2d_errfrac_trace = (deviation ** 2).sum(1) / (latents_mat ** 2).sum(1)
        rot_traj = latents_mat[-1:] * alpha_cumprods[:, None] + \
                   latents_mat[:1] * (1 - alpha_cumprods).sqrt()[:, None]
        deviation_rot = rot_traj - latents_mat[1:]
        rot_errfrac_trace = (deviation_rot ** 2).sum(1) / (latents_mat[1:] ** 2).sum(1)
        proj2d_varfrac_tsr.append(proj2d_errfrac_trace)
        rotate_varfrac_tsr.append(rot_errfrac_trace)

    proj2d_varfrac_tsr = torch.stack(proj2d_varfrac_tsr)
    rotate_varfrac_tsr = torch.stack(rotate_varfrac_tsr)
    errdict = {"proj2d_varfrac": proj2d_varfrac_tsr, "rotate_varfrac": rotate_varfrac_tsr}
    torch.save(errdict, join(figdir, f"{model_id_short}_traj_2dvarfrac.pt"))
    errfrac_df = pd.DataFrame({"proj2d_varfrac": proj2d_varfrac_tsr.mean(0)[1:],
                               "rotate_varfrac": rotate_varfrac_tsr.mean(0)})
    errfrac_df.to_csv(join(figdir, f"{model_id_short}_traj_2dvarfrac.csv"))
    # %
    plt.figure(figsize=(5, 4))
    plt.plot(proj2d_varfrac_tsr.mean(0), label="Project to x0,xT plane")
    plt.fill_between(range(len(proj2d_varfrac_tsr.mean(0))),
                     proj2d_varfrac_tsr.quantile(0.05, 0),
                     proj2d_varfrac_tsr.quantile(0.95, 0), alpha=0.2)
    plt.plot(rotate_varfrac_tsr.mean(0), label="Rotate")
    plt.fill_between(range(len(rotate_varfrac_tsr.mean(0))),
                     rotate_varfrac_tsr.quantile(0.05, 0),
                     rotate_varfrac_tsr.quantile(0.95, 0), alpha=0.2)
    plt.legend()
    plt.xlabel("Time index")
    plt.ylabel("Error fraction")  # |delta x|^2/|x|^2
    plt.title(f"2d & rotation trajectory approximation error fraction"
              f"\n 2d proj err:{proj2d_varfrac_tsr.mean():.4f}   rot err:{rotate_varfrac_tsr.mean():.4f}"
              f"\n{model_id_short} seed {seed_range[0]}-{seed_range[-1]}")
    plt.tight_layout()
    saveallforms(figdir, f"{model_id_short}_traj_2dvarfrac")
    plt.show()
    return errdict, errfrac_df
#%% CelebA HQ
model_id = "google/ddpm-celebahq-256"  # most popular
model_id_short = model_id.split("/")[-1]
pipe = DDIMPipeline.from_pretrained(model_id)
pipe.scheduler.set_timesteps(51)
alpha_cumprods = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
exproot = r"F:\insilico_exps\Diffusion_traj\ddpm-celebahq-256_scheduler\DDIM"
seed_range = range(125, 300)
Face_expvars_df, Face_PCA_dict = PCA_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir)
Face_errdict, Face_errfrac_df = Proj2d_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir, alpha_cumprods)
#%% church
model_id = "google/ddpm-church-256"  # most popular
model_id_short = model_id.split("/")[-1]
exproot = r"F:\insilico_exps\Diffusion_traj\ddpm-church-256_scheduler\DDIM"
pipe = DDIMPipeline.from_pretrained(model_id)
pipe.scheduler.set_timesteps(51)
alpha_cumprods = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
seed_range = range(125, 150)
Church_expvars_df, Church_PCA_dict = PCA_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir)
Church_errdict, Church_errfrac_df = Proj2d_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir, alpha_cumprods)

#%% MNIST
model_id = "dimpo/ddpm-mnist"  # most popular
model_id_short = model_id.split("/")[-1]
pipe = DDIMPipeline.from_pretrained(model_id)
pipe.scheduler.set_timesteps(51)
alpha_cumprods = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
exproot = r"F:\insilico_exps\Diffusion_traj\ddpm-mnist_scheduler\DDIM"
seed_range = range(200, 400)
MNIST_expvars_df, MNIST_PCA_dict = PCA_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir)
MNIST_errdict, MNIST_errfrac_df = Proj2d_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir, alpha_cumprods)

#%% CIFAR10
model_id = "google/ddpm-cifar10-32"
model_id_short = model_id.split("/")[-1]
pipe = DDIMPipeline.from_pretrained(model_id)
pipe.scheduler.set_timesteps(51)
alpha_cumprods = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
exproot = r"F:\insilico_exps\Diffusion_traj\ddpm-cifar10-32_scheduler\DDIM"
seed_range = range(200, 400)
CIFAR_expvars_df, CIFAR_PCA_dict = PCA_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir)
CIFAR_errdict, CIFAR_errfrac_df = Proj2d_data_sweep_synopsis(exproot, seed_range, model_id_short, figdir, alpha_cumprods)


#%% Stable Diffusion
from core.diffusion_geometry_lib import state_PCA_compute
model_id_short = "StableDiffusion"
exproot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_projection"
seed_range = range(100, 150)
prompt_dir_pair = [
    ("a portrait of an aristocrat", "portrait_aristocrat"),
    ("a portrait of an light bulb", "portrait_lightbulb"),
    ("a large box containing an apple and a toy teddy bear", "box_apple_bear"),
    ("a photo of a cat sitting with a dog on a cozy couch", "cat_dog_couch"),
    ("a CG art of a brain composed of eletronic wires and circuits", "brain_wire_circuits"),
    ("a handsome cat dancing Tango with a female dancer in Monet style", "cat_tango_dancer"),
    ("a bug crawling on a textbook under a bright light, photo", "bug_book_photo"),
]
#%%
#%% Compute the PCA for each prompt
xt_PC_expvars = []
deltaxt_PC_expvars = []
eps_PC_expvars = []
eps_uncond_PC_expvars = []
eps_text_PC_expvars = []
cond_col = []
seed_col = []
for prompt, dirname in prompt_dir_pair:
    for seed in tqdm(seed_range):
        savedir = join(exproot, f"{dirname}-seed{seed}")
        # {"expvar": expvar_vec, "U": U, "D": D, "V": V}
        data = torch.load(join(savedir, "latents_noise_trajs.pt"))
        latents_traj = data["latents_traj"]
        residue_traj = data["residue_traj"]
        noise_uncond_traj = data["noise_uncond_traj"]
        noise_text_traj = data["noise_text_traj"]
        #%%
        latent_PCAdict = state_PCA_compute(latents_traj, savedir, savestr="latent")
        latent_diff_PCAdict = state_PCA_compute(latents_traj[1:] - latents_traj[:-1], savedir, savestr="latent_diff")
        noise_pred_PCAdict = state_PCA_compute(residue_traj, savedir, savestr="noise_pred")
        noise_uncond_PCAdict = state_PCA_compute(noise_uncond_traj, savedir, savestr="noise_uncond")
        noise_text_PCAdict = state_PCA_compute(noise_text_traj, savedir, savestr="noise_text")
        #%%
        xt_PC_expvars.append(latent_PCAdict["expvar"])
        deltaxt_PC_expvars.append(latent_diff_PCAdict["expvar"])
        eps_PC_expvars.append(noise_pred_PCAdict["expvar"])
        eps_uncond_PC_expvars.append(noise_uncond_PCAdict["expvar"])
        eps_text_PC_expvars.append(noise_text_PCAdict["expvar"])
        cond_col.append(prompt)
        seed_col.append(seed)
#%%
xt_PC_expvars = torch.stack(xt_PC_expvars)
deltaxt_PC_expvars = torch.stack(deltaxt_PC_expvars)
eps_PC_expvars = torch.stack(eps_PC_expvars)
eps_uncond_PC_expvars = torch.stack(eps_uncond_PC_expvars)
eps_text_PC_expvars = torch.stack(eps_text_PC_expvars)
#%%
expvar_df = pd.DataFrame({
    "xt": xt_PC_expvars.mean(0)[:-1],
    "deltaxt": deltaxt_PC_expvars.mean(0),
    "eps": eps_PC_expvars.mean(0),
    "eps_uncond": eps_uncond_PC_expvars.mean(0),
    "eps_text": eps_text_PC_expvars.mean(0),
})
expvar_df.to_csv(join(figdir, f"{model_id_short}_traj_PCA_expvars_mean.csv"))
torch.save({"xt_PC_expvars": xt_PC_expvars,
            "deltaxt_PC_expvars": deltaxt_PC_expvars,
            "eps_PC_expvars": eps_PC_expvars,
            "eps_uncond_PC_expvars": eps_uncond_PC_expvars,
            "eps_text_PC_expvars": eps_text_PC_expvars,
            "cond": cond_col,
            "seed": seed_col,},
            join(figdir, f"{model_id_short}_traj_PCA_expvars.pt"))
#%%
data = torch.load(join(figdir, f"{model_id_short}_traj_PCA_expvars.pt"))
xt_PC_expvars = data["xt_PC_expvars"]
deltaxt_PC_expvars = data["deltaxt_PC_expvars"]
eps_PC_expvars = data["eps_PC_expvars"]
eps_uncond_PC_expvars = data["eps_uncond_PC_expvars"]
eps_text_PC_expvars = data["eps_text_PC_expvars"]
cond_col = data["cond"]
#%%
plt.figure(figsize=(5, 4))
plt.plot(xt_PC_expvars.mean(dim=0), label="Latent")
plt.plot(deltaxt_PC_expvars.mean(dim=0), label="Diff")
plt.plot(eps_PC_expvars.mean(dim=0), label="Noise")
plt.plot(eps_uncond_PC_expvars.mean(dim=0), label="Noise uncond", linestyle="--", alpha=0.7)
plt.plot(eps_text_PC_expvars.mean(dim=0), label="Noise text", linestyle="-.", alpha=0.7)
plt.legend()
plt.xlabel("PC index")
plt.ylabel("Explained variance")
plt.title(f"Latent, diff, and noise explained variance for\n{model_id_short} seed {seed_range[0]}-{seed_range[-1]}")
plt.tight_layout()
saveallforms(figdir, f"{model_id_short}_PC_expvar")
plt.show()


#%% 2d trajectory goodness of fit and explained variance
model_id_short = "StableDiffusion"
exproot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_projection"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe.scheduler.set_timesteps(51)
alpha_cumprods = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]

seed_range = range(100, 150)
prompt_dir_pair = [
    ("a portrait of an aristocrat", "portrait_aristocrat"),
    ("a portrait of an light bulb", "portrait_lightbulb"),
    ("a large box containing an apple and a toy teddy bear", "box_apple_bear"),
    ("a photo of a cat sitting with a dog on a cozy couch", "cat_dog_couch"),
    ("a CG art of a brain composed of eletronic wires and circuits", "brain_wire_circuits"),
    ("a handsome cat dancing Tango with a female dancer in Monet style", "cat_tango_dancer"),
    ("a bug crawling on a textbook under a bright light, photo", "bug_book_photo"),
]
proj2d_varfrac_tsr = []
rotate_varfrac_tsr = []
cond_col = []
seed_col = []
for prompt, dirname in prompt_dir_pair:
    for seed in tqdm(seed_range):
        savedir = join(exproot, f"{dirname}-seed{seed}")
        data = torch.load(join(savedir, "latents_noise_trajs.pt"))
        latents_traj = data['latents_traj'].float()
        latents_mat = latents_traj.flatten(1)
        latents_traj_proj2d = proj2subspace(latents_mat[[0, -1], :], latents_mat)
        deviation = latents_traj_proj2d - latents_mat
        proj2d_errfrac_trace = (deviation ** 2).sum(1) / (latents_mat ** 2).sum(1)
        rot_traj = latents_mat[-1:] * alpha_cumprods[:, None] + \
                   latents_mat[:1] * (1 - alpha_cumprods).sqrt()[:, None]
        deviation_rot = rot_traj - latents_mat[1:]
        rot_errfrac_trace = (deviation_rot ** 2).sum(1) / (latents_mat[1:] ** 2).sum(1)
        proj2d_varfrac_tsr.append(proj2d_errfrac_trace)
        rotate_varfrac_tsr.append(rot_errfrac_trace)
        cond_col.append(prompt)
        seed_col.append(seed)

proj2d_varfrac_tsr = torch.stack(proj2d_varfrac_tsr)
rotate_varfrac_tsr = torch.stack(rotate_varfrac_tsr)
errdict = {"proj2d_varfrac": proj2d_varfrac_tsr, "rotate_varfrac": rotate_varfrac_tsr,
           "cond": cond_col, "seed": seed_col}
torch.save(errdict, join(figdir, f"{model_id_short}_traj_2dvarfrac.pt"))
errfrac_df = pd.DataFrame({"proj2d_varfrac": proj2d_varfrac_tsr.mean(0)[1:],
                           "rotate_varfrac": rotate_varfrac_tsr.mean(0)})
errfrac_df.to_csv(join(figdir, f"{model_id_short}_traj_2dvarfrac.csv"))
# %
plt.figure(figsize=(5, 4))
plt.plot(proj2d_varfrac_tsr.mean(0), label="Project to x0,xT plane")
plt.fill_between(range(len(proj2d_varfrac_tsr.mean(0))),
                 proj2d_varfrac_tsr.quantile(0.05, 0),
                 proj2d_varfrac_tsr.quantile(0.95, 0), alpha=0.2)
plt.plot(rotate_varfrac_tsr.mean(0), label="Rotate")
plt.fill_between(range(len(rotate_varfrac_tsr.mean(0))),
                 rotate_varfrac_tsr.quantile(0.05, 0),
                 rotate_varfrac_tsr.quantile(0.95, 0), alpha=0.2)
plt.legend()
plt.xlabel("Time index")
plt.ylabel("Error fraction")  # |delta x|^2/|x|^2
plt.title(f"2d & rotation trajectory approximation error fraction"
          f"\n 2d proj err:{proj2d_varfrac_tsr.mean():.4f}   rot err:{rotate_varfrac_tsr.mean():.4f}"
          f"\n{model_id_short} seed {seed_range[0]}-{seed_range[-1]}")
plt.tight_layout()
saveallforms(figdir, f"{model_id_short}_traj_2dvarfrac")
plt.show()

#%% synopsis summarizing all models
model_id_shorts = [
    "ddpm-mnist",
    "ddpm-cifar10-32",
    "ddpm-church-256",
    "ddpm-celebahq-256",
    "StableDiffusion",
]
for model_id_short in model_id_shorts:
    expvar_df = pd.read_csv(join(figdir, f"{model_id_short}_traj_PCA_expvars_mean.csv"), index_col=0)
    print(expvar_df["xt"][1])
    # print(f"{model_id_short} explained variance: {expvar_df['explained_variance_ratio'].mean():.4f}")
for model_id_short in model_id_shorts:
    expvar_df = pd.read_csv(join(figdir, f"{model_id_short}_traj_PCA_expvars_mean.csv"), index_col=0)
    dim_xt = (expvar_df["xt"] < 0.999).sum() + 1
    dim_deltaxt = (expvar_df["deltaxt"] < 0.999).sum() + 1
    dim_eps = (expvar_df["eps"] < 0.999).sum() + 1
    print(f"{dim_xt}\t{dim_deltaxt}\t{dim_eps}")


#%%
from core.diffusion_geometry_lib import proj2subspace
#%% Scracth zone!!
model_id = "google/ddpm-church-256"  # most popular
model_id_short = model_id.split("/")[-1]
exproot = r"F:\insilico_exps\Diffusion_traj\ddpm-church-256_scheduler\DDIM"
seed_range = range(125, 150)
pipe = DDIMPipeline.from_pretrained(model_id)
pipe.scheduler.set_timesteps(51)
alpha_cumprods = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
#%%

#%%
xt_PC_expvars = []
deltaxt_PC_expvars = []
eps_PC_expvars = []
for seed in tqdm(seed_range):
    savedir = join(exproot, f"seed{seed}")
    # {"expvar": expvar_vec, "U": U, "D": D, "V": V}
    data = torch.load(join(savedir, "latent_PCA.pt"))
    xt_PC_expvars.append(data["expvar"].detach().clone())
    # {"expvar_diff": expvar_diff, "U_diff": U_diff, "D_diff": D_diff, "V_diff": V_diff}
    data = torch.load(join(savedir, "latent_diff_PCA.pt"))
    deltaxt_PC_expvars.append(data["expvar_diff"].detach().clone())
    # {"expvar": expvar_noise, "U": U_noise, "D": D_noise, "V": V_noise}
    data = torch.load(join(savedir, "noise_pred_PCA.pt"))
    eps_PC_expvars.append(data["expvar"].detach().clone())

#%%
xt_PC_expvars = torch.stack(xt_PC_expvars)
deltaxt_PC_expvars = torch.stack(deltaxt_PC_expvars)
eps_PC_expvars = torch.stack(eps_PC_expvars)
#%%
torch.save({"xt_PC_expvars": xt_PC_expvars, "deltaxt_PC_expvars": deltaxt_PC_expvars, "eps_PC_expvars": eps_PC_expvars},
           join(figdir, f"{model_id_short}_traj_PCA_expvars.pt"))

expvars_df = pd.DataFrame({"xt": xt_PC_expvars.mean(dim=0)[:-1],
                           "deltaxt": deltaxt_PC_expvars.mean(dim=0),
                           "eps": eps_PC_expvars.mean(dim=0),})
expvars_df.to_csv(join(figdir, f"{model_id_short}_traj_PCA_expvars_mean.csv"))
#%%
plt.figure(figsize=(5, 4))
plt.plot(xt_PC_expvars.mean(dim=0), label="Latent")
plt.plot(deltaxt_PC_expvars.mean(dim=0), label="Diff")
plt.plot(eps_PC_expvars.mean(dim=0), label="Noise")
plt.legend()
plt.xlabel("PC index")
plt.ylabel("Explained variance")
plt.title(f"Latent, diff, and noise explained variance for\n{model_id_short} seed {seed_range[0]}-{seed_range[-1]}")
plt.tight_layout()
plt.savefig(join(figdir, f"{model_id_short}_PC_expvar.png"), dpi=300)
plt.show()