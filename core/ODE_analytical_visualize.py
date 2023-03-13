import os

import matplotlib.pyplot as plt
import torch
from os.path import join
from datasets import load_dataset
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from diffusers import DDIMPipeline
figdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\Theory"
#%%
# traj_dir = r"F:\insilico_exps\Diffusion_traj\ddpm-cifar10-32_scheduler\DDIM"
model_id = "google/ddpm-cifar10-32"
# model_id = "dimpo/ddpm-mnist"  # most popular
model_id_short = model_id.split("/")[-1]
# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)
#%%
pipe.scheduler.set_timesteps(51)
t_traj = pipe.scheduler.timesteps
alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
#%%
plt.figure(figsize=(9,3.5))
plt.subplot(1,3,1)
plt.plot(t_traj, pipe.scheduler.alphas_cumprod[t_traj])
plt.title("alphacum(t)")
plt.xlabel("t")
plt.subplot(1,3,2)
plt.plot(t_traj, pipe.scheduler.alphas[t_traj])
plt.title("alpha(t)")
plt.xlabel("t")
plt.subplot(1,3,3)
plt.plot(t_traj, pipe.scheduler.betas[t_traj])
plt.title("beta(t)")
plt.xlabel("t")
plt.suptitle(f"DDPM-{model_id_short} alpha-beta schedule")
plt.tight_layout()
saveallforms(figdir, f"DDPM-{model_id_short}_alpha-beta_schedule")
plt.show()
#%%
figdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\AlphaSchedule"
os.makedirs(figdir, exist_ok=True)
def plot_alpha_beta_schedule(pipe, Nsteps, figdir, model_id_short):
    pipe.scheduler.set_timesteps(Nsteps)
    t_traj = pipe.scheduler.timesteps
    fig = plt.figure(figsize=(12.5, 3.5))
    plt.subplot(1, 4, 1)
    plt.plot(pipe.scheduler.alphas_cumprod[t_traj], label="alpha2")
    plt.plot(1-pipe.scheduler.alphas_cumprod[t_traj], label="sigma2")
    plt.title("alphacum(t)")
    plt.xlabel("t")
    plt.subplot(1, 4, 2)
    plt.semilogy((1-pipe.scheduler.alphas_cumprod[t_traj]) /
             pipe.scheduler.alphas_cumprod[t_traj],)
    plt.title("sigma2(t)/alpha2(t)")
    plt.xlabel("t")
    plt.subplot(1, 4, 3)
    plt.plot(pipe.scheduler.alphas[t_traj])
    plt.title("alpha(t)")
    plt.xlabel("t")
    plt.subplot(1, 4, 4)
    plt.plot(pipe.scheduler.betas[t_traj])
    plt.title("beta(t)")
    plt.xlabel("t")
    plt.suptitle(f"DDPM-{model_id_short} alpha-beta schedule")
    plt.tight_layout()
    saveallforms(figdir, f"DDPM-{model_id_short}_alpha-beta_schedule")
    plt.show()
    return fig

# model_id = "google/ddpm-bedroom-256"
# model_id = "google/ddpm-ema-bedroom-256"
# model_id = "google/ddpm-church-256"
# model_id = "google/ddpm-ema-church-256"
# model_id = "google/ddpm-ema-celebahq-256"
# model_id = "google/ddpm-celebahq-256" # most popular
model_id = "google/ddpm-cifar10-32"
# model_id = "dimpo/ddpm-mnist"  # most popular
model_id = "google/ddpm-cifar10-32"
for model_id in ["dimpo/ddpm-mnist",
                 "google/ddpm-cifar10-32",
                 "google/ddpm-celebahq-256",
                 "google/ddpm-church-256",]:
    model_id_short = model_id.split("/")[-1]
    # load model and scheduler
    pipe = DDIMPipeline.from_pretrained(model_id)
    plot_alpha_beta_schedule(pipe, 51, figdir, model_id_short)
#%%
def xtproj_coef(Lambda, alphacum_traj):
    """ Projection coefficient for xt on eigenvector u_k of value Lambda """
    if type(Lambda) is not torch.Tensor:
        Lambda = torch.tensor(Lambda).float()
    coef_traj = ((1 + (Lambda - 1) * alphacum_traj) /
                 (1 + (Lambda - 1) * alphacum_traj[0])).sqrt()
    return coef_traj


def x0hat_proj_coef(Lambda, alphacum_traj):
    """ Projection coefficient for x0hat on eigenvector of value Lambda """
    if type(Lambda) is not torch.Tensor:
        Lambda = torch.tensor(Lambda).float()
    coef_traj = alphacum_traj.sqrt() * Lambda / \
                ((1 + (Lambda - 1) * alphacum_traj) *
                 (1 + (Lambda - 1) * alphacum_traj[0])).sqrt()
    return coef_traj

#%%
outdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\AnalyticalNote"
plt.figure(figsize=(5, 4))
for Lambda in [0.01, 0.1, 1, 10, 100, 0.0]:
    Lambda = torch.tensor(Lambda).float()
    coef_traj = Lambda * alphacum_traj / (Lambda * alphacum_traj + 1 - alphacum_traj)# xtproj_coef(Lambda, alphacum_traj)
    plt.plot(coef_traj,
                 label=f"Var={Lambda:.2f}" if Lambda < 1 else f"Var={Lambda:.0f}")
plt.ylabel("coefficient")
plt.xlabel("time step")
plt.title("tilde Lambda modulation of the projected outcome")
plt.legend()
saveallforms(figdir, f"{model_id_short}_tilde_Lambda_diag_modul")
saveallforms(outdir, f"{model_id_short}_tilde_Lambda_diag_modul")
plt.show()

# %%
plt.figure(figsize=(5, 4))
for Lambda in [0.01, 0.1, 1, 10, 100, 0.0]:
    Lambda = torch.tensor(Lambda).float()
    coef_traj = xtproj_coef(Lambda, alphacum_traj)
    plt.plot(coef_traj,
                 label=f"Var={Lambda:.2f}" if Lambda < 1 else f"Var={Lambda:.0f}")
plt.ylabel("coefficient")
plt.xlabel("time step")
plt.title("Projection of x_t along eigen dimension with different variance")
plt.legend()
saveallforms(figdir, f"{model_id_short}_xt_proj_coef")
plt.show()
#%%
plt.figure(figsize=(5, 4))
for Lambda in [0.01, 0.1, 1, 10, 100, 0.0]:
    Lambda = torch.tensor(Lambda).float()
    coef_traj = xtproj_coef(Lambda, alphacum_traj)
    plt.semilogy(coef_traj,
                 label=f"Var={Lambda:.2f}" if Lambda < 1 else f"Var={Lambda:.0f}")
plt.ylabel("coefficient")
plt.xlabel("time step")
plt.title("Projection of x_t along eigen dimension with different variance")
plt.legend()
saveallforms(figdir, f"{model_id_short}_xt_proj_coef_logy")
plt.show()
#%%
plt.figure(figsize=(5, 4))
for Lambda in [0.01, 0.1, 1, 10, 100, 0.0]:
    Lambda = torch.tensor(Lambda).float()
    coef_traj = xtproj_coef(Lambda, alphacum_traj)
    plt.plot(torch.diff(coef_traj).abs(),
                 label=f"Var={Lambda:.2f}" if Lambda < 1 else f"Var={Lambda:.0f}")
plt.ylabel("coefficient")
plt.xlabel("time step")
plt.title("Absolute Speed of change of x_t Projection along eigen dimension with different variance")
plt.legend()
saveallforms(figdir, f"{model_id_short}_xt_proj_coef_deriv_abs")
plt.show()
#%%
plt.figure(figsize=(5, 4))
for Lambda in [0.01, 0.1, 1, 10, 100, 0.0]:
    Lambda = torch.tensor(Lambda).float()
    coef_traj = xtproj_coef(Lambda, alphacum_traj)
    plt.plot(torch.diff(coef_traj),
                 label=f"Var={Lambda:.2f}" if Lambda < 1 else f"Var={Lambda:.0f}")
plt.ylabel("coefficient")
plt.xlabel("time step")
plt.title("Speed of change of x_t Projection along eigen dimension with different variance")
plt.legend()
saveallforms(figdir, f"{model_id_short}_xt_proj_coef_deriv")
plt.show()
#%%
plt.figure(figsize=(5, 4))
for Lambda in [0.01, 0.1, 1, 10, 100]:
    Lambda = torch.tensor(Lambda).float()
    coef_traj = x0hat_proj_coef(Lambda, alphacum_traj)
    plt.plot(coef_traj / Lambda.sqrt(),
                 label=f"Var={Lambda:.2f}" if Lambda < 1 else f"Var={Lambda:.0f}")
plt.ylabel("normalized coefficient (by std. dev.)")
plt.xlabel("time step")
plt.title("Projection of (\hat x_0(x_t) - \mu) along eigen with different variance")
plt.legend()
saveallforms(figdir, f"{model_id_short}_x0hat_proj_coef_norm")
plt.show()
#%%
plt.figure(figsize=(5, 4))
for Lambda in [0.01, 0.1, 1, 10, 100, 0.0]:
    Lambda = torch.tensor(Lambda).float()
    coef_traj = x0hat_proj_coef(Lambda, alphacum_traj)
    plt.plot(torch.diff(coef_traj) / Lambda.sqrt(),
                 label=f"Var={Lambda:.2f}" if Lambda < 1 else f"Var={Lambda:.0f}")
plt.ylabel("normalized coefficient (by std. dev.)")
plt.xlabel("time step")
plt.title("Speed of change of (\hat x_0(x_t) - \mu) projection along eigen with different variance")
plt.legend()
saveallforms(figdir, f"{model_id_short}_x0hat_proj_coef_norm_deriv")
plt.show()