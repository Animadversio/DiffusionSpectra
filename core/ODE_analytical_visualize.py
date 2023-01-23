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
def xtproj_coef(Lambda, alphacum_traj):
    if type(Lambda) is not torch.Tensor:
        Lambda = torch.tensor(Lambda).float()
    coef_traj = ((1 + (Lambda - 1) * alphacum_traj) /
                 (1 + (Lambda - 1) * alphacum_traj[0])).sqrt()
    return coef_traj


def x0hat_proj_coef(Lambda, alphacum_traj):
    if type(Lambda) is not torch.Tensor:
        Lambda = torch.tensor(Lambda).float()
    coef_traj = alphacum_traj.sqrt() * Lambda / \
                ((1 + (Lambda - 1) * alphacum_traj) *
                 (1 + (Lambda - 1) * alphacum_traj[0])).sqrt()
    return coef_traj


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