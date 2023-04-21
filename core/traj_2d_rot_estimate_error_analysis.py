import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from os.path import join
from datasets import load_dataset
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from core.ODE_analytical_lib import *
from diffusers import DDIMPipeline
from core.diffusion_geometry_lib import *
import platform
figdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\Rot_2d_subspace_residual"

#%%
RNDrange = range(200,400)
model_id = "google/ddpm-cifar10-32" # most popular
for model_id, RNDrange in [
    ("dimpo/ddpm-mnist", range(200, 400)),
    ("google/ddpm-cifar10-32", range(200,400)),
    ("google/ddpm-celebahq-256", range(125, 300)),
    ("google/ddpm-church-256", range(125, 150)),
]:
    model_id_short = model_id.split("/")[-1]
    pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
    pipe.scheduler.set_timesteps(51)
    alphacum_traj = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
    #%%
    traj_dir = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_scheduler\DDIM"
    res_ratio_projall_col = []
    res_ratio_proj2_col = []
    for seed in tqdm(RNDrange):
        traj_data = torch.load(join(traj_dir, f"seed{seed}", "state_reservoir.pt"))
        sample_traj = traj_data["latents_traj"]
        res_traj = traj_data['residue_traj']
        t_traj = traj_data['t_traj']
        proj_x0_traj = (sample_traj[:-1] -
                        res_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / \
                  alphacum_traj.sqrt().view(-1, 1, 1, 1)

        x0_vec = sample_traj[-1:].flatten(1)
        res_ratio_projall_vec = []
        res_ratio_proj2_vec = []
        for stepi in range(51):
            res_final_projall = proj2orthospace(sample_traj[:stepi+1].flatten(1), x0_vec)
            res_ratio_projall = res_final_projall.norm(dim=1)**2 / x0_vec.norm(dim=1)**2
            # print(res_ratio_projall)
            res_final_proj2 = proj2orthospace(sample_traj[[0, stepi]].flatten(1), x0_vec)
            res_ratio_proj2 = res_final_proj2.norm(dim=1)**2 / x0_vec.norm(dim=1)**2
            # print(res_ratio_proj2)
            res_ratio_projall_vec.append(res_ratio_projall)
            res_ratio_proj2_vec.append(res_ratio_proj2)

        res_ratio_projall_vec = torch.stack(res_ratio_projall_vec)
        res_ratio_proj2_vec = torch.stack(res_ratio_proj2_vec)

        res_ratio_projall_col.append(res_ratio_projall_vec)
        res_ratio_proj2_col.append(res_ratio_proj2_vec)
    #%%
    res_ratio_projall_col = torch.stack(res_ratio_projall_col)
    res_ratio_proj2_col = torch.stack(res_ratio_proj2_col)
    res_ratio_projall_col.squeeze_()
    res_ratio_proj2_col.squeeze_()
    #%%
    torch.save({
        "res_ratio_projall_col": res_ratio_projall_col,
        "res_ratio_proj2_col": res_ratio_proj2_col,
    }, join(figdir, f"rot_subspace_residual_{model_id_short}.pt"))
    #%%
    # plot with shaded error bar
    plt.figure()
    plt.plot(res_ratio_proj2_col.mean(dim=0).numpy(), label="residue from [x_T,x_t] 2d subspace")
    plt.plot(res_ratio_projall_col.mean(dim=0).numpy(), label="residue from [x_T:x_t] subspace")
    plt.fill_between(range(51),
                        res_ratio_proj2_col.quantile(0.25, dim=0).numpy(),
                        res_ratio_proj2_col.quantile(0.75, dim=0).numpy(),
                        alpha=0.2)
    plt.fill_between(range(51),
                        res_ratio_projall_col.quantile(0.25, dim=0).numpy(),
                        res_ratio_projall_col.quantile(0.75, dim=0).numpy(),
                        alpha=0.2)
    plt.legend()
    plt.xlabel("time steps", fontsize=14)
    plt.ylabel("residual variance ratio", fontsize=14)
    plt.title(f"Deviation of final sample x_0 from 'rotation' subspace\n{model_id}", fontsize=14)
    saveallforms(figdir, f"rot_subspace_residual_{model_id_short}")
    plt.show()

#%%







#%% Scratch zone
plt.plot(res_ratio_projall_vec[:].numpy(), label="residue from [x_T:x_t] subspace")
plt.plot(res_ratio_proj2_vec[:].numpy(), label="residue from [x_T,x_t] 2d subspace")
plt.legend()
plt.xlabel("time steps")
plt.ylabel("residual variance ratio")
plt.title("Deviation from 'rotation' subspace")
plt.show()
#%%
torch.pinverse(sample_traj[[0,5,10,15,20]].flatten(1))
#%%
Amat = sample_traj[:40].flatten(1)
bvec = x0_vec
xproj = bvec @ torch.pinverse(Amat) @ Amat
xres  = bvec - xproj
res_ratio = xres.norm(dim=1)**2 / x0_vec.norm(dim=1)**2
print(res_ratio)
#%%