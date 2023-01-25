import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from os.path import join
from datasets import load_dataset
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
#%% CIFAR1- PCA
savedir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ImageSpacePCA\CIFAR10"
# torch.save({"U": U, "S": S, "V": V, "mean": imgmean, "cov_eigs": cov_eigs},
#            join(savedir, "mnist_pca.pt"))
data = torch.load(join(savedir, "CIFAR10_PCA.pt"))
S, V, imgmean, cov_eigs  = data["S"], data["V"], data["mean"], data["cov_eigs"]
#%%
# cov_eigs = S**2 / (U.shape[0] - 1)
#%%
def norm2img(x):
    return torch.clamp((x + 1) / 2, 0, 1)
#%%
from core.ODE_analytical_lib import *
from diffusers import DDIMPipeline
import platform
model_id = "google/ddpm-cifar10-32" # most popular
model_id_short = model_id.split("/")[-1]
pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.scheduler.set_timesteps(51)
alphacum_traj = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
#%%
traj_dir = r"F:\insilico_exps\Diffusion_traj\ddpm-cifar10-32_scheduler\DDIM"
outdir = r"F:\insilico_exps\Diffusion_traj\cifar10_PCA_theory"
os.makedirs(outdir, exist_ok=True)
traj_collection = []
for seed in tqdm(range(200, 400)):
    traj_data = torch.load(join(traj_dir, f"seed{seed}", "state_reservoir.pt"))
    sample_traj = traj_data["latents_traj"]
    res_traj = traj_data['residue_traj']
    t_traj = traj_data['t_traj']
    proj_x0_traj = (sample_traj[:-1] -
                    res_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / \
              alphacum_traj.sqrt().view(-1, 1, 1, 1)
    # pred_x0_imgs = (pred_x0 + 1) / 2
    # Analytical prediction
    x0_vec = sample_traj[0:1].flatten(1)
    mu_vec = imgmean.flatten(1) * 2 - 1
    # predict xt
    xt_traj, xt0_residue, scaling_coef_ortho, xttraj_coef = \
        xt_ode_solution(x0_vec, mu_vec, V, cov_eigs * 4, alphacum_traj)
    # predict x0hat
    x0hatxt_traj, xttraj_coef, xttraj_coef_modulated = x0hat_ode_solution( \
        x0_vec, mu_vec, V, cov_eigs * 4, alphacum_traj)
    # save trajectoryimages
    save_imgrid(norm2img(x0hatxt_traj.reshape(-1, 3, 32, 32)), join(outdir, f"seed{seed}_x0hat_theory.png"))
    save_imgrid(norm2img(xt_traj.reshape(-1, 3, 32, 32)), join(outdir, f"seed{seed}_xt_theory.png"))
    save_imgrid(norm2img(sample_traj), join(outdir, f"seed{seed}_xt_empir.png"))
    save_imgrid(norm2img(proj_x0_traj), join(outdir, f"seed{seed}_x0hat_empir.png"))
    # if seed == 400:
    #     break
    xt_pred_mse = ((sample_traj[1:].flatten(1) - xt_traj)**2).mean(1)
    x0hat_pred_mse = ((proj_x0_traj.flatten(1) - x0hatxt_traj)**2).mean(1)
    plt.figure()
    plt.plot(xt_pred_mse)
    # plt.plot((sample_traj[1:].flatten(1) - xt_traj).norm(dim=1))
    plt.ylabel("MSE of deviation")
    plt.xlabel("timestep")
    plt.title("L2 norm of deviation between empirical and analytical prediction of xt")
    saveallforms(outdir, f"seed{seed}_xt_deviation_L2")
    plt.show()
    plt.figure()
    plt.plot(x0hat_pred_mse)
    # plt.plot((proj_x0_traj.flatten(1) - x0hatxt_traj).norm(dim=1))
    plt.ylabel("MSE of deviation")
    plt.xlabel("timestep")
    plt.title("L2 norm of deviation between empirical and analytical prediction of x0hat")
    saveallforms(outdir, f"seed{seed}_x0hat_deviation_L2")
    plt.show()
    torch.save({"xt_traj": xt_traj,
                "x0hatxt_traj": x0hatxt_traj,
                "xttraj_coef": xttraj_coef, "xt0_residue": xt0_residue,
                "xttraj_coef_modulated": xttraj_coef_modulated,
                "scaling_coef_ortho": scaling_coef_ortho,
                "xt_pred_mse": xt_pred_mse,
                "x0hat_pred_mse": x0hat_pred_mse,
                }, join(outdir, f"seed{seed}_theory_coef.pt"))
