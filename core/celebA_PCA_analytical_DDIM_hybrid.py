import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from os.path import join
from datasets import load_dataset
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
#%% MNIST PCA
savedir = "E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ImageSpacePCA\celebA"
# torch.save({"U": U, "S": S, "V": V, "mean": imgmean, "cov_eigs": cov_eigs},
#            join(savedir, "mnist_pca.pt"))
data = torch.load(join("F:\insilico_exps\Diffusion_traj", "celebA_dataset_PCA.pt"))
U, S, V, imgmean,  = data["U"], data["S"], data["V"], data["mean"], #, data["cov_eigs"]
cov_eigs = S**2 / (U.shape[0] - 1)
#%%
def norm2img(x):
    return torch.clamp((x + 1) / 2, 0, 1)
#%%
from core.ODE_analytical_lib import *
from diffusers import DDIMPipeline
import platform
model_id = "google/ddpm-celebahq-256" # most popular
model_id_short = model_id.split("/")[-1]
pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.scheduler.set_timesteps(51)
alphacum_traj = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
pipe.to('cuda')
#%%
def sampling_with_skip(unet, scheduler, noisy_sample_init, skip_steps=0):
    # noisy_sample = torch.randn(
    #     batch_size, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size,
    #     generator=generator, device=unet.device
    # ).to(unet.device).to(unet.dtype)
    # noisy_sample_init
    t_traj, sample_traj, residual_traj = [], [], []
    sample = noisy_sample_init
    sample_traj.append(sample.cpu().detach())
    for i, t in enumerate(tqdm(scheduler.timesteps[skip_steps:])):
        # 1. predict noise residual
        with torch.no_grad():
            residual = unet(sample.to(unet.device), t).sample
        # 2. compute previous image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample.to(unet.device)).prev_sample
        residual_traj.append(residual.cpu().detach())
        sample_traj.append(sample.cpu().detach())
        t_traj.append(t)
    t_traj = torch.tensor(t_traj)
    t_traj = t_traj.long().cpu()
    sample_traj = torch.cat(sample_traj, dim=0)
    residual_traj = torch.cat(residual_traj, dim=0)
    return sample.cpu().detach(), sample_traj, residual_traj, t_traj
#%%
from torchvision.utils import save_image, make_grid

traj_dir = r"F:\insilico_exps\Diffusion_traj\ddpm-celebahq-256_scheduler\DDIM"
# outdir = r"F:\insilico_exps\Diffusion_traj\MNIST_PCA_theory"
outdir = r"F:\insilico_exps\Diffusion_traj\celebA_PCA_theory_DDIM_hybrid"
os.makedirs(outdir, exist_ok=True)
traj_collection = []
for seed in tqdm(range(150, 300)):
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
    hybrid_col = {}
    for skip_steps in [10, 15, 20, 25, 30, 40]:
        noisy_sample_init = xt_traj[skip_steps:skip_steps + 1].reshape(1, 3, 256, 256)
        sample_hybrid, _, _, _ = sampling_with_skip(
            pipe.unet, pipe.scheduler, noisy_sample_init, skip_steps=skip_steps)
        hybrid_col[skip_steps] = sample_hybrid
    # %%

    mtg = make_grid(torch.stack([sample_traj[-1],
                                 hybrid_col[10][0],
                                 hybrid_col[15][0],
                                 hybrid_col[20][0],
                                 hybrid_col[25][0],
                                 hybrid_col[30][0],
                                 hybrid_col[40][0],
                                 x0hatxt_traj[-1].reshape(3, 256, 256)], dim=0),
                    nrow=8, padding=0, )

    save_image(mtg, join(outdir, f"seed{seed}_hybrid_cmp.png"))
    torch.save({"hybrid_col": hybrid_col, "sample_traj": sample_traj,
                "x0hatxt_traj": x0hatxt_traj, "xt_traj": xt_traj,
                "t_traj": t_traj},
               join(outdir, f"seed{seed}_hybrid_cmp.pt"))
