import sys
sys.path.append("/home/biw905/Github/DiffusionSpectra")
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from os.path import join
from diffusers import DDIMPipeline
from datasets import load_dataset
from core.ODE_analytical_lib import *
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torchvision.utils import save_image, make_grid
import platform
#%% CIFAR1- PCA
# savedir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ImageSpacePCA\CIFAR10"
# savedir = r"/home/binxu/Datasets"
savedir = r"/home/biw905/Datasets"
# torch.save({"U": U, "S": S, "V": V, "mean": imgmean, "cov_eigs": cov_eigs},
#            join(savedir, "mnist_pca.pt"))
data = torch.load(join(savedir, "CIFAR10_pca.pt"))
S, V, imgmean, cov_eigs  = data["S"], data["V"], data["mean"], data["cov_eigs"]
# cov_eigs = S**2 / (U.shape[0] - 1)
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
model_id = "google/ddpm-cifar10-32" # most popular
model_id_short = model_id.split("/")[-1]
pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.scheduler.set_timesteps(51)
alphacum_traj = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
pipe.to('cuda')
#%%
import argparse
# create argparser that parse the range of random seeds to run
parser = argparse.ArgumentParser()
parser.add_argument('--seed_start', type=int, default=11000)
parser.add_argument('--seed_end', type=int, default=16000)
args = parser.parse_args()
print(args)
#%%
# traj_dir = r"F:\insilico_exps\Diffusion_traj\ddpm-cifar10-32_scheduler\DDIM"
# outdir = r"F:\insilico_exps\Diffusion_traj\cifar10_PCA_theory_hybrid"
# outdir = r"/home/binxu/insilico_exps/Diffusion_traj/cifar10_PCA_theory_hybrid"
outdir = r"/n/scratch3/users/b/biw905/Diffusion_traj/cifar10_PCA_theory_hybrid"
os.makedirs(outdir, exist_ok=True)
mu_vec = imgmean.flatten(1) * 2 - 1
for seed in tqdm(range(args.seed_start, args.seed_end)):
    # for seed in tqdm(range(0, 5)):
    # traj_data = torch.load(join(traj_dir, f"seed{seed}", "state_reservoir.pt"))
    # sample_traj = traj_data["latents_traj"]
    # res_traj = traj_data['residue_traj']
    # t_traj = traj_data['t_traj']
    # x0_vec = sample_traj[0:1].flatten(1)
    # np.random.seed(seed)
    # x0_vec = np.random.randn(1, 3072, dtype=torch.float32,)
    x0_vec = torch.randn(1, 3072, dtype=torch.float32,
                         generator=torch.Generator().manual_seed(seed))
    # predict x_t
    xt_traj, xt0_residue, scaling_coef_ortho, xttraj_coef = \
        xt_ode_solution(x0_vec, mu_vec, V, cov_eigs * 4, alphacum_traj)
    # predict x0hat
    x0hatxt_traj, xttraj_coef, xttraj_coef_modulated = x0hat_ode_solution( \
        x0_vec, mu_vec, V, cov_eigs * 4, alphacum_traj)
    # %
    hybrid_col = {}
    for skip_steps in [0, 10, 20, 25, 30, 40]:
        noisy_sample_init = xt_traj[skip_steps:skip_steps + 1].reshape(1, 3, 32, 32)
        sample_hybrid, sample_traj_hybrid, residual_traj_hybrid, t_traj_hybrid = \
            sampling_with_skip(
            pipe.unet, pipe.scheduler, noisy_sample_init, skip_steps=skip_steps)
        hybrid_col[skip_steps] = sample_hybrid
        if skip_steps == 0:
            sample_traj = sample_traj_hybrid
            residual_traj = residual_traj_hybrid
            t_traj = t_traj_hybrid

    mtg = make_grid(torch.stack([hybrid_col[0][0],
                               hybrid_col[10][0],
                               hybrid_col[20][0],
                               hybrid_col[25][0],
                               hybrid_col[30][0],
                               hybrid_col[40][0],
                               xt_traj[-1].reshape(3, 32, 32)], dim=0),
                                nrow=7, padding=0, )
    save_image(mtg, join(outdir, f"seed{seed}_hybrid_cmp.png"))
    torch.save({"hybrid_col": hybrid_col, "x0hatxt_traj": x0hatxt_traj, "xt_traj": xt_traj,
                "sample_traj": sample_traj, "residual_traj": residual_traj, "t_traj": t_traj,},
               join(outdir, f"seed{seed}_hybrid_cmp.pt"))


