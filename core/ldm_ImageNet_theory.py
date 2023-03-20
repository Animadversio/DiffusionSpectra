import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from os.path import join
from core.utils.plot_utils import saveallforms, save_imgrid, to_imgrid
from core.ODE_analytical_lib import *

#%%

sys.path.append("/home/binxu/Github/latent-diffusion")
sys.path.append('/home/binxu/Github/taming-transformers')
# from taming.models import vqgan
sys.path.append("/home/binxu/Github/latent-diffusion")
#%%
#@title loading utils
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
if os.environ["USER"] == "binxu":
    ldm_root = "/home/binxu/Github/latent-diffusion"
elif os.environ["USER"] == "biw905":
    ldm_root = "/home/biw905/Github/latent-diffusion"


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load(join(ldm_root,"configs/latent-diffusion/cin256-v2.yaml"))
    model = load_model_from_config(config, join(ldm_root,"models/ldm/cin256-v2/model.ckpt"))
    return model


@torch.no_grad()
def decode_batch(model, zs, batch_size=5):
    # with model.ema_scope():
    x = []
    for i in tqdm(range(0, zs.shape[0], batch_size)):
        z_batch = zs[i:i + batch_size].to(model.device)
        x.append(model.decode_first_stage(z_batch).detach().cpu())
    x = torch.cat(x, dim=0)
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    return x


@torch.no_grad()
def decode_batchvec(model, zs, batch_size=5, latent_shape=(3, 64, 64)):
    # with model.ema_scope():
    x = []
    for i in tqdm(range(0, zs.shape[0], batch_size)):
        z_batch = zs[i:i + batch_size].reshape(-1, *latent_shape).to(model.device)
        x.append(model.decode_first_stage(z_batch).detach().cpu())
    x = torch.cat(x, dim=0)
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    return x
#%%
from ldm.models.diffusion.ddim import DDIMSampler

model = get_model()
sampler = DDIMSampler(model)
#%%
PCAdir = r"/home/binxu/DL_Projects/ldm-imagenet/latents_save"
traj_dir = r"/home/binxu/insilico_exps/Diffusion_traj/ldm_imagenet/DDIM"
outdir = r"/home/binxu/insilico_exps/Diffusion_traj/ldm_imagenet_analytical"
saveroot = r"/home/binxu/insilico_exps/Diffusion_traj/ldm_imagenet"
os.makedirs(outdir, exist_ok=True)
#%%
# PCAdir = r"F:\insilico_exps\Diffusion_traj\ldm-imagenet-pca\latents_save"
# traj_dir = r"F:\insilico_exps\Diffusion_traj\ldm_imagenet\DDIM"
# outdir = r"F:\insilico_exps\Diffusion_traj\ldm_imagenet_analytical"
# saveroot = r"F:\insilico_exps\Diffusion_traj\ldm_imagenet"
# os.makedirs(outdir, exist_ok=True)
#%%
ddim_ab = torch.load(join(saveroot, "ddim_alphas_betas.pt"))
alphas_cumprod = ddim_ab["alphas_cumprod"]
alphas_cumprod_prev = ddim_ab["alphas_cumprod_prev"]
betas = ddim_ab["betas"]
#%%
for class_id in range(0, 100):
    PCA_data = torch.load(join(PCAdir, f"class{class_id}_z_pca.pt"))
    U = PCA_data['U']
    V = PCA_data['V']
    S = PCA_data['S']
    imgmean = PCA_data['mean']
    cov_eigs = S**2 / (U.shape[0] - 1)
    #%%
    traj_collection = []
    for RNDseed in tqdm(range(10)):
        #%%
        traj_data = torch.load(join(traj_dir, f"class{class_id:03d}_seed{RNDseed:03d}", "state_traj.pt"))
        z_traj = traj_data['z_traj'].cpu()
        pred_z0_traj = traj_data['pred_z0_traj'].cpu()
        t_traj = traj_data['t_traj']
        idx_traj = 1000 - t_traj - 1
        # raise NotImplementedError
        # pred_x0_imgs = (pred_x0 + 1) / 2
        # Analytical prediction
        alphacum_traj = alphas_cumprod[idx_traj].cpu()
        alphacumprev_traj = alphas_cumprod_prev[idx_traj].cpu()

        zT_vec = z_traj[0:1].flatten(1)
        mu_vec = imgmean[None, :] #.flatten(1) #  * 2 - 1
        # predict xt
        print("Solving ODE for xt...")
        xt_traj, xt0_residue, scaling_coef_ortho, xttraj_coef = \
            xt_ode_solution(zT_vec, mu_vec, V, cov_eigs, alphacum_traj)
        # predict x0hat
        print("Solving ODE for x0hat...")
        x0hatxt_traj, xttraj_coef, xttraj_coef_modulated = x0hat_ode_solution( \
            zT_vec, mu_vec, V, cov_eigs, alphacum_traj)

        # save trajectoryimages
        # print("Decoding images...")
        # imgtraj_xt_ddim = decode_batchvec(model, z_traj)
        # imgtraj_x0hat_ddim = decode_batchvec(model, pred_z0_traj)
        # imgtraj_xt_theory = decode_batchvec(model, xt_traj)
        # imgtraj_x0hat_theory = decode_batchvec(model, x0hatxt_traj)
        # print("Saving images...")
        # save_imgrid(imgtraj_xt_theory, join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_xt_theory.png"))
        # save_imgrid(imgtraj_x0hat_theory, join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_x0hat_theory.png"))
        # save_imgrid(imgtraj_xt_ddim, join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_xt_empir.png"))
        # save_imgrid(imgtraj_x0hat_ddim, join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_x0hat_empir.png"))
        # # if seed == 400:
        # #     break
        # plt.imshow(to_imgrid(imgtraj_x0hat_theory))
        # plt.axis("off")
        # plt.show()
        # plt.imshow(to_imgrid(imgtraj_x0hat_ddim))
        # plt.axis("off")
        # plt.show()
        print("Plotting image differnece...")
        xt_pred_mse = ((z_traj[1:].flatten(1) - xt_traj)**2).mean(1)
        x0hat_pred_mse = ((pred_z0_traj[1:].flatten(1) - x0hatxt_traj)**2).mean(1)
        plt.figure()
        plt.plot(xt_pred_mse)
        # plt.plot((sample_traj[1:].flatten(1) - xt_traj).norm(dim=1))
        plt.ylabel("MSE of deviation")
        plt.xlabel("timestep")
        plt.title("L2 norm of deviation between empirical and analytical prediction of xt")
        saveallforms(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_xt_deviation_L2")
        plt.show()
        plt.figure()
        plt.plot(x0hat_pred_mse)
        # plt.plot((proj_x0_traj.flatten(1) - x0hatxt_traj).norm(dim=1))
        plt.ylabel("MSE of deviation")
        plt.xlabel("timestep")
        plt.title("L2 norm of deviation between empirical and analytical prediction of x0hat")
        saveallforms(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_x0hat_deviation_L2")
        plt.show()
        raise NotImplementedError
        torch.save({"xt_traj": xt_traj,
                    "x0hatxt_traj": x0hatxt_traj,
                    "xttraj_coef": xttraj_coef, "xt0_residue": xt0_residue,
                    "xttraj_coef_modulated": xttraj_coef_modulated,
                    "scaling_coef_ortho": scaling_coef_ortho,
                    "xt_pred_mse": xt_pred_mse,
                    "x0hat_pred_mse": x0hat_pred_mse,
                    }, join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_theory_coef.pt"))
        # raise NotImplementedError
#%% sweep the mse and traj
from collections import defaultdict
xt_pred_mse_all = defaultdict(list)
x0hat_pred_mse_all = defaultdict(list)
for class_id in tqdm(range(100)):
    for RNDseed in range(10):
        theory_data = torch.load(join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_theory_coef.pt"))
        xt_pred_mse = theory_data['xt_pred_mse']
        x0hat_pred_mse = theory_data['x0hat_pred_mse']
        xt_pred_mse_all[class_id].append(xt_pred_mse)
        x0hat_pred_mse_all[class_id].append(x0hat_pred_mse)
xt_pred_mse_all = {k: torch.stack(v) for k, v in xt_pred_mse_all.items()}
x0hat_pred_mse_all = {k: torch.stack(v) for k, v in x0hat_pred_mse_all.items()}
#%%
def plot_mean_with_quantile(data_arr, quantile, label, color=None, ax=None):
    if ax is None:
        ax = plt.gca()
    mean_vec = data_arr.mean(axis=0)
    ax.plot(mean_vec, label=label, color=color, lw=2)  # ttraj,
    ax.fill_between(range(len(mean_vec)),
                    np.quantile(data_arr, quantile[0], axis=0),
                    np.quantile(data_arr, quantile[1], axis=0),
                    alpha=0.3, color=color)
    return ax


figoutdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ldm_ImageNet"
figoutdir = r"/home/binxu/Documents/ldm_ImageNet"
os.makedirs(figoutdir, exist_ok=True)
#%%
plt.figure(figsize=(4, 3.5))
for class_id in range(100):
    # plt.plot(xt_pred_mse_all[class_id].mean(0), label=f"class {class_id}")
    plot_mean_with_quantile(xt_pred_mse_all[class_id], [0.25, 0.75], f"class {class_id}",
                            color=plt.cm.tab10(class_id % 10))
# plt.legend()
plt.ylabel("MSE of deviation")
plt.xlabel("Time step (DDIM)")
plt.title("Deviation of empirical from analytical xt\nLatent Diffusion ImageNet")
plt.tight_layout()
saveallforms(figoutdir, "xt_deviation_MSE_allclass")
plt.show()
#%%
plt.figure(figsize=(4, 3.5))
for class_id in range(100):
    # plt.plot(x0hat_pred_mse_all[class_id].mean(0), label=f"class {class_id}")
    plot_mean_with_quantile(x0hat_pred_mse_all[class_id], [0.25, 0.75], f"class {class_id}",
                            color=plt.cm.tab10(class_id % 10))
# plt.legend()
plt.ylabel("MSE of deviation")
plt.xlabel("Time step (DDIM)")
plt.title("Deviation of empirical from analytical x0hat\nLatent Diffusion ImageNet")
plt.tight_layout()
saveallforms(figoutdir, "x0hat_deviation_MSE_allclass")
plt.show()
#%%
theory_sumdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\Theory"
plt.figure(figsize=(5, 4.5))
Lmbda = 10
for Lmbda in [0.01, 0.1, 1, 10, 100, 0.0, ]:
    plt.plot(np.linspace(0,1,1000), torch.sqrt(Lmbda / (1 + (Lmbda - 1) * alphas_cumprod)).cpu(), label=f"Lambda={Lmbda}")
plt.legend()
# reverse x axis direction
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.ylabel("Amplification factor")
plt.xlabel("Perturbation time t'")
plt.title("Amplification factor of perturbation \nas a function of perturbing time ")
saveallforms(theory_sumdir, "perturb_amplification_factor")
plt.show()
# collect final images as a row
#%% Visualize the latent images
for class_id in range(10, 100):
    for RNDseed in tqdm(range(10)):
        theory_data = torch.load(join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_theory_coef.pt"))
        x0hatxt_traj = theory_data['x0hatxt_traj']
        xt_traj = theory_data['xt_traj']
        traj_data = torch.load(join(traj_dir, f"class{class_id:03d}_seed{RNDseed:03d}", "state_traj.pt"))
        z_traj = traj_data['z_traj'].cpu()
        pred_z0_traj = traj_data['pred_z0_traj'].cpu()
        # raise NotImplementedError
        print("Decoding images...")
        with torch.no_grad():
            # note without clone there will be memory error somehow causing the whole thing to crash
            print("Decoding xt ddim...")
            imgtraj_xt_ddim = decode_batch(model, z_traj[:, 0].clone(), batch_size=6)
            print("Decoding x0hat ddim...")
            imgtraj_x0hat_ddim = decode_batch(model, pred_z0_traj[:, 0].clone(), batch_size=6)
            print("Decoding xt theory...")
            imgtraj_xt_theory = decode_batch(model, xt_traj.reshape(-1, 3, 64, 64).clone(), batch_size=6)
            print("Decoding x0hat theory...")
            imgtraj_x0hat_theory = decode_batch(model, x0hatxt_traj.reshape(-1, 3, 64, 64).clone(), batch_size=6)
        print("Saving images...")
        save_imgrid(imgtraj_xt_theory, join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_xt_theory.jpg"))
        save_imgrid(imgtraj_x0hat_theory, join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_x0hat_theory.jpg"))
        save_imgrid(imgtraj_xt_ddim, join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_xt_empir.jpg"))
        save_imgrid(imgtraj_x0hat_ddim, join(outdir, f"class{class_id:03d}_seed{RNDseed:03d}_x0hat_empir.jpg"))






#%%
alphacum_prev_traj = alphas_cumprod_prev[idx_traj].cpu()
x0hatxt_traj, xttraj_coef, xttraj_coef_modulated = x0hat_ode_solution( \
        zT_vec, mu_vec, V, cov_eigs, alphacum_prev_traj)
imgtraj_x0hat_theory = decode_batchvec(model, x0hatxt_traj)
# plt.imshow(to_imgrid(imgtraj_x0hat_ddim))
plt.imshow(to_imgrid(imgtraj_x0hat_theory))
plt.axis("off")
plt.show()
plt.imshow(to_imgrid(imgtraj_x0hat_ddim))
plt.axis("off")
plt.show()
