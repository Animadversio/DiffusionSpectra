
import os
from os.path import join
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from torchvision.datasets import CelebA, ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
# imagedataset
transform = Compose([ToTensor(),
                     CenterCrop(178),
                     Resize([256,256])
                     ])
imagedataset = ImageFolder(r"E:\Datasets\CelebA", transform=transform)
#%%
from torch.utils.data import DataLoader
dataloader = DataLoader(imagedataset, batch_size=64, shuffle=False, num_workers=4)
imgtsrs = []
for i, (imgs, _) in tqdm(enumerate(dataloader)):
    if i == 500:
        break
    imgtsrs.append(imgs)

imgtsrs = torch.cat(imgtsrs, dim=0)
#%%
imgmean = imgtsrs.mean(dim=(0), keepdim=True)
U_face, S_face, V_face = torch.svd_lowrank((imgtsrs - imgmean).view(imgtsrs.shape[0], -1), q=2000)
torch.save({"U": U_face, "S": S_face, "V": V_face, "mean": imgmean},
           r"F:\insilico_exps\Diffusion_traj\celebA_dataset_PCA.pt")
#%%
show_imgrid(torch.clamp(0.5+V_face[:, 0:100].reshape(3, 256, 256, -1).permute(3,0,1,2)*70,0,1), nrow=10)
#%%
show_imgrid(imgmean)
#%%
"""
img = (sample + 1) / 2
sample = 2 * image - 1
"""
#%%
figdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ImageSpacePCA"
#%%
dataset_name = "CelebA"
os.makedirs(join(figdir, dataset_name), exist_ok=True)
cov_eigs = S_face**2 / (imgtsrs.shape[0] - 1)
plt.figure()
plt.semilogy(cov_eigs * 2**2)
plt.axhline(1, color="red", linestyle="--")
plt.ylabel("Eigenvalue of Covariance Matrix")
plt.xlabel("PC index")
plt.title(f"Eigenvalues of Covariance Matrix of {dataset_name} Dataset\nScaled to sample x space")
saveallforms(join(figdir, dataset_name), f"cov_eigenspectrum_{dataset_name}_log")
plt.show()
plt.figure()
plt.plot(cov_eigs * 2**2)
plt.axhline(1, color="red", linestyle="--")
plt.ylabel("Eigenvalue of Covariance Matrix")
plt.xlabel("PC index")
plt.title(f"Eigenvalues of Covariance Matrix of {dataset_name} Dataset\nScaled to sample x space")
saveallforms(join(figdir, dataset_name), f"cov_eigenspectrum_{dataset_name}")
plt.show()
save_imgrid(imgmean, join(figdir, dataset_name, "img_mean.png"))
#%%

#%%
from diffusers import DDIMPipeline
import platform
model_id = "google/ddpm-celebahq-256" # most popular
# model_id_short = "ddpm-celebahq-256"
model_id_short = model_id.split("/")[-1]
# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.unet.requires_grad_(False).eval().to("cuda")#.half()
pipe.scheduler.set_timesteps(51)
#%%
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_scheduler"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}_scheduler"
else:
    raise RuntimeError("Unknown system")

export_root = rf"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\{model_id_short}"
save_fmt = "jpg"
name_Sampler = "DDIM"
exp_fignames = ["proj_z0_vae_decode.png",
                "sample_diff_lag1_stdnorm_vae_decode.png",
                "samples_all.png",]


for seed in tqdm(range(125, 300)):
    savedir = join(saveroot, name_Sampler, f"seed{seed}")
    traj_data = torch.load(join(savedir, "state_reservoir.pt"))
    sample_traj = traj_data["latents_traj"]
    residual_traj = traj_data["residue_traj"]
    t_traj = traj_data["t_traj"]
    alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
    pred_x0 = (sample_traj[:-1] - residual_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / \
              alphacum_traj.sqrt().view(-1, 1, 1, 1)
    pred_x0_imgs = (pred_x0 + 1) / 2

    # save_imgrid(pred_x0_imgs, join(savedir, "proj_z0_vae_decode.jpg"), nrow=10, )
    show_imgrid(pred_x0_imgs, nrow=10)
    show_imgrid((1 + sample_traj)/2, nrow=10)
    show_imgrid(residual_traj, nrow=10)
    break
#%%
PCcoef = (sample_traj - imgmean * alphacum_traj[0]).flatten(1) @ V_face
PCcoef.shape
#%%
PC_range = range(100, 105)
plt.figure()
plt.plot(PCcoef[:, PC_range])
plt.title("PC Coefficients of Samples traj x(t)")
plt.xlabel("Time")
plt.ylabel("PC Coefficient")
# plt.legend([f"PC{i}" for i in range(10)])
# saveallforms(savedir, "PCcoef")
plt.show()
#%%
# from mpl_axes_aligner import align
from core.ODE_analytical_lib import xt_proj_coef
PC_range = range(0, 15)
for PC_range in [range(0, 10), range(50, 60), range(100,110), range(250,260),
                 range(500, 510), range(1000, 1010), range(1500, 1510), range(1990, 2000)]:
    plt.figure(figsize=(9, 5))
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(PCcoef[:, PC_range])
    plt.xlabel("Time")
    plt.ylabel("PC Coefficient")
    plt.title("Actual")
    ax2 = plt.subplot(1, 2, 2)
    for Lambda, coef_0 in zip(cov_eigs[PC_range], PCcoef[0, PC_range]):
        proj_theory = xt_proj_coef(Lambda * 4, alphacum_traj)
        plt.plot(proj_theory * coef_0, )
    plt.xlabel("Time")
    plt.ylabel("PC Coefficient")
    plt.title("Theoretical")
    plt.suptitle(f"PC Coefficients of Samples traj x(t) for {str(PC_range)}\n"
                 f"Cov eigenvalues: {cov_eigs[PC_range[0]]:.3f}-{cov_eigs[PC_range[-1]]:.3f}")
    plt.legend([f"PC{i}" for i in PC_range], loc="right")
    YLIM = min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax2.set_ylim(YLIM)
    ax1.set_ylim(YLIM)
    saveallforms(join(figdir, dataset_name), f"PCcoef_theory_cmp_PC{PC_range[0]}-{PC_range[-1]}")
    plt.show()
#%% RND dimension
RNDvecs = torch.randn(imgmean.numel(), 10, device=imgmean.device,
                      generator=torch.manual_seed(42))
RNDvecs = RNDvecs / RNDvecs.norm(dim=0, keepdim=True)
RNDproj = (sample_traj - imgmean * alphacum_traj[0]).flatten(1) @ RNDvecs

plt.figure(figsize=(9, 5))
ax1 = plt.subplot(1, 2, 1)
plt.plot(RNDproj)
plt.xlabel("Time")
plt.ylabel("PC Coefficient")
plt.title("Actual")
ax2 = plt.subplot(1, 2, 2)
for RNDvec, coef_0 in zip(RNDvecs.T, RNDproj[0, :]):
    Lambda = ((V_face.T @ RNDvec)**2 * cov_eigs).sum()
    proj_theory = xt_proj_coef(Lambda, alphacum_traj)  # *4
    plt.plot(proj_theory * coef_0, )
plt.xlabel("Time")
plt.ylabel("PC Coefficient")
plt.title("Theoretical")
plt.suptitle(f"PC Coefficients of Samples traj x(t) for {str(PC_range)}\n"
             f"Cov eigenvalues: {cov_eigs[PC_range[0]]:.3f}-{cov_eigs[PC_range[-1]]:.3f}")
# plt.legend([f"RND{i}" for i in PC_range], loc="right")
YLIM = min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax2.set_ylim(YLIM)
ax1.set_ylim(YLIM)
saveallforms(join(figdir, dataset_name), f"RNDProjcoef_theory_cmp_RND")
plt.show()
#%%
pred_x0_PCcoef = (pred_x0_imgs - imgmean).flatten(1) @ V_face
#%%
plt.figure()
plt.plot(pred_x0_PCcoef[:, 1800:1850])
plt.title("PC Coefficients of Projected Outcome \hat x_0(x_t)")
plt.xlabel("Time")
plt.ylabel("PC Coefficient")
# plt.legend([f"PC{i}" for i in range(10)])
# saveallforms(savedir, "PCcoef")
plt.show()

#%%
# show_imgrid(torch.cat([imgmean, pred_x0_imgs[0:1]]))
save_imgrid(torch.cat([imgmean, pred_x0_imgs[0:1]]), join(figdir, dataset_name, "imgmean_samples_cmp.jpg"), nrow=2)
#%%
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity


