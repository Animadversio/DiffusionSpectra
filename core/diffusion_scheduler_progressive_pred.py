

import os
from os.path import join
import json
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler
from diffusers import PNDMPipeline, DDIMScheduler, UNet2DModel, PNDMScheduler
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
#% Utility functions for analysis
from core.diffusion_geometry_lib import proj2subspace, proj2orthospace, subspace_variance, \
    trajectory_geometry_pipeline, latent_PCA_analysis, latent_diff_PCA_analysis, diff_cosine_mat_analysis, PCA_data_visualize
from core.diffusion_traj_analysis_lib import compute_save_diff_imgs_diff, plot_diff_matrix
#%%
# model_id = "fusing/ddim-celeba-hq"

# model_id = "google/ddpm-cat-256"
# model_id = "google/ddpm-ema-cat-256"
# model_id = "google/ddpm-bedroom-256"
# model_id = "google/ddpm-ema-bedroom-256"
# model_id = "google/ddpm-church-256"
# model_id = "google/ddpm-ema-church-256"
# model_id = "google/ddpm-ema-celebahq-256"
# model_id = "google/ddpm-celebahq-256" # most popular
model_id = "google/ddpm-cifar10-32"
# model_id = "dimpo/ddpm-mnist"  # most popular
model_id_short = model_id.split("/")[-1]
# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipe.unet.requires_grad_(False).eval().to("cuda")#.half()

#%%
import matplotlib
matplotlib.use('Agg')
# use the interactive backend

# matplotlib.use('module://backend_interagg')
#%%
def sampling(unet, scheduler, batch_size=1, generator=None):
    noisy_sample = torch.randn(
        batch_size, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size,
        generator=generator, device=unet.device
    ).to(unet.device).to(unet.dtype)
    t_traj, sample_traj, residual_traj = [], [], []
    sample = noisy_sample
    sample_traj.append(sample.cpu().detach())
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        with torch.no_grad():
            residual = unet(sample, t).sample
        # 2. compute previous image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample
        residual_traj.append(residual.cpu().detach())
        sample_traj.append(sample.cpu().detach())
        t_traj.append(t)
    t_traj = torch.tensor(t_traj)
    t_traj = t_traj.long().cpu()
    sample_traj = torch.cat(sample_traj, dim=0)
    residual_traj = torch.cat(residual_traj, dim=0)
    return sample, sample_traj, residual_traj, t_traj

#%%
import platform
from diffusers import LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, \
    EulerDiscreteScheduler, DPMSolverMultistepScheduler
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_scheduler"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}_scheduler"
else:
    raise RuntimeError("Unknown system")


def denorm_sample_renorm(x, mu, std):
    return ((x - x.mean(dim=(1,2,3), keepdims=True)) / x.std(dim=(1,2,3), keepdims=True) * std + mu)


tsteps = 51
for nameCls, SamplerCls in [#("LMSDiscrete", LMSDiscreteScheduler,),
                            #("EulerDiscrete", EulerDiscreteScheduler,),
                            ("DDIM", DDIMScheduler,),
                            #("DPMSolverMultistep", DPMSolverMultistepScheduler,),
                            #("PNDM", PNDMScheduler,),
    ]:
    pipe.scheduler = SamplerCls.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(tsteps)
    for seed in range(200, 400):
        sample, sample_traj, residual_traj, t_traj = sampling(pipe.unet, pipe.scheduler, batch_size=1,
                                                              generator=torch.cuda.manual_seed(seed))
        #%%
        savedir = join(saveroot, nameCls, f"seed{seed}")
        os.makedirs(savedir, exist_ok=True)
        torch.save({"latents_traj": sample_traj,
                    "residue_traj" : residual_traj,
                    "t_traj": t_traj
                    }, join(savedir, "state_reservoir.pt"))
        json.dump({"tsteps": tsteps, "seed": seed}, open(join(savedir, "prompt.json"), "w"))

        images = (sample_traj[-1] / 2 + 0.5).clamp(0, 1)
        save_imgrid(images, join(savedir, f"samples_all.png"))
        #%%
        alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
        pred_x0 = (sample_traj[:-1] - residual_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / \
                  alphacum_traj.sqrt().view(-1, 1, 1, 1)
        pred_x0_imgs = (pred_x0 + 1) / 2

        save_imgrid(pred_x0_imgs, join(savedir, "proj_z0_vae_decode.jpg"), nrow=10, )
        # %%
        mean_fin = sample_traj[-1].mean()
        std_fin = sample_traj[-1].std()
        for lag in [1, 2, 3, 4, 5, 10]:
            print(f"lag {lag}")
            sample_diff = sample_traj[lag:] - sample_traj[:-lag]
            sample_diff_renorm = denorm_sample_renorm(sample_diff, mean_fin, std_fin)
            sampdif_img_traj = (sample_diff_renorm + 1) / 2
            save_imgrid(sampdif_img_traj, join(savedir, f"sample_diff_lag{lag}_stdnorm_vae_decode.jpg"), nrow=10, )

        # %%
        trajectory_geometry_pipeline(sample_traj, savedir, )
        diff_cosine_mat_analysis(sample_traj, savedir, )
        expvar_vec, U, D, V = latent_PCA_analysis(sample_traj, savedir, savestr="latent_traj")
        expvar_diff, U_diff, D_diff, V_diff = latent_diff_PCA_analysis(sample_traj, savedir,
                                                                       savestr="latent_diff")
        PCA_data_visualize(sample_traj, U, D, V, savedir, topcurv_num=8, topImg_num=16,
                           prefix="latent_traj")
        PCA_data_visualize(sample_traj, U_diff, D_diff, V_diff, savedir, topcurv_num=8, topImg_num=16,
                           prefix="latent_diff")
        expvar_noise, U_noise, D_noise, V_noise = latent_PCA_analysis(residual_traj, savedir,
                                                                    savestr="noise_pred",)
        PCA_data_visualize(residual_traj, U_noise, D_noise, V_noise, savedir, topcurv_num=8, topImg_num=16,
                           prefix="noise_pred")
        torch.save({"expvar": expvar_vec, "U": U, "D": D, "V": V},
                   join(savedir, "latent_PCA.pt"))
        torch.save({"expvar_diff": expvar_diff, "U_diff": U_diff, "D_diff": D_diff, "V_diff": V_diff},
                   join(savedir, "latent_diff_PCA.pt"))
        torch.save({"expvar": expvar_noise, "U": U_noise, "D": D_noise, "V": V_noise},
                   join(savedir, "noise_pred_PCA.pt"))
    #     break
    # break
        # %%
        # compute_save_diff_imgs_diff(savedir, range(0, 51, 5), sample_traj)
        # plot_diff_matrix(savedir, range(0, 51, 5), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
        #                  save_sfx="_img_stdnorm", tril=True)
        # # %%
        # compute_save_diff_imgs_diff(savedir, range(0, 16, 1), sample_traj)
        # plot_diff_matrix(savedir, range(0, 16, 1), diff_x_sfx="_img_stdnorm", step_x_sfx="_img_stdnorm",
        #                  save_sfx="_img_stdnorm_early0-15", tril=True)






#%%
pipe.scheduler.set_timesteps(51)
sample, sample_traj, residual_traj, t_traj = sampling(pipe.unet, pipe.scheduler, batch_size=1)

#%%
alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
pred_x0 = (sample_traj[:-1] - residual_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / alphacum_traj.sqrt().view(-1, 1, 1, 1)
pred_x0_imgs = (pred_x0 + 1) / 2
#%%
show_imgrid(pred_x0_imgs, nrow=10, )
#%%
plt.imshow(to_imgrid(pred_x0_imgs))
plt.axis("off")
plt.tight_layout()
plt.show()
#%%

def denorm_sample_mean_std(x):
    return ((x - x.mean(dim=(1,2,3), keepdims=True)) / x.std(dim=(1,2,3), keepdims=True) * 0.4 + 1) / 2


def denorm_sample_std(x):
    return ((x) / x.std(dim=(1,2,3), keepdims=True) * 0.4 + 1) / 2


#%%
show_imgrid(torch.clamp(denorm_sample_std(sample_traj[1:] - sample_traj[:-1]), 0, 1), nrow=10, figsize=(10, 10))
#%%
show_imgrid(torch.clamp(denorm_sample_mean_std(residual_traj[1:]-sample_traj[0:1]), 0, 1), nrow=10, figsize=(10, 10))
#%%
pipe.scheduler.set_timesteps(51)
#%%
pipe.scheduler.set_timesteps(51)
timesteps_orig = pipe.scheduler.timesteps
pipe.scheduler.timesteps = timesteps_orig # torch.cat((timesteps_orig[:25,], timesteps_orig[-1:]))
pipe.scheduler.num_inference_steps = len(pipe.scheduler.timesteps)
#%%

sample, sample_traj, residual_traj, t_traj = sampling(pipe.unet, pipe.scheduler, batch_size=1)
#%%

alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
pred_x0 = (sample_traj - residual_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / alphacum_traj.sqrt().view(-1, 1, 1, 1)
show_imgrid((pred_x0 + 1)/2, nrow=10, figsize=(10, 10))

#%% [markdown]


#%% Sampler setting.
plt.figure(figsize=(8, 3))
plt.subplot(1,3,1)
plt.plot(pipe.scheduler.betas[pipe.scheduler.timesteps])
plt.title("beta")
plt.subplot(1,3,2)
plt.plot(pipe.scheduler.alphas[pipe.scheduler.timesteps])
plt.title("alpha")
plt.subplot(1,3,3)
plt.plot(pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps])
plt.title("alpha_cumprod")
plt.suptitle("betas, alphas, alphas_cumprod")
plt.tight_layout()
plt.show()
#%% Dev zone
# #%%
# # save image
# # image[0].save("ddpm_generated_image.png")
# seed = 99
#
# # for seed in range(200, 400):
# latents_reservoir = []
# t_traj = []
#
# @torch.no_grad()
# def save_latents(i, t, latents):
#     latents_reservoir.append(latents.detach().cpu())
#     t_traj.append(t)
#
# tsteps = 51
# out = pipe(callback=save_latents, num_inference_steps=tsteps,
#            generator=torch.cuda.manual_seed(seed))
# latents_reservoir = torch.cat(latents_reservoir, dim=0)
# t_traj = torch.tensor(t_traj)
