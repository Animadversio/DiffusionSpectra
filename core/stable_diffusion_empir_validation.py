from os.path import join
import torch
from tqdm import tqdm, trange
exproot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_O2\PNDM"

sample_col = []
for RNDseed in trange(500, 2500):
    expdir = join(exproot, f"portrait_aristocrat-seed{RNDseed}_cfg7.5")
    save_dict = torch.load(join(expdir, "latents_noise_trajs.pt"))
    sample = save_dict['latents_traj'][-1].float()
    sample_col.append(sample)
    # print(list(save_dict))
    # raise ValueError

sample_col = torch.concatenate(sample_col)
#%%
test_samples = sample_col[-100:].flatten(1)
train_samples = sample_col[:-100].flatten(1)
#%%
# mean and PCA of train samples
train_mean = train_samples.mean(dim=0,keepdim=True)
train_centered = train_samples - train_mean
U, S, V = torch.svd(train_centered)
cov_eigs = S**2 / (train_samples.shape[0] - 1)
#%%
from diffusers import pipelines, StableDiffusionPipeline, PNDMScheduler, DDIMScheduler
# exproot = r"/home/binxuwang/insilico_exp/Diffusion_Hessian/StableDiffusion"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
# pipeline.to(torch.half)
def dummy_checker(images, **kwargs): return images, False

pipe.safety_checker = dummy_checker
#%%
pipe.scheduler.set_timesteps(51)
alphacum_traj = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
#%%
from core.ODE_analytical_lib import *
from diffusers import DDIMPipeline
mu_vec = train_mean.flatten(1)
xt_col = []
xt_pred_col = []
xt_mse_col = []
for RNDseed in trange(2400, 2500):
    expdir = join(exproot, f"portrait_aristocrat-seed{RNDseed}_cfg7.5")
    save_dict = torch.load(join(expdir, "latents_noise_trajs.pt"))
    xt_traj = save_dict['latents_traj'].float()
    residue_traj = save_dict['residue_traj'].float()
    pred_z0 = (xt_traj[:-1] -
               residue_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1, 1)) / \
              alphacum_traj.sqrt().view(-1, 1, 1, 1, 1)
    xT_vec = xt_traj[0].flatten(1)
    #%%
    # predict xt
    xt_traj_pred, _, _, _ = \
        xt_ode_solution(xT_vec, mu_vec, V, cov_eigs, alphacum_traj)
    # predict x0hat
    x0hatxt_traj_pred, _, _ = x0hat_ode_solution( \
        xT_vec, mu_vec, V, cov_eigs, alphacum_traj)
    xt_col.append(xt_traj)
    xt_pred_col.append(xt_traj_pred)
    xt_mse = ((xt_traj_pred - xt_traj[1:].flatten(1))**2).mean(dim=(1))
    xt_mse_col.append(xt_mse)
#%%

xt_mse = ((xt_traj_pred - xt_traj[1:].flatten(1))**2).mean(dim=(1))
# x0hat_mse = ((x0hatxt_traj_pred - pred_z0.flatten(1))**2).mean(dim=(1))
#%%
xt_mse_arr = torch.stack(xt_mse_col)
#%%
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms, save_imgrid
figdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\StableDiffusion_empir_valid"
plt.figure(figsize=(5.5, 4.5))
plt.plot(xt_mse_arr.T, alpha=0.1, color='k')
plt.plot(xt_mse_arr.mean(dim=0), color='r', lw=2)
plt.ylabel("MSE of xt", fontsize=16)
plt.xlabel("timestep", fontsize=16)
plt.title("MSE of xt vs timestep\n (StableDiffusion v1.5 PNDM cfg 7.5) \n 'a portrait of an aristocrat'", fontsize=16)
plt.tight_layout()
saveallforms(figdir, "SD_xt_mse_vs_timestep")
# plt.plot(x0hat_mse)
plt.show()
#%%
from core.diffusion_traj_analysis_lib import \
    denorm_std, denorm_var, denorm_sample_std, \
    latents_to_image, latentvecs_to_image
# xt_col = []
# xt_pred_col = []
xt_img_col = []
xt_pred_img_col = []
for xt_traj, xt_traj_pred in tqdm(zip(xt_col, xt_pred_col)):
    with torch.no_grad():
        xt_img = latentvecs_to_image(xt_traj[-1].half(), pipe)
        xt_pred_img = latentvecs_to_image(xt_traj_pred[-1:].half(), pipe)
    xt_img_col.append(xt_img)
    xt_pred_img_col.append(xt_pred_img)
#%%
from core.utils.plot_utils import save_imgrid
save_imgrid(torch.concatenate(xt_img_col, 0), join(figdir, "SD_xt_img_col.jpg"), nrow=10)
save_imgrid(torch.concatenate(xt_pred_img_col, 0), join(figdir, "SD_xt_pred_img_col.jpg"), nrow=10)
#%%
save_imgrid(torch.concatenate(xt_img_col[:10], 0), join(figdir, "SD_xt_img_onerow.jpg"), nrow=10)
save_imgrid(torch.concatenate(xt_pred_img_col[:10], 0), join(figdir, "SD_xt_pred_img_onerow.jpg"), nrow=10)

