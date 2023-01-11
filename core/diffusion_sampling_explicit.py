import torch
from tqdm import tqdm
from diffusers import PNDMPipeline, DDIMScheduler, UNet2DModel, PNDMScheduler
from diffusers import DiffusionPipeline
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms


# computation code
def sampling(unet, scheduler, batch_size=1):
    noisy_sample = torch.randn(
        batch_size, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size
    ).to(unet.device).to(unet.dtype)
    t_traj, sample_traj, residual_traj = [], [], []
    sample = noisy_sample

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        with torch.no_grad():
            residual = unet(sample, t).sample

        # 2. compute previous image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample

        residual_traj.append(residual.cpu().detach())
        sample_traj.append(sample.cpu().detach())
        t_traj.append(t)
    return sample, sample_traj, t_traj
#%%
repo_id = "google/ddpm-cifar10-32"  # Note this model has self-attention in it.
# repo_id = "nbonaker/ddpm-celeb-face-32"
model = UNet2DModel.from_pretrained(repo_id)
model.requires_grad_(False).eval().to("cuda")#.half()
scheduler = DDIMScheduler.from_pretrained(repo_id)
scheduler.set_timesteps(num_inference_steps=100, )
#%%
sample, sample_traj, t_traj = sampling(model, scheduler, batch_size=64)
show_imgrid((sample + 1) / 2)
sample_traj = torch.stack(sample_traj)