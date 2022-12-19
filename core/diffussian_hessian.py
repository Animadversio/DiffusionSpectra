import torch
from tqdm import tqdm
from diffusers import PNDMPipeline, DDIMScheduler, UNet2DModel

import os
from os.path import join
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms
savedir = r"F:\insilico_exps\Diffusion_Hessian\cifar10-32"


def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    plt.imshow(image_pil)
    plt.show()
    # display(image_pil)


def visualize_tensor(tsr):
    image_processed = tsr.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_processed[0])
    plt.show()


# computation code
def sampling(model, scheduler, ):
    noisy_sample = torch.randn(
        1, model.config.in_channels, model.config.sample_size, model.config.sample_size
    ).to(model.device).half()
    t_traj, sample_traj, = [], []
    sample = noisy_sample

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        with torch.no_grad():
            residual = model(sample, t).sample

        # 2. compute previous image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample

        sample_traj.append(sample.cpu().detach())
        t_traj.append(t)
    return sample, sample_traj, t_traj


def compute_hessian(model, t, sample, hvp_batch=20):
    input_dim = sample.numel()
    sample_req_vec = sample.clone().detach().flatten().requires_grad_(True)
    sample_req = sample_req_vec.reshape(sample.shape)
    delta_sample = model(sample_req, t).sample
    hess = []
    for i in tqdm(range(0, input_dim, hvp_batch)):
        hess_part = torch.autograd.grad(delta_sample.flatten(), sample_req_vec,
                                        grad_outputs=torch.eye(input_dim, device="cuda")[i:i + hvp_batch, :],
                                        is_grads_batched=True, retain_graph=True, create_graph=False, )
                                        # this is a trick to avoid sum over outputs
        hess.append(hess_part[0])
    hess = torch.cat(hess, dim=0)
    return hess
#%%
# repo_id = "google/ddpm-celebahq-256"
repo_id = "google/ddpm-cifar10-32"
# repo_id = "nbonaker/ddpm-celeb-face-32/unet"
model = UNet2DModel.from_pretrained(repo_id)
model.requires_grad_(False).eval().half().to("cuda")
#%%
scheduler = DDIMScheduler.from_pretrained(repo_id)
# scheduler = PNDMScheduler.from_pretrained(model_id)
scheduler.set_timesteps(num_inference_steps=50)

#%% Visualization code
def SVec_to_imgtsr(SVecs, img_shape=(3, 32, 32), scale=1):
    imgtsr = ((SVecs.T.reshape(-1, *img_shape) * scale) + 1) / 2
    return imgtsr


def visualize_spectrum(S, savedir, name):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(S.cpu().numpy())
    plt.ylabel("Singular Value")
    plt.xlabel("Index")
    plt.subplot(1, 2, 2)
    plt.semilogy(S.cpu().numpy())
    plt.ylabel("Singular Value")
    plt.xlabel("Index")
    plt.tight_layout()
    saveallforms(savedir, "SingularValues"+ name)
    plt.show()


def visualize_SVecs_all(UorV, expdir, name="", scale=15,
                   num_per_page=400, num_per_row=20, reverse=True):
    UorV_show = torch.fliplr(UorV) if reverse else UorV
    for i in range(0, UorV.shape[1], num_per_page):
        save_image(SVec_to_imgtsr(UorV_show[:, i:i+num_per_page] * scale, ),
                   join(expdir, f"{name}_{i}.png"), nrow=num_per_row,)

#%%
savedir = r"F:\insilico_exps\Diffusion_Hessian\cifar10-32"
expdir = join(savedir, "exp%03d"%torch.randint(0, 1000, (1,)).item())
os.makedirs(expdir, exist_ok=True)
sample, _, t_traj = sampling(model, scheduler)
hess = compute_hessian(model, t_traj[-1], sample, hvp_batch=20)
U, S, V = torch.svd(hess.to(torch.float))
torch.save({"S": S, "U": U, "V": V, "H": hess}, join(expdir, "Hess_SUV.pt"))
save_image((sample+0.5)/2, join(expdir, "sample.png"))
visualize_spectrum(S, expdir, "")
visualize_spectrum(1 / S, expdir, "Inv")
visualize_SVecs_all(U, expdir, name="U", scale=15)
visualize_SVecs_all(V, expdir, name="V", scale=15)
#%%




#%%
#%% Dev zone
#%%
# torch.manual_seed(0)

noisy_sample = torch.randn(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
).to(model.device).half()
noisy_sample.shape
t_traj = []
sample_traj = []
sample = noisy_sample

for i, t in enumerate(tqdm(scheduler.timesteps)):
  # 1. predict noise residual
  with torch.no_grad():
      residual = model(sample, t).sample
  # 2. compute previous image and set x_t -> x_t-1
  sample = scheduler.step(residual, t, sample).prev_sample
  sample_traj.append(sample.cpu().detach())
  t_traj.append(t)
  # 3. optionally look at image
  if (i + 1) % 5 == 0:
      display_sample(sample, i + 1)

#%%
# display_sample(sample, i + 1)
display_sample(5*(sample_traj[95]-sample_traj[52]), i + 1)
#%%
show_imgrid(SVec_to_imgtsr(U[:, :100] * 10), nrow=10)
show_imgrid(SVec_to_imgtsr(V[:, :100] * 30), nrow=10)
show_imgrid(SVec_to_imgtsr(V[:, 200:400] * 30), nrow=10)
show_imgrid(SVec_to_imgtsr(V[:, 2400:2800] * 20), nrow=20)
show_imgrid(SVec_to_imgtsr(V[:, 2800:3000] * 10), nrow=20)
show_imgrid(SVec_to_imgtsr(V[:, 2800:3200] * 10), nrow=20)
#%%
input_dim = sample.numel()
sample_req_vec = sample.clone().detach().flatten().requires_grad_(True)
sample_req = sample_req_vec.reshape(sample.shape)
delta_sample = model(sample_req, 0).sample
#%%
hess = []
for i in tqdm(range(0,input_dim,)):
    hess_part = torch.autograd.grad(delta_sample.flatten(), sample_req_vec,
        grad_outputs=torch.eye(input_dim,device="cuda")[i, :], # this is a trick to avoid sum over outputs
        retain_graph=True, create_graph=False, )
    hess.append(hess_part[0])
hess = torch.stack(hess, dim=0)
# 1.5 G, 38 sec for 32x32x3 Hessian computation
#%%
hvp_batch = 20
hess = []
for i in tqdm(range(0, input_dim, hvp_batch)):
    hess_part = torch.autograd.grad(delta_sample.flatten(), sample_req_vec,
        grad_outputs=torch.eye(input_dim, device="cuda")[i:i+hvp_batch, :], # this is a trick to avoid sum over outputs
        is_grads_batched=True, retain_graph=True, create_graph=False, )
    hess.append(hess_part[0])
hess = torch.cat(hess, dim=0)
#%%
U,S,V = torch.svd(hess.to(torch.float))
#%%
visualize_tensor(15*U[:, -1].reshape(sample.shape))
#%%
#%%
# this is not working since it will explode the memory  ( x 3072 memory cost )
input_dim = sample_req_vec.size()[0]
hess_mat = torch.autograd.grad(delta_sample.flatten(), sample_req_vec,
    grad_outputs=torch.eye(input_dim,device="cuda"),
    is_grads_batched=True, retain_graph=True, create_graph=False, )
# this is a trick to avoid sum over outputs


#%%

#%% Dev zone
# model_id = "nbonaker/ddpm-celeb-face-32"
# load model and scheduler
model_id = "google/ddpm-cifar10-32"
# model_id = "nbonaker/ddpm-celeb-face-32"
# load model and scheduler
ddpm = PNDMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
# run pipeline in inference (sample random noise and denoise)
image = ddpm(num_inference_steps=100).images[0]

# save image
image.save("ddpm_generated_image.png")
