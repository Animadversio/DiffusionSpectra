#%%
import sys
sys.path.append("/home/binxu/Github/latent-diffusion")
sys.path.append('/home/binxu/Github/taming-transformers')
# from taming.models import vqgan
#%%
# %cd /home/binxu/Github/latent-diffusion
#
# !mkdir -p models/ldm/cin256-v2/
# !wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
#%%
# %cd /home/binxu/Github/latent-diffusion
#%%
#@title loading utils
import os
import torch
from omegaconf import OmegaConf
from os.path import join
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
#%%
from ldm.models.diffusion.ddim import DDIMSampler

model = get_model()
sampler = DDIMSampler(model)
#%%

import os
from tqdm import tqdm, trange
from os.path import join
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

@torch.no_grad()
def decode_batch(model, zs, batch_size=5):
    with model.ema_scope():
        x = []
        for i in range(0, zs.shape[0], batch_size):
            x.append(model.decode_first_stage(zs[i:i + batch_size]).cpu())
        x = torch.cat(x, dim=0)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    return x
#%%
if os.environ["USER"] == "binxu":
    saveroot = "/home/binxu/insilico_exps/Diffusion_traj/ldm_imagenet"
elif os.environ["USER"] == "biw905":
    saveroot = "/n/scratch3/users/b/biw905/Diffusion_traj/ldm_imagenet"


classes = range(600, 1000)  # 448, 992  define classes to be sampled here
# classes = range(100, 1000)  # 448, 992  define classes to be sampled here
classes = range(325, 500)  # 448, 992  define classes to be sampled here
classes = range(500, 600)  # 448, 992  define classes to be sampled here
RNDrng = range(0, 10)  # define random seeds here
n_samples_per_class = 1

ddim_steps = 50
ddim_eta = 0.0
scale = 3.0  # for unconditional guidance

# all_samples = list()

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
        )

        for class_label in classes:
            xc = torch.tensor(n_samples_per_class * [class_label])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            print(
                f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            for RNDseed in tqdm(RNDrng):
                savedir = join(saveroot, "DDIM", f"class{class_label:03d}_seed{RNDseed:03d}")
                os.makedirs(savedir, exist_ok=True)
                print(f"Class {class_label}  RNDseed={RNDseed}")

                x_T = torch.randn(n_samples_per_class, 3, 64, 64, device=model.device,
                                  generator=torch.cuda.manual_seed(RNDseed))
                samples_ddim, samp_traj = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 x_T=x_T,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 log_every_t=1,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta)

                x_samples_ddim = decode_batch(model, samples_ddim, batch_size=10)
                z_traj = torch.stack(samp_traj["x_inter"], dim=0)
                pred_z0_traj = torch.stack(samp_traj["pred_x0"], dim=0)
                pred_x0_traj = decode_batch(model, pred_z0_traj[:, 0, :], batch_size=10)
                xt_traj = decode_batch(model, z_traj[:, 0, :], batch_size=10)
                save_image(x_samples_ddim, join(savedir, f"samples.png"))
                save_image(make_grid(pred_x0_traj, nrow=10),
                           join(savedir, f"pred_x0_decode.jpg"))
                save_image(make_grid(xt_traj, nrow=10),
                            join(savedir, f"xt_decode.jpg"))
                torch.save({"z_traj": z_traj,
                            "pred_z0_traj": pred_z0_traj,
                            "t_traj": sampler.ddim_timesteps},
                           join(savedir, "state_traj.pt"))






#%%
# display as grid
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=n_samples_per_class)

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
Image.fromarray(grid.astype(np.uint8))
#%%
z_traj = torch.stack(samp_traj["x_inter"], dim=0).cpu()
pred_z0_traj = torch.stack(samp_traj["pred_x0"], dim=0).cpu()
#%%

#%%
pred_x0_traj = decode_batch(model, pred_z0_traj[:, 0, :], batch_size=8)
#%%
#%%
pred_x0_traj = decode_batch(model, pred_z0_traj[:, 5, :].cuda(), batch_size=10)
xt_traj = decode_batch(model, z_traj[:, 5, :].cuda(), batch_size=10)
#%%
grid_pred_x0 = make_grid(pred_x0_traj, nrow=10).permute(1, 2, 0)
grid_xt = make_grid(xt_traj, nrow=10).permute(1, 2, 0)
# grid = rearrange(grid, 'c h w -> h w c')
plt.figure()
plt.imshow(grid_pred_x0)
plt.axis('off')
plt.tight_layout()
plt.show()
plt.figure()
plt.imshow(grid_xt)
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
plt.imshow(grid.astype(np.uint8))
plt.show()

