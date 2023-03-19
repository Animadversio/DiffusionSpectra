# !git clone https://github.com/CompVis/latent-diffusion.git
# !git clone https://github.com/CompVis/taming-transformers
# !pip install -e ./taming-transformers
# !pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
# !pip install pytorch-lightning==1.7.7
# !pip install clip kornia
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
import torch
from omegaconf import OmegaConf
from os.path import join
from ldm.util import instantiate_from_config
ldm_root = "/home/binxu/Github/latent-diffusion"

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
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

classes = [25, 187, ]  # 448, 992  define classes to be sampled here
n_samples_per_class = 3

ddim_steps = 51
ddim_eta = 0.0  # note eta = 0.0 is the deterministic version of DDIM, no noise added except the starting point.
scale = 3.0  # for unconditional guidance

all_samples = list()

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
        )

        for class_label in classes:
            print(
                f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class * [class_label])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            x_T = torch.randn(n_samples_per_class, 3, 64, 64, device=model.device,
                              generator=torch.cuda.manual_seed(0))
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

            x_samples_ddim = model.decode_first_stage(samples_ddim).cpu()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                         min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)

# display as grid
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=n_samples_per_class)

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
Image.fromarray(grid.astype(np.uint8))
#%%
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(grid.astype(np.uint8))
plt.show()
#%%
z_traj = torch.stack(samp_traj["x_inter"], dim=0).cpu()
pred_z0_traj = torch.stack(samp_traj["pred_x0"], dim=0).cpu()
#%%

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
pred_x0_traj = decode_batch(model, pred_z0_traj[:, 0, :], batch_size=8)
#%%
import matplotlib.pyplot as plt
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

