#%%
import sys

import matplotlib.pyplot as plt

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
from ldm.models.diffusion.ddim import DDIMSampler

model = get_model()
sampler = DDIMSampler(model)
#%%
with torch.no_grad():
    z = model.first_stage_model.encode(torch.randn(32, 3, 256, 256).cuda()).cpu()
#%%
# define ImageNet dataset and dataloader
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# We then pass the original dataset and the indices we are interested in
ImageNet_root = "/home/binxu/Datasets/imagenet/train"
#%%
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

dataset = ImageFolder(ImageNet_root, transform=train_transform)
#%%
# find subset of ImageNet corresponding to class 1
#%%
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.imshow(ToPILImage()(dataset[-1][0]))
plt.show()
#%%
savedir = r"/home/binxu/DL_Projects/ldm-imagenet/latents_save"
for class_id in tqdm(range(1, 100)):
    class_mask = (torch.tensor(dataset.targets) == class_id)
    class_dataset = Subset(dataset, class_mask.nonzero().flatten())
    dataloader = DataLoader(class_dataset, batch_size=32, shuffle=False, num_workers=8)
    zs = []
    for i, (imgtsrs, labels) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            z = model.first_stage_model.encode(2 * imgtsrs.cuda() - 1.0).cpu()
        zs.append(z)

    zs = torch.cat(zs, dim=0)
    torch.save({"zs": zs}, join(savedir, f"class{class_id}_zs.pt"))
    # compute mean and covariance of zs
    z_flat = zs.flatten(1)
    z_mean = z_flat.mean(dim=0)
    # compute PCA of z_flat in torch
    U, S, V = torch.svd(z_flat - z_mean)
    torch.save({"mean": z_mean, "U": U, "S": S, "V": V}, join(savedir, f"class{class_id}_z_pca.pt"))
    # z_cov = torch.einsum("ij,ik->jk", z_flat - z_mean, z_flat - z_mean) / (z_flat.shape[0] - 1)


#%%
PCtsrs = V[:, :5].reshape(3, 64, 64, -1).permute(3, 0, 1, 2)
#%%
with torch.no_grad():
    PCimgs = model.first_stage_model.decode(50*PCtsrs.cuda()).cpu()
#%%
from torchvision.utils import make_grid, save_image
make_grid(PCimgs, nrow=5)
#%%
plt.imshow((1+make_grid(PCimgs, nrow=5).permute(1, 2, 0))/2)
plt.show()
