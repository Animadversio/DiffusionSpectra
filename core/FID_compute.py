import sys
import os
from os.path import join
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from pytorch_gan_metrics import get_inception_score, get_fid
from pytorch_gan_metrics.utils import ImageDataset
from pytorch_gan_metrics.core  import torch_cov, get_inception_feature, calculate_inception_score, calculate_frechet_distance
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from core.utils import saveallforms, showimg, show_imgrid, save_imgrid
if sys.platform == "linux" and os.getlogin() == 'binxuwang':
    savedir = "/home/binxuwang/DL_Projects/GAN-fids"
else:
    savedir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\hybrid_FID_scores"
#%%
# img_size = 256
# INroot = r"E:\Datasets\imagenet-valid\valid"
# transform = Compose([ToTensor()])
# # dataset = ImageDataset(r"E:\Datasets\imagenet-valid\valid")
# INdataset = ImageDataset(root=INroot, transform=transform)

#%%
# create a CIFAR10 dataset without the labels
class CIFAR10NoLabels(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.cifar10 = CIFAR10(root, train=train, transform=transform,
                               target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, _ = self.cifar10[index]  # Ignore label
        return img

    def __len__(self):
        return len(self.cifar10)

class CropTransform:
    """Crop a PIL Image.

    Args:
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    """

    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return transforms.functional.crop(img, self.top, self.left, self.height, self.width)


datalabel = "CIFAR10"
transform = Compose([ToTensor()])
CFdataset = CIFAR10NoLabels(root=r"E:\Datasets", transform=transform)
#%% INet
batch_size = 256
num_workers = 0  # os.cpu_count()
loader = DataLoader(CFdataset, batch_size=batch_size, num_workers=num_workers)
acts, probs = get_inception_feature(
    loader, dims=[2048, 1008], use_torch=False, verbose=True)
mu = torch.mean(torch.from_numpy(acts), dim=0).cpu().numpy()
sigma = torch_cov(torch.from_numpy(acts), rowvar=False).cpu().numpy()
np.savez_compressed(Path(savedir)/rf"{datalabel}_inception_stats.npz", mu=mu, sigma=sigma)
inception_score, IS_std = calculate_inception_score(torch.from_numpy(probs), 10, use_torch=True)
np.savez_compressed(Path(savedir)/rf"{datalabel}_IS_stats.npz", IS=inception_score, IS_std=IS_std)
# Inception score: 211.13711547851562 +- 3.3677103519439697
print(f"Inception Score {inception_score:.4f}+-{IS_std:.4f}")
#%%
datalabel = "CIFAR10"
with np.load(join(savedir, f"{datalabel}_inception_stats.npz")) as f:
    mu_CIFAR = f["mu"]
    sigma_CIFAR = f["sigma"]
#%%


hybrid_samples_root = r"F:\insilico_exps\Diffusion_traj\cifar10_PCA_theory_hybrid"
cropi = 2
transform = Compose([CropTransform(top=0, left=32 * cropi,
                                   height=32, width=32),
                     ToTensor()])
hyrbiddataset = ImageDataset(root=hybrid_samples_root, transform=transform)
#%%
# inspect image
plt.imshow(hyrbiddataset[0].permute(1, 2, 0))
plt.show()
#%%
import pandas as pd
from tqdm import tqdm, trange
skipstep_list = [0, 10, 20, 25, 30, 40, 50]
# cropi = 2
batch_size = 256
num_workers = 0
stats_col = []
for cropi, skipstep in tqdm(enumerate(skipstep_list)):
    datalabel = f"CIFAR10_hybrid_skip{skipstep}"
    transform = Compose([CropTransform(top=0, left=32 * cropi,
                                       height=32, width=32),
                         ToTensor()])
    hyrbiddataset = ImageDataset(root=hybrid_samples_root, transform=transform)
    loader_hybrid = DataLoader(hyrbiddataset, batch_size=batch_size, num_workers=num_workers)
    acts, probs = get_inception_feature(
        loader_hybrid, dims=[2048, 1008], use_torch=True, verbose=True)
    # acts = torch.from_numpy(acts)
    # probs = torch.from_numpy(probs)
    mu = torch.mean(acts, dim=0).cpu().numpy()
    sigma = torch_cov(acts, rowvar=False).cpu().numpy()
    np.savez_compressed(Path(savedir)/rf"{datalabel}_inception_stats.npz", mu=mu, sigma=sigma)
    inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
    print(f"Inception Score {inception_score:.4f}+-{IS_std:.4f}")
    fid_w_CIFAR = calculate_frechet_distance(mu, sigma, mu_CIFAR, sigma_CIFAR, eps=1e-6)
    print("FID", fid_w_CIFAR)
    np.savez_compressed(Path(savedir) / rf"{datalabel}_IS_FID_stats.npz",
                        IS=inception_score, IS_std=IS_std, FID=fid_w_CIFAR)
    stats_col.append({"skipstep": skipstep, "IS": inception_score, "IS_std": IS_std, "FID": fid_w_CIFAR})
#%%
stats_df = pd.DataFrame(stats_col)
stats_df.to_csv(Path(savedir)/"CIFAR10_hybrid_IS_FID_stats_synopsis.csv")
#%%
import pickle as pkl
# from core.utils.dnnlib_utils import open_url
detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
detector_kwargs = dict(return_features=True)
feature_dim = 2048
device = "cuda"
with open_url(detector_url, ) as f:
    detector_net = pkl.load(f).to(device)
    #%%
# imageset_str = "FC6_std4"
# FG4_fun = lambda batch_size: \
#         FG.visualize(4 * torch.randn(batch_size, 4096, device="cuda"))
# FC6_loader = GANDataloader(FG4_fun, batch_size=40, total_imgnum=50000)
# with torch.no_grad():
#     acts, probs = get_inception_feature(FC6_loader, dims=[2048, 1008], use_torch=True, verbose=True)
# mu = torch.mean(acts, dim=0).cpu().numpy()
# sigma = torch_cov(acts, rowvar=False).cpu().numpy()
# np.savez_compressed(join(savedir, f"{imageset_str}_inception_stats.npz"), mu=mu, sigma=sigma)
# inception_score, IS_std = calculate_inception_score(probs, 10, use_torch=True)
# np.savez(join(savedir, f"{imageset_str}_IS_stats.npz"), IS=inception_score, IS_std=IS_std)
# fid_w_INet = calculate_frechet_distance(mu, sigma, mu_INet, sigma_INet, eps=1e-6)
# print(imageset_str)
# print("FID", fid_w_INet)
# print("Inception Score", inception_score, IS_std)