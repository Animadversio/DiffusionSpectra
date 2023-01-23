
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


