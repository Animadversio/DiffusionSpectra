import os

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from os.path import join
from datasets import load_dataset
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage, make_grid_np
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
#%%
imgdir = r"F:\insilico_exps\Diffusion_traj\MNIST_PCA_theory"
imgdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ImageSpacePCA\MNIST"
outdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\TheoryValMNIST"

theory_col = []
empir_col = []
for seed in range(330,345):
    mtg_theory = plt.imread(join(imgdir, f"seed{seed}_x0hat_theory.png"))
    mtg_empir = plt.imread(join(imgdir, f"seed{seed}_xt_empir.png"))
    final_theory = crop_from_montage(mtg_theory, imgid=50, imgsize=32, pad=2)
    final_empir = crop_from_montage(mtg_empir, imgid=50, imgsize=32, pad=2)
    theory_col.append(final_theory)
    empir_col.append(final_empir)
#%%
plt.imsave(join(outdir, "theory_x0hat_T.png"), make_grid_np(theory_col, nrow=len(theory_col)))
plt.imsave(join(outdir, "empirical_x0hat_T.png"), make_grid_np(empir_col, nrow=len(empir_col)))
#%%
plt.imshow(make_grid_np(empir_col, nrow=len(theory_col)))
plt.show()

