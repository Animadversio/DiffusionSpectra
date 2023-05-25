import json
import math
import os
from os.path import join
from tqdm import tqdm
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from diffusers import pipelines, StableDiffusionPipeline
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from core.utils.montage_utils import make_grid_np
from pathlib import Path

import platform
if platform.system() == "Windows":
    saveroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_perturb"
elif platform.system() == "Linux":
    saveroot = r"/home/binxuwang/insilico_exp/Diffusion_traj/StableDiffusion_perturb"
else:
    raise RuntimeError("Unknown system")
def make_save_montage_pert_T_scale(savedir, idxs, savesuffix, prefix="PC", tsteps=range(0, 51, 5),
                              scales=(-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0,), ):
    """ Util function to montage images on the plane spanned by perturb time and scale """
    os.makedirs(join(savedir, "summary"), exist_ok=True)
    for idx in tqdm(idxs):
        img_col = []
        for inject_step in tsteps:
            for pert_scale in scales:
                if pert_scale == 0.0:
                    img = plt.imread(join(savedir, f"sample_orig.png"))
                else:
                    if prefix == "PC":
                        img = plt.imread(join(savedir, f"sample_{prefix}{idx:02d}_T{inject_step:02d}_scale{pert_scale:.1f}.png"))
                    elif prefix == "RND":
                        img = plt.imread(join(savedir, f"sample_{prefix}{idx:03d}_T{inject_step:02d}_scale{pert_scale:.1f}.png"))
                img_col.append(img[:, :, :3])
        mtg = make_grid_np(img_col, nrow=len(scales))
        if prefix == "PC":
            plt.imsave(join(savedir, "summary", f"sample_mtg_{prefix}{idx:02d}{savesuffix}.jpg"), mtg)
        elif prefix == "RND":
            plt.imsave(join(savedir, "summary", f"sample_mtg_{prefix}{idx:03d}{savesuffix}.jpg"), mtg)
#%%
savedir = Path(saveroot) / "box_apple_bear-seed130"
make_save_montage_pert_T_scale(savedir, [2, 3, 4], "part", prefix="RND", tsteps=range(5, 20, 5),
                              scales=(-10, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0), )

