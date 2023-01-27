import os
import shutil
from os.path import join
import torch
from tqdm import tqdm
inroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_scheduler"
outroot = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\GuidanceSampler"
#%%
expdir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_scheduler\PNDM\portrait_aristocrat-seed100_cfg1.0"
data = torch.load(join(expdir, "latent_diff_PCA.pt")) # "latent_traj_PCA.pt"
list(data.keys())
#%%
#%% Compare the ExpVar of PCA
cfg = 1.0  # 7.5
expvar_col = {}
expvar_diff_col = {}
for cfg in [1.0, 7.5]:
    for nameCls in ["DDIM",
                    "PNDM",
                    "DPMSolverMultistep",
                    "LMSDiscrete",
                    "EulerDiscrete",
                    ]:
        expvar_col[(nameCls, cfg)] = []
        expvar_diff_col[(nameCls, cfg)] = []
        for prompt_brief in ["portrait_lightbulb",
                             "portrait_aristocrat", ]:
            for seed in tqdm(range(101, 115)):
                expdir = join(inroot, nameCls, f"{prompt_brief}-seed{seed}_cfg{cfg:.1f}")
                data = torch.load(join(expdir, "latent_diff_PCA.pt"))
                expvar_diff_col[(nameCls, cfg)].append(data["expvar_diff"])
                data = torch.load(join(expdir, "latent_PCA.pt"))
                expvar_col[(nameCls, cfg)].append(data["expvar"])
        expvar_col[(nameCls, cfg)] = torch.stack(expvar_col[(nameCls, cfg)])
        expvar_diff_col[(nameCls, cfg)] = torch.stack(expvar_diff_col[(nameCls, cfg)])
    #%%
torch.save({"expvar_col":expvar_col,
            "expvar_diff_col":expvar_diff_col}, join(outroot, f"expvar_SamplerCfg_cmp.pt"))
#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
#%%
# plt.subplots(figsize=(15, 6))
figh, axs = plt.subplots(2, 5, figsize=(15, 6))
for ci, cfg in enumerate([1.0, 7.5]):
    for si, nameCls in enumerate(["DDIM",
                    "PNDM",
                    "DPMSolverMultistep",
                    "LMSDiscrete",
                    "EulerDiscrete",
                    ]):
        # plt.plot(expvar_col[(nameCls, cfg)].mean(0), label=f"{nameCls} cfg={cfg}")
        # plt.plot(expvar_diff_col[(nameCls, cfg)].mean(0), label=f"{nameCls} cfg={cfg}")
        axs[ci, si].plot(expvar_col[(nameCls, cfg)].mean(0), label=f"trajectory")
        axs[ci, si].plot(expvar_diff_col[(nameCls, cfg)].mean(0), label=f"trajectory difference")
        # fill between 25% and 75% quantiles
        axs[ci, si].fill_between(range(len(expvar_col[(nameCls, cfg)].mean(0))),
                                 expvar_col[(nameCls, cfg)].quantile(0.05, 0),
                                 expvar_col[(nameCls, cfg)].quantile(0.95, 0), alpha=0.2)
        axs[ci, si].fill_between(range(len(expvar_diff_col[(nameCls, cfg)].mean(0))),
                                 expvar_diff_col[(nameCls, cfg)].quantile(0.05, 0),
                                 expvar_diff_col[(nameCls, cfg)].quantile(0.95, 0), alpha=0.2)
        axs[ci, si].set_title(f"{nameCls} cfg={cfg}")
        if si == 0:
            axs[ci, si].set_ylabel("Explained variance")
        if ci == 1:
            axs[ci, si].set_xlabel("PC")
        if si ==0 and ci == 1:
            axs[ci, si].legend()


plt.tight_layout()
plt.show()
#%%
from core.utils.plot_utils import saveallforms
saveallforms(outroot, "PCexpvar_SamplerCfg_synopsis", figh)



#%%
for cfg in [1.0, 7.5]:
    for nameCls in [
                    "PNDM",
                    ]:
        for prompt_brief in ["portrait_lightbulb",
                             "portrait_aristocrat", ]:
            for seed in tqdm(range(101, 115)):
                expdir = join(inroot, nameCls, f"{prompt_brief}-seed{seed}_cfg{cfg:.1f}")
                shutil.copy(join(expdir, "latent_diff_topPC_imgs_vae_decode.png"),
                            join(outroot, f"{nameCls}_{prompt_brief}_seed{seed}_cfg{cfg:.1f}_latent_diff_topPC_imgs_vae_decode.png"))
                shutil.copy(join(expdir, "sample.png"),
                            join(outroot, f"{nameCls}_{prompt_brief}_seed{seed}_cfg{cfg:.1f}_sample.png"))

