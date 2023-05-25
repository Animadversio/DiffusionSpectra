from pathlib import Path
from os.path import join
import torch
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import numpy as np
exactdir = r"F:\insilico_exps\Diffusion_traj\cifar_uncond_gmm_exact"
exactdir = r"F:\insilico_exps\Diffusion_traj\cifar_uncond_gmm_exact_normalized"
figdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\MNIST_CIFAR_gmm_simul_plot"
outdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\TheoryValCIFAR_Application"
""" Compare the distance between DDIM, GMM, and Unigauss """
#%%
x0_ddims = []
x0_exacts = []
x0_gmms = []
x0_unis = []
for RNDseed in trange(200):
    x0_ddim = plt.imread(Path(exactdir)/f"uncond_RND{RNDseed:03d}_DDIM.png")[:, :, :3]
    x0_exact = plt.imread(Path(exactdir)/f"uncond_RND{RNDseed:03d}_exact.png")[:, :, :3]
    x0_gmm = plt.imread(Path(exactdir)/f"uncond_RND{RNDseed:03d}_gmm.png")[:, :, :3]
    x0_uni = plt.imread(Path(exactdir)/f"uncond_RND{RNDseed:03d}_unigauss.png")[:, :, :3]
    # add to stacks
    x0_ddims.append(x0_ddim)
    x0_exacts.append(x0_exact)
    x0_gmms.append(x0_gmm)
    x0_unis.append(x0_uni)

x0_ddims = np.stack(x0_ddims)
x0_exacts = np.stack(x0_exacts)
x0_gmms = np.stack(x0_gmms)
x0_unis = np.stack(x0_unis)
#%%
ddim2exact = np.mean((x0_ddims - x0_exacts)**2, axis=(1,2,3))
ddim2gmm = np.mean((x0_ddims - x0_gmms)**2, axis=(1,2,3))
ddim2uni = np.mean((x0_ddims - x0_unis)**2, axis=(1,2,3))
#%%
# save row of montage
from core.utils.montage_utils import make_grid_np
# idx = np.random.randint(0, 200, size=10, dtype=int)
idx = np.arange(125, 140,)
x0_ddim_mtg = make_grid_np(list(x0_ddims[idx]), nrow=15, padding=1)
x0_exact_mtg = make_grid_np(list(x0_exacts[idx]), nrow=15, padding=1)
x0_gmm_mtg = make_grid_np(list(x0_gmms[idx]), nrow=15, padding=1)
x0_gauss_mtg = make_grid_np(list(x0_unis[idx]), nrow=15, padding=1)
plt.imsave(Path(outdir)/"CIFAR10_uncond_DDIM.png", x0_ddim_mtg)
plt.imsave(Path(outdir)/"CIFAR10_uncond_exact.png", x0_exact_mtg)
plt.imsave(Path(outdir)/"CIFAR10_uncond_gmm.png", x0_gmm_mtg)
plt.imsave(Path(outdir)/"CIFAR10_uncond_unigauss.png", x0_gauss_mtg)
#%%
# show all mtgs
plt.figure(figsize=[12, 3])
plt.subplot(411)
plt.imshow(x0_exact_mtg);plt.axis("off")
plt.subplot(412)
plt.imshow(x0_ddim_mtg);plt.axis("off")
plt.subplot(413)
plt.imshow(x0_gmm_mtg);plt.axis("off")
plt.subplot(414)
plt.imshow(x0_gauss_mtg);plt.axis("off")
plt.tight_layout()
plt.show()
#%%
# paired strip plot, add paired lines for each row
import seaborn as sns
import pandas as pd
from core.utils.plot_utils import saveallforms
df = pd.DataFrame({"exact":ddim2exact, "gmm":ddim2gmm, "uni":ddim2uni})
df.to_csv(join(figdir, "CIFAR10_uncond_gmm_gauss_exact_DDIM_cmp.csv"))
df.to_csv(join(outdir, "CIFAR10_uncond_gmm_gauss_exact_DDIM_cmp.csv"))
df = df.melt()
plt.figure(figsize=[4, 5])
sns.stripplot(data=df, x="variable", y="value", jitter=0.2, alpha=0.35)
sns.pointplot(data=df, x="variable", y="value", join=False, color="black",
              capsize=0.2, errwidth=1)
plt.ylabel("MSE")
plt.xlabel("Score Model")
plt.title("CIFAR10 Uncond. analytical vs DDIM")
plt.tight_layout()
saveallforms([figdir,outdir], "CIFAR10_uncond_gmm_gauss_exact_DDIM_cmp")
plt.show()

#%%
# paired strip plot, add paired lines for each row
xjit = np.random.normal(0, 0.1, size=ddim2exact.shape)
plt.figure(figsize=[4, 5])
plt.plot(np.arange(3)[:,None]+xjit[None,:],
         np.stack([ddim2uni, ddim2gmm, ddim2exact, ], axis=0),
         color="black", alpha=0.05)
plt.scatter(xjit, ddim2uni, s=25, alpha=0.35, label="Gaussian")
plt.scatter(xjit+1, ddim2gmm, s=25, alpha=0.35, label="GMM")
plt.scatter(xjit+2, ddim2exact, s=25, alpha=0.35, label="Exact")
plt.errorbar([0, 1, 2], [np.mean(ddim2uni), np.mean(ddim2gmm), np.mean(ddim2exact)],
                yerr=[np.std(ddim2uni)/np.sqrt(len(ddim2uni)), np.std(ddim2gmm)/np.sqrt(len(ddim2gmm)), np.std(ddim2exact)/np.sqrt(len(ddim2exact))],
                fmt="o", capsize=10, color="black", alpha=0.5, lw=1, capthick=1)
plt.xticks([0, 1, 2], ["Gaussian", "GMM", "Exact"], fontsize=12)
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Score Model")
plt.title("CIFAR10 Uncond. analytical vs DDIM\nimage prediction")
plt.tight_layout()
saveallforms([figdir,outdir], "CIFAR10_uncond_gmm_gauss_exact_DDIM_paired_strip")
plt.show()
#%%
# paired t test for each row, with mean and sem of each cole
from scipy.stats import ttest_rel
print("exact vs gmm", ttest_rel(ddim2exact, ddim2gmm),)
print("exact vs uni", ttest_rel(ddim2exact, ddim2uni))
print("gmm vs uni", ttest_rel(ddim2gmm, ddim2uni))
#%%
