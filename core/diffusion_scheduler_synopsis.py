import os
import platform
from diffusers import LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, \
    EulerDiscreteScheduler, DPMSolverMultistepScheduler
from os.path import join
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms, save_imgrid, to_imgrid, make_grid
from core.utils.montage_utils import build_montages, make_grid_np

model_id_short = "StableDiffusion"
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_scheduler"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}_scheduler"
else:
    raise RuntimeError("Unknown system")
#%%
import numpy as np
def uniformize_img_size(img_col, size=None):
    if len(set([img.shape[:2] for img in img_col])) == 1:
        return img_col
    if size is None:
        # use the biggest size
        size = max([img.shape[:2] for img in img_col])
        # if all images are the same size, then use that size
    # pad the image to the size
    img_col = [np.pad(img, ((0, size[0] - img.shape[0]), (0, size[1] - img.shape[1]), (0, 0)), "constant")
                for img in img_col]
    return img_col
#%%
figure_names = [
    "cosine_mat_latent_diff_lag1.png",
    "cosine_mat_latent_diff_lag2.png",
    "cosine_trace_w_init_end_latent.png",
    "cosine_trace_w_init_end_latent_diff_lag1.png",
    "latent_trajectory_2d_proj.png",
    "latent_diff_2d_proj.png",
    "latent_traj_topPC_imgs_vae_decode.png",
    "latent_diff_PCA_expvar.png",
    "latent_diff_PCA_projcurve.png",
    "latent_diff_PCA_projcurve_norm1.png",
    "latent_diff_PC1_PC2_proj.png",
    "latent_diff_topPC_imgs_vae_decode.png",
    "noise_pred_traj_PCA_expvar.png",
    "noise_pred_traj_PCA_projcurve.png",
    "noise_pred_traj_PCA_projcurve_norm1.png",
    "noise_pred_traj_topPC_imgs_vae_decode.png",
    "sample.png",
    "proj_z0_vae_decode.png",
]
import os
prompt_brief = "portrait_aristocrat"
for seed in range(100, 115):
    os.makedirs(join(saveroot, "synopsis", f"{prompt_brief}-seed{seed}"), exist_ok=True)
    for fignm in figure_names:
        img_col = []
        for guidance in [1.0, 7.5]:
            for nameCls, SamplerCls in [("DDIM", DDIMScheduler,),
                                        ("PNDM", PNDMScheduler,),
                                        ("DPMSolverMultistep", DPMSolverMultistepScheduler,),
                                        ("LMSDiscrete", LMSDiscreteScheduler,),
                                        ("EulerDiscrete", EulerDiscreteScheduler,),
                                         ]:
                savedir = join(saveroot, nameCls, f"{prompt_brief}-seed{seed}_cfg{guidance}")
                img = plt.imread(join(savedir, fignm))
                img_col.append(img[:, :, :3])
        img_col = uniformize_img_size(img_col, size=None)
        mtg = make_grid_np(img_col, nrow=5, padding=8)
        plt.imsave(join(saveroot, "synopsis", f"{prompt_brief}-seed{seed}",
                        fignm.split(".")[0]+"_syn.png"), mtg)
        # plt.imshow(mtg)
        # plt.axis("off")
        # plt.tight_layout()
        # plt.show()

#%%
