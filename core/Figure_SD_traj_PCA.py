import os
import shutil
from os.path import join
inroot = r"F:\insilico_exps\Diffusion_traj\StableDiffusion_scheduler"
outroot = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\SD_PCA_vis"

fignms = [
    "noise_text_traj_topPC_imgs_vae_decode.png",
    "noise_uncond_traj_topPC_imgs_vae_decode.png",
    "noise_pred_traj_topPC_imgs_vae_decode.png",
    "latent_traj_topPC_imgs_vae_decode.png",
    "latent_diff_topPC_imgs_vae_decode.png",
    "sample.png",
]
expnms = [
    # r"PNDM\portrait_aristocrat-seed108_cfg7.5",
    # r"PNDM\portrait_aristocrat-seed105_cfg7.5",
    # r"PNDM\portrait_lightbulb-seed112_cfg7.5",
    r"PNDM\portrait_aristocrat-seed111_cfg7.5"
]
for expnm in expnms:
    os.makedirs(join(outroot, expnm), exist_ok=True)
    for fignm in fignms:
        shutil.copy2(join(inroot, expnm, fignm), join(outroot, expnm))