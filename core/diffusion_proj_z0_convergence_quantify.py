import platform
from os.path import join
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lpips import LPIPS
from core.utils.montage_utils import crop_from_montage, crop_all_from_montage
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid

# exportroot = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffusionConvergence"
Dist = LPIPS(net='squeeze').cuda().eval()
Dist.requires_grad_(False)
#%%
import matplotlib
matplotlib.use('Agg')
# use the interactive backend
# matplotlib.use('module://backend_interagg')
#%%
dist_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffusionConvergence"
#%%
def compute_lpips_dist_from_mtg(mtg_img, final_img, savename, savedir,
                                totalnum=52, imgsize=512, pad=2):
    projimg_col = crop_all_from_montage(mtg_img, totalnum, imgsize=imgsize, pad=pad)
    projimg_tsr = np.stack(projimg_col)
    finalimg_tsr = final_img[None, :, :, :]
    projimg_tsr = torch.tensor(projimg_tsr).permute(0, 3, 1, 2).float() / 255
    finalimg_tsr = torch.tensor(finalimg_tsr).permute(0, 3, 1, 2).float() / 255
    Dist.spatial = False
    distvec = Dist(projimg_tsr.cuda(), finalimg_tsr.cuda()).detach().cpu()
    Dist.spatial = True
    distmaps = Dist(projimg_tsr.cuda(), finalimg_tsr.cuda()).detach().cpu()
    save_imgrid(distmaps,
                join(savedir, f"{savename}_proj_z0_diff2final.jpg"),
                nrow=10, )
    torch.save({"distvec": distvec},  # "distmaps": distmaps,
               join(savedir, f"{savename}_proj_z0_distsave.pt"))

    plt.figure(figsize=(6, 4))
    plt.plot(distvec[:, 0, 0, 0].detach().cpu().numpy())
    plt.xlabel("time step")
    plt.ylabel("LPIPS distance")
    plt.title(f"Distance between projected decoded z0 hat and final image")
    plt.savefig(
        join(savedir, f"{savename}_proj_z0_dist2final_trace.jpg"))
    plt.show()
    return distvec, distmaps
#%%
padding = 2
imgsize = 512
name_Sampler = "PNDM"
model_id_short = "StableDiffusion"
export_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\StableDiffusionTraj"
os.makedirs(join(dist_root, model_id_short), exist_ok=True)
prompt_dir_pair = [
    ("a portrait of an aristocrat", "portrait_aristocrat"),
    ("a portrait of an light bulb", "portrait_lightbulb"),
    ("a large box containing an apple and a toy teddy bear", "box_apple_bear"),
    ("a photo of a cat sitting with a dog on a cozy couch", "cat_dog_couch"),
    ("a CG art of a brain composed of eletronic wires and circuits", "brain_wire_circuits"),
    ("a handsome cat dancing Tango with a female dancer in Monet style", "cat_tango_dancer"),
    ("a bug crawling on a textbook under a bright light, photo", "bug_book_photo"),
]
for prompt, dirname in prompt_dir_pair:
    for seed in range(100, 150):
        mtg_img = plt.imread(join(export_root, f"{dirname}_{name_Sampler}_seed{seed}_proj_z0_vae_decode_new.jpg", ))
        final_img = plt.imread(join(export_root, f"{dirname}_{name_Sampler}_seed{seed}_samples.jpg"))
        distvec, distmaps = compute_lpips_dist_from_mtg(mtg_img, final_img,
                f"{dirname}_{name_Sampler}_seed{seed}", savedir=join(dist_root, model_id_short),imgsize=512)

#%%
padding = 2
imgsize = 256
name_Sampler = "DDIM"
model_id_short = "ddpm-celebahq-256"
export_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\FaceTrajectory"
os.makedirs(join(dist_root, model_id_short), exist_ok=True)
for seed in range(150, 300):
    mtg_img = plt.imread(join(export_root, f"{name_Sampler}_seed{seed}_proj_z0_vae_decode.jpg", ))
    final_img = plt.imread(join(export_root, f"{name_Sampler}_seed{seed}_samples_all.jpg"))
    distvec, distmaps = compute_lpips_dist_from_mtg(mtg_img, final_img,
        f"{name_Sampler}_seed{seed}", savedir=join(dist_root, model_id_short), imgsize=256, totalnum=51)

#%%
padding = 2
imgsize = 256
name_Sampler = "DDIM"
model_id_short = "ddpm-church-256"
export_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\ChurchTrajectory"
os.makedirs(join(dist_root, model_id_short), exist_ok=True)
for seed in range(125, 150):
    mtg_img = plt.imread(join(export_root, f"{name_Sampler}_seed{seed}_proj_z0_vae_decode.jpg", ))
    final_img = plt.imread(join(export_root, f"{name_Sampler}_seed{seed}_samples_all.jpg"))
    distvec, distmaps = compute_lpips_dist_from_mtg(mtg_img, final_img,
        f"{name_Sampler}_seed{seed}", savedir=join(dist_root, model_id_short), imgsize=256, totalnum=51)

#%%
padding = 2
imgsize = 32
name_Sampler = "DDIM"
model_id_short = "ddpm-mnist"
export_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\MNISTTrajectory"
os.makedirs(join(dist_root, model_id_short), exist_ok=True)
for seed in range(200, 400):
    mtg_img = plt.imread(join(export_root, f"{name_Sampler}_seed{seed}_proj_z0_vae_decode.jpg", ))
    final_img = plt.imread(join(export_root, f"{name_Sampler}_seed{seed}_samples_all.jpg"))
    distvec, distmaps = compute_lpips_dist_from_mtg(mtg_img, final_img,
        f"{name_Sampler}_seed{seed}", savedir=join(dist_root, model_id_short), imgsize=32, totalnum=51)

#%%
padding = 2
imgsize = 32
name_Sampler = "DDIM"
model_id_short = "ddpm-cifar10-32"
export_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\CIFARTrajectory"
os.makedirs(join(dist_root, model_id_short), exist_ok=True)
for seed in range(200, 400):
    mtg_img = plt.imread(join(export_root, f"{name_Sampler}_seed{seed}_proj_z0_vae_decode.jpg", ))
    final_img = plt.imread(join(export_root, f"{name_Sampler}_seed{seed}_samples_all.jpg"))
    distvec, distmaps = compute_lpips_dist_from_mtg(mtg_img, final_img,
        f"{name_Sampler}_seed{seed}", savedir=join(dist_root, model_id_short), imgsize=32, totalnum=51)

#%%

# show_imgrid(distmaps, nrow=10,)


