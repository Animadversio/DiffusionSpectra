"""Simple script to export figures to a different format and combine """
import os
import platform
from os.path import join
import matplotlib.pyplot as plt
import torch
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from tqdm import tqdm
#%%

"""Separate a path into file name and extension"""
def change_format(path, new_ext):
    """Change the file format of a file"""
    name, ext = os.path.splitext(path)
    return name + "." + new_ext
#%%
model_id_short = "ddpm-celebahq-256"
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_scheduler"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}_scheduler"
else:
    raise RuntimeError("Unknown system")
export_root = rf"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\{model_id_short}"
save_fmt = "jpg"
name_Sampler = "DDIM"
exp_fignames = ["proj_z0_vae_decode.png",
                "sample_diff_lag1_stdnorm_vae_decode.png",
                "samples_all.png",]
for seed in range(210, 300):
    savedir = join(saveroot, name_Sampler, f"seed{seed}")
    for fignm in exp_fignames:
        if os.path.exists(join(savedir, fignm)):
            img = plt.imread(join(savedir, fignm))
        elif os.path.exists(change_format(join(savedir, fignm), "jpg")):
            img = plt.imread(change_format(join(savedir, fignm), "jpg"))
        else:
            raise RuntimeError(f"Can't find {fignm} in {savedir}")
        plt.imsave(join(export_root,
            change_format(f"{name_Sampler}_seed{seed}_{fignm}", save_fmt)), img)

for seed in tqdm(range(125, 300)):
    savedir = join(saveroot, name_Sampler, f"seed{seed}")
    traj_data = torch.load(join(savedir, "state_reservoir.pt"))
    save_imgrid((1 + traj_data["latents_traj"])/2,
                join(export_root, f"{name_Sampler}_seed{seed}_sample_traj.jpg"), nrow=10)
    # save_imgrid(traj_data["residue_traj"],
    #             join(export_root, f"{name_Sampler}_seed{seed}_sample_traj.jpg"), nrow=10)
#%%
model_id_short = "ddpm-church-256"
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_scheduler"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}_scheduler"
else:
    raise RuntimeError("Unknown system")
export_root = rf"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\{model_id_short}"
name_Sampler = "DDIM"
save_fmt = "jpg"
exp_fignames = ["proj_z0_vae_decode.png",
                "sample_diff_lag1_stdnorm_vae_decode.png",
                "samples_all.png",]

for seed in range(125, 150):
    savedir = join(saveroot, name_Sampler, f"seed{seed}")
    for fignm in exp_fignames:
        if os.path.exists(join(savedir, fignm)):
            img = plt.imread(join(savedir, fignm))
        elif os.path.exists(change_format(join(savedir, fignm), "jpg")):
            img = plt.imread(change_format(join(savedir, fignm), "jpg"))
        else:
            raise RuntimeError(f"Can't find {fignm} in {savedir}")
        plt.imsave(join(export_root,
            change_format(f"{name_Sampler}_seed{seed}_{fignm}", save_fmt)), img)

for seed in tqdm(range(125, 150)):
    savedir = join(saveroot, name_Sampler, f"seed{seed}")
    traj_data = torch.load(join(savedir, "state_reservoir.pt"))
    save_imgrid((1 + traj_data["latents_traj"])/2,
                join(export_root, f"{name_Sampler}_seed{seed}_sample_traj.jpg"), nrow=10)
#%%
model_id_short = "ddpm-mnist"
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_scheduler"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}_scheduler"
else:
    raise RuntimeError("Unknown system")
export_root = rf"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\{model_id_short}"
os.makedirs(export_root, exist_ok=True)
name_Sampler = "DDIM"
save_fmt = "jpg"
exp_fignames = ["proj_z0_vae_decode.png",
                "sample_diff_lag1_stdnorm_vae_decode.png",
                "samples_all.png",]

# for seed in range(200, 400):
#     savedir = join(saveroot, name_Sampler, f"seed{seed}")
#     for fignm in exp_fignames:
#         if os.path.exists(join(savedir, fignm)):
#             img = plt.imread(join(savedir, fignm))
#         elif os.path.exists(change_format(join(savedir, fignm), "jpg")):
#             img = plt.imread(change_format(join(savedir, fignm), "jpg"))
#         else:
#             raise RuntimeError(f"Can't find {fignm} in {savedir}")
#         plt.imsave(join(export_root,
#             change_format(f"{name_Sampler}_seed{seed}_{fignm}", save_fmt)), img)

for seed in tqdm(range(200, 400)):
    savedir = join(saveroot, name_Sampler, f"seed{seed}")
    traj_data = torch.load(join(savedir, "state_reservoir.pt"))
    save_imgrid((1 + traj_data["latents_traj"]) / 2,
                join(export_root, f"{name_Sampler}_seed{seed}_sample_traj.jpg"), nrow=10)
#%%
model_id_short = "ddpm-cifar10-32"
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_scheduler"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}_scheduler"
else:
    raise RuntimeError("Unknown system")
export_root = rf"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\{model_id_short}"
os.makedirs(export_root, exist_ok=True)
name_Sampler = "DDIM"
save_fmt = "jpg"
exp_fignames = ["proj_z0_vae_decode.png",
                "sample_diff_lag1_stdnorm_vae_decode.png",
                "samples_all.png",]

# for seed in range(200, 400):
#     savedir = join(saveroot, name_Sampler, f"seed{seed}")
#     for fignm in exp_fignames:
#         if os.path.exists(join(savedir, fignm)):
#             img = plt.imread(join(savedir, fignm))
#         elif os.path.exists(change_format(join(savedir, fignm), "jpg")):
#             img = plt.imread(change_format(join(savedir, fignm), "jpg"))
#         else:
#             raise RuntimeError(f"Can't find {fignm} in {savedir}")
#         plt.imsave(join(export_root,
#             change_format(f"{name_Sampler}_seed{seed}_{fignm}", save_fmt)), img)

for seed in tqdm(range(200, 400)):
    savedir = join(saveroot, name_Sampler, f"seed{seed}")
    traj_data = torch.load(join(savedir, "state_reservoir.pt"))
    save_imgrid((1 + traj_data["latents_traj"]) / 2,
                join(export_root, f"{name_Sampler}_seed{seed}_sample_traj.jpg"), nrow=10)
#%%
model_id_short = "StableDiffusion"
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_projection"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}_projection"
else:
    raise RuntimeError("Unknown system")


export_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\DiffTrajectory\StableDiffusion"
prompt_dir_pair = [
    ("a portrait of an aristocrat", "portrait_aristocrat"),
    ("a portrait of an light bulb", "portrait_lightbulb"),
    ("a large box containing an apple and a toy teddy bear", "box_apple_bear"),
    ("a photo of a cat sitting with a dog on a cozy couch", "cat_dog_couch"),
    ("a CG art of a brain composed of eletronic wires and circuits", "brain_wire_circuits"),
    ("a handsome cat dancing Tango with a female dancer in Monet style", "cat_tango_dancer"),
    ("a bug crawling on a textbook under a bright light, photo", "bug_book_photo"),
]

save_fmt = "jpg"
exp_fignames = [#"proj_z0_vae_decode.png",
                "proj_z0_vae_decode_new.png",
                "samples.png",]

name_Sampler = "PNDM"
for prompt, dirname in prompt_dir_pair:
    for seed in range(100, 150):
        savedir = join(saveroot, f"{dirname}-seed{seed}")
        for fignm in exp_fignames:
            if os.path.exists(join(savedir, fignm)):
                img = plt.imread(join(savedir, fignm))
            elif os.path.exists(change_format(join(savedir, fignm), "jpg")):
                img = plt.imread(change_format(join(savedir, fignm), "jpg"))
            else:
                if fignm == "samples.png":
                    img = plt.imread(join(savedir, "sample.png"))
                else:
                    raise RuntimeError(f"Can't find {fignm} in {savedir}")
            plt.imsave(join(export_root,
                change_format(f"{dirname}_{name_Sampler}_seed{seed}_{fignm}", save_fmt)), img)

