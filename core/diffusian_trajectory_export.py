import os
import platform
from os.path import join
import matplotlib.pyplot as plt

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
export_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\FaceTrajectory"
save_fmt = "jpg"
exp_fignames = ["proj_z0_vae_decode.png",
                "sample_diff_lag1_stdnorm_vae_decode.png",
                "samples_all.png",]

name_Sampler = "DDIM"
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

#%%
#%%
model_id_short = "ddpm-church-256"
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}_scheduler"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}_scheduler"
else:
    raise RuntimeError("Unknown system")
export_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\ChurchTrajectory"
save_fmt = "jpg"
exp_fignames = ["proj_z0_vae_decode.png",
                "sample_diff_lag1_stdnorm_vae_decode.png",
                "samples_all.png",]

name_Sampler = "DDIM"
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


#%%
model_id_short = "StableDiffusion"
if platform.system() == "Windows":
    saveroot = rf"F:\insilico_exps\Diffusion_traj\{model_id_short}"
elif platform.system() == "Linux":
    saveroot = rf"/home/binxuwang/insilico_exp/Diffusion_traj/{model_id_short}"
else:
    raise RuntimeError("Unknown system")


export_root = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\StableDiffusionTraj"
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
exp_fignames = ["proj_z0_vae_decode.png",
                "samples.png",]

name_Sampler = "PNDM"
for prompt, dirname in prompt_dir_pair:
    for seed in range(100, 125):
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

