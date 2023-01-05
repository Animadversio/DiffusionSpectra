import os
from os.path import join
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from diffusers import pipelines, StableDiffusionPipeline
# exproot = r"/home/binxuwang/insilico_exp/Diffusion_Hessian/StableDiffusion"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
#%%
# pipe = pipeline
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
# pipeline.to(torch.half)
#%%
# with torch.autocast("cuda"):
latents_reservoir = []
@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())


out = pipe("a cute and classy mice wearing dress and heels", callback=save_latents,
           num_inference_steps=51)
out.images[0].show()
latents_reservoir = torch.cat(latents_reservoir, dim=0)
#%%
def denorm_std(x):
    return ((x - x.mean()) / x.std() * 0.4 + 1) / 2


def denorm_sample_std(x):
    return ((x - x.mean(dim=0, keepdims=True)) / x.std(dim=0, keepdims=True) * 0.4 + 1) / 2


def denorm_var(x, mu, std):
    return (x - x.mean()) / x.std() * std + mu


def latents_to_image(latents, pipe):
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents.to(pipe.vae.device)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image.cpu()
#%%
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid

show_imgrid(denorm_std((latents_reservoir[0] - latents_reservoir[10]).unsqueeze(1)), nrow=2,)
#%%
show_imgrid(denorm_std((latents_reservoir[50] - latents_reservoir[25]).unsqueeze(1)), nrow=2,)
#%%
show_imgrid(denorm_std((latents_reservoir[7] - latents_reservoir[2]).unsqueeze(1)), nrow=2,)

#%%
# img_interim = pipe.vae.decode((latents_reservoir[7] - latents_reservoir[2]).unsqueeze(0).cuda()).sample.cpu()
img_interim = latents_to_image(2*(latents_reservoir[7] - latents_reservoir[2])[None, :], pipe)
show_imgrid(img_interim, nrow=1,)
#%%
img_interim = latents_to_image(20.0*(latents_reservoir[5] - latents_reservoir[0])[None, :], pipe)
show_imgrid(img_interim, nrow=1, )
#%%
img_interim = latents_to_image(10.0*(latents_reservoir[50] - latents_reservoir[20])[None, :], pipe)
show_imgrid(img_interim, nrow=1, )
#%%
savedir = r"F:\insilico_exps\Diffusion_traj\StableDiffusion\mice_dress_heels1"
os.makedirs(savedir, exist_ok=True)
#%% record the inter mediate images
mean_fin = latents_reservoir[-1].mean(dim=0)
std_fin = latents_reservoir[-1].std(dim=0)
for i in range(0, 51, 5):
    latent = latents_reservoir[i]
    save_imgrid(denorm_std(latent[:, None, :, :]),
                    join(savedir, f"diffusion_step_{i:02d}_latent_stdnorm.png"), nrow=2)
    img = latents_to_image(latent[None, :], pipe)
    save_imgrid(img, join(savedir, f"diffusion_step_{i:02d}_vae_decode.png"))
    for j in range(0, 51, 5):
        latent_diff = latents_reservoir[j] - latents_reservoir[i]
        save_imgrid(denorm_std(latent_diff[:, None, :, :]),
                    join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_latent_stdnorm.png"), nrow=2, )
        img_interim = latents_to_image(latent_diff[None, :], pipe)
        save_imgrid(img_interim, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode.png"), nrow=1, )
        img_interim_denorm = latents_to_image(denorm_var(latent_diff[None, :], mean_fin, std_fin), pipe)
        save_imgrid(img_interim_denorm, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode_stdfinal.png"), nrow=1, )

#%%
def compute_save_diff_imgs(savedir, step_list, latents_reservoir):
    for i in step_list:
        latent = latents_reservoir[i]
        save_imgrid(denorm_std(latent[:, None, :, :]),
                        join(savedir, f"diffusion_step_{i:02d}_latent_stdnorm.png"), nrow=2)
        img = latents_to_image(latent[None, :], pipe)
        save_imgrid(img, join(savedir, f"diffusion_step_{i:02d}_vae_decode.png"))
        for j in step_list:
            latent_diff = latents_reservoir[j] - latents_reservoir[i]
            save_imgrid(denorm_std(latent_diff[:, None, :, :]),
                        join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_latent_stdnorm.png"), nrow=2, )
            img_interim = latents_to_image(latent_diff[None, :], pipe)
            save_imgrid(img_interim, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode.png"), nrow=1, )
            img_interim_denorm = latents_to_image(denorm_var(latent_diff[None, :], mean_fin, std_fin), pipe)
            save_imgrid(img_interim_denorm, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode_stdfinal.png"), nrow=1, )


compute_save_diff_imgs(savedir, range(0, 21, 2), latents_reservoir)
#%%
def plot_diff_matrix(savedir, step_list, diff_x_sfx="", step_x_sfx="", save_sfx="", tril=True):
    """

    :param savedir: directory to load the components and to save the figures
    :param step_list: the time steps to fetch the existing data
    :param diff_x_sfx: the suffix of the difference image file name
    :param step_x_sfx: the suffix of the step image file name
    :param save_sfx:    the suffix of the saved figure file name
    :param tril:   whether to plot the lower triangle of the matrix
    :return:
    """
    nrow = len(step_list)
    figh, axs = plt.subplots(nrow, nrow, figsize=(2 * nrow, 2 * nrow + 2))
    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")

    for i, stepi in enumerate(step_list):
        for j, stepj in enumerate(step_list):
            if i + 1 == nrow:
                continue
            if i >= j and tril:
                continue
            axs[i + 1, j].imshow(plt.imread(join(savedir,
                                                 f"diffusion_traj_{stepj:02d}-{stepi:02d}{diff_x_sfx}.png")))  # make_grid(denorm(sample_traj[j] - sample_traj[i]),).permute(1, 2, 0)
            axs[i + 1, j].set_title(f"{stepj}-{stepi}")

    for i, stepi in enumerate(step_list):
        axs[0, i, ].imshow(plt.imread(join(savedir, f"diffusion_step_{stepi:02d}{step_x_sfx}.png")))
        axs[0, i, ].set_title(f"t={stepi}")

    plt.suptitle(f"x difference along Trajectories diff={diff_x_sfx}, step={step_x_sfx}", fontsize=16)
    plt.tight_layout()
    saveallforms(savedir, f"diffusion_traj_diff_mtg{save_sfx}", figh)
    plt.show()
    return figh


plot_diff_matrix(savedir, range(0, 51, 5),
                 diff_x_sfx="_vae_decode", step_x_sfx="_vae_decode", save_sfx="_vae_decode")
plot_diff_matrix(savedir, range(0, 51, 5),
                 diff_x_sfx="_vae_decode_stdfinal", step_x_sfx="_vae_decode", save_sfx="_vae_decode_stdfinal")
plot_diff_matrix(savedir, range(0, 51, 5),
                 diff_x_sfx="_latent_stdnorm", step_x_sfx="_latent_stdnorm", save_sfx="_latent_stdnorm")


#%%
plot_diff_matrix(savedir, range(0, 21, 2),
                 diff_x_sfx="_vae_decode", step_x_sfx="_vae_decode", save_sfx="_vae_decode_early0-20")
plot_diff_matrix(savedir, range(0, 21, 2),
                 diff_x_sfx="_vae_decode_stdfinal", step_x_sfx="_vae_decode", save_sfx="_vae_decode_stdfinal_early0-20")
plot_diff_matrix(savedir, range(0, 21, 2),
                 diff_x_sfx="_latent_stdnorm", step_x_sfx="_latent_stdnorm", save_sfx="_latent_stdnorm_early0-20")




#%% dev zone
diff_x_sfx = "_latent_stdnorm"
# diff_x_sfx = "_vae_decode"
step_x_sfx = "_latent_stdnorm"

figh, axs = plt.subplots(11, 11, figsize=(20, 21))
for i, ax in enumerate(axs.flatten()):
    ax.axis("off")

for i, stepi in enumerate([*range(0, 51, 5)]):
    for j, stepj in enumerate([*range(0, 51, 5)]):
        if i >= j:
            continue
        axs[i + 1, j].imshow(plt.imread(join(savedir, f"diffusion_traj_{stepj:02d}-{stepi:02d}{diff_x_sfx}.png"))) # make_grid(denorm(sample_traj[j] - sample_traj[i]),).permute(1, 2, 0)
        axs[i + 1, j].set_title(f"{stepj}-{stepi}")

for i, stepi in enumerate([*range(0, 51, 5)]):
    axs[0, i,].imshow(plt.imread(join(savedir, f"diffusion_step_{stepi:02d}{step_x_sfx}.png")))
    axs[0, i,].set_title(f"t={stepi}")

plt.suptitle("x difference along Trajectories", fontsize=16)
plt.tight_layout()
saveallforms(savedir, f"diffusion_traj_diff_mtg{diff_x_sfx}", figh)
plt.show()

