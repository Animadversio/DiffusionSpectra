import json
import math
import os
from os.path import join
from tqdm import tqdm
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid


"""Normalizing states"""
def denorm_std(x):
    return ((x - x.mean()) / x.std() * 0.4 + 1) / 2


def denorm_sample_std(x):
    return ((x - x.mean(dim=0, keepdims=True)) / x.std(dim=0, keepdims=True) * 0.4 + 1) / 2


def denorm_var(x, mu, std):
    return (x - x.mean()) / x.std() * std + mu


"""turning latents to images for LDM"""
def latents_to_image(latents, pipe):
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents.to(pipe.vae.device).to(pipe.vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image.cpu()


def latentvecs_to_image(latents, pipe, latent_shape=(4, 64, 64)):
    if len(latents.shape) == 2:
        latents = latents.reshape(latents.shape[0], *latent_shape)
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents.to(pipe.vae.device).to(pipe.vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image.cpu()


def compute_save_diff_imgs_diff(savedir, step_list, latents_reservoir, triu=True):
    """
    compute the difference code and decode the image

    :param savedir:
    :param step_list:
    :param latents_reservoir:
    :param triu:
    :return:
    """
    for i in step_list:
        latent = latents_reservoir[i]
        save_imgrid(denorm_std(latent[None, :, :, :]),
                        join(savedir, f"diffusion_step_{i:02d}_img_stdnorm.png"), nrow=2)
        for j in step_list:
            if triu and j <= i:
                continue
            latent_diff = latents_reservoir[j] - latents_reservoir[i]
            save_imgrid(denorm_std(latent_diff[None, :, :, :]),
                        join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_img_stdnorm.png"), nrow=2, )


def compute_save_diff_imgs_ldm(savedir, step_list, latents_reservoir, pipe, triu=True):
    """
    compute the difference code and decode the image

    :param savedir:
    :param step_list:
    :param latents_reservoir:
    :param triu:
    :return:
    """
    mean_fin = latents_reservoir[-1].mean()  #(dim=0)
    std_fin = latents_reservoir[-1].std()  #(dim=0)
    for i in step_list:
        latent = latents_reservoir[i]
        save_imgrid(denorm_std(latent[:, None, :, :]),
                        join(savedir, f"diffusion_step_{i:02d}_latent_stdnorm.png"), nrow=2)
        img = latents_to_image(latent[None, :], pipe)
        save_imgrid(img, join(savedir, f"diffusion_step_{i:02d}_vae_decode.png"))
        for j in step_list:
            if triu and j <= i:
                continue
            latent_diff = latents_reservoir[j] - latents_reservoir[i]
            save_imgrid(denorm_std(latent_diff[:, None, :, :]),
                        join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_latent_stdnorm.png"), nrow=2, )
            img_interim = latents_to_image(latent_diff[None, :], pipe)
            save_imgrid(img_interim, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode.png"), nrow=1, )
            img_interim_denorm = latents_to_image(denorm_var(latent_diff[None, :], mean_fin, std_fin), pipe)
            save_imgrid(img_interim_denorm, join(savedir, f"diffusion_traj_{j:02d}-{i:02d}_vae_decode_stdfinal.png"), nrow=1, )


def plot_diff_matrix(savedir, step_list, diff_x_sfx="", step_x_sfx="", save_sfx="", tril=True):
    """ Plot the matrix of figures showing the vector difference images.

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