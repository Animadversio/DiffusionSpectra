import torch


"""Geometric utils """
def proj2subspace(A, b):
    """ Project b onto the subspace spanned by A
    Assume, A, b are both row vectors
    """
    return (A.T @ (torch.linalg.inv(A@A.T) @ (A @ b.T))).T


def proj2orthospace(A, b):
    """ Project b onto the subspace spanned by A
    Assume, A, b are both row vectors
    """
    return b - proj2subspace(A, b)


def subspace_variance(X, subspace):
    """ Calculate the variance of X projected onto the subspace
    """
    if X.ndim != 2:
        X = X.flatten(1)
    if subspace.ndim != 2:
        subspace = subspace.flatten(1)
    X_proj = proj2subspace(subspace, X)
    var_ratio = X_proj.norm(dim=1)**2 / X.norm(dim=1)**2
    return var_ratio, 1 - var_ratio


import math
from torchmetrics.functional import pairwise_cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from core.utils.plot_utils import saveallforms
def trajectory_geometry_pipeline(latents_reservoir, savedir):
    init_latent = latents_reservoir[:1].flatten(1).float()
    end_latent = latents_reservoir[-1:].flatten(1).float()
    init_end_cosine = pairwise_cosine_similarity(init_latent, end_latent).item()
    init_end_angle = math.acos(init_end_cosine)
    init_end_ang_deg = init_end_angle / math.pi * 180
    unitbasis1 = end_latent / end_latent.norm()  # end state
    unitbasis2 = proj2orthospace(end_latent, init_latent)  # init noise that is ortho to the end state
    unitbasis2 = unitbasis2 / unitbasis2.norm()  # unit normalize
    proj_coef1 = torch.matmul(latents_reservoir.flatten(1).float(), unitbasis1.T)
    proj_coef2 = torch.matmul(latents_reservoir.flatten(1).float(), unitbasis2.T)
    residue = latents_reservoir.flatten(1).float() - (proj_coef1 @ unitbasis1 + proj_coef2 @ unitbasis2)
    residue_frac = residue.norm(dim=1) ** 2 / latents_reservoir.flatten(1).float().norm(dim=1) ** 2

    """plot latent space trajectory on the 2d plane spanned by the initial and final states"""
    plt.figure(figsize=(6, 6.5))
    plt.plot([0, proj_coef1[0].item()], [0, proj_coef2[0].item()], label="noise init", color="r")
    plt.plot([0, proj_coef1[-1].item()], [0, proj_coef2[-1].item()], label="final latent", color="g")
    plt.scatter(proj_coef1, proj_coef2, label="latent trajectory")
    plt.axis("equal")
    plt.axhline(0, color="k", linestyle="--", lw=0.5)
    plt.axvline(0, color="k", linestyle="--", lw=0.5)
    plt.legend()
    plt.xlabel("projection with z_T")
    plt.ylabel("projection with ortho part of z_0")
    plt.title(
        f"latent trajectory in 2d projection space (z0,zT)\ninit end cosine={init_end_cosine:.3f} angle={init_end_ang_deg:.1f} deg")
    saveallforms(savedir, f"latent_trajectory_2d_proj", plt.gcf())
    plt.show()
    # %
    """The geometry of the differences"""
    plt.figure(figsize=(6, 6.5))
    plt.scatter(proj_coef1[1:] - proj_coef1[:-1], proj_coef2[1:] - proj_coef2[:-1], c=range(len(proj_coef2[1:])), label="latent diff")
    plt.plot(proj_coef1[1:] - proj_coef1[:-1], proj_coef2[1:] - proj_coef2[:-1], color="k", alpha=0.5)
    plt.axhline(0, color="k", linestyle="--", lw=0.5)
    plt.axvline(0, color="k", linestyle="--", lw=0.5)
    plt.axline((0, 0), slope=proj_coef2[0].item() / proj_coef1[0].item(),
               color="r", linestyle="--", label="init noise direction")
    plt.axline((0, 0), slope=0,
               color="g", linestyle="--", label="final latent direction")
    plt.legend()
    plt.axis("equal")
    plt.xlabel("projection with z_T")
    plt.ylabel("projection with ortho part of z_0")
    plt.title(
        f"latent diff (z_t+1 - z_t) in 2d projection space (z0,zT)\ninit end cosine={init_end_cosine:.3f} angle={init_end_ang_deg:.1f} deg")
    saveallforms(savedir, f"latent_diff_2d_proj", plt.gcf())
    plt.show()
    # %
    """There is little variance outside the subspace spanned by the initial and final states"""
    plt.figure()
    plt.plot(residue_frac)
    plt.title("fraction of residue ortho to the 2d subspace spanned by z_0 and z_T")
    plt.xlabel("t")
    plt.ylabel("fraction of var")
    saveallforms(savedir, f"latent_trajectory_2d_proj_residue_trace", plt.gcf())
    plt.show()
    # %
    """There is little variance of vector norm"""
    plt.figure()
    plt.plot(latents_reservoir.flatten(1).float().norm(dim=1))
    plt.title("Norm of latent states")
    plt.xlabel("t")
    plt.ylabel("L2 norm")
    saveallforms(savedir, f"latent_trajectory_norm_trace", plt.gcf())
    plt.show()
    """There is little variance of vector norm"""
    plt.figure()
    plt.plot((latents_reservoir[1:] - latents_reservoir[:-1]).flatten(1).float().norm(dim=1))
    plt.title("Norm of latent states diff")
    plt.xlabel("t")
    plt.ylabel("L2 norm")
    saveallforms(savedir, f"latent_diff_norm_trace", plt.gcf())
    plt.show()



def latent_PCA_analysis(latents_reservoir, savedir,
                        proj_planes=[(0, 1)]):
    latents_mat = latents_reservoir.flatten(1).double()
    latents_mat = latents_mat - latents_mat.mean(dim=0)
    U, D, V = torch.svd(latents_mat, )
    expvar_vec = torch.cumsum(D ** 2 / (D ** 2).sum(), dim=0)

    plt.figure()
    plt.plot(expvar_vec)
    plt.xlabel("PC")
    plt.ylabel("explained variance")
    plt.title("Explained variance of the latent trajectory")
    saveallforms(savedir, "latent_traj_PCA_expvar", plt.gcf())
    plt.show()

    for PCi, PCj in proj_planes:
        projvar = (D[[PCi, PCj]] ** 2).sum() / (D ** 2).sum()
        plt.figure(figsize=(6, 6.5))
        plt.plot(U[:, PCi] * D[PCi], U[:, PCj] * D[PCj], "-")
        plt.scatter(U[:, PCi] * D[PCi], U[:, PCj] * D[PCj], c=range(len(U)))
        plt.xlabel(f"PC{PCi + 1}")
        plt.ylabel(f"PC{PCj + 1}")
        plt.axis("equal")
        plt.title(f"Latent state projection onto the 2 PCs (PC{PCi + 1} vs PC{PCj + 1})"
                  f"\n{projvar:.2%} of the variance is explained")
        saveallforms(savedir, f"latent_traj_PC{PCi + 1}_PC{PCj + 1}_proj", plt.gcf())
        plt.show()
    return expvar_vec, U, D, V


def latent_diff_PCA_analysis(latents_reservoir, savedir,
             proj_planes=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
    """PCA of the latent steps taken """
    latents_mat = latents_reservoir.flatten(1).double()
    latents_diff_mat = latents_mat[1:] - latents_mat[:-1]
    latents_diff_mat = latents_diff_mat - latents_diff_mat.mean(dim=0)
    U_diff, D_diff, V_diff = torch.svd(latents_diff_mat, )
    expvar_diff = torch.cumsum(D_diff ** 2 / (D_diff ** 2).sum(), dim=0)
    plt.figure()
    plt.plot(expvar_diff)
    plt.xlabel("PC")
    plt.ylabel("explained variance")
    plt.title("Explained variance of the latent difference")
    saveallforms(savedir, "latent_diff_PCA_expvar", plt.gcf())
    plt.show()
    """Project the latent steps to different PC planes."""
    for PCi, PCj in proj_planes:
        projvar = (D_diff[[PCi, PCj]] ** 2).sum() / (D_diff ** 2).sum()
        plt.figure(figsize=(6, 6.5))
        plt.plot(U_diff[:, PCi] * D_diff[PCi], U_diff[:, PCj] * D_diff[PCj], ":k", lw=1)
        plt.scatter(U_diff[:, PCi] * D_diff[PCi], U_diff[:, PCj] * D_diff[PCj], c=range(len(U_diff)))
        plt.xlabel(f"PC{PCi + 1}")
        plt.ylabel(f"PC{PCj + 1}")
        plt.axis("equal")
        plt.title(f"Latent Step projection onto the 2 PCs (PC{PCi + 1} vs PC{PCj + 1})"
                  f"\n{projvar:.2%} of the variance is explained")
        saveallforms(savedir, f"latent_diff_PC{PCi + 1}_PC{PCj + 1}_proj", plt.gcf())
        plt.show()
    return expvar_diff, U_diff, D_diff, V_diff


""" Correlogram of the latent state difference """
def diff_lag(x, lag=1, ):
    assert lag >= 1
    return x[lag:] - x[:-lag]


def avg_cosine_sim_mat(X):
    cosmat = pairwise_cosine_similarity(X,)
    idxs = torch.tril_indices(cosmat.shape[0], cosmat.shape[1], offset=-1)
    cosmat_vec = cosmat[idxs[0], idxs[1]]
    return cosmat, cosmat_vec.mean()


def diff_cosine_mat_analysis(latents_reservoir, savedir, lags=(1,2,3,4,5,10)):
    for lag in lags:
        cosmat, cosmat_avg = avg_cosine_sim_mat(diff_lag(latents_reservoir, lag).flatten(1).float())
        figh = plt.figure(figsize=(7, 6))
        sns.heatmap(cosmat, cmap="coolwarm", vmin=-1, vmax=1)
        plt.axis("image")
        plt.title(
            f"cosine similarity matrix of latent states diff z_t+{lag} - z_t\n avg cosine={cosmat_avg:.3f} lag={lag}")
        plt.xlabel("t1")
        plt.ylabel("t2")
        saveallforms(savedir, f"cosine_mat_latent_diff_lag{lag}", figh)
        plt.show()

    for lag in lags:
        cosvec_end = pairwise_cosine_similarity(diff_lag(latents_reservoir, lag).flatten(1).float(),
                                                latents_reservoir[-1:].flatten(1).float())
        cosvec_init = pairwise_cosine_similarity(diff_lag(latents_reservoir, lag).flatten(1).float(),
                                                 latents_reservoir[:1].flatten(1).float())
        figh = plt.figure()
        plt.plot(cosvec_end, label="with end z_T")
        plt.plot(cosvec_init, label="with init z_0")
        plt.axhline(0, color="k", linestyle="--")
        plt.title(f"cosine similarity of latent states diff z_t+{lag} - z_t with z_0, z_T")
        plt.xlabel("t")
        plt.ylabel("cosine similarity")
        plt.legend()
        saveallforms(savedir, f"cosine_trace_w_init_end_latent_diff_lag{lag}", figh)
        plt.show()

    cosvec_end = pairwise_cosine_similarity(latents_reservoir.flatten(1).float(),
                                            latents_reservoir[-1:].flatten(1).float())
    cosvec_init = pairwise_cosine_similarity(latents_reservoir.flatten(1).float(),
                                             latents_reservoir[:1].flatten(1).float())
    figh = plt.figure()
    plt.plot(cosvec_end, label="with end z_T")
    plt.plot(cosvec_init, label="with init z_0")
    plt.axhline(0, color="k", linestyle="--")
    plt.title(f"cosine similarity of latent states z_t with z_0, z_T")
    plt.xlabel("t")
    plt.ylabel("cosine similarity")
    plt.legend()
    saveallforms(savedir, f"cosine_trace_w_init_end_latent", figh)
    plt.show()


