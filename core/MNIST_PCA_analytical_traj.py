import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from os.path import join
from datasets import load_dataset
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
#%% MNIST PCA
savedir = "E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ImageSpacePCA\MNIST"
# torch.save({"U": U, "S": S, "V": V, "mean": imgmean, "cov_eigs": cov_eigs},
#            join(savedir, "mnist_pca.pt"))
data = torch.load(join(savedir, "mnist_pca.pt"))
U, S, V, imgmean, cov_eigs = data["U"], data["S"], data["V"], data["mean"], data["cov_eigs"]
#%%
# use Agg backend
# plt.switch_backend('agg')
# use pycharm backend
plt.switch_backend('module://backend_interagg')
#%%
def norm2img(x):
    return torch.clamp((x + 1) / 2, 0, 1)
#%%
from core.ODE_analytical_lib import *
from diffusers import DDIMPipeline
# load model and scheduler and Alpha schedule
model_id = "dimpo/ddpm-mnist"
model_id_short = model_id.split("/")[-1]
pipe = DDIMPipeline.from_pretrained(model_id)
pipe.scheduler.set_timesteps(51)
alphacum_traj = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps]
#%%
traj_dir = r"F:\insilico_exps\Diffusion_traj\ddpm-mnist_scheduler\DDIM"
outdir = r"F:\insilico_exps\Diffusion_traj\MNIST_PCA_theory"
os.makedirs(outdir, exist_ok=True)
traj_collection = []
for seed in tqdm(range(200, 400)):
    traj_data = torch.load(join(traj_dir, f"seed{seed}", "state_reservoir.pt"))
    sample_traj = traj_data["latents_traj"]
    res_traj = traj_data['residue_traj']
    t_traj = traj_data['t_traj']
    proj_x0_traj = (sample_traj[:-1] -
                    res_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / \
              alphacum_traj.sqrt().view(-1, 1, 1, 1)
    # pred_x0_imgs = (pred_x0 + 1) / 2
    # Analytical prediction
    x0_vec = sample_traj[0:1].flatten(1)
    mu_vec = imgmean.flatten(1) * 2 - 1
    # predict xt
    xt_traj, xt0_residue, scaling_coef_ortho, xttraj_coef = \
        xt_ode_solution(x0_vec, mu_vec, V, cov_eigs * 4, alphacum_traj)
    # predict x0hat
    x0hatxt_traj, xttraj_coef, xttraj_coef_modulated = x0hat_ode_solution( \
        x0_vec, mu_vec, V, cov_eigs * 4, alphacum_traj)
    # save trajectoryimages
    save_imgrid(norm2img(x0hatxt_traj.reshape(-1, 3, 32, 32)), join(outdir, f"seed{seed}_x0hat_theory.png"))
    save_imgrid(norm2img(xt_traj.reshape(-1, 3, 32, 32)), join(outdir, f"seed{seed}_xt_theory.png"))
    save_imgrid(norm2img(sample_traj), join(outdir, f"seed{seed}_xt_empir.png"))
    save_imgrid(norm2img(proj_x0_traj), join(outdir, f"seed{seed}_x0hat_empir.png"))
    # if seed == 400:
    #     break
    xt_pred_mse = ((sample_traj[1:].flatten(1) - xt_traj)**2).mean(1)
    x0hat_pred_mse = ((proj_x0_traj.flatten(1) - x0hatxt_traj)**2).mean(1)
    plt.figure()
    plt.plot(xt_pred_mse)
    # plt.plot((sample_traj[1:].flatten(1) - xt_traj).norm(dim=1))
    plt.ylabel("MSE of deviation")
    plt.xlabel("timestep")
    plt.title("L2 norm of deviation between empirical and analytical prediction of xt")
    saveallforms(outdir, f"seed{seed}_xt_deviation_L2")
    plt.show()
    plt.figure()
    plt.plot(x0hat_pred_mse)
    # plt.plot((proj_x0_traj.flatten(1) - x0hatxt_traj).norm(dim=1))
    plt.ylabel("MSE of deviation")
    plt.xlabel("timestep")
    plt.title("L2 norm of deviation between empirical and analytical prediction of x0hat")
    saveallforms(outdir, f"seed{seed}_x0hat_deviation_L2")
    plt.show()
    torch.save({"xt_traj": xt_traj,
                "x0hatxt_traj": x0hatxt_traj,
                "xttraj_coef": xttraj_coef, "xt0_residue": xt0_residue,
                "xttraj_coef_modulated": xttraj_coef_modulated,
                "scaling_coef_ortho": scaling_coef_ortho,
                "xt_pred_mse": xt_pred_mse,
                "x0hat_pred_mse": x0hat_pred_mse,
                }, join(outdir, f"seed{seed}_theory_coef.pt"))
#%%
show_imgrid(norm2img(x0hatxt_traj.reshape(-1, 3, 32, 32)))
#%%
show_imgrid(norm2img(x0hatxt_traj.reshape(-1, 3, 32, 32)))
show_imgrid(norm2img(xt_traj.reshape(-1, 3, 32, 32)))
show_imgrid(norm2img(sample_traj))
show_imgrid(norm2img(proj_x0_traj))
#%%
figdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\TheoryValMNIST"

#%% Compute the MSE of prediction.
outdir = r"F:\insilico_exps\Diffusion_traj\MNIST_PCA_theory"
xt_pred_mse_col = []
x0hat_pred_mse_col = []
for seed in tqdm(range(200, 400)):
    data = torch.load(join(outdir, f"seed{seed}_theory_coef.pt"))
    xt_pred_mse = data["xt_pred_mse"]
    x0hat_pred_mse = data["x0hat_pred_mse"]
    xt_pred_mse_col.append(xt_pred_mse)
    x0hat_pred_mse_col.append(x0hat_pred_mse)
xt_pred_mse_col = torch.stack(xt_pred_mse_col)
x0hat_pred_mse_col = torch.stack(x0hat_pred_mse_col)
#%%
plt.figure(figsize=[4,4])
plt.plot(xt_pred_mse_col.T, alpha=0.05, color="k")
# plot shaded errorbar of 95 confidence interval
plt.plot(xt_pred_mse_col.mean(0), color="k")
plt.fill_between(range(xt_pred_mse_col.shape[1]),
                 xt_pred_mse_col.quantile(0.05, dim=0),
                xt_pred_mse_col.quantile(0.95, dim=0),
                 alpha=0.2, color="r")
plt.ylabel("MSE of deviation")
plt.xlabel("timestep")
plt.title("L2 norm of deviation between empirical and analytical prediction of xt")
saveallforms(figdir, f"xt_deviation_L2_synopsis")
plt.show()
plt.figure(figsize=[4,4])
plt.plot(x0hat_pred_mse_col.T, alpha=0.05, color="k")
# plot shaded errorbar of 95 confidence interval
plt.plot(x0hat_pred_mse_col.mean(0), color="k")
plt.fill_between(range(x0hat_pred_mse_col.shape[1]),
                 x0hat_pred_mse_col.quantile(0.05, dim=0),
                x0hat_pred_mse_col.quantile(0.95, dim=0),
                 alpha=0.2, color="r")
plt.ylabel("MSE of deviation")
plt.xlabel("timestep")
plt.title("L2 norm of deviation between empirical and analytical prediction of x_0hat")
saveallforms(figdir, f"x0hat_deviation_L2_synopsis")
plt.show()

#%% compute the deviation vector
outdir = r"F:\insilico_exps\Diffusion_traj\MNIST_PCA_theory"
xt_error_col = []
x0hat_error_col = []
for seed in tqdm(range(200, 400)):
    data = torch.load(join(outdir, f"seed{seed}_theory_coef.pt"))
    xt_traj = data["xt_traj"]
    x0hatxt_traj = data["x0hatxt_traj"]
    traj_data = torch.load(join(traj_dir, f"seed{seed}", "state_reservoir.pt"))
    sample_traj = traj_data["latents_traj"]
    res_traj = traj_data['residue_traj']
    t_traj = traj_data['t_traj']
    proj_x0_traj = (sample_traj[:-1] -
                    res_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / \
                   alphacum_traj.sqrt().view(-1, 1, 1, 1)

    xt_error = (xt_traj - sample_traj[1:].flatten(1))
    x0hat_error = (x0hatxt_traj - proj_x0_traj.flatten(1))

    xt_error_col.append(xt_error)
    x0hat_error_col.append(x0hat_error)
xt_error_col = torch.stack(xt_error_col)
x0hat_error_col = torch.stack(x0hat_error_col)
#%%
xt_error_proj_coef = torch.einsum("ijk,kl->ijl", xt_error_col, V)
x0hat_error_proj_coef = torch.einsum("ijk,kl->ijl", x0hat_error_col, V)
#%%
xt_Sqrerror_fraction = (xt_error_proj_coef ** 2) / (xt_error_proj_coef ** 2).sum(-1,keepdim=True)
x0hat_Sqrerror_fraction = (x0hat_error_proj_coef ** 2) / (x0hat_error_proj_coef ** 2).sum(-1,keepdim=True)
#%%
PCrng = range(800)
plt.figure(figsize=[4, 4])
# plt.plot(xt_Sqrerror_fraction[:, -1, PCrng].T,
#          alpha=0.01, color="k", label=None)
plt.plot(cov_eigs[PCrng,] / cov_eigs.sum(),
         alpha=0.7, color="g", label="PC explained variance fraction", lw=2)
plt.plot(xt_Sqrerror_fraction.mean(0).T[PCrng, -1],
         alpha=0.7, color="r", label="mean error fraction", lw=2)
plt.fill_between(PCrng,
                xt_Sqrerror_fraction[:, -1, PCrng].quantile(0.05, dim=0),
                xt_Sqrerror_fraction[:, -1, PCrng].quantile(0.95, dim=0),
                alpha=0.2, color="r", label="95% CI")
plt.xlabel("PC index")
plt.ylabel("Fraction")
plt.title("Fraction of squared error along eigen vector for endpoint x0")
plt.legend()
saveallforms(figdir, f"x0_deviation_proj_errfrac_synopsis")
plt.show()
#%%
PCrng = range(800)
plt.figure(figsize=[4, 4])
plt.plot(cov_eigs.T[PCrng, ],
         alpha=0.9, color="g", label="Data Variance")
plt.plot((xt_error_proj_coef[:, -1, PCrng]**2).mean(0).T,
         alpha=0.9, color="r", label="squared deviation of X0")
# plt.fill_between(PCrng,
#                 (xt_error_proj_coef[:, -1, PCrng]**2).quantile(0.05, dim=0),
#                 (xt_error_proj_coef[:, -1, PCrng]**2).quantile(0.95, dim=0),
#                 alpha=0.2, color="r", label="95% CI")
plt.xlabel("PC index")
plt.ylabel("Squared deviation along eigen vector")
plt.title("Squared error along eigen vector for endpoint x0")
plt.legend()
saveallforms(figdir, f"xt_deviation_proj_sqrerror_synopsis")
plt.show()
#%%
EPS = 1e-2
plt.figure(figsize=[4, 4])
# plt.plot(cov_eigs.T[PCrng, ],
#          alpha=0.9, color="g", label="Data Variance")
plt.plot(((xt_error_proj_coef[:, -1, PCrng]**2).mean(0) + EPS) / (cov_eigs.T[PCrng, ] + EPS),)
plt.xlabel("PC index")
plt.show()
#%%
EPS = 1E-2
plt.figure(figsize=[4, 4])
# plt.plot(cov_eigs.T[PCrng, ],
#          alpha=0.9, color="g", label="Data Variance")
plt.plot(((xt_error_proj_coef[:, -1, PCrng].abs()).mean(0)) / (cov_eigs.sqrt()[PCrng, ] + EPS),)
plt.xlabel("PC index")
plt.show()
#%%
plt.figure(figsize=[4, 4])
plt.plot(cov_eigs.T[:800, ], alpha=0.5, color="k")
plt.show()
#%%
y_traj = sample_traj[1:] - \
         alphacum_traj[:, None, None, None].sqrt() * (imgmean * 2 - 1)
projtraj_empir = y_traj.flatten(1) @ V
#%%
plt.plot(xttraj_coef[:, 600:615])
plt.show()
plt.plot(projtraj_empir[:, 600:615])
plt.show()
#%%
#%%
outdir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\TheoryValMNIST"
for PC_range in [range(0, 10), range(10, 20), range(20, 30), range(30, 40), range(50, 60),
                 range(100, 110), range(250,260), range(500, 510), range(1000, 1010), ]:
    plt.figure(figsize=(9, 5))
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(projtraj_empir[:, PC_range].numpy())
    plt.xlabel("Time")
    plt.ylabel("PC Coefficient")
    plt.title("Empirical")
    ax2 = plt.subplot(1, 2, 2)
    plt.plot(xttraj_coef[:, PC_range].numpy())
    plt.xlabel("Time")
    plt.ylabel("PC Coefficient")
    plt.title("Theoretical")
    plt.suptitle(f"PC Coefficients of Samples traj x(t) for {str(PC_range)}\n"
                 f"Cov eigenvalues: {cov_eigs[PC_range[0]]:.3f}-{cov_eigs[PC_range[-1]]:.3f}")
    plt.legend([f"PC{i}" for i in PC_range], loc="right")
    YLIM = min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax2.set_ylim(YLIM)
    ax1.set_ylim(YLIM)
    saveallforms(savedir, f"seed{seed}_PCcoef_theory_cmp_PC{PC_range[0]}-{PC_range[-1]}")
    plt.show()
#%%

#%%
# PC_proj_Xt = (sample_traj.flatten(1) - imgmean.expand(-1,3,-1,-1).flatten())@ V[:,]




#%% dev zone
t0 = 0
xt0 = sample_traj[0:1].flatten(1)
x_mean = imgmean.flatten(1)
Lambdas = cov_eigs * 4
U = V


xt0_dev = xt0 - x_mean * alphacum_traj[t0].sqrt()  # (N, D)
# projection of xt0 on the eigenvectors
xt0_coef = xt0_dev @ U
# the out of plane component of xt0
xt0_residue = xt0_dev - xt0_coef @ U.T
# coefficients for the projection of xt on the eigenvectors
scaling_coef = ((1 + alphacum_traj[:, None] @ (Lambdas[None, :] - 1)) /
                (1 + alphacum_traj[t0] * (Lambdas[None, :] - 1))
                ).sqrt()
scaling_coef_ortho = ((1 - alphacum_traj) / (1 - alphacum_traj[t0]) ).sqrt()
# multiply the initial condition
xttraj_coef = scaling_coef * xt0_coef  # shape: (T step, n eigen)
# add the residue
xt_traj =   alphacum_traj[:, None].sqrt() @ x_mean \
          + scaling_coef_ortho[:, None] @ xt0_residue \
          + xttraj_coef @ U.T  # shape: (T step, n eigen)