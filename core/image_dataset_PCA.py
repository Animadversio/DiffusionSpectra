import matplotlib.pyplot as plt
import torch
from os.path import join
from datasets import load_dataset
from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
#%%
mnistdata = load_dataset("mnist")
cifar10data = load_dataset("cifar10")
#%%
def transforms(examples):
    examples["pixel_values"] = [ToTensor()(image) for image in examples["img"]]
    return examples

cifar10data_tsr = cifar10data.map(transforms, remove_columns=["img"], batched=True)
cifar10data_tsr.set_format(type="torch", columns=["pixel_values", "label",])
#%%
def transforms(examples):
    examples["pixel_values"] = [Resize([32,32])(ToTensor()(image),) for image in examples["image"]]
    return examples


mnistdata_tsr = mnistdata.map(transforms, remove_columns=["image"], batched=True)
mnistdata_tsr.set_format(type="torch", columns=["pixel_values", "label",])

#%% MNIST
savedir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ImageSpacePCA\MNIST"
#%%
imgtsr = mnistdata_tsr["train"]["pixel_values"].expand(-1,3,-1,-1)
#%%
imgmean = imgtsr.mean(dim=0, keepdim=True)
U, S, V = torch.svd((imgtsr - imgmean).flatten(1))
cov_eigs = S**2 / (imgtsr.shape[0] - 1)
#%%
torch.save({"U": U, "S": S, "V": V, "mean": imgmean, "cov_eigs": cov_eigs},
           join(savedir, "mnist_pca.pt"))
#%%
save_imgrid(imgmean, join(savedir, "mnist_mean.png"))
save_imgrid(torch.clamp(0.5+V.reshape(3, 32, 32, 3072)\
                        .permute(3, 0, 1, 2)[:100]*10, 0, 1),
            join(savedir, "mnist_0-100PC.jpg"), nrow=10)
save_imgrid(imgtsr[:100], join(savedir, "mnist_samples_real.jpg"), nrow=10)
#%%
plt.semilogy(cov_eigs)
plt.xlabel("PC index")
plt.ylabel("covariance")
plt.show()
#%%
#%%
show_imgrid(torch.clamp(0.5+V.reshape(3, 32, 32, 3072)\
                        .permute(3, 0, 1, 2)[:200]*10, 0, 1), nrow=10)
#%%
show_imgrid(imgtsr[:64])
#%%


#%%

#%%
#%%

#%%



#%%
plt.plot(PC_proj_Xt)
plt.show()
#%%
ratio = PC_proj_Xt[-1,:] / PC_proj_Xt[0,:]
# plt.scatter(ratio, cov_eigs)
# plt.show()
plt.plot(ratio)
plt.show()
#%%

#%% CIFAR10
savedir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ImageSpacePCA\CIFAR10"
imgtsr_cifar = cifar10data_tsr["train"]["pixel_values"]
#%%
imgmean = imgtsr_cifar.mean(dim=0, keepdim=True)
U_c, S_c, V_c = torch.svd((imgtsr_cifar - imgmean).flatten(1))
cov_eigs = S_c ** 2 / (imgtsr_cifar.shape[0] - 1)
#%%
torch.save({"U": U_c, "S": S_c, "V": V_c, "mean": imgmean, "cov_eigs": cov_eigs},
           join(savedir, "CIFAR10_pca.pt"))
#%%
save_imgrid(imgmean, join(savedir, "CIFAR10_mean.png"))
save_imgrid(torch.clamp(0.5+V_c.reshape(3, 32, 32, 3072)\
                        .permute(3, 0, 1, 2)[:100]*10, 0, 1),
            join(savedir, "CIFAR10_0-100PC.jpg"), nrow=10)
save_imgrid(imgtsr_cifar[:100], join(savedir, "CIFAR10_samples_real.jpg"), nrow=10)
#%%



#%%
show_imgrid(imgmean)
show_imgrid(torch.clamp(0.5+V_c.reshape(3, 32, 32, 3072)\
                        .permute(3, 0, 1, 2)[:200]*4, 0, 1), nrow=10)





#%% Scratch zone
from diffusers import DDIMPipeline
traj_dir = r"F:\insilico_exps\Diffusion_traj\ddpm-cifar10-32_scheduler\DDIM"
model_id = "google/ddpm-cifar10-32"  # most popular
# model_id = "dimpo/ddpm-mnist"
model_id_short = model_id.split("/")[-1]
# load model and scheduler
pipe = DDIMPipeline.from_pretrained(model_id)
#%%
traj_collection = []
for seed in range(202, 400):
    traj_data = torch.load(join(traj_dir, f"seed{seed}", "state_reservoir.pt"))
    sample_traj = traj_data["latents_traj"]
    residual_traj = traj_data['residue_traj']
    t_traj = traj_data['t_traj']
    alphacum_traj = pipe.scheduler.alphas_cumprod[t_traj]
    pred_x0 = (sample_traj[:-1] - residual_traj * (1 - alphacum_traj).sqrt().view(-1, 1, 1, 1)) / \
              alphacum_traj.sqrt().view(-1, 1, 1, 1)
    pred_x0_imgs = (pred_x0 + 1) / 2
    break
#%%
PC_proj_Xt = sample_traj.flatten(1) @ V_c[:,:]
PC_proj_dotXt = residual_traj.flatten(1) @ V_c[:,:]
PC_proj_X0hat = pred_x0_imgs.flatten(1) @ V_c[:,:]
PC_proj_X0hat_bar = (pred_x0_imgs - imgmean).flatten(1) @ V_c[:,:]
#%%
plt.plot(PC_proj_X0hat[:, 30:41])
plt.axhline(0, color="k", linestyle="--")
plt.show()
#%%
plt.plot(PC_proj_X0hat_bar[:, 300:310])
plt.axhline(0, color="k", linestyle="--")
plt.show()
#%%
plt.plot(PC_proj_Xt[:, 100:120])
plt.plot((1 - alphacum_traj).sqrt())
plt.axhline(0, color="k", linestyle="--")
plt.show()
#%%
plt.axhline(0, color="k", linestyle="--")
plt.show()

#%%
#%%
def xtproj_coef_batch(Lambda, alphacum_traj):
    """ Projection coefficient for xt on eigenvector u_k of value Lambda """
    if type(Lambda) is not torch.Tensor:
        Lambda = torch.tensor(Lambda).float()
    coef_traj = ((1 + (Lambda - 1) * alphacum_traj) /
                 (1 + (Lambda - 1) * alphacum_traj[0])).sqrt()
    return coef_traj

init_projcoef = (sample_traj[0:1] - alphacum_traj[0] * imgmean).flatten(1) @ V_c
coef_trajs = [xtproj_coef_batch(Lambda, alphacum_traj) for Lambda in cov_eig]
coef_trajs = torch.stack(coef_trajs, dim=1)
coef_trajs = coef_trajs * init_projcoef
#%%
plt.plot(coef_trajs[:, 0:30])
plt.axhline(0, color="k", linestyle="--")
plt.show()
#%%
img_traj_analy = coef_trajs[:,] @ V_c.T
final_img_analy = coef_trajs[-1:,] @ V_c.T
#%%
# show_imgrid((1+final_img_analy.reshape(1, 3, 32, 32))/2)
show_imgrid((1+img_traj_analy.reshape(-1, 3, 32, 32))/2)
#%%
show_imgrid((1+sample_traj)/2)
#%%
show_imgrid((1+sample_traj[-1])/2)
#%%
plt.plot((sample_traj[:-1].flatten(1) - img_traj_analy).norm(dim=1))
plt.show()
#%%

pred_devia_coefs = (img_traj_analy - sample_traj[:-1].flatten(1)) @ V_c[:, :]
#%%
plt.plot(pred_devia_coefs.T[:, 50].abs(), alpha=0.7)  # , '.'
# plt.axhline(0, color="k", linestyle="--")
plt.ylabel("abs deviation from prediction")
plt.xlabel("eigenvalue index")
plt.title("deviation of prediction from true trajectory along eigenvectors")
plt.show()

#%%
plt.plot(pred_devia_coefs.T[:, 50].abs() / cov_eig.sqrt(), alpha=0.7)  # , '.'
# plt.axhline(0, color="k", linestyle="--")
plt.ylabel("abs deviation from prediction")
plt.xlabel("eigenvalue index")
plt.title("deviation of prediction from true trajectory along eigenvectors")
plt.show()
