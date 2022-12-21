
import os
from os.path import join
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from diffusers import pipelines, StableDiffusionPipeline
exproot = r"/home/binxuwang/insilico_exp/Diffusion_Hessian/StableDiffusion"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
#%% post hoc analysis of the saved data
expdir = join(exproot, "exp2941")
data = torch.load(join(expdir, "latent_model_traj_hessian.pt"))
#%%
latent_traj = data["latent_traj"]
t_traj = data["t_traj"]
cond_grad = data["cond_grad"]
cond_hessian = data["cond_hessian"]
tokens_ids_val = data["tokens_ids_val"]
prompt = data["prompt"]
#%%
cosine_mat = []
for i in range(0, 11):
    cosvec = torch.cosine_similarity(
        cond_hessian[i:i+1,:,:].flatten(1).float(),
        cond_hessian[:,:,:].flatten(1).float())
    cosine_mat.append(cosvec)
cosine_mat = torch.stack(cosine_mat, dim=0)
#%%
plt.figure(figsize=(8, 7))
sns.heatmap(cosine_mat, annot=True, fmt=".2f", xticklabels=t_traj, yticklabels=t_traj)
plt.xlabel("t")
plt.axis("image")
plt.title("cosine similarity of hessian matrix at different t")
plt.tight_layout()
plt.savefig(join(expdir, "hessian_mat_cosine_matrix.png"))
plt.show()
#%%
grad_cosine_mat = []
for i in range(0, 11):
    cosvec = torch.cosine_similarity(
        cond_grad[i:i+1, 0, :, :].flatten(1).float(),
        cond_grad[:, 0, :, :].flatten(1).float())
    grad_cosine_mat.append(cosvec)
grad_cosine_mat = torch.stack(grad_cosine_mat, dim=0)
#%%
plt.figure(figsize=(8, 7))
sns.heatmap(grad_cosine_mat, annot=True, fmt=".2f", xticklabels=t_traj, yticklabels=t_traj)
plt.xlabel("t")
plt.axis("image")
plt.title("cosine similarity of gradient matrix to conditional signal at different t")
plt.tight_layout()
plt.savefig(join(expdir, "grad_mat_cosine_matrix.png"))
plt.show()

#%%
# valid_tokens = [pipe.tokenizer.convert_ids_to_tokens(i) for i in tokens_ids_val]
grad_val_cosine_mat = []
for i in range(0, 11):
    cosvec = torch.cosine_similarity(
        cond_grad[i:i+1, 0, 1:len(tokens_ids_val)-1, :].flatten(1).float(),
        cond_grad[:, 0, 1:len(tokens_ids_val)-1, :].flatten(1).float())
    grad_val_cosine_mat.append(cosvec)
grad_val_cosine_mat = torch.stack(grad_val_cosine_mat, dim=0)
#%%
plt.figure(figsize=(8, 7))
sns.heatmap(grad_val_cosine_mat, annot=True, fmt=".2f", xticklabels=t_traj, yticklabels=t_traj)
plt.xlabel("t")
plt.axis("image")
plt.title("cosine similarity of gradient matrix to conditional signal (word token) at different t")
plt.tight_layout()
plt.savefig(join(expdir, "grad_mat_val_cosine_matrix.png"))
plt.show()
#%%
# valid_tokens = [pipe.tokenizer.convert_ids_to_tokens(i) for i in tokens_ids_val]
grad_norm_cosine_mat = []
for i in range(0, 11):
    cosvec = torch.cosine_similarity(
        cond_grad[i:i+1, 0, 1:len(tokens_ids_val)-1, :].float().norm(dim=-1),
        cond_grad[:, 0, 1:len(tokens_ids_val)-1, :].float().norm(dim=-1))
    grad_norm_cosine_mat.append(cosvec)
grad_norm_cosine_mat = torch.stack(grad_norm_cosine_mat, dim=0)
#%%
plt.figure(figsize=(8, 7))
sns.heatmap(grad_norm_cosine_mat, annot=True, fmt=".2f", xticklabels=t_traj, yticklabels=t_traj)
plt.xlabel("t")
plt.axis("image")
plt.title("cosine similarity of gradient norm to conditional signal (word token) at different t")
plt.tight_layout()
plt.savefig(join(expdir, "grad_norm_val_cosine_matrix.png"))
plt.show()
#%%
tokens_val = [pipe.tokenizer.convert_ids_to_tokens(i.item()) for i in tokens_ids_val]
gradnorm = cond_grad[:, 0, :, :].float().norm(dim=-1)
plt.figure()
sns.heatmap(torch.log10(gradnorm.T)[:len(tokens_val) + 1, :], yticklabels=tokens_val, xticklabels=t_traj)
plt.title("log10 of gradient norm to tokens")
plt.xlabel("t")
plt.tight_layout()
plt.savefig(join(expdir, "grad_norm_by_t_map.png"))
plt.show()
#%%
for i, t in enumerate(t_traj):
    hess = cond_hessian[i, :, :].float() + cond_hessian[i, :, :].float().T
    eigval, eigvec = torch.linalg.eigh(hess) # torch.linalg.eig(hess)
    plt.figure()
    plt.semilogy(eigval.real)
    plt.title("Hessian eigen value spectrum\nt=%d" % t)
    # sns.heatmap(torch.log10(cond_hessian_col[i, :, :].float().norm(dim=-1).T)[:len(tokens_val) + 1, :], yticklabels=tokens_val, xticklabels=tokens_val)
    plt.tight_layout()
    plt.savefig(join(expdir, "eigval_real_T%d.png" % t))
    plt.show()
    plt.figure()
    plt.semilogy(eigval.abs())
    plt.title("Hessian eigen value abs spectrum\nt=%d" % t)
    # sns.heatmap(torch.log10(cond_hessian_col[i, :, :].float().norm(dim=-1).T)[:len(tokens_val) + 1, :], yticklabels=tokens_val, xticklabels=tokens_val)
    plt.tight_layout()
    plt.savefig(join(expdir, "eigval_abs_T%d.png" % t))
    plt.show()
#%%
eigval_col = []
for i, t in enumerate(t_traj):
    hess = cond_hessian[i, :, :].float() + cond_hessian[i, :, :].float().T
    eigval, eigvec = torch.linalg.eigh(hess)  # torch.linalg.eig(hess)
    eigval_col.append(eigval)
eigval_col = torch.stack(eigval_col, dim=0)
#%%
# compute effective dimension
eigval_sort = torch.fliplr(eigval_col.abs().sort(dim=-1, ).values)
tot_var = (eigval_sort**2).sum(dim=-1)
var_ratio = (eigval_sort**2).cumsum(dim=-1) / tot_var[:, None]
#%%
plt.figure()
plt.plot(t_traj, (var_ratio<0.90).sum(dim=1), label="90%")
plt.plot(t_traj, (var_ratio<0.95).sum(dim=1), label="95%")
plt.plot(t_traj, (var_ratio<0.99).sum(dim=1), label="99%")
plt.plot(t_traj, (var_ratio<0.999).sum(dim=1), label="99.9%")
plt.legend()
plt.xlabel("t")
plt.ylabel("effective dimension")
plt.title("effective dimension of Hessian")
plt.tight_layout()
plt.savefig(join(expdir, "Hessian_effective_dim_by_t.png"))
plt.show()
