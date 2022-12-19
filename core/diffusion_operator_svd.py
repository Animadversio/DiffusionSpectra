import torch
from tqdm import tqdm
from diffusers import PNDMPipeline, DDIMScheduler, UNet2DModel
#%%
# repo_id = "google/ddpm-celebahq-256"
repo_id = "google/ddpm-cifar10-32"
# repo_id = "nbonaker/ddpm-celeb-face-32/unet"
model = UNet2DModel.from_pretrained(repo_id)
model.requires_grad_(False).eval().half().to("cuda")
#%%
scheduler = DDIMScheduler.from_pretrained(repo_id)
# scheduler = PNDMScheduler.from_pretrained(model_id)
scheduler.set_timesteps(num_inference_steps=50)
#%%
#%%
import numpy as np
from scipy.sparse.linalg import svds, eigs, LinearOperator
class JacobianVectorProduct_bp(LinearOperator):

    def __init__(self, model, t, sample, ):
        self.model = model
        self.t = t
        self.sample = sample
        self.input_dim = sample.numel()
        self.sample_req_vec = sample.clone().detach().flatten().requires_grad_(True)
        super(JacobianVectorProduct_bp, self).__init__(dtype=np.float32, shape=(self.input_dim, self.input_dim))
        sample_req = self.sample_req_vec.reshape(self.sample.shape)
        delta_sample = self.model(sample_req, self.t).sample
        self.delta_sample = delta_sample

    def _matvec(self, v):
        v = torch.from_numpy(v).to(self.sample.device)

        hess_part = torch.autograd.grad(self.delta_sample.flatten(), self.sample_req_vec,
                                        grad_outputs=v.flatten(),
                                        retain_graph=True, create_graph=False, )
        return hess_part[0].cpu().detach().numpy()

    def _matmat(self, vs):
        vs = torch.from_numpy(vs).to(self.sample.device).half()

        hess_part = torch.autograd.grad(self.delta_sample.flatten(), self.sample_req_vec,
                                        grad_outputs=vs.T,
                                        is_grads_batched=True,
                                        retain_graph=True, create_graph=False, )
        return hess_part[0].cpu().detach().numpy().T

    def _rmatvec(self, x):
        x = torch.from_numpy(x).to(self.sample.device)
        return torch.autograd.grad(self.delta_sample.flatten(), self.sample_req_vec,
                                   grad_outputs=x.flatten(),
                                   retain_graph=True, create_graph=False, )[0].cpu().detach().numpy()


class JacobianVectorProduct_bpfp(LinearOperator):

    def __init__(self, model, t, sample, ):
        self.model = model
        self.t = t
        self.sample = sample
        self.input_dim = sample.numel()
        self.sample_req_vec = sample.clone().detach().flatten().requires_grad_(True)
        super(JacobianVectorProduct_bpfp, self).__init__(dtype=np.float32, shape=(self.input_dim, self.input_dim))
        sample_req = self.sample_req_vec.reshape(self.sample.shape)
        delta_sample = self.model(sample_req, self.t).sample
        self.delta_sample = delta_sample

    def _rmatvec(self, v):
        v = torch.from_numpy(v).to(self.sample.device).half()

        hess_part = torch.autograd.grad(self.delta_sample.flatten(), self.sample_req_vec,
                                        grad_outputs=v.flatten(),
                                        retain_graph=True, create_graph=False, )
        return hess_part[0].cpu().detach().numpy()

    def _rmatmat(self, vs):
        vs = torch.from_numpy(vs).to(self.sample.device).half()
        hess_part = torch.autograd.grad(self.delta_sample.flatten(), self.sample_req_vec,
                                        grad_outputs=vs.T,
                                        is_grads_batched=True,
                                        retain_graph=True, create_graph=False, )
        return hess_part[0].cpu().detach().numpy().T

    def _matvec(self, x, EPS=1E-4):
        x = torch.from_numpy(x).to(self.sample.device).half()
        with torch.no_grad():
            delta_sample_pert = self.model(self.sample + EPS * x.reshape(self.sample.shape), self.t).sample
        jvp = (delta_sample_pert - self.delta_sample.detach()) / EPS
        # print(jvp.shape)
        return jvp.flatten().cpu().detach().numpy()

    def _matmat(self, xs, EPS=1E-4):
        xs = torch.from_numpy(xs).to(self.sample.device).half()
        with torch.no_grad():
            delta_sample_pert = self.model(self.sample + EPS * xs.reshape(-1, *self.sample.shape[1:]), self.t).sample
        jvp = (delta_sample_pert - self.delta_sample.detach()) / EPS
        jvp = torch.flatten(jvp, start_dim=1).cpu().detach().numpy()
        # print(jvp.shape)
        return jvp.T
#%%
sample, _, t_traj = sampling(model, scheduler)
save_image((sample+0.5)/2, join(expdir, "sample.png"))
#%%
JVP = JacobianVectorProduct_bp(model, t_traj[-1], sample)
#%%
# Uk, Sk, Vkt = svds(JVP, k=10, which="SM")
#%% time the line above
#%%
from time import time
start = time()
Uk, Sk, Vkt = svds(JVP, k=80, which="SM", solver='lobpcg')
end = time()
print(end - start)
# Arpack  LM  20 top svds 18.11
# Arpack  LM  40 top svds 33.56
# lobpcg  LM  40 top svds 30.316
# lobpcg  SM  10 bottom svds  8.21  seems not accuracy
# lobpcg  SM  20 bottom svds  16.01
# lobpcg  SM  50 bottom svds  37.03
# lobpcg  SM  80 bottom svds  58.68
# lobpcg  SM  80 bottom svds  45.93 (with matmat implmeneted)
#%%
savedir = r"F:\insilico_exps\Diffusion_Hessian\cifar10-32"
expdir = join(savedir, "exp%03d"%torch.randint(0, 1000, (1,)).item())
os.makedirs(expdir, exist_ok=True)
#%%
visualize_SVecs_all(torch.tensor(Vkt.T), expdir, name="V_sprs", scale=15)
visualize_SVecs_all(torch.tensor(Uk), expdir, name="U_sprs", scale=15)
#%%
JVP_fp = JacobianVectorProduct_bpfp(model, t_traj[-1], sample)
#%%
Vec = np.random.randn(32*32*3, 3) / 100
#%%
JVP_fp.T @ Vec
#%%
start = time()
Uk, Sk, Vkt = svds(JVP_fp, k=160, which="SM", solver='lobpcg')#lobpcg
end = time()
print(end - start)
# lobpcg  SM  20 bottom svds  5.985903263092041 (with matmat, rmatmat implmeneted)
# lobpcg  SM  100 bottom svds  25.3198  (with matmat, rmatmat implmeneted)
# lobpcg  SM  160 bottom svds  38.66  (with matmat, rmatmat implmeneted)
# arpack  LM  80 top svds  6.57  (with matmat, rmatmat implmeneted)
#%%
visualize_SVecs_all(torch.tensor(Vkt.T), expdir, name="V_bpfp_top", scale=15)
visualize_SVecs_all(torch.tensor(Uk), expdir, name="U_bpfp_top", scale=15)
#%%
visualize_SVecs_all(torch.tensor(Vkt.T), expdir, name="V_bpfp_lob", scale=15)
visualize_SVecs_all(torch.tensor(Uk), expdir, name="U_bpfp_lob", scale=15)