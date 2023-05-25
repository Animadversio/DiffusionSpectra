import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from tqdm import trange, tqdm
from pathlib import Path
from easydict import EasyDict as edict
import pandas as pd
import pickle as pkl
from core.utils.plot_utils import saveallforms
import numpy as np
import matplotlib.pyplot as plt
from core.gmm_special_diffusion_lib import f_VP_vec, alpha, beta, \
    GMM_scores, GMM_density, exact_delta_gmm_reverse_diff
from scipy.integrate import solve_ivp
from scipy.special import softmax, log_softmax

trajdir = Path("F:\insilico_exps\Diffusion_traj\mnist_uncond_gmm_exact")
df_syn = []
for SEED in trange(400):
    df = pd.read_csv(trajdir/rf"uncond_RND{SEED:03d}_softmax_stat.csv", index_col=0)
    df_syn.append(df)
