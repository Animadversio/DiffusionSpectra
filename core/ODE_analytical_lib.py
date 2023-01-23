import torch


def xtproj_coef(Lambda, alphacum_traj):
    """ Projection coefficient for xt on eigenvector u_k of value Lambda """
    if type(Lambda) is not torch.Tensor:
        Lambda = torch.tensor(Lambda).float()
    coef_traj = ((1 + (Lambda - 1) * alphacum_traj) /
                 (1 + (Lambda - 1) * alphacum_traj[0])).sqrt()
    return coef_traj


def x0hat_proj_coef(Lambda, alphacum_traj):
    """ Projection coefficient for x0hat on eigenvector of value Lambda """
    if type(Lambda) is not torch.Tensor:
        Lambda = torch.tensor(Lambda).float()
    coef_traj = alphacum_traj.sqrt() * Lambda / \
                ((1 + (Lambda - 1) * alphacum_traj) *
                 (1 + (Lambda - 1) * alphacum_traj[0])).sqrt()
    return coef_traj
