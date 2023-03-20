import torch


def xt_proj_coef(Lambda, alphacum_traj):
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


def xt_ode_solution(xt0, x_mean, U, Lambdas, alphacum_traj, t0=0):
    """ Computes the trajectory of xt for a given xt0

        xt0: initial condition for xt, shape 1 x n
        x_mean: mean of the distribution, shape 1 x n
        U: eigenvectors of the covariance matrix, each column is an eigenvector.
        Lambdas: eigenvalues of the covariance matrix, 1d tensor
        alphacum_traj: cumulative sum of the alphas, this is the alphacumprod in ddim, ddpm,
            the alpha in our paper in the square root of this
        t0: initial time, default 0, it should be integer to index alphacum_traj.
    """
    # if xt0.ndim == 1:
    #     xt0 = xt0.unsqueeze(0)
    # if x_mean.ndim == 1:
    #     x_mean = x_mean.unsqueeze(0)

    # minus the scaled mean
    xt0_dev = xt0 - x_mean * alphacum_traj[t0].sqrt()  # (N, D)
    # projection of xt0 on the eigenvectors
    xt0_coef = xt0_dev @ U
    # the out of plane component of xt0
    xt0_residue = xt0_dev - xt0_coef @ U.T
    # coefficients for the projection of xt on the eigenvectors
    scaling_coef = ((1 + alphacum_traj[:, None] @ (Lambdas[None, :] - 1)) /
                    (1 + alphacum_traj[t0] * (Lambdas[None, :] - 1))
                    ).sqrt()
    scaling_coef_ortho = ((1 - alphacum_traj) / (1 - alphacum_traj[t0])).sqrt()
    # multiply the initial condition
    xttraj_coef = scaling_coef * xt0_coef  # shape: (T step, n eigen)
    # add the residue
    xt_traj = alphacum_traj[:, None].sqrt() @ x_mean \
              + scaling_coef_ortho[:, None] @ xt0_residue \
              + xttraj_coef @ U.T  # shape: (T step, n eigen)
    return xt_traj, xt0_residue, scaling_coef_ortho, xttraj_coef


def x0hat_ode_solution(xt0, x_mean, U, Lambdas, alphacum_traj, t0=0):
    """ Computes the trajectory of xt for a given xt0

        xt0: initial condition for xt, shape 1 x n
        x_mean: mean of the distribution, shape 1 x n
        U: eigenvectors of the covariance matrix, each column is an eigenvector.
        Lambdas: eigenvalues of the covariance matrix, 1d tensor
        alphacum_traj: cumulative sum of the alphas, this is the alphacumprod in ddim, ddpm,
            the alpha in our paper in the square root of this
        t0: initial time, default 0, it should be integer to index alphacum_traj.
    """
    # if xt0.ndim == 1:
    #     xt0 = xt0.unsqueeze(0)
    # if x_mean.ndim == 1:
    #     x_mean = x_mean.unsqueeze(0)

    # minus the scaled mean
    xt0_dev = xt0 - x_mean * alphacum_traj[t0].sqrt()  # (N, D)
    # projection of xt0 on the eigenvectors
    xt0_coef = xt0_dev @ U
    # the out of plane component of xt0
    # xt0_residue = xt0_dev - xt0_coef @ U.t()
    # coefficients for the projection of xt on the eigenvectors
    scaling_coef = ((1 + alphacum_traj[:, None] * (Lambdas[None, :] - 1)) /
                    (1 + alphacum_traj[t0] * (Lambdas[None, :] - 1))
                    ).sqrt()  # shape: (T step, n eigen)
    # scaling_coef_ortho = ((1 - alphacum_traj) / (1 - alphacum_traj[t0]) ).sqrt()
    # multiply the initial condition
    xttraj_coef = scaling_coef * xt0_coef  # c(t)  shape: (T step, n eigen)
    # multiply the modulation factor for each eigenvector
    modulation = alphacum_traj[:, None] * Lambdas[None, :] / \
                 (1 + alphacum_traj[:, None] * (Lambdas[None, :] - 1))
    xttraj_coef_modulated = xttraj_coef * modulation / alphacum_traj[:, None].sqrt()  # c(t) * Lambda / (1 + c(t) * (Lambda - 1))
    # add the residue
    xt_traj =   x_mean \
              + xttraj_coef_modulated @ U.T  # shape: (T step, n eigen)
    return xt_traj, xttraj_coef, xttraj_coef_modulated

#%%
