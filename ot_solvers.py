# ot_solvers.py

import numpy as np
import ot

from constants import (
    LAM_H,
    LAM_R,
)

from math_utils import (
    logsumexp,
    softmax_from_logweights,
)


# computes q*(r) that is responding to a fixed human trajectory
def solve_response(p_r, costs, lam_resp):
    logw = np.log(p_r + 1e-300) - costs / lam_resp
    return softmax_from_logweights(logw)


def solve_joint_kl(gamma_ind, C, lam_joint):
    log_gamma = np.log(gamma_ind + 1e-300) - C / lam_joint
    return np.exp(log_gamma - logsumexp(log_gamma.ravel()))




# EXPONENTIATED GRADIENT DESCENT USING del J/ del gamma.
# USING THIS WE FIND GAMMA*
def solve_marginal_kl_custom(
        p_h,
        p_r,
        C,
        lam_h=LAM_H,
        lam_r=LAM_R,
        n_iter=5000,
        eta=0.04,
    ):
# construct init gamma = ph*pr
    gamma = np.outer(p_h, p_r).copy()
    eps = 1e-300

    for _ in range(n_iter):
        # alpha, beta = gammah, gammar
        alpha = gamma.sum(axis=1)
        beta = gamma.sum(axis=0)
# del J/del gamma = C + lam_h(log(gammah/ph) +1) 
#                     + lam_r(log(gammar/pr) + 1)
# optimal transport is convex, so gradient finds global optima 
# subject to numerical issues
        grad = (
            C
            + lam_h * (
                np.log(alpha[:, None] + eps)
                - np.log(p_h[:, None] + eps)
                + 1.0
            )
            + lam_r * (
                np.log(beta[None, :] + eps)
                - np.log(p_r[None, :] + eps)
                + 1.0
            )
        )
# exponentiate gradient descent update used instead of 
# standard GD because were optimizing a distribution, 
# and distributions cannot be negative
        gamma *= np.exp(-eta * grad)
        # normalization
        gamma /= gamma.sum()
# returns gamma* = argmin_gamma [E_gamma[c] + KL on marginal terms]
    return gamma






# USE POT LIBRARY TO FIND GAMMA*
def solve_marginal_kl_pot(
        p_h,
        p_r,
        C,
        lam_h=LAM_H,
        lam_r=LAM_R,
        reg=1e-1,
        num_iter=1000,
    ):
    # LIBRARY CALL: POT / ot.unbalanced.sinkhorn_unbalanced
    import ot

    gamma = ot.unbalanced.sinkhorn_unbalanced(
        p_h,
        p_r,
        C,
        reg=reg,
        reg_m=(lam_h, lam_r),
        reg_type="kl",
        numItermax=num_iter,
        stopThr=1e-12,
    )

    gamma = np.asarray(gamma, dtype=float)
    gamma /= gamma.sum()

    return gamma




def solve_marginal_kl(
        p_h,
        p_r,
        C,
        lam_h=LAM_H,
        lam_r=LAM_R,
        backend="custom",
    ):
    if backend == "custom":
        return solve_marginal_kl_custom(
            p_h,
            p_r,
            C,
            lam_h=lam_h,
            lam_r=lam_r,
        )

    if backend == "pot":
        return solve_marginal_kl_pot(
            p_h,
            p_r,
            C,
            lam_h=lam_h,
            lam_r=lam_r,
        )

    raise ValueError(f"Unknown OT backend: {backend}")