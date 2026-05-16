# ot_solvers.py

import numpy as np

from constants import (
    LAM_H,
    LAM_R,
)

from math_utils import (
    logsumexp,
    softmax_from_logweights,
)


def solve_response(p_r, costs, lam_resp):
    logw = np.log(p_r + 1e-300) - costs / lam_resp
    return softmax_from_logweights(logw)


def solve_joint_kl(gamma_ind, C, lam_joint):
    log_gamma = np.log(gamma_ind + 1e-300) - C / lam_joint
    return np.exp(log_gamma - logsumexp(log_gamma.ravel()))


def solve_marginal_kl(
        p_h,
        p_r,
        C,
        lam_h=LAM_H,
        lam_r=LAM_R,
        n_iter=5000,
        eta=0.04,
    ):
    gamma = np.outer(p_h, p_r).copy()
    eps = 1e-300

    for _ in range(n_iter):
        alpha = gamma.sum(axis=1)
        beta = gamma.sum(axis=0)

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

        gamma *= np.exp(-eta * grad)
        gamma /= gamma.sum()

    return gamma