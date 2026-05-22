# models.py

import numpy as np

from constants import (
    LAM_RESP_SAMPLE,
    LAM_RESP_MARG,
    LAM_JOINT,
)

from costs import response_cost_vector_from_name
from ot_solvers import (
    solve_response,
    solve_joint_kl,
    solve_marginal_kl,
)


def build_model_distributions(
        p_h,
        p_r,
        C,
        cost_name,
        h_linear,
        R,
        nominal_time_discount=False,
        ot_backend="custom",
    ):
    gamma_ind = np.outer(p_h, p_r)

    q_r_sample = solve_response(
        p_r,
        response_cost_vector_from_name(
            cost_name,
            h_linear,
            R,
            nominal_time_discount=nominal_time_discount,
        ),
        LAM_RESP_SAMPLE,
    )

    q_r_marg = solve_response(
        p_r,
        np.sum(p_h[:, None] * C, axis=0),
        LAM_RESP_MARG,
    )

    gamma_joint = solve_joint_kl(gamma_ind, C, LAM_JOINT)

    gamma_marg = solve_marginal_kl(
        p_h,
        p_r,
        C,
        backend=ot_backend,
    )

    gamma_resp_sample = np.zeros_like(gamma_ind)
    i_star = int(np.argmax(p_h))
    gamma_resp_sample[i_star, :] = q_r_sample

    gamma_resp_marg = p_h[:, None] * q_r_marg[None, :]

    return {
        "ind": gamma_ind,
        "resp_sample": gamma_resp_sample,
        "resp_marg": gamma_resp_marg,
        "joint": gamma_joint,
        "marg": gamma_marg,
        "q_r_sample": q_r_sample,
        "q_r_marg": q_r_marg,
        "i_star": i_star,
    }