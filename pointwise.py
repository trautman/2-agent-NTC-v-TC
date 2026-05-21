# pointwise.py

import numpy as np

from constants import (
    ALPHA_H,
    ALPHA_R,
)

from math_utils import normalize_matrix

from trajectories import trajectory_deviation_costs


def solve_pointwise_pair(H, R, h_linear, r_linear, cost_matrix):
    normalized_cost = normalize_matrix(cost_matrix)
    dev_h = normalize_matrix(trajectory_deviation_costs(H, h_linear))
    dev_r = normalize_matrix(trajectory_deviation_costs(R, r_linear))

    J_pair = (
        normalized_cost
        + ALPHA_H * dev_h[:, None]
        + ALPHA_R * dev_r[None, :]
    )

    return np.unravel_index(np.argmin(J_pair), J_pair.shape)