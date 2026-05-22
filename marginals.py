# marginals.py

import numpy as np

from constants import LATERAL_LEVELS
from constants import LAM_PREF
from math_utils import softmax_from_logweights


def compute_agent_marginals(H, R):
    pref_h = np.array([preference_cost(h) for h in H])
    pref_r = np.array([preference_cost(r) for r in R])

    p_h = softmax_from_logweights(-LAM_PREF * pref_h)
    p_r = softmax_from_logweights(-LAM_PREF * pref_r)

    return p_h, p_r


def make_centerline_trajectory(start, goal, T):
    return np.column_stack([
        np.linspace(start[0], goal[0], T),
        np.linspace(start[1], goal[1], T),
    ])


def lateral_profile(tau, profile_id=0):
    if profile_id == 0:
        return np.sin(np.pi * tau)
    if profile_id == 1:
        return np.sin(np.pi * tau) ** 1.35
    if profile_id == 2:
        return 16.0 * (tau ** 2) * ((1.0 - tau) ** 2)
    if profile_id == 3:
        return (tau ** 0.8) * ((1.0 - tau) ** 1.25)
    raise ValueError("Unknown profile_id")


def build_structured_library(start, goal, T=31):
    tau = np.linspace(0.0, 1.0, T)
    base = make_centerline_trajectory(start, goal, T)
    sign_x = (
        np.sign(goal[0] - start[0])
        if abs(goal[0] - start[0]) >= abs(goal[1] - start[1])
        else 1.0
    )

    trajectories = []
    metadata = []

    for profile_id in [0, 1]:
        traj = base.copy()
        trajectories.append(traj)
        metadata.append({"side": 0, "max_dev": 0.0, "profile_id": profile_id})

    for side in [-1.0, 1.0]:
        for dmax in LATERAL_LEVELS[1:]:
            for profile_id in [0, 1, 2, 3]:
                profile = lateral_profile(tau, profile_id)
                lateral = side * dmax * profile / np.max(np.abs(profile))
                long_basis = 16.0 * (tau ** 2) * ((1.0 - tau) ** 2)
                longi = 0.004 * (profile_id - 1.5) * long_basis

                traj = base.copy()
                traj[:, 0] += sign_x * longi
                traj[:, 1] += lateral
                traj[0] = start
                traj[-1] = goal

                trajectories.append(traj)
                metadata.append({
                    "side": int(side),
                    "max_dev": float(dmax),
                    "profile_id": profile_id,
                })

    return np.array(trajectories), metadata


def preference_cost(traj):
    y = traj[:, 1]
    dy = np.diff(y)
    ddy = np.diff(y, n=2)
    max_dev = np.max(np.abs(y))

    return (
        3.0 * max_dev ** 2
        + 12.0 * max(0.0, max_dev - 0.30) ** 2
        + 30.0 * max(0.0, max_dev - 0.60) ** 2
        + 1.4 * np.sum(dy ** 2)
        + 2.8 * np.sum(ddy ** 2)
    )


def trajectory_deviation_costs(trajs, linear_traj):
    return np.array([
        float(np.mean(np.linalg.norm(tr - linear_traj, axis=1)))
        for tr in trajs
    ])


def build_snapshot(snapshot_dist, T=31):
    start_h = np.array([-snapshot_dist / 2.0, 0.0])
    goal_h = np.array([snapshot_dist / 2.0, 0.0])

    start_r = np.array([snapshot_dist / 2.0, 0.0])
    goal_r = np.array([-snapshot_dist / 2.0, 0.0])

    H, meta_h = build_structured_library(start_h, goal_h, T=T)
    R, meta_r = build_structured_library(start_r, goal_r, T=T)

    h_linear = make_centerline_trajectory(start_h, goal_h, T)
    r_linear = make_centerline_trajectory(start_r, goal_r, T)

    return H, R, h_linear, r_linear, meta_h, meta_r