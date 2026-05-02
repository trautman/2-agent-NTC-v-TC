import csv
import shutil
import os
from pathlib import Path
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter

OUTDIR = Path("ntc_tc_sim_outputs_v20")
OUTDIR.mkdir(exist_ok=True)

DEFAULT_FIELD_DISTANCES = np.arange(0.5, 10.0 + 0.001, 0.5)
DEFAULT_MOVIE_DISTANCES = DEFAULT_FIELD_DISTANCES[::-1]
LATERAL_LEVELS = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00], dtype=float)

TOP_K = 8
COLLISION_DISTANCE_M = 0.5
NO_BENEFIT_EPS = 0.05

LAM_PREF = 1.6
LAM_RESP_SAMPLE = 0.25
LAM_RESP_MARG = 0.25
LAM_JOINT = 0.45
LAM_H = 0.30
LAM_R = 0.30
ALPHA_H = LAM_H
ALPHA_R = LAM_R

METRIC_ORDER = [
    "NOMINAL_COST",
    "NUM_COLLISIONS",
    "MDP",
    "ASD",
    "IMBALANCE",
    "PSC",
    "PATH_EFF",
    "CONTROL_EFFORT",
]

METRIC_LABELS = {
    "NOMINAL_COST": "cost",
    "NUM_COLLISIONS": "num_collisions",
    "MDP": "MDP",
    "ASD": "ASD",
    "IMBALANCE": "imbalance",
    "PSC": "PSC",
    "PATH_EFF": "path_efficiency",
    "CONTROL_EFFORT": "control_effort",
}

METRIC_YLABELS = {
    "NOMINAL_COST": "Delta cost",
    "NUM_COLLISIONS": "Delta num_collisions",
    "MDP": "Delta MDP (m)",
    "ASD": "Delta ASD (m)",
    "IMBALANCE": "Delta imbalance (m)",
    "PSC": "Delta PSC",
    "PATH_EFF": "Delta path efficiency",
    "CONTROL_EFFORT": "Delta control effort",
}

METRIC_BETTER = {
    "NOMINAL_COST": "smaller",
    "NUM_COLLISIONS": "smaller",
    "MDP": "larger",
    "ASD": "larger",
    "IMBALANCE": "smaller",
    "PSC": "larger",
    "PATH_EFF": "larger",
    "CONTROL_EFFORT": "smaller",
}

COST_ORDER = [
    "C_NOMINAL",
    "C_NUM_COLLISIONS",
    "C_MDP",
    "C_ASD",
    "C_IMBALANCE",
    "C_PSC",
    "C_CONTROL_EFFORT",
    "C_COMBINED",
]

COST_LABELS = {
    "C_NOMINAL": "c_nominal",
    "C_NUM_COLLISIONS": "c_num_collisions",
    "C_MDP": "c_MDP",
    "C_ASD": "c_ASD",
    "C_IMBALANCE": "c_imbalance",
    "C_PSC": "c_PSC",
    "C_CONTROL_EFFORT": "c_control_effort",
    "C_COMBINED": "c_combined",
}

DEFAULT_CONFIG_PATH = Path("simple_ntc_tc_sim_config.yaml")


def load_config(path=DEFAULT_CONFIG_PATH):
    default_config = {
        "costs_to_run": ["C_NOMINAL"],
        "make_metric_pages": True,
        "make_pointwise_vs_ot_pages": True,
        "make_snapshot_pngs": False,
        "make_movies": True,
        "movie_costs": ["C_NOMINAL"],
        "movie_metric_sets": {
            "metrics_safety": ["NOMINAL_COST", "NUM_COLLISIONS", "ASD", "MDP"],
            "metrics_coord_effort": ["IMBALANCE", "PSC", "PATH_EFF", "CONTROL_EFFORT"],
        },
        "snapshot_distances": [10.0, 7.5, 5.0, 3.5, 2.5, 1.5, 0.5],
        "s_min": 0.5,
        "s_max": 10.0,
        "s_step": 0.5,
        "make_expected_metric_pages": True,
        "make_gamma_cost_comparison_pages": False,
        "parallel": False,
        "max_workers": None,
    }

    if not path.exists():
        return default_config

    with open(path, "r") as f:
        loaded = yaml.safe_load(f) or {}

    config = default_config.copy()
    config.update(loaded)
    return config


def validate_config(config):
    for cost_name in config["costs_to_run"]:
        if cost_name not in COST_ORDER:
            raise ValueError(f"Unknown cost in costs_to_run: {cost_name}")
    for cost_name in config["movie_costs"]:
        if cost_name not in COST_ORDER:
            raise ValueError(f"Unknown cost in movie_costs: {cost_name}")
    for _, metrics in config["movie_metric_sets"].items():
        for metric in metrics:
            if metric not in METRIC_ORDER:
                raise ValueError(f"Unknown metric in movie_metric_sets: {metric}")
    if config["s_step"] <= 0:
        raise ValueError("s_step must be positive")
    if config["s_max"] < config["s_min"]:
        raise ValueError("s_max must be >= s_min")


def build_distance_grid(config):
    distances = np.arange(float(config["s_min"]), float(config["s_max"]) + 1e-9, float(config["s_step"]))
    return np.round(distances, 10)


def make_centerline_trajectory(start, goal, T):
    return np.column_stack([np.linspace(start[0], goal[0], T), np.linspace(start[1], goal[1], T)])


def logsumexp(arr):
    m = np.max(arr)
    return m + np.log(np.sum(np.exp(arr - m)))


def softmax_from_logweights(logw):
    return np.exp(logw - logsumexp(logw))


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
    sign_x = np.sign(goal[0] - start[0]) if abs(goal[0] - start[0]) >= abs(goal[1] - start[1]) else 1.0

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
                metadata.append({"side": int(side), "max_dev": float(dmax), "profile_id": profile_id})

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


def nominal_pairwise_cost(tr_h, tr_r):
    d = np.linalg.norm(tr_h - tr_r, axis=1)
    t = np.arange(1, len(tr_h) + 1)
    w = 1.0 / t
    comfort_barrier = 1.0 / (1.0 + np.exp(12.0 * (d - 0.65)))
    overlap = np.exp(-(d / 0.22) ** 2)
    return float(np.sum(w * (3.0 * comfort_barrier + 7.0 * overlap)))


def metric_mdp(tr_h, tr_r):
    return float(np.min(np.linalg.norm(tr_h - tr_r, axis=1)))


def metric_asd(tr_h, tr_r):
    return float(np.mean(np.linalg.norm(tr_h - tr_r, axis=1)))


def path_length(tr):
    return float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1)))


def straight_distance(tr):
    return float(np.linalg.norm(tr[-1] - tr[0]))


def metric_path_efficiency_pair(tr_h, tr_r):
    eff_h = straight_distance(tr_h) / max(path_length(tr_h), 1e-12)
    eff_r = straight_distance(tr_r) / max(path_length(tr_r), 1e-12)
    return 0.5 * (eff_h + eff_r)


def metric_control_effort_pair(tr_h, tr_r):
    ah = np.diff(tr_h, n=2, axis=0)
    ar = np.diff(tr_r, n=2, axis=0)
    return float(np.sum(np.linalg.norm(ah, axis=1) ** 2) + np.sum(np.linalg.norm(ar, axis=1) ** 2))


def metric_imbalance_pair(tr_h, tr_r):
    return abs(float(np.max(np.abs(tr_h[:, 1]))) - float(np.max(np.abs(tr_r[:, 1]))))


def sign_with_zero(x, eps=1e-9):
    if x > eps:
        return 1.0
    if x < -eps:
        return -1.0
    return 0.0


# def metric_psc_pair(tr_h, tr_r):
#     mid_h = tr_h[len(tr_h) // 2, 1]
#     mid_r = tr_r[len(tr_r) // 2, 1]
#     return float(-sign_with_zero(mid_h) * sign_with_zero(mid_r))
def metric_psc_pair(tr_h, tr_r):
    y_h = tr_h[:, 1]
    y_r = tr_r[:, 1]

    # Ignore endpoints because trajectories begin/end on the centerline.
    y_h = y_h[1:-1]
    y_r = y_r[1:-1]

    psc_t = np.array([
        -sign_with_zero(yh) * sign_with_zero(yr)
        for yh, yr in zip(y_h, y_r)
    ])

    return float(np.mean(psc_t))

def metric_collision_pair(tr_h, tr_r):
    return 1.0 if metric_mdp(tr_h, tr_r) <= COLLISION_DISTANCE_M else 0.0


def normalize_matrix(mat, eps=1e-12):
    mn = float(np.min(mat))
    mx = float(np.max(mat))
    return (mat - mn) / (mx - mn + eps)


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


def compute_pairwise_metric_matrices(H, R):
    N, M = len(H), len(R)
    mats = {name: np.zeros((N, M)) for name in METRIC_ORDER}
    for i in range(N):
        for j in range(M):
            h, r = H[i], R[j]
            mats["NOMINAL_COST"][i, j] = nominal_pairwise_cost(h, r)
            mats["NUM_COLLISIONS"][i, j] = metric_collision_pair(h, r)
            mats["MDP"][i, j] = metric_mdp(h, r)
            mats["ASD"][i, j] = metric_asd(h, r)
            mats["IMBALANCE"][i, j] = metric_imbalance_pair(h, r)
            mats["PSC"][i, j] = metric_psc_pair(h, r)
            mats["PATH_EFF"][i, j] = metric_path_efficiency_pair(h, r)
            mats["CONTROL_EFFORT"][i, j] = metric_control_effort_pair(h, r)
    return mats


def compute_response_sample_metric_vectors(h_linear, R):
    vecs = {name: np.zeros(len(R)) for name in METRIC_ORDER}
    for j, r in enumerate(R):
        h = h_linear
        vecs["NOMINAL_COST"][j] = nominal_pairwise_cost(h, r)
        vecs["NUM_COLLISIONS"][j] = metric_collision_pair(h, r)
        vecs["MDP"][j] = metric_mdp(h, r)
        vecs["ASD"][j] = metric_asd(h, r)
        vecs["IMBALANCE"][j] = metric_imbalance_pair(h, r)
        vecs["PSC"][j] = metric_psc_pair(h, r)
        vecs["PATH_EFF"][j] = metric_path_efficiency_pair(h, r)
        vecs["CONTROL_EFFORT"][j] = metric_control_effort_pair(h, r)
    return vecs


def metric_to_cost_matrix(metric_name, metric_matrix):
    if metric_name in ["NOMINAL_COST", "NUM_COLLISIONS", "IMBALANCE", "CONTROL_EFFORT"]:
        return metric_matrix.copy()
    if metric_name in ["MDP", "ASD", "PATH_EFF"]:
        return -metric_matrix
    if metric_name == "PSC":
        return (1.0 - metric_matrix) / 2.0
    raise ValueError(f"Unknown metric_name={metric_name}")


def build_cost_matrices(metric_mats):
    costs = {
        "C_NOMINAL": metric_mats["NOMINAL_COST"].copy(),
        "C_NUM_COLLISIONS": metric_to_cost_matrix("NUM_COLLISIONS", metric_mats["NUM_COLLISIONS"]),
        "C_MDP": metric_to_cost_matrix("MDP", metric_mats["MDP"]),
        "C_ASD": metric_to_cost_matrix("ASD", metric_mats["ASD"]),
        "C_IMBALANCE": metric_to_cost_matrix("IMBALANCE", metric_mats["IMBALANCE"]),
        "C_PSC": metric_to_cost_matrix("PSC", metric_mats["PSC"]),
        "C_CONTROL_EFFORT": metric_to_cost_matrix("CONTROL_EFFORT", metric_mats["CONTROL_EFFORT"]),
    }
    combined_terms = [normalize_matrix(costs[name]) for name in [
        "C_NOMINAL", "C_NUM_COLLISIONS", "C_MDP", "C_ASD", "C_IMBALANCE", "C_PSC", "C_CONTROL_EFFORT"
    ]]
    costs["C_COMBINED"] = sum(combined_terms) / len(combined_terms)
    return costs


def response_cost_vector_from_name(cost_name, h_linear, R):
    vecs = compute_response_sample_metric_vectors(h_linear, R)
    if cost_name == "C_NOMINAL":
        return vecs["NOMINAL_COST"]
    if cost_name == "C_NUM_COLLISIONS":
        return vecs["NUM_COLLISIONS"]
    if cost_name == "C_MDP":
        return -vecs["MDP"]
    if cost_name == "C_ASD":
        return -vecs["ASD"]
    if cost_name == "C_IMBALANCE":
        return vecs["IMBALANCE"]
    if cost_name == "C_PSC":
        return (1.0 - vecs["PSC"]) / 2.0
    if cost_name == "C_CONTROL_EFFORT":
        return vecs["CONTROL_EFFORT"]
    if cost_name == "C_COMBINED":
        component_costs = [
            vecs["NOMINAL_COST"], vecs["NUM_COLLISIONS"], -vecs["MDP"], -vecs["ASD"],
            vecs["IMBALANCE"], (1.0 - vecs["PSC"]) / 2.0, vecs["CONTROL_EFFORT"]
        ]
        return sum(normalize_matrix(v) for v in component_costs) / len(component_costs)
    raise ValueError(f"Unknown cost_name={cost_name}")


def trajectory_deviation_costs(trajs, linear_traj):
    return np.array([float(np.mean(np.linalg.norm(tr - linear_traj, axis=1))) for tr in trajs])


def solve_response(p_r, costs, lam_resp):
    logw = np.log(p_r + 1e-300) - costs / lam_resp
    return softmax_from_logweights(logw)


def solve_joint_kl(gamma_ind, C, lam_joint):
    log_gamma = np.log(gamma_ind + 1e-300) - C / lam_joint
    return np.exp(log_gamma - logsumexp(log_gamma.ravel()))


def solve_marginal_kl(p_h, p_r, C, lam_h=LAM_H, lam_r=LAM_R, n_iter=5000, eta=0.04):
    gamma = np.outer(p_h, p_r).copy()
    eps = 1e-300
    for _ in range(n_iter):
        alpha = gamma.sum(axis=1)
        beta = gamma.sum(axis=0)
        grad = C + lam_h * (np.log(alpha[:, None] + eps) - np.log(p_h[:, None] + eps) + 1.0) + lam_r * (np.log(beta[None, :] + eps) - np.log(p_r[None, :] + eps) + 1.0)
        gamma *= np.exp(-eta * grad)
        gamma /= gamma.sum()
    return gamma


def expected_joint(gamma, mat):
    return float(np.sum(gamma * mat))


def expected_robot(q_r, vec):
    return float(np.sum(q_r * vec))


def solve_pointwise_pair(H, R, h_linear, r_linear, cost_matrix):
    normalized_cost = normalize_matrix(cost_matrix)
    dev_h = normalize_matrix(trajectory_deviation_costs(H, h_linear))
    dev_r = normalize_matrix(trajectory_deviation_costs(R, r_linear))
    J_pair = normalized_cost + ALPHA_H * dev_h[:, None] + ALPHA_R * dev_r[None, :]
    return np.unravel_index(np.argmin(J_pair), J_pair.shape)


def compute_expected_metrics_for_models(H, R, h_linear, r_linear, cost_name):
    pref_h = np.array([preference_cost(h) for h in H])
    pref_r = np.array([preference_cost(r) for r in R])
    p_h = softmax_from_logweights(-LAM_PREF * pref_h)
    p_r = softmax_from_logweights(-LAM_PREF * pref_r)

    metric_mats = compute_pairwise_metric_matrices(H, R)
    response_vecs = compute_response_sample_metric_vectors(h_linear, R)
    cost_mats = build_cost_matrices(metric_mats)
    C = cost_mats[cost_name]

    gamma_ind = np.outer(p_h, p_r)
    q_r_sample = solve_response(p_r, response_cost_vector_from_name(cost_name, h_linear, R), LAM_RESP_SAMPLE)
    q_r_marg = solve_response(p_r, np.sum(p_h[:, None] * C, axis=0), LAM_RESP_MARG)
    gamma_joint = solve_joint_kl(gamma_ind, C, LAM_JOINT)
    gamma_marg = solve_marginal_kl(p_h, p_r, C)

    E = {model: {} for model in ["ind", "resp_sample", "resp_marg", "joint", "marg"]}
    for metric in METRIC_ORDER:
        E["ind"][metric] = expected_joint(gamma_ind, metric_mats[metric])
        E["resp_sample"][metric] = expected_robot(q_r_sample, response_vecs[metric])
        E["resp_marg"][metric] = float(np.sum((p_h[:, None] * q_r_marg[None, :]) * metric_mats[metric]))
        E["joint"][metric] = expected_joint(gamma_joint, metric_mats[metric])
        E["marg"][metric] = expected_joint(gamma_marg, metric_mats[metric])

    i_gamma, j_gamma = np.unravel_index(np.argmax(gamma_marg), gamma_marg.shape)
    i_pair, j_pair = solve_pointwise_pair(H, R, h_linear, r_linear, C)

    pair_metric_values = {metric: float(metric_mats[metric][i_pair, j_pair]) for metric in METRIC_ORDER}
    gamma_metric_values = {metric: float(metric_mats[metric][i_gamma, j_gamma]) for metric in METRIC_ORDER}
    pair_minus_gamma = {metric: pair_metric_values[metric] - gamma_metric_values[metric] for metric in METRIC_ORDER}

    return {
        "p_h": p_h, "p_r": p_r,
        "model_dists": {
            "ind": gamma_ind, "resp_sample": q_r_sample, "resp_marg": (p_h, q_r_marg),
            "joint": gamma_joint, "marg": gamma_marg,
        },
        "E": E,
        "i_gamma": i_gamma, "j_gamma": j_gamma, "i_pair": i_pair, "j_pair": j_pair,
        "pair_metric_values": pair_metric_values,
        "gamma_metric_values": gamma_metric_values,
        "pair_minus_gamma": pair_minus_gamma,
    }


def collaboration_delta(metric, E_model, E_ref):
    return E_ref - E_model if METRIC_BETTER[metric] == "larger" else E_model - E_ref


def top_joint_pairs(gamma, k=TOP_K):
    idx = np.argsort(gamma.ravel())[::-1][:k]
    nR = gamma.shape[1]
    return [(rank, flat_idx // nR, flat_idx % nR) for rank, flat_idx in enumerate(idx, start=1)]


def top_robot_indices(q_r, k=TOP_K):
    idx = np.argsort(q_r)[::-1][:k]
    return [(rank, j) for rank, j in enumerate(idx, start=1)]


def metric_line(prefix, metrics_dict, ref_dict=None):
    order = ["MDP", "ASD", "PATH_EFF"]
    if ref_dict is None:
        return ", ".join([f"{prefix}[{METRIC_LABELS[m]}]={metrics_dict[m]:.3f}" for m in order])
    return ", ".join([f"{prefix}[{METRIC_LABELS[m]}]={collaboration_delta(m, metrics_dict[m], ref_dict[m]):.3f}" for m in order])


def plot_pair(ax, H, R, i, j, color, linewidth=2.0, linestyle="-", alpha=1.0):
    # Both trajectories use the same style so color encodes only the solution type.
    ax.plot(H[i][:, 0], H[i][:, 1], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    ax.plot(R[j][:, 0], R[j][:, 1], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def add_solution_legend(ax):
    handles = [
        Line2D([0], [0], color="green", lw=2.0, label="top K OT samples"),
        Line2D([0], [0], color="red", lw=3.0, label="OT mode: argmax gamma"),
        Line2D([0], [0], color="black", lw=3.0, label="pointwise optimum: argmin J_pair"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=7)


def make_row(snapshot_dist, cost_name, sol):
    row = {
        "distance_m": snapshot_dist,
        "cost_name": cost_name,
        "cost_label": COST_LABELS[cost_name],
        "i_gamma": sol["i_gamma"], "j_gamma": sol["j_gamma"],
        "i_pair": sol["i_pair"], "j_pair": sol["j_pair"],
    }
    for model in ["ind", "resp_sample", "resp_marg", "joint", "marg"]:
        for metric in METRIC_ORDER:
            row[f"E_{model}_{metric}"] = sol["E"][model][metric]
    for model in ["ind", "resp_sample", "resp_marg", "joint"]:
        for metric in METRIC_ORDER:
            row[f"DeltaE_{model}_{metric}"] = collaboration_delta(metric, sol["E"][model][metric], sol["E"]["marg"][metric])
    for metric in METRIC_ORDER:
        row[f"pair_minus_gamma_{metric}"] = sol["pair_minus_gamma"][metric]
        row[f"pointwise_pair_{metric}"] = sol["pair_metric_values"][metric]
        row[f"ot_mode_pair_{metric}"] = sol["gamma_metric_values"][metric]
    return row


def compute_row_only(snapshot_dist, cost_name):
    H, R, h_linear, r_linear, _, _ = build_snapshot(snapshot_dist)
    sol = compute_expected_metrics_for_models(H, R, h_linear, r_linear, cost_name)
    return make_row(snapshot_dist, cost_name, sol)


def save_metrics_csv(rows):
    path = OUTDIR / "snapshot_metrics_v20.csv"
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def model_plot_style(model_key):
    styles = {
        "ind": {"marker": "o", "linestyle": "-"},
        "resp_sample": {"marker": "s", "linestyle": "--"},
        "resp_marg": {"marker": "^", "linestyle": "-."},
        "joint": {"marker": "D", "linestyle": ":"},
    }
    return styles[model_key]


def find_no_benefit_cutoff(xs, panel_series, eps=NO_BENEFIT_EPS):
    """
    Returns the smallest s such that all model curves remain inside
    [-eps, eps] for that s and every larger s.

    If no such cutoff exists, returns None.
    """
    arr = np.vstack(panel_series)
    inside = np.all(np.abs(arr) <= eps, axis=0)

    for k in range(len(xs)):
        if np.all(inside[k:]):
            return xs[k]

    return None

def save_metric_page(rows_for_cost, cost_name):
    xs = [row["distance_m"] for row in rows_for_cost]
    fig, axes = plt.subplots(4, 2, figsize=(15.0, 16.0), sharex=False)
    axes = axes.ravel()
    model_keys = [
        ("ind", "Perf diff: NTC_marg - TC: p_h p_r"),
        ("resp_sample", "Perf diff: NTC_marg - TC: q_r*delta(h-h*)"),
        ("resp_marg", "Perf diff: NTC_marg - TC: q_r*p_h"),
        ("joint", "Perf diff: NTC_marg - NTC: KL(joint)"),
    ]
    for ax, metric in zip(axes, METRIC_ORDER):
        panel_vals = []
        panel_series = []
        for model_key, label in model_keys:
            # ys = [row[f"DeltaE_{model_key}_{metric}"] for row in rows_for_cost]
            ys = [row[f"DeltaE_{model_key}_{metric}"] for row in rows_for_cost]
            panel_vals.extend(ys)
            panel_series.append(np.array(ys, dtype=float))
            style = model_plot_style(model_key)
            if metric == "PSC":
                style = model_plot_style(model_key)
                ax.plot(
                    xs,
                    ys,
                    marker=style["marker"],
                    linestyle="-",
                    linewidth=2.0,
                    markersize=5,
                    label=label,
                )
            else:
                ax.plot(xs, ys, marker="o", linestyle="-", linewidth=2.0, label=label)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        # ax.axhspan(-0.05, 0.05, color="gray", alpha=0.15)
        ax.axhspan(
            -NO_BENEFIT_EPS,
            NO_BENEFIT_EPS,
            color="gray",
            alpha=0.15,
            label="negligible benefit band"
        )
        s_cutoff = find_no_benefit_cutoff(xs, panel_series)

        if s_cutoff is not None:
            ax.axvline(
                s_cutoff,
                color="purple",
                linewidth=2.0,
                linestyle=":",
                label=f"s*={s_cutoff:g}"
            )
        ax.set_title(f"Collaboration benefit: {METRIC_LABELS[metric]}")
        ax.set_ylabel(METRIC_YLABELS[metric])
        ax.set_xlabel("Start separation s (m)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45)
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Metric improvements for optimization cost: {COST_LABELS[cost_name]}", y=0.995)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"metric_page_{COST_LABELS[cost_name]}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_expected_metric_page(rows_for_cost, cost_name):
    xs = [row["distance_m"] for row in rows_for_cost]
    fig, axes = plt.subplots(4, 2, figsize=(15.0, 16.0), sharex=False)
    axes = axes.ravel()
    model_keys = [("ind", "TC: p_h p_r"), ("resp_sample", "TC: q_r*delta(h-h*)"), ("resp_marg", "TC: q_r*p_h"), ("joint", "NTC: KL(joint)"), ("marg", "NTC: KL(marginals)")]
    for ax, metric in zip(axes, METRIC_ORDER):
        for model_key, label in model_keys:
            ys = [row[f"E_{model_key}_{metric}"] for row in rows_for_cost]
            if model_key == "marg":
                ax.plot(xs, ys, marker="*", linestyle="-", linewidth=2.5, label=label)
            else:
                style = model_plot_style(model_key)
                ax.plot(xs, ys, marker=style["marker"], linestyle=style["linestyle"], linewidth=1.8, label=label)
        ax.set_title(f"Expected {METRIC_LABELS[metric]} by model")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xlabel("Start separation s (m)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45)
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Raw expected metrics for optimization cost: {COST_LABELS[cost_name]}", y=0.995)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"expected_metric_page_{COST_LABELS[cost_name]}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
def save_expected_metric_page(rows_for_cost, cost_name):
    xs = [row["distance_m"] for row in rows_for_cost]

    fig, axes = plt.subplots(4, 2, figsize=(15.0, 16.0), sharex=False)
    axes = axes.ravel()

    model_keys = [
        ("ind", "TC: p_h p_r"),
        ("resp_sample", "TC: q_r*delta(h-h*)"),
        ("resp_marg", "TC: q_r*p_h"),
        ("joint", "NTC: KL(joint)"),
        ("marg", "NTC: KL(marginals)"),
    ]

    for ax, metric in zip(axes, METRIC_ORDER):
        ref = np.array([row[f"E_marg_{metric}"] for row in rows_for_cost], dtype=float)

        panel_vals = []
        for model_key, label in model_keys:
            ys = np.array([row[f"E_{model_key}_{metric}"] for row in rows_for_cost], dtype=float)
            panel_vals.extend(list(ys))

            if model_key == "marg":
                ax.plot(
                    xs,
                    ys,
                    linewidth=3.0,
                    label=label
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    linewidth=2.0,
                    label=label
                )

        max_abs = max(max(abs(v) for v in panel_vals), 1e-12)
        band = 0.10 * max_abs

        ax.fill_between(
            xs,
            ref - band,
            ref + band,
            color="gray",
            alpha=0.12,
            label="within 10% of NTC KL(marginals)"
        )

        ax.set_title(f"Expected {METRIC_LABELS[metric]} values")
        ax.set_ylabel(f"E[{METRIC_LABELS[metric]}]")
        ax.set_xlabel("Start separation s (m)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45)
        ax.grid(True, alpha=0.25)

    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Expected metric values for optimization cost: {COST_LABELS[cost_name]}",
        y=0.995
    )
    fig.tight_layout()
    fig.savefig(
        OUTDIR / f"expected_metric_page_{COST_LABELS[cost_name]}.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)





def save_gamma_cost_comparison_page(rows_by_cost, costs_to_compare):
    """
    Compare the NTC KL(marginals) models indexed by optimization cost.

    Each curve is one gamma*_c model.
    Each panel is one metric: E_{gamma*_c}[metric | s].

    This is not a DeltaE/collaboration-benefit plot.
    """
    if not costs_to_compare:
        return

    first_cost = costs_to_compare[0]
    xs = [row["distance_m"] for row in rows_by_cost[first_cost]]

    fig, axes = plt.subplots(4, 2, figsize=(16.0, 16.0), sharex=False)
    axes = axes.ravel()

    for ax, metric in zip(axes, METRIC_ORDER):
        for cost_name in costs_to_compare:
            rows_for_cost = rows_by_cost[cost_name]
            ys = [row[f"E_marg_{metric}"] for row in rows_for_cost]

            ax.plot(
                xs,
                ys,
                linestyle="-",
                linewidth=2.0,
                label=f"gamma*_{COST_LABELS[cost_name]}",
            )

        ax.set_title(f"Expected {METRIC_LABELS[metric]} for each gamma*_c model")
        ax.set_ylabel(f"E_gamma*[{METRIC_LABELS[metric]}]")
        ax.set_xlabel("Start separation s (m)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45)
        ax.grid(True, alpha=0.25)

    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Comparison of NTC KL(marginals) models indexed by optimization cost",
        y=0.995
    )
    fig.tight_layout()
    fig.savefig(
        OUTDIR / "expected_metric_page_gamma_cost_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)





def save_pair_vs_gamma_page(rows_for_cost, cost_name):
    xs = [row["distance_m"] for row in rows_for_cost]
    fig, axes = plt.subplots(4, 2, figsize=(15.0, 16.0), sharex=False)
    axes = axes.ravel()
    for ax, metric in zip(axes, METRIC_ORDER):
        ys = [row[f"pair_minus_gamma_{metric}"] for row in rows_for_cost]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label="pointwise optimum - OT mode")
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", label="no difference")
        ax.set_title(f"pointwise optimum - OT mode: {METRIC_LABELS[metric]}")
        ax.set_ylabel(f"Delta {METRIC_LABELS[metric]}")
        ax.set_xlabel("Start separation s (m)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45)
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Pointwise optimizer vs OT modal pair for cost: {COST_LABELS[cost_name]}", y=0.995)
    fig.text(0.5, 0.006, "Plotted value = metric(h_pair*, r_pair*) - metric(h_gamma*, r_gamma*); (h_pair*, r_pair*) = argmin J_pair, J_pair = c_norm + alpha_h d_h_norm + alpha_r d_r_norm; (h_gamma*, r_gamma*) = argmax gamma_NTC_marg", ha="center", fontsize=9)
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.985])
    fig.savefig(OUTDIR / f"pointwise_vs_ot_mode_{COST_LABELS[cost_name]}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_snapshot_five_panel(snapshot_dist, cost_name):
    H, R, h_linear, r_linear, _, _ = build_snapshot(snapshot_dist)
    sol = compute_expected_metrics_for_models(H, R, h_linear, r_linear, cost_name)
    E_ref = sol["E"]["marg"]
    fig = plt.figure(figsize=(14.0, 12.5))
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    fig.add_subplot(gs[1, 2]).axis("off")
    panels = [("TC: p_h p_r", "ind"), ("TC: q_r*delta(h-h*)", "resp_sample"), ("TC: q_r*p_h", "resp_marg"), ("NTC: KL(joint)", "joint"), ("NTC: KL(marginals)", "marg")]
    for ax, (title, kind) in zip(axes, panels):
        ax.axhline(0.0, linewidth=1, color="gray")
        if kind == "ind":
            for _, i, j in top_joint_pairs(sol["model_dists"]["ind"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.8, alpha=0.65)
        elif kind == "resp_sample":
            ax.plot(h_linear[:, 0], h_linear[:, 1], color="green", linestyle=":", linewidth=2.2, alpha=0.75)
            for _, j in top_robot_indices(sol["model_dists"]["resp_sample"]):
                ax.plot(R[j][:, 0], R[j][:, 1], color="green", linewidth=1.8, alpha=0.65)
        elif kind == "resp_marg":
            p_h, q_r = sol["model_dists"]["resp_marg"]
            top_h = np.argsort(p_h)[::-1][:TOP_K]
            top_r = [j for _, j in top_robot_indices(q_r)]
            for i, j in zip(top_h, top_r):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.8, alpha=0.65)
        elif kind == "joint":
            for _, i, j in top_joint_pairs(sol["model_dists"]["joint"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.8, alpha=0.65)
        else:
            for _, i, j in top_joint_pairs(sol["model_dists"]["marg"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.6, alpha=0.50)
            plot_pair(ax, H, R, sol["i_gamma"], sol["j_gamma"], color="red", linewidth=3.4)
            plot_pair(ax, H, R, sol["i_pair"], sol["j_pair"], color="black", linewidth=3.0)
            add_solution_legend(ax)
        E_metrics = sol["E"][kind]
        ax.set_title(f"{title} | s={snapshot_dist:.1f}m | {COST_LABELS[cost_name]}\n{metric_line('E', E_metrics)}\n{metric_line('DeltaE', E_metrics, E_ref)}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_aspect("equal")
        ax.set_ylim(-1.25, 1.25)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("y")
    axes[3].set_ylabel("y")
    fig.suptitle("Top K solutions: green; OT mode: red bold; pointwise optimum: black bold", y=0.99)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"snapshot_{COST_LABELS[cost_name]}_{str(snapshot_dist).replace('.', '_')}m.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_five_panel_on_axes(fig, axes, snapshot_dist, cost_name):
    H, R, h_linear, r_linear, _, _ = build_snapshot(snapshot_dist)
    sol = compute_expected_metrics_for_models(H, R, h_linear, r_linear, cost_name)
    E_ref = sol["E"]["marg"]
    panels = [("TC: p_h p_r", "ind"), ("TC: q_r*delta(h-h*)", "resp_sample"), ("TC: q_r*p_h", "resp_marg"), ("NTC: KL(joint)", "joint"), ("NTC: KL(marginals)", "marg")]
    for ax in axes:
        ax.clear()
        ax.axhline(0.0, linewidth=1, color="gray")
    for ax, (title, kind) in zip(axes, panels):
        if kind == "ind":
            for _, i, j in top_joint_pairs(sol["model_dists"]["ind"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.7, alpha=0.6)
        elif kind == "resp_sample":
            ax.plot(h_linear[:, 0], h_linear[:, 1], color="green", linestyle=":", linewidth=2.0, alpha=0.75)
            for _, j in top_robot_indices(sol["model_dists"]["resp_sample"]):
                ax.plot(R[j][:, 0], R[j][:, 1], color="green", linewidth=1.7, alpha=0.6)
        elif kind == "resp_marg":
            p_h, q_r = sol["model_dists"]["resp_marg"]
            top_h = np.argsort(p_h)[::-1][:TOP_K]
            top_r = [j for _, j in top_robot_indices(q_r)]
            for i, j in zip(top_h, top_r):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.7, alpha=0.6)
        elif kind == "joint":
            for _, i, j in top_joint_pairs(sol["model_dists"]["joint"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.7, alpha=0.6)
        else:
            for _, i, j in top_joint_pairs(sol["model_dists"]["marg"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.4, alpha=0.45)
            plot_pair(ax, H, R, sol["i_gamma"], sol["j_gamma"], color="red", linewidth=3.2)
            plot_pair(ax, H, R, sol["i_pair"], sol["j_pair"], color="black", linewidth=2.8)
            add_solution_legend(ax)
        E_metrics = sol["E"][kind]
        ax.set_title(f"{title} | s={snapshot_dist:.1f}m\n{metric_line('E', E_metrics)}\n{metric_line('DeltaE', E_metrics, E_ref)}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_aspect("equal")
        ax.set_ylim(-1.25, 1.25)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("y")
    axes[2].set_ylabel("y")
    axes[4].set_ylabel("y")
    fig.suptitle(f"{COST_LABELS[cost_name]} | green=topK, red=OT mode, black=pointwise optimum", y=0.985)


def render_metric_panels_on_axes(field_axes, rows_for_cost, current_s, metric_subset):
    xs = [row["distance_m"] for row in rows_for_cost]
    model_keys = [("ind", "TC: p_h p_r"), ("resp_sample", "TC: q_r*delta(h-h*)"), ("resp_marg", "TC: q_r*p_h"), ("joint", "NTC: KL(joint)")]
    for ax, metric in zip(field_axes, metric_subset):
        ax.clear()
        panel_vals = []
        panel_series = []
        for model_key, label in model_keys:
            ys = [row[f"DeltaE_{model_key}_{metric}"] for row in rows_for_cost]
            panel_vals.extend(ys)
            panel_series.append(np.array(ys, dtype=float))
            style = model_plot_style(model_key)
            ax.plot(xs, ys, marker="o", linestyle="-", linewidth=2.0, label=label)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

        # ax.axhspan(-0.05, 0.05, color="gray", alpha=0.15)
        ax.axhspan(
            -NO_BENEFIT_EPS,
            NO_BENEFIT_EPS,
            color="gray",
            alpha=0.15
        )
        s_cutoff = find_no_benefit_cutoff(xs, panel_series)

        if s_cutoff is not None:
            ax.axvline(
                s_cutoff,
                color="purple",
                linewidth=2.0,
                linestyle=":",
                label=f"s*={s_cutoff:g}"
            )
        ax.axvline(current_s, color="red", linewidth=2.0, linestyle=":")
        ax.set_title(f"Collaboration benefit: {METRIC_LABELS[metric]}")
        ax.set_ylabel(METRIC_YLABELS[metric])
        ax.set_xlabel("s (m)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45)
        ax.grid(True, alpha=0.25)

    if len(field_axes) > 0:
        field_axes[0].legend(loc="best", fontsize=7)


def save_evolution_movie(rows_by_cost, cost_name, metric_subset, suffix, movie_distances):
    rows_for_cost = rows_by_cost[cost_name]
    fig = plt.figure(figsize=(15.0, 20.0))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 1.05, 0.80, 0.80], hspace=0.38, wspace=0.18)
    behavior_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])]
    field_axes = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]
    s_text = fig.text(0.02, 0.992, "", ha="left", va="top", fontsize=18, fontweight="bold", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=4.0))
    movie_frames = [float(s) for s in movie_distances]
    def update(s):
        render_five_panel_on_axes(fig, behavior_axes, s, cost_name)
        render_metric_panels_on_axes(field_axes, rows_for_cost, s, metric_subset)
        s_text.set_text(f"s = {s:.1f} m")
        return [s_text]
    anim = FuncAnimation(fig, update, frames=movie_frames, interval=700, blit=False, repeat=False)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        from matplotlib.animation import FFMpegWriter
        video_path = OUTDIR / f"evolution_{COST_LABELS[cost_name]}_{suffix}.mp4"
        writer = FFMpegWriter(fps=5, bitrate=1800)
        anim.save(video_path, writer=writer)
    else:
        gif_path = OUTDIR / f"evolution_{COST_LABELS[cost_name]}_{suffix}.gif"
        anim.save(gif_path, writer=PillowWriter(fps=1.5))
    plt.close(fig)



def compute_cost_block(cost_name, field_distances):
    rows_for_cost = []
    for dist in field_distances:
        row = compute_row_only(dist, cost_name)
        rows_for_cost.append(row)
    return cost_name, rows_for_cost


def main():
    config = load_config()
    validate_config(config)

    costs_to_run = list(config["costs_to_run"])
    movie_costs = [c for c in config["movie_costs"] if c in costs_to_run]
    field_distances = build_distance_grid(config)
    movie_distances = field_distances[::-1]

    print("Config:")
    print(f"  costs_to_run: {[COST_LABELS[c] for c in costs_to_run]}")
    print(f"  movie_costs: {[COST_LABELS[c] for c in movie_costs]}")
    print(f"  make_snapshot_pngs: {config['make_snapshot_pngs']}")
    print(f"  make_movies: {config['make_movies']}")
    print(f"  s grid: {field_distances[0]:g} to {field_distances[-1]:g} by {config['s_step']:g}")

    rows = []
    rows_by_cost = {cost_name: [] for cost_name in costs_to_run}

    use_parallel = bool(config.get("parallel", False)) and len(costs_to_run) > 1
    if use_parallel:
        requested = config.get("max_workers", None)
        max_workers = requested or min(len(costs_to_run), os.cpu_count() or 1)
        max_workers = min(max_workers, len(costs_to_run))
        print(f"Using {max_workers} worker processes for {len(costs_to_run)} cost blocks")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_cost_block, cost_name, field_distances): cost_name
                for cost_name in costs_to_run
            }

            for future in as_completed(futures):
                cost_name = futures[future]
                completed_cost_name, rows_for_cost = future.result()
                rows_by_cost[completed_cost_name] = rows_for_cost
                rows.extend(rows_for_cost)
                print(f"Finished cost: {COST_LABELS[cost_name]}")
    else:
        for cost_name in costs_to_run:
            print(f"Computing cost: {COST_LABELS[cost_name]}")
            completed_cost_name, rows_for_cost = compute_cost_block(cost_name, field_distances)
            rows_by_cost[completed_cost_name] = rows_for_cost
            rows.extend(rows_for_cost)

    rows.sort(key=lambda r: (COST_ORDER.index(r["cost_name"]), r["distance_m"]))


    if (
        config["make_metric_pages"]
        or config.get("make_expected_metric_pages", True)
        or config["make_pointwise_vs_ot_pages"]
    ):
        for cost_name in costs_to_run:
            if config.get("make_expected_metric_pages", True):
                print(f"Writing expected metric page for cost: {COST_LABELS[cost_name]}")
                save_expected_metric_page(rows_by_cost[cost_name], cost_name)

            if config["make_metric_pages"]:
                print(f"Writing DeltaE metric page for cost: {COST_LABELS[cost_name]}")
                save_metric_page(rows_by_cost[cost_name], cost_name)

            if config["make_pointwise_vs_ot_pages"]:
                print(f"Writing pointwise-vs-OT page for cost: {COST_LABELS[cost_name]}")
                save_pair_vs_gamma_page(rows_by_cost[cost_name], cost_name)


    if config.get("make_gamma_cost_comparison_pages", False):
        print("Writing gamma cost-comparison expected metric page")
        save_gamma_cost_comparison_page(rows_by_cost, costs_to_run)

    save_metrics_csv(rows)

    if config["make_snapshot_pngs"]:
        for cost_name in costs_to_run:
            print(f"Writing snapshot PNGs for cost: {COST_LABELS[cost_name]}")
            for dist in config["snapshot_distances"]:
                save_snapshot_five_panel(float(dist), cost_name)

    if config["make_movies"]:
        for cost_name in movie_costs:
            print(f"Writing movies for cost: {COST_LABELS[cost_name]}")
            for suffix, metric_subset in config["movie_metric_sets"].items():
                save_evolution_movie(
                    rows_by_cost,
                    cost_name,
                    metric_subset,
                    suffix,
                    movie_distances
                )

    print("Wrote outputs to:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
