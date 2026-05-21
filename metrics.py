# metrics.py

import numpy as np

from constants import (
    COLLISION_DISTANCE_M,
    METRIC_ORDER,
)



def nominal_pairwise_cost(tr_h, tr_r, nominal_time_discount=False):
    d = np.linalg.norm(tr_h - tr_r, axis=1)
    t = np.arange(1, len(tr_h) + 1)

    if nominal_time_discount:
        w = 1.0 / t
    else:
        w = np.ones_like(t, dtype=float)

    comfort_barrier = 1.0 / (1.0 + np.exp(12.0 * (d - 0.65)))
    overlap = np.exp(-(d / 0.22) ** 2)
    return float(np.sum(w * (3.0 * comfort_barrier + 7.0 * overlap)))


def closest_approach(tr_h, tr_r):
    d = np.linalg.norm(tr_h - tr_r, axis=1)
    idx = int(np.argmin(d))
    t = idx + 1  # 1-indexed time, consistent with nominal cost
    return float(d[idx]), t


def metric_mdp(tr_h, tr_r):
    d_min, _ = closest_approach(tr_h, tr_r)
    return d_min


def metric_mdp_discounted(tr_h, tr_r):
    d_min, t_min = closest_approach(tr_h, tr_r)
    return float(d_min / float(t_min))


def metric_asd(tr_h, tr_r):
    return float(np.mean(np.linalg.norm(tr_h - tr_r, axis=1)))

def pairwise_distance_time_matrix(H, R):
    N, M = len(H), len(R)
    T = H.shape[1]
    D = np.zeros((N, M, T))

    for i in range(N):
        for j in range(M):
            D[i, j, :] = np.linalg.norm(H[i] - R[j], axis=1)

    return D


def expected_distance_over_time(gamma, D):
    return np.sum(gamma[:, :, None] * D, axis=(0, 1))


def collision_risk_over_time(gamma, D, threshold=COLLISION_DISTANCE_M):
    return np.sum(gamma[:, :, None] * (D < threshold), axis=(0, 1))

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

def metric_collision_pair(tr_h, tr_r, discount_collision_by_time=False):
    d_min, t_min = closest_approach(tr_h, tr_r)

    if d_min > COLLISION_DISTANCE_M:
        return 0.0

    if discount_collision_by_time:
        return 1.0 / float(t_min)

    return 1.0

def compute_pairwise_metric_matrices(
        H,
        R,
        nominal_time_discount=False,
        discount_metrics_by_time=False,
    ):
    N, M = len(H), len(R)
    # mats = {name: np.zeros((N, M)) for name in METRIC_ORDER}
    PAIRWISE_METRICS = [name for name in METRIC_ORDER if name != "COUPLING_GAIN"]
    mats = {name: np.zeros((N, M)) for name in PAIRWISE_METRICS}
    for i in range(N):
        for j in range(M):
            h, r = H[i], R[j]
            mats["NOMINAL_COST"][i, j] = nominal_pairwise_cost(
                h,
                r,
                nominal_time_discount=nominal_time_discount,
            )
            mats["NUM_COLLISIONS"][i, j] = metric_collision_pair(
                h,
                r,
                discount_collision_by_time=discount_metrics_by_time,
            )

            if discount_metrics_by_time:
                mats["MDP"][i, j] = metric_mdp_discounted(h, r)
            else:
                mats["MDP"][i, j] = metric_mdp(h, r)
            mats["ASD"][i, j] = metric_asd(h, r)
            mats["IMBALANCE"][i, j] = metric_imbalance_pair(h, r)
            mats["PSC"][i, j] = metric_psc_pair(h, r)
            mats["PATH_EFF"][i, j] = metric_path_efficiency_pair(h, r)
            # mats["CONTROL_EFFORT"][i, j] = metric_control_effort_pair(h, r)
    return mats


def compute_response_sample_metric_vectors(
        h_linear,
        R,
        nominal_time_discount=False,
        discount_metrics_by_time=False,
    ):
    # vecs = {name: np.zeros(len(R)) for name in METRIC_ORDER}
    PAIRWISE_METRICS = [name for name in METRIC_ORDER if name != "COUPLING_GAIN"]
    vecs = {name: np.zeros(len(R)) for name in PAIRWISE_METRICS}
    for j, r in enumerate(R):
        h = h_linear
        vecs["NOMINAL_COST"][j] = nominal_pairwise_cost(
            h,
            r,
            nominal_time_discount=nominal_time_discount,
        )
        vecs["NUM_COLLISIONS"][j] = metric_collision_pair(
            h,
            r,
            discount_collision_by_time=discount_metrics_by_time,
        )

        if discount_metrics_by_time:
            vecs["MDP"][j] = metric_mdp_discounted(h, r)
        else:
            vecs["MDP"][j] = metric_mdp(h, r)
        vecs["ASD"][j] = metric_asd(h, r)
        vecs["IMBALANCE"][j] = metric_imbalance_pair(h, r)
        vecs["PSC"][j] = metric_psc_pair(h, r)
        vecs["PATH_EFF"][j] = metric_path_efficiency_pair(h, r)
        # vecs["CONTROL_EFFORT"][j] = metric_control_effort_pair(h, r)
    return vecs


def expected_joint(gamma, mat):
    return float(np.sum(gamma * mat))


def expected_robot(q_r, vec):
    return float(np.sum(q_r * vec))

def compute_time_indexed_metrics(gamma, D):
    expected_d_t = expected_distance_over_time(gamma, D)
    collision_risk_t = collision_risk_over_time(gamma, D)

    return {
        "MIN_EXPECTED_DISTANCE": float(np.min(expected_d_t)),
        "MEAN_EXPECTED_DISTANCE": float(np.mean(expected_d_t)),
        "MAX_COLLISION_RISK": float(np.max(collision_risk_t)),
    }

