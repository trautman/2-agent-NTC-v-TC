# constants.py

from pathlib import Path
import numpy as np

OUTDIR = Path("ntc_tc_sim_outputs_v20")
OUTDIR.mkdir(exist_ok=True)

DEFAULT_FIELD_DISTANCES = np.arange(0.5, 10.0 + 0.001, 0.5)
DEFAULT_MOVIE_DISTANCES = DEFAULT_FIELD_DISTANCES[::-1]
LATERAL_LEVELS = np.array(
    [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00],
    dtype=float,
)

TOP_K = 8
COLLISION_DISTANCE_M = 0.6
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
    "COUPLING_GAIN",
    "NUM_COLLISIONS",
    "MDP",
    "ASD",
    "MIN_EXPECTED_DISTANCE",
    "MEAN_EXPECTED_DISTANCE",
    "MAX_COLLISION_RISK",
    "IMBALANCE",
    "PSC",
    "PATH_EFF",
]


METRIC_LABELS = {
    "NOMINAL_COST": "cost",
    "COUPLING_GAIN": "coupling_gain",
    "NUM_COLLISIONS": "num_collisions",
    "MDP": "MDP",
    "ASD": "ASD",
    "IMBALANCE": "imbalance",
    "PSC": "PSC",
    "PATH_EFF": "path_efficiency",
    "MIN_EXPECTED_DISTANCE": "min_expected_distance",
    "MEAN_EXPECTED_DISTANCE": "mean_expected_distance",
    "MAX_COLLISION_RISK": "max_collision_risk",
}

METRIC_YLABELS = {
    "NOMINAL_COST": "Delta cost",
    "COUPLING_GAIN": "Delta coupling gain",
    "NUM_COLLISIONS": "Delta num_collisions",
    "MDP": "Delta MDP (m)",
    "ASD": "Delta ASD (m)",
    "IMBALANCE": "Delta imbalance (m)",
    "PSC": "Delta PSC",
    "PATH_EFF": "Delta path efficiency",
    "MIN_EXPECTED_DISTANCE": "Delta min_expected_distance (m)",
    "MEAN_EXPECTED_DISTANCE": "Delta mean_expected_distance (m)",
    "MAX_COLLISION_RISK": "Delta max_collision_risk",
}


METRIC_BETTER = {
    "NOMINAL_COST": "smaller",
    "COUPLING_GAIN": "larger",
    "NUM_COLLISIONS": "smaller",
    "MDP": "larger",
    "ASD": "larger",
    "IMBALANCE": "smaller",
    "PSC": "larger",
    "PATH_EFF": "larger",
    "MIN_EXPECTED_DISTANCE": "larger",
    "MEAN_EXPECTED_DISTANCE": "larger",
    "MAX_COLLISION_RISK": "smaller",
}

COST_ORDER = [
    "C_NOMINAL",
    "C_NUM_COLLISIONS",
    "C_MDP",
    "C_ASD",
    "C_IMBALANCE",
    "C_PSC",
    # "C_CONTROL_EFFORT",
    "C_COMBINED",
]

COST_LABELS = {
    "C_NOMINAL": "c_nominal",
    "C_NUM_COLLISIONS": "c_num_collisions",
    "C_MDP": "c_MDP",
    "C_ASD": "c_ASD",
    "C_IMBALANCE": "c_imbalance",
    "C_PSC": "c_PSC",
    # "C_CONTROL_EFFORT": "c_control_effort",
    "C_COMBINED": "c_combined",
}

DEFAULT_CONFIG_PATH = Path("simple_ntc_tc_sim_config.yaml")