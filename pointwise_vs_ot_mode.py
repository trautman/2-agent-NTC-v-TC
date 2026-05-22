# pointwise_vs_ot_mode.py

import numpy as np

from constants import METRIC_ORDER


NON_PAIRWISE_METRICS = [
    "COUPLING_GAIN",
    "MIN_EXPECTED_DISTANCE",
    "MEAN_EXPECTED_DISTANCE",
    "MAX_COLLISION_RISK",
]


def compare_pointwise_and_ot_mode_pairs(
        metric_mats,
        i_pair,
        j_pair,
        i_gamma,
        j_gamma,
    ):
    pointwise_pair_metric_values = {}
    ot_mode_pair_metric_values = {}
    pointwise_minus_ot_mode = {}

    for metric in METRIC_ORDER:
        if metric in NON_PAIRWISE_METRICS:
            pointwise_pair_metric_values[metric] = np.nan
            ot_mode_pair_metric_values[metric] = np.nan
            pointwise_minus_ot_mode[metric] = np.nan
        else:
            pointwise_pair_metric_values[metric] = float(
                metric_mats[metric][i_pair, j_pair]
            )
            ot_mode_pair_metric_values[metric] = float(
                metric_mats[metric][i_gamma, j_gamma]
            )
            pointwise_minus_ot_mode[metric] = (
                pointwise_pair_metric_values[metric]
                - ot_mode_pair_metric_values[metric]
            )

    return (
        pointwise_pair_metric_values,
        ot_mode_pair_metric_values,
        pointwise_minus_ot_mode,
    )