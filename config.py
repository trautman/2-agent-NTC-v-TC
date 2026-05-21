# config.py

import yaml
import numpy as np

from constants import (
    DEFAULT_CONFIG_PATH,
    METRIC_ORDER,
    COST_ORDER,
)


def load_config(path=DEFAULT_CONFIG_PATH):
    default_config = {
        "costs_to_run": ["C_NOMINAL"],
        "make_individual_metric_plots": False,
        "make_metric_pages": True,
        "make_pointwise_vs_ot_pages": True,
        "make_snapshot_pngs": False,
        "make_movies": True,
        "movie_costs": ["C_NOMINAL"],
        "movie_metric_sets": {
            "metrics_safety": ["NOMINAL_COST", "NUM_COLLISIONS", "ASD", "MDP"],
            "metrics_coord_effort": ["IMBALANCE", "PSC", "PATH_EFF", "CONTROL_EFFORT"],
        },
        "snapshot_distances": [10.0, 7.5, 5.0, 3.5, 2.5, 1.5, 1.0],
        "s_min": 1.0,
        "s_max": 10.0,
        "s_step": 0.5,
        "make_expected_metric_pages": True,
        "make_gamma_cost_comparison_pages": False,
        "make_coupling_gain_comparison_pages": False,
        "parallel": False,
        "max_workers": None,
        "nominal_time_discount": False,
        "discount_metrics_by_time": False,
        "metrics_to_plot": METRIC_ORDER.copy(),
        "models_to_plot": ["ind", "resp_sample", "resp_marg", "marg"],
        "no_benefit_rel_threshold": 0.05,
        "show_bands": True,
        "make_marginal_pngs": True,
        "ot_backend": "custom",
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

    for metric_name in config["metrics_to_plot"]:
        if metric_name not in METRIC_ORDER:
            raise ValueError(f"Unknown metric in metrics_to_plot: {metric_name}")

    if config["s_step"] <= 0:
        raise ValueError("s_step must be positive")

    if config["s_max"] < config["s_min"]:
        raise ValueError("s_max must be >= s_min")

    if config["ot_backend"] not in ["custom", "pot"]:
        raise ValueError(f"Unknown ot_backend: {config['ot_backend']}")


def build_distance_grid(config):
    distances = np.arange(
        float(config["s_min"]),
        float(config["s_max"]) + 1e-9,
        float(config["s_step"]),
    )
    return np.round(distances, 10)