# expected_metric_values.py

from math_utils import kl_divergence

from metrics import (
    compute_time_indexed_metrics,
    expected_joint,
    expected_robot,
)

from constants import METRIC_ORDER


def generate_expected_values(
        metric_matrices,
        D_time,
        gamma_ind,
        gamma_resp_sample,
        gamma_resp_marg,
        gamma_joint,
        gamma_marg,
        q_r_sample,
    ):


    joint_metric_matrices = metric_matrices["joint"]
    response_metric_vectors = metric_matrices["response"]

    coupling_gain = {
        "ind": kl_divergence(gamma_ind, gamma_ind),
        "resp_sample": kl_divergence(gamma_resp_sample, gamma_ind),
        "resp_marg": kl_divergence(gamma_resp_marg, gamma_ind),
        "joint": kl_divergence(gamma_joint, gamma_ind),
        "marg": kl_divergence(gamma_marg, gamma_ind),
    }

    time_metrics = {
        "ind": compute_time_indexed_metrics(gamma_ind, D_time),
        "resp_sample": compute_time_indexed_metrics(gamma_resp_sample, D_time),
        "resp_marg": compute_time_indexed_metrics(gamma_resp_marg, D_time),
        "joint": compute_time_indexed_metrics(gamma_joint, D_time),
        "marg": compute_time_indexed_metrics(gamma_marg, D_time),
    }

    E = {
        model: {}
        for model in ["ind", "resp_sample", "resp_marg", "joint", "marg"]
    }

    for metric in METRIC_ORDER:
        if metric == "COUPLING_GAIN":
            for model in E:
                E[model][metric] = coupling_gain[model]

        elif metric in time_metrics["ind"]:
            for model in E:
                E[model][metric] = time_metrics[model][metric]

        else:
            E["ind"][metric] = expected_joint(gamma_ind, joint_metric_matrices[metric])
            E["resp_sample"][metric] = expected_robot(
                q_r_sample,
                response_metric_vectors[metric],
            )
            E["resp_marg"][metric] = expected_joint(
                gamma_resp_marg,
                joint_metric_matrices[metric],
            )
            E["joint"][metric] = expected_joint(
                gamma_joint,
                joint_metric_matrices[metric],
            )
            E["marg"][metric] = expected_joint(
                gamma_marg,
                joint_metric_matrices[metric],
            )

    return E