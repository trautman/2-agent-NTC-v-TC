from math_utils import normalize_matrix
from metrics import compute_response_sample_metric_vectors




def metric_to_cost_matrix(metric_name, metric_matrix):
    if metric_name == "COUPLING_GAIN":
        raise ValueError("COUPLING_GAIN is distribution-level and cannot be used as a pairwise cost.")
    if metric_name in ["NOMINAL_COST", "NUM_COLLISIONS", "IMBALANCE", "CONTROL_EFFORT"]:
        return metric_matrix.copy()
    if metric_name in ["MDP", "ASD", "PATH_EFF"]:
        return -metric_matrix
    if metric_name == "PSC":
        return (1.0 - metric_matrix) / 2.0
    raise ValueError(f"Unknown metric_name={metric_name}")


def generate_cost_matrices(metric_matrices):
    costs = {
        "C_NOMINAL": metric_matrices["NOMINAL_COST"].copy(),
        "C_NUM_COLLISIONS": metric_to_cost_matrix("NUM_COLLISIONS", metric_matrices["NUM_COLLISIONS"]),
        "C_MDP": metric_to_cost_matrix("MDP", metric_matrices["MDP"]),
        "C_ASD": metric_to_cost_matrix("ASD", metric_matrices["ASD"]),
        "C_IMBALANCE": metric_to_cost_matrix("IMBALANCE", metric_matrices["IMBALANCE"]),
        "C_PSC": metric_to_cost_matrix("PSC", metric_matrices["PSC"]),
    }
    
    combined_terms = [normalize_matrix(costs[name]) for name in [
        "C_NOMINAL",
        "C_NUM_COLLISIONS",
        "C_MDP",
        "C_ASD",
        "C_IMBALANCE",
        "C_PSC",
    ]]
    costs["C_COMBINED"] = sum(combined_terms) / len(combined_terms)
    return costs


def response_cost_vector_from_name(cost_name, h_linear, R, nominal_time_discount=False):
    vecs = compute_response_sample_metric_vectors(
        h_linear,
        R,
        nominal_time_discount=nominal_time_discount,
    )
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
    # if cost_name == "C_CONTROL_EFFORT":
    #     return vecs["CONTROL_EFFORT"]
    if cost_name == "C_COMBINED":
        component_costs = [
            vecs["NOMINAL_COST"],
            vecs["NUM_COLLISIONS"],
            -vecs["MDP"],
            -vecs["ASD"],
            vecs["IMBALANCE"],
            (1.0 - vecs["PSC"]) / 2.0,
        ]
        return sum(normalize_matrix(v) for v in component_costs) / len(component_costs)
    raise ValueError(f"Unknown cost_name={cost_name}")




