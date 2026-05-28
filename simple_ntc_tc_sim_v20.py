import csv
import shutil
import os
from pathlib import Path
# import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter



# REFACTOR
from constants import (
    OUTDIR,
    DEFAULT_FIELD_DISTANCES,
    DEFAULT_MOVIE_DISTANCES,
    LATERAL_LEVELS,
    TOP_K,
    COLLISION_DISTANCE_M,
    NO_BENEFIT_EPS,
    LAM_PREF,
    LAM_RESP_SAMPLE,
    LAM_RESP_MARG,
    LAM_JOINT,
    LAM_H,
    LAM_R,
    ALPHA_H,
    ALPHA_R,
    METRIC_ORDER,
    METRIC_LABELS,
    METRIC_YLABELS,
    METRIC_BETTER,
    COST_ORDER,
    COST_LABELS,
    DEFAULT_CONFIG_PATH,
)

from config_support_functions import (
    load_config,
    validate_config,
    build_distance_grid,
)


from marginals import (
    generate_marginals,
    generate_trajectory_samples,
    preference_cost,
    trajectory_deviation_costs,
)


from math_utils import (
    logsumexp,
    softmax_from_logweights,
    kl_divergence,
    normalize_matrix,
)


from ot_solvers import (
    solve_response,
    solve_joint_kl,
    solve_marginal_kl,
)


from metrics import (
    nominal_pairwise_cost,
    closest_approach,
    metric_mdp,
    metric_mdp_discounted,
    metric_asd,
    pairwise_distance_time_matrix,
    expected_distance_over_time,
    collision_risk_over_time,
    path_length,
    straight_distance,
    metric_path_efficiency_pair,
    metric_control_effort_pair,
    metric_imbalance_pair,
    sign_with_zero,
    metric_psc_pair,
    metric_collision_pair,
    generate_metric_matrices,
    expected_joint,
    expected_robot,
    compute_time_indexed_metrics,
)



from costs import (
    metric_to_cost_matrix,
    generate_cost_matrices,
    response_cost_vector_from_name,
)


from pointwise import solve_pointwise_pair


from plotting import (
    top_joint_pairs,
    top_robot_indices,
    plot_pair,
    add_solution_legend,
    model_plot_style,
    find_no_benefit_cutoff,
    save_marginal_page,
    save_metric_page,
    save_expected_metric_page,
    save_individual_metric_plots,
    save_gamma_cost_comparison_page,
    save_coupling_gain_comparison_page,
    save_pair_vs_gamma_page,
)




def unpack_marginals(marginals):
    return (
        marginals["samples_h"],
        marginals["samples_r"],
        marginals["p_h"],
        marginals["p_r"],
        marginals["h_linear"],
        marginals["r_linear"],
        marginals["meta_h"],
        marginals["meta_r"],
    )





def save_snapshot_five_panel(snapshot_dist, cost_name):
    H, R, h_linear, r_linear, _, _ = generate_trajectory_samples(snapshot_dist)
    sol = compute_expected_metrics_for_models(snapshot_dist, cost_name)

    E_ref = sol["expected_values"]["marg"]
    fig = plt.figure(figsize=(14.0, 12.5))
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    fig.add_subplot(gs[1, 2]).axis("off")
    panels = [("TC: p_h p_r", "ind"), ("TC: q_r*delta(h-h*)", "resp_sample"), ("TC: q_r*p_h", "resp_marg"), ("NTC: KL(joint)", "joint"), ("NTC: KL(marginals)", "marg")]
    for ax, (title, kind) in zip(axes, panels):
        ax.axhline(0.0, linewidth=1, color="gray")
        if kind == "ind":
            for _, i, j in top_joint_pairs(sol["joints"]["ind"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.8, alpha=0.65)
        elif kind == "resp_sample":
            ax.plot(h_linear[:, 0], h_linear[:, 1], color="green", linestyle=":", linewidth=2.2, alpha=0.75)
            for _, j in top_robot_indices(sol["joints"]["resp_sample"]):
                ax.plot(R[j][:, 0], R[j][:, 1], color="green", linewidth=1.8, alpha=0.65)
        elif kind == "resp_marg":
            p_h, q_r = sol["joints"]["resp_marg"]
            top_h = np.argsort(p_h)[::-1][:TOP_K]
            top_r = [j for _, j in top_robot_indices(q_r)]
            for i, j in zip(top_h, top_r):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.8, alpha=0.65)
        elif kind == "joint":
            for _, i, j in top_joint_pairs(sol["joints"]["joint"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.8, alpha=0.65)
        else:
            for _, i, j in top_joint_pairs(sol["joints"]["marg"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.6, alpha=0.50)
            plot_pair(ax, H, R, sol["i_gamma"], sol["j_gamma"], color="red", linewidth=3.4)
            plot_pair(ax, H, R, sol["i_pair"], sol["j_pair"], color="black", linewidth=3.0)
            add_solution_legend(ax)
        E_metrics = sol["expected_values"][kind]
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
    H, R, h_linear, r_linear, _, _ = generate_trajectory_samples(snapshot_dist)
    sol = compute_expected_metrics_for_models(snapshot_dist, cost_name)
    
    E_ref = sol["expected_values"]["marg"]
    panels = [("TC: p_h p_r", "ind"), ("TC: q_r*delta(h-h*)", "resp_sample"), ("TC: q_r*p_h", "resp_marg"), ("NTC: KL(joint)", "joint"), ("NTC: KL(marginals)", "marg")]
    for ax in axes:
        ax.clear()
        ax.axhline(0.0, linewidth=1, color="gray")
    for ax, (title, kind) in zip(axes, panels):
        if kind == "ind":
            for _, i, j in top_joint_pairs(sol["joints"]["ind"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.7, alpha=0.6)
        elif kind == "resp_sample":
            ax.plot(h_linear[:, 0], h_linear[:, 1], color="green", linestyle=":", linewidth=2.0, alpha=0.75)
            for _, j in top_robot_indices(sol["joints"]["resp_sample"]):
                ax.plot(R[j][:, 0], R[j][:, 1], color="green", linewidth=1.7, alpha=0.6)
        elif kind == "resp_marg":
            p_h, q_r = sol["joints"]["resp_marg"]
            top_h = np.argsort(p_h)[::-1][:TOP_K]
            top_r = [j for _, j in top_robot_indices(q_r)]
            for i, j in zip(top_h, top_r):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.7, alpha=0.6)
        elif kind == "joint":
            for _, i, j in top_joint_pairs(sol["joints"]["joint"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.7, alpha=0.6)
        else:
            for _, i, j in top_joint_pairs(sol["joints"]["marg"]):
                plot_pair(ax, H, R, i, j, color="green", linewidth=1.4, alpha=0.45)
            plot_pair(ax, H, R, sol["i_gamma"], sol["j_gamma"], color="red", linewidth=3.2)
            plot_pair(ax, H, R, sol["i_pair"], sol["j_pair"], color="black", linewidth=2.8)
            add_solution_legend(ax)
        E_metrics = sol["expected_values"][kind]
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



def compute_cost_block(
        cost_name,
        field_distances,
        nominal_time_discount=False,
        discount_metrics_by_time=False,
        ot_backend="custom",
    ):
    rows_for_cost = []
    for dist in field_distances:
        row = compute_row_only(
            dist,
            cost_name,
            nominal_time_discount=nominal_time_discount,
            discount_metrics_by_time=discount_metrics_by_time,
            ot_backend=ot_backend,
        )
        rows_for_cost.append(row)
    return cost_name, rows_for_cost












from models import generate_joints
from expected_metric_values import generate_expected_values
from pointwise_vs_ot_mode import compare_pointwise_and_ot_mode_pairs

def compute_expected_metrics_for_models(
        state,
        cost_name,
        nominal_time_discount=False,
        discount_metrics_by_time=False,
        ot_backend="custom",
    ):

    marginals = generate_marginals(state)

    H = marginals["samples_h"]
    R = marginals["samples_r"]
    p_h = marginals["p_h"]
    p_r = marginals["p_r"]
    h_linear = marginals["h_linear"]
    r_linear = marginals["r_linear"]

    metric_matrices = generate_metric_matrices(
        H,
        R,
        h_linear=h_linear,
        nominal_time_discount=nominal_time_discount,
        discount_metrics_by_time=discount_metrics_by_time,
    )

    distance_time_matrix = pairwise_distance_time_matrix(H, R)

    cost_matrices = generate_cost_matrices(metric_matrices["joint"])
    cost_matrix = cost_matrices[cost_name]

    joints = generate_joints(
        p_h,
        p_r,
        cost_matrix,
        cost_name,
        h_linear,
        R,
        nominal_time_discount=nominal_time_discount,
        ot_backend=ot_backend,
    )

    gamma_ind = joints["ind"]
    gamma_resp_sample = joints["resp_sample"]
    gamma_resp_marg = joints["resp_marg"]
    gamma_joint = joints["joint"]
    gamma_marg = joints["marg"]
    q_r_sample = joints["q_r_sample"]
    q_r_marg = joints["q_r_marg"]

    expected_values = generate_expected_values(
        metric_matrices,
        distance_time_matrix,
        gamma_ind,
        gamma_resp_sample,
        gamma_resp_marg,
        gamma_joint,
        gamma_marg,
        q_r_sample,
    )

    i_gamma, j_gamma = np.unravel_index(np.argmax(gamma_marg), gamma_marg.shape)
    i_pair, j_pair = solve_pointwise_pair(H, R, h_linear, r_linear, cost_matrix)

    (
    pointwise_pair_metric_values,
    ot_mode_pair_metric_values,
    pointwise_minus_ot_mode,
        ) = compare_pointwise_and_ot_mode_pairs(
            metric_matrices["joint"],
            i_pair,
            j_pair,
            i_gamma,
            j_gamma,
        )

    return {
        "p_h": p_h,
        "p_r": p_r,

        "joints": {
            "ind": gamma_ind,
            "resp_sample": q_r_sample,
            "resp_marg": (p_h, q_r_marg),
            "joint": gamma_joint,
            "marg": gamma_marg,
        },

        "expected_values": expected_values,

        "i_gamma": i_gamma,
        "j_gamma": j_gamma,
        "i_pair": i_pair,
        "j_pair": j_pair,
        "pointwise_pair_metric_values": pointwise_pair_metric_values,
        "ot_mode_pair_metric_values": ot_mode_pair_metric_values,
        "pointwise_minus_ot_mode": pointwise_minus_ot_mode,
    }





def collaboration_delta(metric, E_model, E_ref):
    return E_ref - E_model if METRIC_BETTER[metric] == "larger" else E_model - E_ref



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
            row[f"E_{model}_{metric}"] = sol["expected_values"][model][metric]
    for model in ["ind", "resp_sample", "resp_marg", "joint"]:
        for metric in METRIC_ORDER:
            row[f"DeltaE_{model}_{metric}"] = collaboration_delta(metric, sol["expected_values"][model][metric], sol["expected_values"]["marg"][metric])
    for metric in METRIC_ORDER:
        row[f"pointwise_minus_ot_mode_{metric}"] = sol["pointwise_minus_ot_mode"][metric]
        row[f"pointwise_pair_{metric}"] = sol["pointwise_pair_metric_values"][metric]
        row[f"ot_mode_pair_{metric}"] = sol["ot_mode_pair_metric_values"][metric]
    return row


def compute_row_only(
        snapshot_dist,
        cost_name,
        nominal_time_discount=False,
        discount_metrics_by_time=False,
        ot_backend="custom",
    ):
    sol = compute_expected_metrics_for_models(
        snapshot_dist,
        cost_name,
        nominal_time_discount=nominal_time_discount,
        discount_metrics_by_time=discount_metrics_by_time,
        ot_backend=ot_backend,
    )
    return make_row(snapshot_dist, cost_name, sol)


def save_metrics_csv(rows):
    path = OUTDIR / "snapshot_metrics_v20.csv"
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_line(prefix, metrics_dict, ref_dict=None):
    order = ["MDP", "ASD", "PATH_EFF"]
    if ref_dict is None:
        return ", ".join([f"{prefix}[{METRIC_LABELS[m]}]={metrics_dict[m]:.3f}" for m in order])
    return ", ".join([f"{prefix}[{METRIC_LABELS[m]}]={collaboration_delta(m, metrics_dict[m], ref_dict[m]):.3f}" for m in order])











def main():
    config = load_config()
    validate_config(config)

    costs_to_run = list(config["costs_to_run"])
    movie_costs = [c for c in config["movie_costs"] if c in costs_to_run]
    field_distances = build_distance_grid(config)
    movie_distances = field_distances[::-1]
    nominal_time_discount = bool(config.get("nominal_time_discount", False))
    discount_metrics_by_time = bool(config.get("discount_metrics_by_time", False))
    ot_backend = config.get("ot_backend", "custom")

    print("Config:")
    print(f"  costs_to_run: {[COST_LABELS[c] for c in costs_to_run]}")
    print(f"  movie_costs: {[COST_LABELS[c] for c in movie_costs]}")
    print(f"  make_snapshot_pngs: {config['make_snapshot_pngs']}")
    print(f"  make_movies: {config['make_movies']}")
    print(f"  s grid: {field_distances[0]:g} to {field_distances[-1]:g} by {config['s_step']:g}")
    print(f"  nominal_time_discount: {nominal_time_discount}")
    print(f"  discount_metrics_by_time: {discount_metrics_by_time}")
    print(f"  ot_backend: {ot_backend}")

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
                executor.submit(
                    compute_cost_block,
                    cost_name,
                    field_distances,
                    nominal_time_discount,
                    discount_metrics_by_time,
                    ot_backend,
                ): cost_name
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
            completed_cost_name, rows_for_cost = compute_cost_block(
                cost_name,
                field_distances,
                nominal_time_discount,
                discount_metrics_by_time,
                ot_backend,
            )
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
                save_metric_page(
                    rows_by_cost[cost_name],
                    cost_name,
                    float(config.get("no_benefit_rel_threshold", 0.05)),
                    config.get("show_bands", True),
                )

            if config["make_pointwise_vs_ot_pages"]:
                print(f"Writing pointwise-vs-OT page for cost: {COST_LABELS[cost_name]}")
                save_pair_vs_gamma_page(rows_by_cost[cost_name], cost_name)

            if config.get("make_individual_metric_plots", False):
                print(f"Writing individual metric plots for cost: {COST_LABELS[cost_name]}")
                save_individual_metric_plots(
                    rows_by_cost[cost_name],
                    cost_name,
                    config["metrics_to_plot"],
                    config["models_to_plot"],
                    float(config.get("no_benefit_rel_threshold", 0.05)),
                    config.get("show_bands", True),
                )


    if config.get("make_gamma_cost_comparison_pages", False):
        print("Writing gamma cost-comparison expected metric page")
        save_gamma_cost_comparison_page(rows_by_cost, costs_to_run)

    if config.get("make_coupling_gain_comparison_pages", False):
        print("Writing coupling gain comparison page")
        save_coupling_gain_comparison_page(rows_by_cost, costs_to_run)

    save_metrics_csv(rows)

    if config["make_snapshot_pngs"]:
        for cost_name in costs_to_run:
            print(f"Writing snapshot PNGs for cost: {COST_LABELS[cost_name]}")
            for dist in config["snapshot_distances"]:
                save_snapshot_five_panel(float(dist), cost_name)

    if config.get("make_marginal_pngs", False):
        print("Writing marginal PNGs")
        for dist in config["snapshot_distances"]:
            save_marginal_page(float(dist))

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
