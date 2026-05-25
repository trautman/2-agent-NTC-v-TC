import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from constants import (
    OUTDIR,
    TOP_K,
    NO_BENEFIT_EPS,
    LAM_PREF,
    METRIC_ORDER,
    METRIC_LABELS,
    METRIC_YLABELS,
    METRIC_BETTER,
    COST_LABELS,
)

from marginals import (
    generate_trajectory_samples,
    preference_cost,
)

from math_utils import softmax_from_logweights



def top_joint_pairs(gamma, k=TOP_K):
    idx = np.argsort(gamma.ravel())[::-1][:k]
    nR = gamma.shape[1]
    return [(rank, flat_idx // nR, flat_idx % nR) for rank, flat_idx in enumerate(idx, start=1)]


def top_robot_indices(q_r, k=TOP_K):
    idx = np.argsort(q_r)[::-1][:k]
    return [(rank, j) for rank, j in enumerate(idx, start=1)]


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



def plot_marginal(ax, trajectories, probs, title):
    ax.axhline(0.0, linewidth=1, color="gray")

    pmax = max(float(np.max(probs)), 1e-12)

    for tr, p in sorted(zip(trajectories, probs), key=lambda x: x[1]):
        weight = float(p / pmax)
        ax.plot(
            tr[:, 0],
            tr[:, 1],
            linewidth=0.4 + 4.0 * weight,
            alpha=0.08 + 0.85 * weight,
            color="green",
        )

    imax = int(np.argmax(probs))
    ax.plot(
        trajectories[imax][:, 0],
        trajectories[imax][:, 1],
        linewidth=3.5,
        color="red",
        label=f"MAP, p={probs[imax]:.4f}",
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


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

def model_plot_style(model_key):
    styles = {
        "ind": {"marker": "o", "linestyle": "-"},
        "resp_sample": {"marker": "s", "linestyle": "--"},
        "resp_marg": {"marker": "^", "linestyle": "-."},
        "joint": {"marker": "D", "linestyle": ":"},
    }
    return styles[model_key]



def save_metric_page(rows_for_cost, cost_name, rel_threshold=0.05, show_bands=True):
    xs = [row["distance_m"] for row in rows_for_cost]
    fig, axes = plt.subplots(6, 2, figsize=(15.0, 16.0), sharex=False)
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
        
        if show_bands:
            thresholds = no_benefit_thresholds(rows_for_cost, metric, rel_threshold)

            ax.fill_between(
                xs,
                -thresholds,
                thresholds,
                color="gray",
                alpha=0.15,
                label="within 5% of NTC KL(marginals)",
            )

            s_cutoff = find_no_benefit_cutoff_relative(xs, panel_series, thresholds)

            if s_cutoff is not None:
                ax.axvline(
                    s_cutoff,
                    color="purple",
                    linewidth=2.0,
                    linestyle=":",
                    label=f"end of coordination benefit: s={s_cutoff:g}",
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




def save_marginal_page(snapshot_dist):
    H, R, h_linear, r_linear, _, _ = generate_trajectory_samples(snapshot_dist)

    pref_h = np.array([preference_cost(h) for h in H])
    pref_r = np.array([preference_cost(r) for r in R])

    p_h = softmax_from_logweights(-LAM_PREF * pref_h)
    p_r = softmax_from_logweights(-LAM_PREF * pref_r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    plot_marginal(
        axes[0],
        H,
        p_h,
        f"Human marginal p_h | s={snapshot_dist:.1f}m",
    )

    plot_marginal(
        axes[1],
        R,
        p_r,
        f"Robot marginal p_r | s={snapshot_dist:.1f}m",
    )

    fig.suptitle("Marginal trajectory distributions: thickness/opacity = probability")
    fig.tight_layout()

    outpath = OUTDIR / f"marginals_{str(snapshot_dist).replace('.', '_')}m.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)



def save_expected_metric_page(rows_for_cost, cost_name):
    xs = [row["distance_m"] for row in rows_for_cost]

    fig, axes = plt.subplots(6, 2, figsize=(15.0, 16.0), sharex=False)
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


def save_individual_expected_metric_plot(
        rows_for_cost,
        cost_name,
        metric,
        models_to_plot,
    ):
    xs = [row["distance_m"] for row in rows_for_cost]

    model_labels = {
        "ind": "TC: p_h p_r",
        "resp_sample": "TC: q_r*delta(h-h*)",
        "resp_marg": "TC: q_r*p_h",
        "joint": "NTC: KL(joint)",
        "marg": "NTC: KL(marginals)",
    }

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for model_key in models_to_plot:
        ys = [row[f"E_{model_key}_{metric}"] for row in rows_for_cost]

        if model_key == "marg":
            ax.plot(xs, ys, linewidth=3.0, label=model_labels[model_key])
        else:
            ax.plot(xs, ys, linewidth=2.0, label=model_labels[model_key])

    ax.set_title(f"Expected {METRIC_LABELS[metric]} values")
    ax.set_ylabel(f"E[{METRIC_LABELS[metric]}]")
    ax.set_xlabel("Start separation s (m)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUTDIR / f"expected_metric_{metric}_c_{COST_LABELS[cost_name]}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def best_tc_value_for_metric(row, metric):
    tc_models = ["ind", "resp_sample", "resp_marg"]
    vals = [row[f"E_{model}_{metric}"] for model in tc_models]

    if METRIC_BETTER[metric] == "larger":
        return max(vals)
    return min(vals)


def no_benefit_thresholds(rows_for_cost, metric, rel_threshold):
    thresholds = []

    for row in rows_for_cost:
        ntc = row[f"E_marg_{metric}"]
        thresholds.append(float(rel_threshold) * abs(float(ntc)))

    return np.array(thresholds, dtype=float)



def find_no_benefit_cutoff_relative(xs, panel_series, thresholds):
    arr = np.vstack(panel_series)
    inside = np.all(np.abs(arr) <= thresholds[None, :], axis=0)

    for k in range(len(xs)):
        if np.all(inside[k:]):
            return xs[k]

    return None




def save_individual_delta_metric_plot(
        rows_for_cost,
        cost_name,
        metric,
        models_to_plot,
        rel_threshold,
        show_bands,
    ):
    xs = [row["distance_m"] for row in rows_for_cost]

    model_labels = {
        "ind": "Perf diff: NTC_marg - TC: p_h p_r",
        "resp_sample": "Perf diff: NTC_marg - TC: q_r*delta(h-h*)",
        "resp_marg": "Perf diff: NTC_marg - TC: q_r*p_h",
        "joint": "Perf diff: NTC_marg - NTC: KL(joint)",
        "marg": "NTC: KL(marginals)",
    }

    delta_models = [m for m in models_to_plot if m != "marg"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    panel_series = []

    for model_key in delta_models:
        ys = np.array(
            [row[f"DeltaE_{model_key}_{metric}"] for row in rows_for_cost],
            dtype=float,
        )
        panel_series.append(ys)

        ax.plot(
            xs,
            ys,
            marker="o",
            linestyle="-",
            linewidth=2.0,
            label=model_labels[model_key],
        )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    if show_bands:
        thresholds = no_benefit_thresholds(rows_for_cost, metric, rel_threshold)

        ax.fill_between(
            xs,
            -thresholds,
            thresholds,
            color="gray",
            alpha=0.15,
            label="within 5% of NTC KL(marginals)",
        )

        s_cutoff = find_no_benefit_cutoff_relative(xs, panel_series, thresholds)

        if s_cutoff is not None:
            ax.axvline(
                s_cutoff,
                color="purple",
                linewidth=2.0,
                linestyle=":",
                label=f"end of coordination benefit: s={s_cutoff:g}",
            )

    ax.set_title(f"Collaboration benefit: {METRIC_LABELS[metric]}")
    ax.set_ylabel(METRIC_YLABELS[metric])
    ax.set_xlabel("Start separation s (m)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUTDIR / f"metric_{metric}_c_{COST_LABELS[cost_name]}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_individual_metric_plots(
        rows_for_cost,
        cost_name,
        metrics_to_plot,
        models_to_plot,
        rel_threshold,
        show_bands,
    ):
    for metric in metrics_to_plot:
        save_individual_expected_metric_plot(
            rows_for_cost,
            cost_name,
            metric,
            models_to_plot,
        )
        save_individual_delta_metric_plot(
            rows_for_cost,
            cost_name,
            metric,
            models_to_plot,
            rel_threshold,
            show_bands,
        )


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

    fig, axes = plt.subplots(6, 2, figsize=(16.0, 16.0), sharex=False)
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


def save_coupling_gain_comparison_page(rows_by_cost, costs_to_compare):
    """
    Plot coupling gain for each gamma*_c model across s.

    coupling_gain = KL(gamma*_c || p_h p_r)

    Each curve = one cost-indexed NTC KL(marginals) model
    """
    if not costs_to_compare:
        return

    first_cost = costs_to_compare[0]
    xs = [row["distance_m"] for row in rows_by_cost[first_cost]]

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    for cost_name in costs_to_compare:
        rows_for_cost = rows_by_cost[cost_name]
        ys = [row["E_marg_COUPLING_GAIN"] for row in rows_for_cost]

        ax.plot(
            xs,
            ys,
            linewidth=2.0,
            linestyle="-",
            label=f"gamma*_{COST_LABELS[cost_name]}",
        )

    ax.set_title("Coupling gain: KL(gamma*_c || p_h p_r)")
    ax.set_ylabel("coupling gain (KL)")
    ax.set_xlabel("Start separation s (m)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45)
    ax.grid(True, alpha=0.25)

    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUTDIR / "coupling_gain_comparison_page.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_pair_vs_gamma_page(rows_for_cost, cost_name):
    xs = [row["distance_m"] for row in rows_for_cost]
    fig, axes = plt.subplots(6, 2, figsize=(15.0, 16.0), sharex=False)
    axes = axes.ravel()
    for ax, metric in zip(axes, METRIC_ORDER):
        ys = [row[f"pointwise_minus_ot_mode_{metric}"] for row in rows_for_cost]
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










