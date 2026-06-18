"""Plot ablation-study radar charts for MCFD-ML.

The figure uses mean accuracies from the ablation table only. Standard
deviations are intentionally kept in the manuscript table/caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    "w/o meta",
    "w/o adv",
    "w/o CORAL",
    "w/o Domacc",
    "w/o HSIC",
    "w/o rec",
    "MCFD-ML",
]

TASK_VALUES = {
    "Task 1": [99.58, 93.83, 99.42, 99.35, 99.38, 99.35, 99.44],
    "Task 2": [95.32, 74.46, 97.85, 95.54, 95.62, 95.91, 98.88],
    "Task 3": [99.56, 95.55, 99.41, 99.21, 99.44, 99.27, 99.61],
    "Task 4": [94.16, 70.91, 95.26, 94.98, 96.39, 95.90, 97.66],
    "Task 5": [99.99, 80.44, 99.83, 99.91, 99.84, 99.88, 99.83],
    "Task 6": [99.89, 99.17, 98.84, 99.87, 99.83, 99.89, 99.76],
    "Task 7": [98.22, 66.81, 98.08, 98.29, 98.18, 98.28, 98.37],
    "Task 8": [99.92, 89.43, 99.87, 99.87, 99.84, 99.95, 99.77],
}

STYLE = {
    "w/o meta": {"color": "#6B7280", "ls": "-", "lw": 1.15, "alpha": 0.96},
    "w/o adv": {"color": "#1F77B4", "ls": (0, (4, 2)), "lw": 1.55, "alpha": 0.98},
    "w/o CORAL": {"color": "#F28E2B", "ls": (0, (2, 2)), "lw": 1.15, "alpha": 0.96},
    "w/o Domacc": {"color": "#59A14F", "ls": (0, (6, 2)), "lw": 1.15, "alpha": 0.96},
    "w/o HSIC": {"color": "#B07AA1", "ls": (0, (1, 2)), "lw": 1.15, "alpha": 0.96},
    "w/o rec": {"color": "#00A6A6", "ls": (0, (3, 1, 1, 1)), "lw": 1.15, "alpha": 0.96},
    "MCFD-ML": {"color": "#D62728", "ls": "-", "lw": 2.25, "alpha": 1.0},
}

R_MIN = 65
R_MAX = 100
R_TICKS = [70, 80, 90, 95, 98, 100]
RADIAL_GAMMA = 0.58
FONT_SIZE_PT = 12.0


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": FONT_SIZE_PT,
            "axes.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def closed(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[:1]])


def scale_accuracy(values: np.ndarray) -> np.ndarray:
    """Compress lower accuracies and expand differences near 100%."""
    values = np.asarray(values, dtype=float)
    clipped = np.clip(values, R_MIN, R_MAX)
    normalized_error = (R_MAX - clipped) / (R_MAX - R_MIN)
    return R_MIN + (R_MAX - R_MIN) * (1 - normalized_error**RADIAL_GAMMA)


def panel_matrix(tasks: list[str]) -> np.ndarray:
    matrix = np.array([TASK_VALUES[task] for task in tasks], dtype=float)
    if matrix.shape != (len(tasks), len(METHODS)):
        raise ValueError(f"Unexpected matrix shape: {matrix.shape}")
    return matrix


def draw_radar_panel(
    ax: plt.Axes,
    tasks: list[str],
) -> tuple[list[plt.Line2D], plt.Line2D]:
    values = panel_matrix(tasks)
    scaled_values = scale_accuracy(values)
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False)
    angles_closed = closed(angles)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")
    ax.set_ylim(R_MIN, R_MAX)
    ax.set_xticks(angles)
    ax.set_xticklabels(tasks, fontsize=FONT_SIZE_PT)
    ax.tick_params(axis="x", pad=6)
    ax.set_yticks(scale_accuracy(np.array(R_TICKS)))
    ax.set_yticklabels([str(tick) for tick in R_TICKS], fontsize=FONT_SIZE_PT, color="#4A4A4A")
    ax.set_rlabel_position(25)
    ax.grid(color="#D8DCE0", linewidth=0.55)
    ax.spines["polar"].set_color("#9EA4AA")
    ax.spines["polar"].set_linewidth(0.65)

    handles: list[plt.Line2D] = []
    for method_index, method in enumerate(METHODS):
        method_values = scaled_values[:, method_index]
        style = STYLE[method]
        (line,) = ax.plot(
            angles_closed,
            closed(method_values),
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            alpha=style["alpha"],
            marker="o",
            markersize=2.6 if method == "MCFD-ML" else 2.1,
            markerfacecolor="white" if method != "MCFD-ML" else style["color"],
            markeredgewidth=0.6,
            markeredgecolor=style["color"],
            label=method,
            zorder=5 if method == "MCFD-ML" else 3,
        )
        if method == "MCFD-ML":
            ax.fill(angles_closed, closed(method_values), color=style["color"], alpha=0.105, zorder=1)
        handles.append(line)

    best_by_task = values.max(axis=1)
    best_mask = np.isclose(values, best_by_task[:, None], atol=1e-9)
    for task_index, angle in enumerate(angles):
        for method_index in np.flatnonzero(best_mask[task_index]):
            ax.scatter(
                angle,
                scaled_values[task_index, method_index],
                s=32,
                facecolor="white",
                edgecolor="#111111",
                linewidth=0.75,
                clip_on=False,
                zorder=8,
            )

    best_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#111111",
        markeredgewidth=0.75,
        markersize=4.7,
        label="Task-wise best mean",
    )

    ax.set_title(
        "Ablation study across DIRG and MAFAULDA\n"
        "Tasks 1-4: DIRG; Tasks 5-8: MAFAULDA; radial scale expands 95-100%",
        fontsize=FONT_SIZE_PT,
        fontweight="bold",
        pad=18,
        linespacing=1.45,
    )
    return handles, best_handle


def validate_table_values() -> None:
    expected = {
        ("Task 2", "w/o Domacc"): 95.54,
        ("Task 7", "MCFD-ML"): 98.37,
        ("Task 5", "w/o adv"): 80.44,
        ("Task 8", "w/o rec"): 99.95,
    }
    for (task, method), expected_value in expected.items():
        actual_value = TASK_VALUES[task][METHODS.index(method)]
        if not np.isclose(actual_value, expected_value):
            raise ValueError(f"{task} / {method}: expected {expected_value}, got {actual_value}")


def build_figure() -> plt.Figure:
    configure_matplotlib()
    validate_table_values()

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(8.6, 7.9),
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    all_tasks = list(TASK_VALUES)
    method_handles, best_handle = draw_radar_panel(ax, all_tasks)
    legend_handles = method_handles + [best_handle]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.07),
        fontsize=FONT_SIZE_PT,
        handlelength=2.35,
        columnspacing=1.05,
        handletextpad=0.42,
    )
    fig.subplots_adjust(left=0.075, right=0.925, top=0.86, bottom=0.25)
    return fig


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "figures"
    output_dir.mkdir(exist_ok=True)
    output_base = output_dir / "ablation_radar"

    fig = build_figure()
    fig.savefig(f"{output_base}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{output_base}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for suffix in ("svg", "pdf", "tiff"):
        path = output_base.with_suffix(f".{suffix}")
        print(f"Wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
