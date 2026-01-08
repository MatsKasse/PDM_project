import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Core Metric Computations
# ============================================================

def compute_path_length(xy):
    """Total Euclidean path length."""
    diffs = np.diff(xy, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


def compute_execution_path(history):
    """Extract executed (x,y) trajectory from history."""
    return np.array([[h[0][0], h[0][1]] for h in history])


def goal_reached(history, goal, threshold):
    final_pos = history[-1][0][:2]
    return np.linalg.norm(final_pos - np.array(goal)) <= threshold


def goal_reach_time(history, dt):
    return len(history) * dt


def path_efficiency(executed_length, planned_length):
    if planned_length == 0:
        return np.nan
    return executed_length / planned_length


# ============================================================
# Main Evaluation Wrapper
# ============================================================

def evaluate_trial(
    history,
    planned_path_xy,
    goal,
    goal_threshold=0.2,
    dt=0.08
):
    executed_xy = compute_execution_path(history)

    metrics = {}
    metrics["success"] = goal_reached(history, goal, goal_threshold)
    metrics["goal_time"] = goal_reach_time(history, dt)
    metrics["executed_path_length"] = compute_path_length(executed_xy)
    metrics["planned_path_length"] = compute_path_length(planned_path_xy)
    metrics["path_efficiency"] = path_efficiency(
        metrics["executed_path_length"],
        metrics["planned_path_length"]
    )

    return metrics, executed_xy


# ============================================================
# Batch Evaluation (Multiple Runs)
# ============================================================

def evaluate_multiple_runs(
    histories,
    planned_path_xy,
    goal,
    goal_threshold=0.2,
    dt=0.08
):
    records = []
    trajectories = []

    for i, hist in enumerate(histories):
        metrics, traj = evaluate_trial(
            hist, planned_path_xy, goal, goal_threshold, dt
        )
        metrics["run"] = i
        records.append(metrics)
        trajectories.append(traj)

    df = pd.DataFrame(records)
    return df, trajectories


# ============================================================
# Plotting Utilities
# ============================================================

def plot_trajectories(trajectories, planned_path):
    plt.figure()
    for traj in trajectories:
        plt.plot(traj[:,0], traj[:,1], alpha=0.4)
    plt.plot(planned_path[:,0], planned_path[:,1], 'k--', linewidth=2, label="Planned path")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Executed Trajectories")
    plt.legend()
    plt.grid()
    plt.show()


def plot_metrics(df):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].boxplot(df["goal_time"])
    axs[0].set_title("Goal Reach Time [s]")

    axs[1].boxplot(df["executed_path_length"])
    axs[1].set_title("Executed Path Length [m]")

    axs[2].boxplot(df["path_efficiency"])
    axs[2].set_title("Path Efficiency")

    for ax in axs:
        ax.grid()

    plt.tight_layout()
    plt.show()

def plot_metrics_test(metrics_list, resolutions):
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].boxplot(metrics_list["computation_times"])
    axs[0].set_title("A* Computation Time [s]")

    axs[1].boxplot(metrics_list["path_lengths"])
    axs[1].set_title("Computed Path Length [m]")

    axs_2_twin = axs[2].twinx()
    axs[2].plot(resolutions, np.array(metrics_list["computation_times"]), 'o-')
    axs_2_twin.plot(resolutions, metrics_list["path_lengths"], 'ro-')
    axs_2_twin.set_ylim(max(metrics_list["path_lengths"]) * 0.85, max(metrics_list["path_lengths"]) * 1.15)
    axs_2_twin.set_ylabel("Path Length [m]", color='r')
    axs[2].set_title("Computation Time vs Resolution")
    axs[2].set_xlabel("Grid Resolution [m]")
    axs[2].set_ylabel("Computation Time [s]", color='b')

    for ax in axs:
        ax.grid()

    plt.tight_layout()
    plt.show()

# ============================================================
# Summary Table (LaTeX-ready)
# ============================================================

def summarize_metrics(df):
    summary = df.agg({
        "success": "mean",
        "goal_time": ["mean", "std"],
        "executed_path_length": ["mean", "std"],
        "planned_path_length": "mean",
        "path_efficiency": ["mean", "std"]
    })

    summary.loc["mean", "success"] *= 100
    return summary
