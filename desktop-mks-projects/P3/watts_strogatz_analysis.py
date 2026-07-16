"""
watts_strogatz_analysis.py

Watts-Strogatz part of Lab 03.

This script generates Watts-Strogatz graphs G(N,k,p), computes the graph
statistics required in the lab, and saves the main tables and figures used in
the report.

Run with:
    python watts_strogatz_analysis.py
"""

from pathlib import Path
from collections import Counter
import csv

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

N = 200
K = 6

P_VALUES = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

# The lab recommends 1000 realizations if feasible.
N_REALIZATIONS = 1000

SEED = 12345
OUTPUT_DIR = Path("ws_results")


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def largest_component_subgraph(G):
    """Return G itself if connected, otherwise its largest connected component."""
    if nx.is_connected(G):
        return G

    largest_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_nodes).copy()


def graph_statistics(G):
    """
    Compute the graph statistics required in the lab.

    If G is disconnected, diameter and average path length are computed on the
    largest connected component, as stated in the lab instructions.
    """
    degrees = [degree for _, degree in G.degree()]

    components = list(nx.connected_components(G))
    largest_component_size = max(len(component) for component in components)

    H = largest_component_subgraph(G)

    stats = {
        "average_degree": np.mean(degrees),
        "is_connected": int(nx.is_connected(G)),
        "number_of_components": len(components),
        "largest_component_size": largest_component_size,
        "diameter": nx.diameter(H),
        "average_path_length": nx.average_shortest_path_length(H),
        "average_clustering": nx.average_clustering(G),
    }

    return stats, degrees


def mean_and_std(values):
    """Return mean and sample standard deviation."""
    values = np.array(values, dtype=float)

    if len(values) == 1:
        return values[0], 0.0

    return np.mean(values), np.std(values, ddof=1)


def save_csv(filename, rows, columns):
    """Save a list of dictionaries as a CSV file."""
    with open(OUTPUT_DIR / filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def p_label(p):
    """Return a compact label for p."""
    if p == 0:
        return "0"
    return f"{p:g}"


# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

def run_simulation():
    """Generate graphs and collect all statistics."""
    rng = np.random.default_rng(SEED)

    all_rows = []
    degree_counts = {p: Counter() for p in P_VALUES}

    for p in P_VALUES:
        print(f"Running p = {p_label(p)}")

        for realization in range(N_REALIZATIONS):
            graph_seed = int(rng.integers(0, 2**32 - 1))

            G = nx.watts_strogatz_graph(N, K, p, seed=graph_seed)

            stats, degrees = graph_statistics(G)
            degree_counts[p].update(degrees)

            row = {
                "p": p,
                "realization": realization,
                **stats,
            }
            all_rows.append(row)

    return all_rows, degree_counts


def summarize(all_rows):
    """Average the statistics over realizations for each p-value."""
    summary = []

    for p in P_VALUES:
        rows = [row for row in all_rows if row["p"] == p]

        average_degree_mean, average_degree_std = mean_and_std(
            [row["average_degree"] for row in rows]
        )
        clustering_mean, clustering_std = mean_and_std(
            [row["average_clustering"] for row in rows]
        )
        path_mean, path_std = mean_and_std(
            [row["average_path_length"] for row in rows]
        )
        diameter_mean, diameter_std = mean_and_std(
            [row["diameter"] for row in rows]
        )
        components_mean, components_std = mean_and_std(
            [row["number_of_components"] for row in rows]
        )
        largest_component_mean, largest_component_std = mean_and_std(
            [row["largest_component_size"] for row in rows]
        )

        fraction_connected = np.mean([row["is_connected"] for row in rows])

        summary.append({
            "p": p,
            "realizations": len(rows),
            "average_degree_mean": average_degree_mean,
            "average_degree_std": average_degree_std,
            "average_clustering_mean": clustering_mean,
            "average_clustering_std": clustering_std,
            "average_path_length_mean": path_mean,
            "average_path_length_std": path_std,
            "diameter_mean": diameter_mean,
            "diameter_std": diameter_std,
            "fraction_connected": fraction_connected,
            "number_of_components_mean": components_mean,
            "number_of_components_std": components_std,
            "largest_component_size_mean": largest_component_mean,
            "largest_component_size_std": largest_component_std,
        })

    C0 = summary[0]["average_clustering_mean"]
    L0 = summary[0]["average_path_length_mean"]

    for row in summary:
        row["C_over_C0"] = row["average_clustering_mean"] / C0
        row["L_over_L0"] = row["average_path_length_mean"] / L0
        row["C_over_C0_std"] = row["average_clustering_std"] / C0
        row["L_over_L0_std"] = row["average_path_length_std"] / L0

    return summary


# ------------------------------------------------------------
# Save tables
# ------------------------------------------------------------

def save_results(all_rows, summary, degree_counts):
    """Save CSV tables used in the report."""
    save_csv(
        "ws_all_realizations.csv",
        all_rows,
        [
            "p",
            "realization",
            "average_degree",
            "is_connected",
            "number_of_components",
            "largest_component_size",
            "diameter",
            "average_path_length",
            "average_clustering",
        ],
    )

    save_csv(
        "ws_summary.csv",
        summary,
        [
            "p",
            "realizations",
            "average_degree_mean",
            "average_degree_std",
            "average_clustering_mean",
            "average_clustering_std",
            "average_path_length_mean",
            "average_path_length_std",
            "diameter_mean",
            "diameter_std",
            "fraction_connected",
            "number_of_components_mean",
            "number_of_components_std",
            "largest_component_size_mean",
            "largest_component_size_std",
            "C_over_C0",
            "L_over_L0",
            "C_over_C0_std",
            "L_over_L0_std",
        ],
    )

    degree_rows = []

    for p, counter in degree_counts.items():
        total = sum(counter.values())

        for degree in sorted(counter):
            degree_rows.append({
                "p": p,
                "degree": degree,
                "count": counter[degree],
                "probability": counter[degree] / total,
            })

    save_csv(
        "ws_degree_distribution.csv",
        degree_rows,
        ["p", "degree", "count", "probability"],
    )


# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------

def p_axis():
    """Use equally spaced x-values and label them by p."""
    x = np.arange(len(P_VALUES))
    labels = [p_label(p) for p in P_VALUES]
    return x, labels


def plot_small_world(summary):
    """
    Plot the main small-world result:
    normalized clustering and normalized average path length.
    """
    x, labels = p_axis()

    C = [row["C_over_C0"] for row in summary]
    L = [row["L_over_L0"] for row in summary]

    C_std = [row["C_over_C0_std"] for row in summary]
    L_std = [row["L_over_L0_std"] for row in summary]

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, C, yerr=C_std, marker="o", capsize=3, label="C(p) / C(0)")
    plt.errorbar(x, L, yerr=L_std, marker="s", capsize=3, label="L(p) / L(0)")
    plt.xticks(x, labels)
    plt.xlabel("rewiring probability p")
    plt.ylabel("normalized value")
    plt.title("Watts-Strogatz small-world effect")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_normalized_small_world.png", dpi=300)
    plt.close()


def plot_components(summary):
    """Plot connectedness and largest component size."""
    x, labels = p_axis()

    fraction_connected = [row["fraction_connected"] for row in summary]
    largest_component_fraction = [
        row["largest_component_size_mean"] / N for row in summary
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(x, fraction_connected, "o-", label="fraction connected")
    plt.plot(
        x,
        largest_component_fraction,
        "s-",
        label="largest component size / N",
    )
    plt.xticks(x, labels)
    plt.xlabel("rewiring probability p")
    plt.ylabel("fraction")
    plt.ylim(-0.05, 1.05)
    plt.title("Connected components")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_component_stats.png", dpi=300)
    plt.close()


def plot_degree_histograms(degree_counts):
    """Plot pooled degree distributions for a few representative p-values."""
    selected_p_values = [0, 0.01, 0.1, 1]

    fig, axes = plt.subplots(
        len(selected_p_values),
        1,
        figsize=(8, 10),
        sharex=True,
    )

    for ax, p in zip(axes, selected_p_values):
        counter = degree_counts[p]

        degrees = sorted(counter)
        counts = np.array([counter[degree] for degree in degrees])
        probabilities = counts / counts.sum()

        ax.bar(degrees, probabilities)
        ax.set_ylabel("probability")
        ax.set_title(f"degree distribution, p = {p_label(p)}")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("degree")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_degree_histograms.png", dpi=300)
    plt.close()


def save_figures(summary, degree_counts):
    """Save only the figures used in the report."""
    plot_small_world(summary)
    plot_components(summary)
    plot_degree_histograms(degree_counts)


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_rows, degree_counts = run_simulation()
    summary = summarize(all_rows)

    save_results(all_rows, summary, degree_counts)
    save_figures(summary, degree_counts)

    print("Done.")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
