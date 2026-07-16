"""
spectralConvergence.py

Markov-chain part of Lab 03.

The lab uses the column-stochastic convention

    x_{t+1} = P x_t.

This script studies P0 and the lazy matrices P0.5 and P0.8. It computes the
stationary distribution, eigenvalues, second-largest eigenvalue modulus,
spectral gap, and total variation convergence curves.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

OUTPUT_DIR = Path("markov_results")

MAX_STEPS = 200
TOLERANCE = 1e-3

INITIAL_STATES = [0, 1, 2, 3]


# ------------------------------------------------------------
# Matrices
# ------------------------------------------------------------

P0 = np.array([
    [0.60, 0.20, 0.15, 0.10],
    [0.20, 0.50, 0.20, 0.20],
    [0.10, 0.20, 0.45, 0.20],
    [0.10, 0.10, 0.20, 0.50],
])


def lazy_matrix(P, alpha):
    """Return P_alpha = alpha*I + (1-alpha)*P."""
    n = P.shape[0]
    return alpha * np.eye(n) + (1 - alpha) * P


matrices = {
    "P0": P0,
    "P0.5": lazy_matrix(P0, 0.5),
    "P0.8": lazy_matrix(P0, 0.8),
}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def check_column_stochastic(P):
    """Check nonnegative entries and column sums equal to 1."""
    nonnegative = np.all(P >= -1e-12)
    column_sums = P.sum(axis=0)
    columns_sum_to_one = np.allclose(column_sums, np.ones(P.shape[1]))
    return nonnegative and columns_sum_to_one


def stationary_distribution(P):
    """
    Compute pi from P*pi = pi.

    Since P is column-stochastic, the stationary distribution is the right
    eigenvector corresponding to eigenvalue 1.
    """
    eigenvalues, eigenvectors = np.linalg.eig(P)

    index = np.argmin(np.abs(eigenvalues - 1))
    pi = np.real(eigenvectors[:, index])

    if pi.sum() < 0:
        pi = -pi

    pi = pi / pi.sum()
    return pi


def spectral_data(P):
    """Return eigenvalues, mu = |lambda_2|, and gap = 1 - mu."""
    eigenvalues = np.linalg.eigvals(P)

    # Sort eigenvalues by decreasing absolute value.
    eigenvalues_sorted = eigenvalues[np.argsort(-np.abs(eigenvalues))]

    mu = abs(eigenvalues_sorted[1])
    gap = 1 - mu

    return eigenvalues_sorted, mu, gap


def total_variation(x, pi):
    """Compute delta(t) = 1/2 sum_i |x_i - pi_i|."""
    return 0.5 * np.sum(np.abs(x - pi))


def simulate(P, pi, initial_state):
    """Simulate x_{t+1} = P*x_t from one concentrated initial state."""
    n = P.shape[0]

    x = np.zeros(n)
    x[initial_state] = 1.0

    times = np.arange(MAX_STEPS + 1)
    distances = []

    for _ in times:
        distances.append(total_variation(x, pi))
        x = P @ x

    return times, np.array(distances)


def first_step_below(distances, tolerance):
    """Return the first time where distance < tolerance."""
    below = np.where(distances < tolerance)[0]

    if len(below) == 0:
        return "not reached"

    return int(below[0])


def save_csv(filename, rows, columns):
    """Save a list of dictionaries to a CSV file."""
    with open(OUTPUT_DIR / filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_matrix_convergence(label, mu, simulation_results):
    """
    Plot total variation distance for one matrix.

    The dashed line is proportional to mu^t. It is used to compare the slope
    on the logarithmic plot, not to match the exact constant.
    """
    plt.figure(figsize=(8, 5))

    largest_initial_distance = max(distances[0] for _, distances in simulation_results)

    for initial_state, (times, distances) in zip(INITIAL_STATES, simulation_results):
        plt.semilogy(
            times,
            np.maximum(distances, 1e-16),
            label=f"initial state {initial_state}",
        )

    reference = largest_initial_distance * mu**times

    plt.semilogy(
        times,
        np.maximum(reference, 1e-16),
        "--",
        label=r"proportional to $\mu^t$",
    )

    plt.axhline(TOLERANCE, linestyle=":", label=r"$10^{-3}$")

    plt.xlabel("t")
    plt.ylabel(r"total variation distance $\delta(t)$")
    plt.title(f"Convergence to stationarity for {label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    safe_label = label.replace(".", "_")
    plt.savefig(OUTPUT_DIR / f"fig_convergence_{safe_label}.png", dpi=300)
    plt.close()


def plot_worst_case(all_worst_curves):
    """Plot the worst initial-state distance for all matrices in one figure."""
    plt.figure(figsize=(8, 5))

    for label, times, worst_distances in all_worst_curves:
        plt.semilogy(times, np.maximum(worst_distances, 1e-16), label=label)

    plt.axhline(TOLERANCE, linestyle=":", label=r"$10^{-3}$")
    plt.xlabel("t")
    plt.ylabel("largest total variation distance")
    plt.title("Worst-case convergence over initial states")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_worst_case_convergence.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    matrix_rows = []
    stationary_rows = []
    eigenvalue_rows = []
    summary_rows = []
    convergence_rows = []
    all_worst_curves = []

    for label, P in matrices.items():
        if not check_column_stochastic(P):
            raise ValueError(f"{label} is not column-stochastic.")

        pi = stationary_distribution(P)
        eigenvalues, mu, gap = spectral_data(P)

        matrix_rows.append({
            "label": label,
            "min_entry": np.min(P),
            "column_sum_0": P[:, 0].sum(),
            "column_sum_1": P[:, 1].sum(),
            "column_sum_2": P[:, 2].sum(),
            "column_sum_3": P[:, 3].sum(),
            "max_error_in_Ppi_minus_pi": np.max(np.abs(P @ pi - pi)),
        })

        stationary_rows.append({
            "label": label,
            "pi_0": pi[0],
            "pi_1": pi[1],
            "pi_2": pi[2],
            "pi_3": pi[3],
            "sum_pi": pi.sum(),
        })

        for order, eigenvalue in enumerate(eigenvalues, start=1):
            eigenvalue_rows.append({
                "label": label,
                "order_by_modulus": order,
                "real_part": np.real(eigenvalue),
                "imaginary_part": np.imag(eigenvalue),
                "modulus": abs(eigenvalue),
            })

        simulation_results = []
        first_steps = []

        for initial_state in INITIAL_STATES:
            times, distances = simulate(P, pi, initial_state)
            simulation_results.append((times, distances))

            first_step = first_step_below(distances, TOLERANCE)
            first_steps.append(first_step)

            for t, delta in zip(times, distances):
                convergence_rows.append({
                    "label": label,
                    "initial_state": initial_state,
                    "t": int(t),
                    "delta": delta,
                    "mu_power_reference": distances[0] * mu**t,
                })

        plot_matrix_convergence(label, mu, simulation_results)

        all_distances = np.array([distances for _, distances in simulation_results])
        worst_distances = np.max(all_distances, axis=0)
        all_worst_curves.append((label, times, worst_distances))

        numerical_steps = [s for s in first_steps if isinstance(s, int)]

        if len(numerical_steps) == len(first_steps):
            observed_step = max(numerical_steps)
        else:
            observed_step = "not reached"

        summary_rows.append({
            "label": label,
            "mu": mu,
            "spectral_gap": gap,
            "observed_worst_step_below_1e_minus_3": observed_step,
        })

    plot_worst_case(all_worst_curves)

    save_csv(
        "matrix_checks.csv",
        matrix_rows,
        [
            "label",
            "min_entry",
            "column_sum_0",
            "column_sum_1",
            "column_sum_2",
            "column_sum_3",
            "max_error_in_Ppi_minus_pi",
        ],
    )

    save_csv(
        "stationary_distributions.csv",
        stationary_rows,
        ["label", "pi_0", "pi_1", "pi_2", "pi_3", "sum_pi"],
    )

    save_csv(
        "eigenvalues.csv",
        eigenvalue_rows,
        ["label", "order_by_modulus", "real_part", "imaginary_part", "modulus"],
    )

    save_csv(
        "convergence_summary.csv",
        summary_rows,
        ["label", "mu", "spectral_gap", "observed_worst_step_below_1e_minus_3"],
    )

    save_csv(
        "total_variation_curves.csv",
        convergence_rows,
        ["label", "initial_state", "t", "delta", "mu_power_reference"],
    )

    print("Done.")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
