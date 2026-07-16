"""
Task 3: Order in the Vicsek model

This script:
1. Implements the Vicsek model with periodic boundary conditions.
2. Simulates particles for several noise values eta.
3. Computes the polarization/order parameter Phi(t).
4. Saves snapshots of the particle system for low and high noise.
5. Saves a plot of Phi(t) as a function of time.
6. Saves a plot of the time-averaged polarization as a function of eta.

The code is written to follow the lab instruction directly:
- particles move in the periodic square [0, 1)^2,
- each particle aligns with particles within distance R,
- noise is added to the average direction,
- the global order is measured using the polarization Phi(t).
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------

N = 500
T = 1000

V0 = 0.03
R = 0.1

ETA_VALUES = [
    0.0,
    0.05,
    0.1,
    np.pi / 2,
    np.pi,
    2 * np.pi
]

FIGURE_DIR = "figures_task3"

# We use a fixed seed so that the results are reproducible.
SEED = 2026


# ---------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------

def eta_label(eta):
    """
    Create a readable label for a noise value eta.
    """
    if np.isclose(eta, 0.0):
        return "0"
    if np.isclose(eta, 0.05):
        return "0.05"
    if np.isclose(eta, 0.1):
        return "0.1"
    if np.isclose(eta, np.pi / 2):
        return "pi/2"
    if np.isclose(eta, np.pi):
        return "pi"
    if np.isclose(eta, 2 * np.pi):
        return "2pi"
    return f"{eta:.3f}"


def eta_filename(eta):
    """
    Create a file-safe label for a noise value eta.
    """
    return eta_label(eta).replace("/", "_").replace(".", "p")


def polarization(theta):
    """
    Compute the polarization Phi(t).

    Each particle direction is represented by the unit vector

        u_i = (cos(theta_i), sin(theta_i)).

    The polarization is the length of the average direction vector:

        Phi = | (1/N) sum_i u_i |.

    Phi is close to 1 if the particles are aligned.
    Phi is close to 0 if the particles move in random directions.
    """
    mean_cos = np.mean(np.cos(theta))
    mean_sin = np.mean(np.sin(theta))

    return np.sqrt(mean_cos**2 + mean_sin**2)


# ---------------------------------------------------------------------
# 3. Vicsek model simulation
# ---------------------------------------------------------------------

def simulate_vicsek(initial_positions, initial_theta, eta):
    """
    Simulate the Vicsek model for one noise value eta.

    Parameters
    ----------
    initial_positions : numpy array of shape (N, 2)
        Initial particle positions in the square [0, 1)^2.

    initial_theta : numpy array of shape (N,)
        Initial particle directions, given as angles in [0, 2*pi).

    eta : float
        Noise strength.

    Returns
    -------
    positions : numpy array of shape (N, 2)
        Final particle positions.

    theta : numpy array of shape (N,)
        Final particle directions.

    phi_values : numpy array of shape (T + 1,)
        Polarization values from time 0 to time T.
    """
    rng = np.random.default_rng(SEED + int(10000 * eta))

    positions = initial_positions.copy()
    theta = initial_theta.copy()

    phi_values = np.zeros(T + 1)
    phi_values[0] = polarization(theta)

    R_squared = R**2

    for t in range(1, T + 1):

        # -------------------------------------------------------------
        # Periodic distance using the minimum image convention.
        #
        # The domain is [0,1)^2 with periodic boundary conditions.
        # If two particles are close across a boundary, they should still
        # count as neighbours.
        #
        # Example:
        # x = 0.99 and x = 0.01 are distance 0.02 apart periodically,
        # not distance 0.98 apart.
        # -------------------------------------------------------------

        dx = positions[:, 0, None] - positions[:, 0][None, :]
        dy = positions[:, 1, None] - positions[:, 1][None, :]

        dx = dx - np.round(dx)
        dy = dy - np.round(dy)

        distance_squared = dx**2 + dy**2

        neighbours = distance_squared < R_squared

        # -------------------------------------------------------------
        # Average direction of neighbours.
        #
        # Instead of averaging angles directly, we average the vectors
        # (cos(theta), sin(theta)).
        # -------------------------------------------------------------

        sum_cos = neighbours @ np.cos(theta)
        sum_sin = neighbours @ np.sin(theta)

        average_angle = np.arctan2(sum_sin, sum_cos)

        # -------------------------------------------------------------
        # Add noise.
        #
        # The random variables xi_i are uniformly distributed in
        # [-1/2, 1/2], so the noise term is eta * xi_i.
        # -------------------------------------------------------------

        noise = eta * rng.uniform(-0.5, 0.5, size=N)

        theta = average_angle + noise
        theta = theta % (2 * np.pi)

        # -------------------------------------------------------------
        # Move particles forward and apply periodic boundary conditions.
        # The modulo operation keeps all positions inside [0,1)^2.
        # -------------------------------------------------------------

        velocity = V0 * np.column_stack((np.cos(theta), np.sin(theta)))
        positions = (positions + velocity) % 1.0

        phi_values[t] = polarization(theta)

    return positions, theta, phi_values


# ---------------------------------------------------------------------
# 4. Plotting functions
# ---------------------------------------------------------------------

def plot_snapshot(positions, theta, eta):
    """
    Save a snapshot showing particle positions and velocity directions.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    plt.figure(figsize=(6, 6))

    plt.quiver(
        positions[:, 0],
        positions[:, 1],
        np.cos(theta),
        np.sin(theta),
        angles="xy",
        scale_units="xy",
        scale=35,
        width=0.003
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Vicsek model snapshot at t = {T}, eta = {eta_label(eta)}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()

    filename = f"snapshot_eta_{eta_filename(eta)}.png"
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def plot_polarization_time(all_phi_values):
    """
    Save a plot of Phi(t) for all noise values.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    time = np.arange(T + 1)

    plt.figure(figsize=(8, 5))

    for eta, phi_values in all_phi_values.items():
        plt.plot(time, phi_values, label=rf"$\eta = {eta_label(eta)}$")

    plt.xlabel("Time step")
    plt.ylabel(r"Polarization $\Phi(t)$")
    plt.title(r"Polarization as a function of time")
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_DIR, "polarization_vs_time.png"), dpi=300)
    plt.close()


def plot_average_polarization(eta_values, average_phi_values):
    """
    Save a plot of the time-averaged polarization as a function of eta.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    plt.figure(figsize=(7, 5))

    plt.plot(eta_values, average_phi_values, marker="o")

    plt.xlabel(r"Noise strength $\eta$")
    plt.ylabel(r"Time-averaged polarization $\langle \Phi \rangle$")
    plt.title(r"Average order parameter as a function of noise")
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_DIR, "average_polarization_vs_eta.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 5. Main program
# ---------------------------------------------------------------------

def main():
    """
    Run the complete computation for Task 3.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Same initial condition for all eta-values.
    # This makes the comparison between different noise levels fairer.
    initial_positions = rng.random((N, 2))
    initial_theta = rng.uniform(0, 2 * np.pi, size=N)

    all_phi_values = {}
    average_phi_values = []

    # We average Phi(t) over the second half of the simulation.
    # This avoids giving too much weight to the initial transient.
    transient_cutoff = T // 2

    print("Task 3: Vicsek model")
    print("--------------------")
    print(f"N  = {N}")
    print(f"T  = {T}")
    print(f"v0 = {V0}")
    print(f"R  = {R}")
    print()

    print("eta        final Phi       average Phi after transient")
    print("-------------------------------------------------------")

    for eta in ETA_VALUES:
        final_positions, final_theta, phi_values = simulate_vicsek(
            initial_positions,
            initial_theta,
            eta
        )

        all_phi_values[eta] = phi_values

        average_phi = np.mean(phi_values[transient_cutoff:])
        average_phi_values.append(average_phi)

        print(f"{eta_label(eta):>5s}      {phi_values[-1]:8.4f}              {average_phi:8.4f}")

        # The lab asks for snapshots for at least two different noise values.
        # We save one low-noise snapshot and one high-noise snapshot.
        if np.isclose(eta, 0.05) or np.isclose(eta, 2 * np.pi):
            plot_snapshot(final_positions, final_theta, eta)

    plot_polarization_time(all_phi_values)
    plot_average_polarization(np.array(ETA_VALUES), np.array(average_phi_values))

    print()
    print(f"Figures saved in folder: {FIGURE_DIR}")
    print("Saved files:")
    print("- snapshot_eta_0p05.png")
    print("- snapshot_eta_2pi.png")
    print("- polarization_vs_time.png")
    print("- average_polarization_vs_eta.png")


if __name__ == "__main__":
    main()