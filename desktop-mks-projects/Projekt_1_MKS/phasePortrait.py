"""
phasePortrait.py

Iterate the coupled map, discard a transient, and plot the long-time orbit
in the phase plane (x_n, y_n).
"""

import os
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CHANGE THESE SETTINGS IF NEEDED
# ============================================================
PARAMETER_SETS = [
    (2.8, 2.9, 0.1),
    (3.1, 3.4, 0.3),
    (3.85, 3.95, 0.2),
]

x0 = 0.21
y0 = 0.37
transient_steps = 1000
keep_steps = 4000
output_folder = "figures"
# ============================================================



def safe_number(x):
    return f"{x:.2f}".replace(".", "p")

def F(x,y, r1, r2, eps):
    """
    One iteration of the coupled logistic map.
    z = [x, y]
    """

    x_next = (1 - eps) * r1 * x * (1 - x) + eps * r2 * y * (1 - y)
    y_next = (1 - eps) * r2 * y * (1 - y) + eps * r1 * x * (1 - x)

    return np.array([x_next, y_next], dtype=float)

def iterate_map(x0, y0, r1, r2, eps, n_steps):
    """
    Return the full orbit of length n_steps+1, including the initial point.
    """
    orbit = np.zeros((n_steps + 1, 2), dtype=float)
    orbit[0] = [x0, y0]

    x, y = x0, y0
    for n in range(n_steps):
        x, y = F(x,y, r1, r2, eps)
        orbit[n + 1] = [x, y]

    return orbit



def make_phase_portrait(r1, r2, eps, x0, y0, transient_steps, keep_steps, output_folder):
    total_steps = transient_steps + keep_steps
    orbit = iterate_map(x0, y0, r1, r2, eps, total_steps)

    long_time_orbit = orbit[transient_steps + 1:]
    x_values = long_time_orbit[:, 0]
    y_values = long_time_orbit[:, 1]

    plt.figure(figsize=(7, 6))
    plt.plot(x_values, y_values, ".", markersize=5)
    plt.xlabel(r"$x_n$")
    plt.ylabel(r"$y_n$")
    plt.title(f"Phase portrait: r1={r1}, r2={r2}, eps={eps}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    filename = (
        f"phase_r1_{safe_number(r1)}_r2_{safe_number(r2)}_eps_{safe_number(eps)}.png"
    )
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300)
    plt.show()

    print(f"Saved figure to: {filepath}")



def main():
    for r1, r2, eps in PARAMETER_SETS:
        make_phase_portrait(
            r1, r2, eps, x0, y0, transient_steps, keep_steps, output_folder
        )


if __name__ == "__main__":
    main()
