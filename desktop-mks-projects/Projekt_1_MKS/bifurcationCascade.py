"""
bifurcationCascade.py

Reproduce a bifurcation cascade for the coupled system.
Fix r1 = 3.1 and eps = 0.3, vary r2, discard a transient,
and plot long-time x_n values against r2.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CHANGE THESE SETTINGS IF NEEDED
# ============================================================
r1 = 3.1
eps = 0.3
r2_min = 2.8
r2_max = 3.97
number_of_r2_values = 800
transient_steps = 1000
keep_steps = 250
x0 = 0.21
y0 = 0.37
output_folder = "figures"
# ============================================================



def safe_number(x):
    return f"{x:.2f}".replace(".", "p")

def F(x, y, r1, r2, eps):
    """
    One iteration of the coupled logistic map.
    z = [x, y]
    """
    x_next = (1 - eps) * r1 * x * (1 - x) + eps * r2 * y * (1 - y)
    y_next = (1 - eps) * r2 * y * (1 - y) + eps * r1 * x * (1 - x)

    return np.array([x_next, y_next], dtype=float)

def orbit_tail_for_parameter(r1, r2, eps, x0, y0, transient_steps, keep_steps):
    x, y = x0, y0

    for _ in range(transient_steps):
        x, y = F(x, y, r1, r2, eps)

    x_tail = np.zeros(keep_steps)
    for k in range(keep_steps):
        x, y = F(x, y, r1, r2, eps)
        x_tail[k] = x

    return x_tail



def main():
    r2_values = np.linspace(r2_min, r2_max, number_of_r2_values)

    x_plot = []
    r2_plot = []

    for r2 in r2_values:
        x_tail = orbit_tail_for_parameter(r1, r2, eps, x0, y0, transient_steps, keep_steps)
        x_plot.extend(x_tail)
        r2_plot.extend([r2] * keep_steps)

    plt.figure(figsize=(10, 6))
    plt.plot(r2_plot, x_plot, ".", markersize=0.5)
    plt.xlabel(r"$r_2$")
    plt.ylabel(r"Long-time values of $x_n$")
    plt.title(f"Bifurcation cascade for r1={r1}, eps={eps}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    filename = f"bifurcation_r1_{safe_number(r1)}_eps_{safe_number(eps)}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300)
    plt.show()

    print(f"Saved figure to: {filepath}")


if __name__ == "__main__":
    main()
