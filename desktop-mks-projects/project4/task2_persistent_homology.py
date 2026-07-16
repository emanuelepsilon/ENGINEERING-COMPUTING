"""
Task 2: Persistent homology of a planar point cloud

This script:
1. Reads the points from points_lab04.txt.
2. Constructs the Vietoris--Rips complex for many values of r.
3. Builds the boundary matrices d1 and d2 over Z2.
4. Computes beta0(r) and beta1(r).
5. Saves the required plots.

The method follows the lab instruction:
- vertices are the 20 points,
- edges are pairs of points with distance < r,
- triangles are triples where all three pairwise distances are < r,
- Betti numbers are computed from ranks of boundary matrices over Z2.
"""

import os
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Basic input/output settings
# ---------------------------------------------------------------------

DATA_FILE = "points_lab04.txt"
FIGURE_DIR = "figures_task2"


# ---------------------------------------------------------------------
# 2. Linear algebra over Z2
# ---------------------------------------------------------------------

def rank_mod2(matrix):
    """
    Compute the rank of a matrix over Z2.

    Over Z2, all entries are 0 or 1 and addition is done modulo 2.
    This means that 1 + 1 = 0.

    The function uses Gaussian elimination, but row addition is replaced
    by XOR, which is the same as addition modulo 2 for 0/1 entries.
    """
    A = np.array(matrix, dtype=np.uint8) % 2

    if A.size == 0:
        return 0

    n_rows, n_cols = A.shape
    rank = 0
    pivot_row = 0

    for col in range(n_cols):
        # Find a row with a 1 in the current column.
        pivot = None
        for row in range(pivot_row, n_rows):
            if A[row, col] == 1:
                pivot = row
                break

        # If no pivot exists in this column, move to the next column.
        if pivot is None:
            continue

        # Move the pivot row into position.
        if pivot != pivot_row:
            A[[pivot_row, pivot]] = A[[pivot, pivot_row]]

        # Remove all other 1's in this column using row addition mod 2.
        for row in range(n_rows):
            if row != pivot_row and A[row, col] == 1:
                A[row, :] = A[row, :] ^ A[pivot_row, :]

        rank += 1
        pivot_row += 1

        if pivot_row == n_rows:
            break

    return rank


# ---------------------------------------------------------------------
# 3. Vietoris--Rips complex construction
# ---------------------------------------------------------------------

def i(points, r):
    """
    Build the part of the Vietoris--Rips complex needed for beta0 and beta1.

    For this lab we only need:
    - vertices: the points themselves,
    - edges: pairs of points with distance < r,
    - triangles: triples of points where all three edges are present.

    Parameters
    ----------
    points : numpy array of shape (n_points, 2)
        The point cloud.
    r : float
        The scale parameter.

    Returns
    -------
    edges : list of tuples
        Each edge is represented as (i, j) with i < j.
    triangles : list of tuples
        Each triangle is represented as (i, j, k) with i < j < k.
    """
    n_points = len(points)

    edges = []
    for i, j in combinations(range(n_points), 2):
        distance = np.linalg.norm(points[i] - points[j])
        if distance < r:
            edges.append((i, j))

    edge_set = set(edges)

    triangles = []
    for i, j, k in combinations(range(n_points), 3):
        if ((i, j) in edge_set and
            (i, k) in edge_set and
            (j, k) in edge_set):
            triangles.append((i, j, k))

    return edges, triangles


# ---------------------------------------------------------------------
# 4. Boundary matrices over Z2
# ---------------------------------------------------------------------

def boundary_matrix_1(n_vertices, edges):
    """
    Construct the boundary matrix d1: C1 -> C0.

    Rows correspond to vertices.
    Columns correspond to edges.

    If an edge is (i, j), then its boundary consists of the two endpoints
    i and j. Over Z2, the corresponding column therefore has two 1's.
    """
    d1 = np.zeros((n_vertices, len(edges)), dtype=np.uint8)

    for col, (i, j) in enumerate(edges):
        d1[i, col] = 1
        d1[j, col] = 1

    return d1


def boundary_matrix_2(edges, triangles):
    """
    Construct the boundary matrix d2: C2 -> C1.

    Rows correspond to edges.
    Columns correspond to triangles.

    If a triangle is (i, j, k), then its boundary consists of the three
    edges (i, j), (i, k), and (j, k). Over Z2, the corresponding column
    has three 1's.
    """
    edge_to_row = {edge: row for row, edge in enumerate(edges)}
    d2 = np.zeros((len(edges), len(triangles)), dtype=np.uint8)

    for col, (i, j, k) in enumerate(triangles):
        triangle_edges = [(i, j), (i, k), (j, k)]

        for edge in triangle_edges:
            row = edge_to_row[edge]
            d2[row, col] = 1

    return d2


# ---------------------------------------------------------------------
# 5. Betti number computation
# ---------------------------------------------------------------------

def compute_betti_numbers(points, r):
    """
    Compute beta0 and beta1 for the Vietoris--Rips complex at scale r.

    The formulas used are:

        beta0 = dim(C0) - rank(d1)

        beta1 = dim(C1) - rank(d1) - rank(d2)

    where all ranks are computed over Z2.
    """
    n_vertices = len(points)

    edges, triangles = i(points, r)

    d1 = boundary_matrix_1(n_vertices, edges)
    d2 = boundary_matrix_2(edges, triangles)

    rank_d1 = rank_mod2(d1)
    rank_d2 = rank_mod2(d2)

    dim_C0 = n_vertices
    dim_C1 = len(edges)

    beta0 = dim_C0 - rank_d1
    beta1 = dim_C1 - rank_d1 - rank_d2

    return beta0, beta1, len(edges), len(triangles), rank_d1, rank_d2


# ---------------------------------------------------------------------
# 6. Choice of r-values
# ---------------------------------------------------------------------

def choose_r_values(points):
    """
    Choose representative r-values.

    The Vietoris--Rips complex changes only when r crosses a pairwise
    distance between two points. Therefore, instead of using a random grid,
    we use midpoints between consecutive pairwise distances.

    This avoids evaluating exactly at a critical distance. That is useful
    because the lab uses the strict condition distance < r.
    """
    distances = []

    for i, j in combinations(range(len(points)), 2):
        distances.append(np.linalg.norm(points[i] - points[j]))

    distances = np.array(sorted(set(np.round(distances, 12))))

    r_values = [0.0]

    for k in range(len(distances) - 1):
        midpoint = 0.5 * (distances[k] + distances[k + 1])
        r_values.append(midpoint)

    # Add one value slightly larger than the largest pairwise distance.
    r_values.append(distances[-1] + 0.05)

    return np.array(r_values)


# ---------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------

def plot_point_cloud(points):
    """
    Save a plot of the input point cloud.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1])

    # Add point labels. This is useful for checking the data.
    for i, (x, y) in enumerate(points):
        plt.text(x + 0.015, y + 0.015, str(i), fontsize=8)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Point cloud")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "point_cloud.png"), dpi=300)
    plt.close()


def plot_betti_numbers(r_values, beta0_values, beta1_values):
    """
    Save the two required Betti-number plots.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.step(r_values, beta0_values, where="mid")
    plt.scatter(r_values, beta0_values, s=10)
    plt.xlabel("Scale parameter r")
    plt.ylabel(r"$\beta_0(r)$")
    plt.title(r"Number of connected components, $\beta_0(r)$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "beta0_vs_r.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.step(r_values, beta1_values, where="mid")
    plt.scatter(r_values, beta1_values, s=10)
    plt.xlabel("Scale parameter r")
    plt.ylabel(r"$\beta_1(r)$")
    plt.title(r"Number of one-dimensional holes, $\beta_1(r)$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "beta1_vs_r.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 8. Main program
# ---------------------------------------------------------------------

def main():
    """
    Run the complete computation for Task 2.
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. "
            "Place points_lab04.txt in the same folder as this Python file."
        )

    points = np.loadtxt(DATA_FILE)

    plot_point_cloud(points)

    r_values = choose_r_values(points)

    beta0_values = []
    beta1_values = []

    print("Task 2: Persistent homology of the point cloud")
    print("------------------------------------------------")
    print(f"Number of points: {len(points)}")
    print(f"Number of r-values used: {len(r_values)}")
    print()

    print("Selected values:")
    print("r        beta0   beta1   edges   triangles   rank(d1)   rank(d2)")
    print("----------------------------------------------------------------")

    previous_beta0 = None
    previous_beta1 = None

    for r in r_values:
        beta0, beta1, n_edges, n_triangles, rank_d1, rank_d2 = compute_betti_numbers(points, r)

        beta0_values.append(beta0)
        beta1_values.append(beta1)

        # Print a row only when beta0 or beta1 changes.
        # This keeps the terminal output short and useful for the report.
        if beta0 != previous_beta0 or beta1 != previous_beta1:
            print(
                f"{r:0.4f}   {beta0:5d}   {beta1:5d}   "
                f"{n_edges:5d}   {n_triangles:9d}   "
                f"{rank_d1:8d}   {rank_d2:8d}"
            )

        previous_beta0 = beta0
        previous_beta1 = beta1

    beta0_values = np.array(beta0_values)
    beta1_values = np.array(beta1_values)

    plot_betti_numbers(r_values, beta0_values, beta1_values)

    max_beta1 = np.max(beta1_values)
    r_at_max_beta1 = r_values[beta1_values == max_beta1]

    print()
    print("Summary:")
    print(f"Maximum beta1 value: {max_beta1}")
    print(
        "This maximum occurs for r-values approximately between "
        f"{r_at_max_beta1[0]:0.4f} and {r_at_max_beta1[-1]:0.4f}."
    )

    first_connected_indices = np.where(beta0_values == 1)[0]
    if len(first_connected_indices) > 0:
        first_connected_r = r_values[first_connected_indices[0]]
        print(f"The complex first becomes connected at about r = {first_connected_r:0.4f}.")

    print()
    print(f"Figures saved in folder: {FIGURE_DIR}")
    print("Saved files:")
    print("- point_cloud.png")
    print("- beta0_vs_r.png")
    print("- beta1_vs_r.png")


if __name__ == "__main__":
    main()