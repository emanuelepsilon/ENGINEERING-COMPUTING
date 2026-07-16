import numpy as np
from scipy.optimize import root

GLOBAL_TOL = 1e-10 

# ------------------------------------------------------------
# Coupled logistic map
# ------------------------------------------------------------
def F(z, r1, r2, eps):
    """
    One iteration of the coupled logistic map.
    z = [x, y]
    """
    x, y = z

    x_next = (1 - eps) * r1 * x * (1 - x) + eps * r2 * y * (1 - y)
    y_next = (1 - eps) * r2 * y * (1 - y) + eps * r1 * x * (1 - x)

    return np.array([x_next, y_next], dtype=float)


def Fp(z, p, r1, r2, eps):
    """
    Apply the map F p times.
    This gives F^p(z).
    """
    z = np.array(z, dtype=float)

    for _ in range(p):
        z = F(z, r1, r2, eps)

    return z


# ------------------------------------------------------------
# Nonlinear system G_p(z) = F^p(z) - z
# ------------------------------------------------------------
def G(z, p, r1, r2, eps):
    """
    Residual for period-p points.
    We solve G(z) = 0.
    """
    return Fp(z, p, r1, r2, eps) - np.array(z, dtype=float)


# ------------------------------------------------------------
# Check minimal period
# ------------------------------------------------------------
def has_minimal_period(z, p, r1, r2, eps, tol=GLOBAL_TOL):
    """
    Return True if z has minimal period p.
    """
    # Must return after p steps
    if np.linalg.norm(G(z, p, r1, r2, eps), ord=np.inf) > tol:
        return False

    # Must not return earlier
    for k in range(1, p):
        if np.linalg.norm(G(z, k, r1, r2, eps), ord=np.inf) < tol:
            return False

    return True


# ------------------------------------------------------------
# Orbit utilities
# ------------------------------------------------------------
def build_orbit(z, p, r1, r2, eps):
    """
    Return the p points in the orbit starting from z.
    """
    orbit = []
    current = np.array(z, dtype=float)

    for _ in range(p):
        orbit.append(current.copy())
        current = F(current, r1, r2, eps)

    return orbit


def same_orbit(orbit1, orbit2, tol=GLOBAL_TOL):
    """
    Two orbits are treated as the same if one point from orbit1
    is very close to one point from orbit2.
    """
    for a in orbit1:
        for b in orbit2:
            if np.linalg.norm(a - b, ord=np.inf) < tol:
                return True
    return False


# ------------------------------------------------------------
# Initial guesses
# ------------------------------------------------------------
def make_initial_guesses(n=9):
    """
    Create a simple grid of initial guesses in [0,1]x[0,1].
    We avoid exactly 0 and 1.
    """
    grid = np.linspace(0.01, 0.99, n)
    guesses = []

    for x0 in grid:
        for y0 in grid:
            guesses.append(np.array([x0, y0], dtype=float))
    return guesses


# ------------------------------------------------------------
# Find periodic orbits
# ------------------------------------------------------------
def find_periodic_orbits(p, r1, r2, eps, guesses, tol_root=GLOBAL_TOL):
    """
    Find distinct periodic orbits of minimal period p.
    """
    orbits = []
    converging_guesses = []

    for guess in guesses:
        sol = root(lambda z: G(z, p, r1, r2, eps), guess)

        z = np.array(sol.x, dtype=float)

        # Check that the residual is small
        if np.linalg.norm(G(z, p, r1, r2, eps), ord=np.inf) > tol_root:
            continue

        # Keep only solutions in the physical region [0,1]^2
        if np.any(z < 0) or np.any(z > 1):
            continue

        # Check minimal period
        if not has_minimal_period(z, p, r1, r2, eps):
            continue

        orbit = build_orbit(z, p, r1, r2, eps)

        # Avoid printing the same orbit many times
        new_orbit = True
        for i in range(len(orbits)):
            if same_orbit(orbit, orbits[i]):
                converging_guesses[i].append(guess)
                new_orbit = False
                break

        if new_orbit:
            orbits.append(orbit)
            converging_guesses.append([guess])

    return orbits, converging_guesses


# ------------------------------------------------------------
# Direct iteration for comparison
# ------------------------------------------------------------
def long_time_orbit(z0, r1, r2, eps, transient=500, keep=12):
    """
    Iterate the map from one initial point.
    First discard a transient, then keep later iterates.
    """
    z = np.array(z0, dtype=float)

    for _ in range(transient):
        z = F(z, r1, r2, eps)

    points = []
    for _ in range(keep):
        z = F(z, r1, r2, eps)
        points.append(z.copy())

    return points


def observed_period(points, max_period=6, tol=GLOBAL_TOL):
    """
    Very simple check of the period seen in direct iteration.
    Returns p if the list of points repeats with period p.
    Otherwise returns None.
    """
    m = len(points)

    for p in range(1, max_period + 1):
        ok = True
        for i in range(m - p):
            if np.linalg.norm(points[i] - points[i + p], ord=np.inf) > tol:
                ok = False
                break
        if ok:
            return p

    return None


# ------------------------------------------------------------
# Printing
# ------------------------------------------------------------
def print_orbit(orbit, decimals=10):
    for j, point in enumerate(orbit, start=1):
        x, y = point
        print(f"    Point {j}: ({x:.{decimals}f}, {y:.{decimals}f})")


def print_guesses(guess_list, decimals=10, max_show=10):
    print(f"    Number of converging initial guesses: {len(guess_list)}")

    for guess in guess_list[:max_show]:
        x0, y0 = guess
        print(f"      ({x0:.{decimals}f}, {y0:.{decimals}f})")

    if len(guess_list) > max_show:
        print("      ...")


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
if __name__ == "__main__":
    parameter_sets = [
        (3.1, 3.4, 0.3),
        (3.1, 3.55, 0.3),
        (3.1, 3.8, 0.3),
    ]

    guesses = make_initial_guesses(n=9)

    # One ordinary initial point for direct iteration comparison
    z0_iteration = [0.2, 0.4]

    for r1, r2, eps in parameter_sets:
        print("=" * 72)
        print(f"Parameters: r1 = {r1}, r2 = {r2}, eps = {eps}")
        print("=" * 72)

        # Period 2
        print("\nSearching for minimal period-2 orbits...")
        orbits2, guesses2 = find_periodic_orbits(2, r1, r2, eps, guesses)

        if len(orbits2) == 0:
            print("  No period-2 orbit found with these initial guesses.")
        else:
            print(f"  Number of distinct period-2 orbits found: {len(orbits2)}")
            for i, orbit in enumerate(orbits2):
                print(f"\n  Orbit {i + 1}:")
                print_orbit(orbit)
                print("  Initial guesses that converged to this orbit:")
                print_guesses(guesses2[i])

        # Period 3
        print("\nSearching for minimal period-3 orbits...")
        orbits3, guesses3 = find_periodic_orbits(3, r1, r2, eps, guesses)

        if len(orbits3) == 0:
            print("  No period-3 orbit found with these initial guesses.")
        else:
            print(f"  Number of distinct period-3 orbits found: {len(orbits3)}")
            for i, orbit in enumerate(orbits3):
                print(f"\n  Orbit {i + 1}:")
                print_orbit(orbit)
                print("  Initial guesses that converged to this orbit:")
                print_guesses(guesses3[i])

        # Direct iteration comparison
        print("\nDirect iteration comparison from one initial point:")
        points = long_time_orbit(z0_iteration, r1, r2, eps, transient=500, keep=12)
        p_obs = observed_period(points, max_period=6)

        if p_obs is None:
            print("  No small period (up to 6) detected from direct iteration.")
        else:
            print(f"  Direct iteration seems to approach a period-{p_obs} orbit.")

        print("  Last iterates after the transient:")
        for point in points[:6]:
            print(f"    ({point[0]:.10f}, {point[1]:.10f})")

        print()