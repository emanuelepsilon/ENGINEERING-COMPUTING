import numpy as np


def coupled_map(x, y, r1, r2, eps):
    """
    One iteration of the coupled logistic map.

    Parameters
    ----------
    x, y : float
        Current population values.
    r1, r2 : float
        Growth parameters.
    eps : float
        Coupling strength.

    Returns
    -------
    numpy.ndarray with shape (2,)
        [x_{n+1}, y_{n+1}]
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
        x, y = coupled_map(x, y, r1, r2, eps)
        orbit[n + 1] = [x, y]

    return orbit



def iterate_from_point(z, r1, r2, eps, n_steps):
    """
    Same as iterate_map, but the initial point is given as z = [x0, y0].
    """
    return iterate_map(z[0], z[1], r1, r2, eps, n_steps)



def F_iterate(z, p, r1, r2, eps):
    """
    Compute F^p(z), where p is the number of iterations.
    """
    x, y = float(z[0]), float(z[1])
    for _ in range(p):
        x, y = coupled_map(x, y, r1, r2, eps)
    return np.array([x, y], dtype=float)



def G_periodic(z, p, r1, r2, eps):
    """
    Nonlinear system G_p(z) = F^p(z) - z.
    Roots of this function are points on a period-p orbit,
    but they may also include smaller periods, so that must be checked later.
    """
    return F_iterate(z, p, r1, r2, eps) - np.array(z, dtype=float)



def fixed_point_residual(z, r1, r2, eps):
    """
    Residual for fixed points: F(z) - z.
    """
    return G_periodic(z, 1, r1, r2, eps)



def is_close(a, b, tol=1e-8):
    return np.linalg.norm(np.array(a) - np.array(b), ord=np.inf) < tol



def minimal_period(z, max_period_to_check, r1, r2, eps, tol=1e-7):
    """
    Return the minimal period of z among 1, 2, ..., max_period_to_check,
    if one is detected within tolerance.

    If no period up to max_period_to_check is found, return None.
    """
    z = np.array(z, dtype=float)
    for p in range(1, max_period_to_check + 1):
        if is_close(F_iterate(z, p, r1, r2, eps), z, tol=tol):
            return p
    return None



def orbit_points_from_seed(z, p, r1, r2, eps):
    """
    Starting from a point z, return the p points
    z, F(z), F^2(z), ..., F^{p-1}(z).
    """
    points = [np.array(z, dtype=float)]
    current = np.array(z, dtype=float)
    for _ in range(1, p):
        current = F_iterate(current, 1, r1, r2, eps)
        points.append(current.copy())
    return points
