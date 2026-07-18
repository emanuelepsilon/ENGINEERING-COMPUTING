from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
REPORTS_DIR = ROOT / "reports"


def ricker(t: float, frequency: float, delay: float) -> float:
    tau = np.pi * frequency * (t - delay)
    return float((1.0 - 2.0 * tau * tau) * np.exp(-tau * tau))


def laplacian_4th_order(u: np.ndarray, dx: float) -> np.ndarray:
    lap = np.zeros_like(u)

    lap[2:-2, 2:-2] = (
        -u[2:-2, 4:]
        + 16.0 * u[2:-2, 3:-1]
        - 30.0 * u[2:-2, 2:-2]
        + 16.0 * u[2:-2, 1:-3]
        - u[2:-2, :-4]
        - u[4:, 2:-2]
        + 16.0 * u[3:-1, 2:-2]
        - 30.0 * u[2:-2, 2:-2]
        + 16.0 * u[1:-3, 2:-2]
        - u[:-4, 2:-2]
    ) / (12.0 * dx * dx)

    # Second-order fallback close to the boundary.
    lap[1:-1, 1:-1] = np.where(
        lap[1:-1, 1:-1] == 0.0,
        (
            u[1:-1, 2:]
            + u[1:-1, :-2]
            + u[2:, 1:-1]
            + u[:-2, 1:-1]
            - 4.0 * u[1:-1, 1:-1]
        )
        / (dx * dx),
        lap[1:-1, 1:-1],
    )

    return lap


def build_velocity_model(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)

    c = np.full((n, n), 1.0)
    inclusion = (xx - 0.62) ** 2 + (yy - 0.52) ** 2 < 0.13**2
    slow_layer = (yy > 0.72) & (xx > 0.18) & (xx < 0.86)
    c[inclusion] = 1.55
    c[slow_layer] = 0.72
    return c


def build_sponge(n: int, width: int = 18, strength: float = 0.018) -> np.ndarray:
    damping = np.zeros((n, n))
    for i in range(n):
        distance = min(i, n - 1 - i)
        if distance < width:
            value = strength * ((width - distance) / width) ** 2
            damping[i, :] += value
            damping[:, i] += value
    return damping


def render_wavefield(field: np.ndarray, velocity: np.ndarray, step: int, energy: float) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=100)
    ax.contour(velocity, levels=[0.8, 1.2], colors=["#facc15", "#34d399"], linewidths=1.1, alpha=0.8)
    im = ax.imshow(field, cmap="RdBu_r", vmin=-0.18, vmax=0.18, origin="lower", interpolation="bilinear")
    ax.set_title("2D Acoustic Wave Equation")
    ax.text(
        0.02,
        0.97,
        f"step = {step:04d}   discrete energy = {energy:.3e}",
        transform=ax.transAxes,
        va="top",
        color="#e5edf5",
        bbox={"facecolor": "#111827", "alpha": 0.78, "edgecolor": "none"},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pressure")
    fig.tight_layout()
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image


def render_receivers(times: list[float], traces: np.ndarray, visible: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=100)
    labels = ["R1", "R2", "R3", "R4"]
    for idx in range(traces.shape[1]):
        ax.plot(times[:visible], traces[:visible, idx] + idx * 0.16, label=labels[idx], linewidth=1.7)
    ax.set_title("Synthetic Receiver Traces")
    ax.set_xlabel("time")
    ax.set_ylabel("normalized pressure + offset")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image


def render_energy(times: list[float], energy: list[float], visible: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=100)
    ax.plot(times[:visible], energy[:visible], color="#34d399", linewidth=2.0)
    ax.set_title("Discrete Energy Diagnostic")
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image


def save_gif(frames: list[Image.Image], path: Path, duration: int) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True, duration=duration, loop=0)


def write_csv(path: Path, rows: list[list[float]], headers: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def simulate() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    n = 128
    dx = 1.0 / (n - 1)
    velocity = build_velocity_model(n)
    damping = build_sponge(n)
    dt = 0.42 * dx / (np.sqrt(2.0) * velocity.max())
    steps = 560

    p_now = np.zeros((n, n))
    p_old = np.zeros((n, n))
    source = (n // 2, n // 4)
    receivers = [(34, 86), (54, 86), (74, 86), (94, 86)]

    times: list[float] = []
    energy: list[float] = []
    traces: list[list[float]] = []
    wave_frames: list[Image.Image] = []

    for step in range(steps):
        t = step * dt
        lap = laplacian_4th_order(p_now, dx)
        p_next = (
            (2.0 - damping) * p_now
            - (1.0 - damping) * p_old
            + (dt * dt) * (velocity * velocity) * lap
        )
        p_next[source] += (dt * dt) * 850.0 * ricker(t, frequency=24.0, delay=0.08)

        p_next[0, :] = 0.0
        p_next[-1, :] = 0.0
        p_next[:, 0] = 0.0
        p_next[:, -1] = 0.0

        velocity_like = (p_next - p_old) / (2.0 * dt)
        grad_y, grad_x = np.gradient(p_now, dx)
        discrete_energy = float(0.5 * np.sum(velocity_like**2 + velocity**2 * (grad_x**2 + grad_y**2)) * dx * dx)

        times.append(t)
        energy.append(discrete_energy)
        traces.append([float(p_now[y, x]) for x, y in receivers])

        if step % 12 == 0:
            wave_frames.append(render_wavefield(p_now, velocity, step, discrete_energy))

        p_old, p_now = p_now, p_next

    trace_array = np.array(traces)
    max_abs = np.max(np.abs(trace_array)) or 1.0
    trace_array = trace_array / max_abs

    receiver_frames = [
        render_receivers(times, trace_array, visible)
        for visible in np.linspace(16, len(times), 36).astype(int)
    ]
    energy_frames = [
        render_energy(times, energy, visible)
        for visible in np.linspace(16, len(times), 36).astype(int)
    ]

    save_gif(wave_frames, ASSETS_DIR / "acoustic-wavefield.gif", duration=70)
    save_gif(receiver_frames, ASSETS_DIR / "acoustic-receivers.gif", duration=70)
    save_gif(energy_frames, ASSETS_DIR / "acoustic-energy.gif", duration=70)

    write_csv(
        REPORTS_DIR / "receiver_traces.csv",
        [[times[i], *trace_array[i].tolist()] for i in range(len(times))],
        ["time", "receiver_1", "receiver_2", "receiver_3", "receiver_4"],
    )
    write_csv(REPORTS_DIR / "energy_history.csv", [[times[i], energy[i]] for i in range(len(times))], ["time", "energy"])

    summary = (
        "2D acoustic wave equation simulation\n"
        f"grid_points: {n} x {n}\n"
        f"time_steps: {steps}\n"
        f"dx: {dx:.6f}\n"
        f"dt: {dt:.6f}\n"
        "velocity_model: homogeneous background with fast circular inclusion and slow layer\n"
        "boundary_model: damped absorbing sponge layer\n"
        "source: Ricker wavelet\n"
        f"receiver_count: {len(receivers)}\n"
    )
    (REPORTS_DIR / "simulation_summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print("Generated GIFs:")
    for path in sorted(ASSETS_DIR.glob("*.gif")):
        print(f"- {path.name}: {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    simulate()
