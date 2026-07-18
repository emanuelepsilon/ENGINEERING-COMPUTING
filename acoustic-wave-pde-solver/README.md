# Acoustic Wave PDE Solver

High-level numerical simulation project for the 2D acoustic wave equation:

```text
p_tt = c(x, y)^2 (p_xx + p_yy) + s(x, y, t)
```

The solver uses a high-order finite-difference stencil in the interior, a damped absorbing layer near the boundaries, a Ricker wavelet source, a heterogeneous velocity field, receiver sampling, and diagnostic plots.

## Why This Is Here

This is a compact computational physics project designed to show practical numerical modeling: discretizing a partial differential equation, enforcing stability through the Courant-Friedrichs-Lewy condition, simulating wave propagation, and checking numerical behavior with energy/receiver diagnostics.

## Outputs

| Wavefield simulation | Receiver traces | Energy diagnostics |
| --- | --- | --- |
| ![2D acoustic wave propagation](assets/acoustic-wavefield.gif) | ![Receiver traces](assets/acoustic-receivers.gif) | ![Energy diagnostic](assets/acoustic-energy.gif) |

## Run

```powershell
python solver.py
```

Generated files:

- `assets/acoustic-wavefield.gif`
- `assets/acoustic-receivers.gif`
- `assets/acoustic-energy.gif`
- `reports/receiver_traces.csv`
- `reports/energy_history.csv`
- `reports/simulation_summary.txt`

## Numerical Ingredients

- 2D scalar acoustic wave equation
- fourth-order central finite differences in the interior
- second-order fallback near boundaries
- absorbing sponge layer to reduce boundary reflections
- heterogeneous velocity model with a circular fast inclusion
- Ricker wavelet source
- receiver line for synthetic waveform traces
- discrete energy tracking
