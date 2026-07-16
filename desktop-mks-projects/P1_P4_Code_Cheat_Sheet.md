# MCS Code Cheat Sheet (P1–P4)

This sheet is built for oral defense prep: each project lists the main code engine, the critical functions, and exactly what to say when asked.

## Project 1 — Coupled Logistic Map (`Projekt_1_MKS`)

Core idea: everything is built on one map update `F(x,y)` for two coupled populations.

### 1) `coupled_map` in `common.py`
- What it does: computes one time-step update of `(x_n, y_n)` to `(x_{n+1}, y_{n+1})`.
- Why it matters: this is the model itself; every other script (fixed points, periodic orbits, portraits, bifurcation) depends on this rule.
- What to say: “This function is the mathematical system encoded in code. If this changes, all downstream dynamics change.”

### 2) `F_iterate`, `G_periodic`, `minimal_period` in `common.py`
- `F_iterate`: computes `F^p(z)` by repeated iteration.
- `G_periodic`: computes residual `F^p(z)-z`; roots are period-`p` candidates.
- `minimal_period`: checks whether a candidate returns earlier (to avoid false period labels).
- What to say: “Periodic orbits are found as roots of a residual, then filtered for true minimal period.”

### 3) `find_periodic_orbits` in `periodicPoints.py`
- What it does: runs root-finding from many initial guesses, keeps physical roots in `[0,1]^2`, enforces minimal period, removes duplicate orbits.
- Why it matters: this is the numerical engine for Task 2 (period-2 and period-3 search).

### 4) `make_phase_portrait` in `phasePortrait.py`
- What it does: iterates long trajectory, discards transient, plots tail in `(x_n, y_n)`.
- Why it matters: reveals attractor type (fixed point, periodic set, irregular/chaotic cloud).

### 5) `orbit_tail_for_parameter` + loop over `r2` in `bifurcationCascade.py`
- What it does: for each `r2`, discard transient and keep long-time `x_n` values.
- Why it matters: builds bifurcation diagram showing branch splitting and complexity growth.

---

## Project 2 — Cellular Automata (`P2`)

Core idea: local update rules + designed initial patterns produce computation and complex waves.

### Part A: Game of Life gates (`and_gate.py`, `or_gate.py`, `not_gate.py`)

### 1) `compute_next_generation`
- What it does: applies Conway rules synchronously to entire grid.
- Why it matters: this is the physics engine; gates are not hard-coded logic statements, they emerge via collisions under this update.

### 2) `create_*glider`, `create_stopper_*`, `create_eater_*`
- What they do: define pattern geometry (signal sources, blockers, sinks).
- Why they matter: gate truth behavior comes from geometry/timing of gliders.
- What to say: “Inputs select which stoppers are inserted, changing collisions and whether output glider survives.”

### Part B: Neuron CA (`neuron_branching.py`)

### 3) `update`
- What it does: applies 3-state rules: `READY -> FIRING` (if trigger), `FIRING -> RESTING`, `RESTING -> READY`.
- Why it matters: encodes excitable-medium dynamics and wave directionality.

### 4) `task_a_moving_pattern`, `task_b_moving_and_creating`, `task_c_periodic_motion`
- What they do: handcrafted initial conditions for the three required behaviors.
- Why they matter: demonstrates control of macroscopic behavior via microstate design.

### Part C: GHCA (`GHCA.py`)

### 5) `get_next_state`
- What it does: GHCA update with states `0..N-1` and excited range `1..e`.
- Rules:
  - state `0` -> `1` if any von Neumann neighbor is excited, else stays `0`.
  - states `1..N-2` increment by one.
  - state `N-1` resets to `0`.
- Why it matters: defines periodic excitable cycle.

### 6) `check_cycle`
- What it does: hashes full grid state and detects first repeated state.
- Why it matters: directly computes transient length and period.

---

## Project 3 — Networks + Markov Convergence (`P3`)

Core idea: Part 1 studies small-world graph structure; Part 2 studies spectral control of Markov mixing.

### Part A: Watts-Strogatz (`watts_strogatz_analysis.py`)

### 1) `run_simulation`
- What it does: for each rewiring probability `p`, generates many realizations and collects metrics.
- Why it matters: Monte Carlo backbone; avoids one-realization bias.

### 2) `graph_statistics`
- What it does: computes degree stats, connectivity/components, largest component size, diameter, average path length, clustering.
- Important detail: if disconnected, diameter/path are computed on largest connected component.
- Why it matters: this matches lab convention and gives robust comparable metrics.

### 3) `summarize`
- What it does: aggregates means/std over realizations and computes normalized `C(p)/C(0)` and `L(p)/L(0)`.
- Why it matters: normalized metrics are the quantitative small-world evidence.

### Part B: Markov chain (`spectralConvergence.py`)

### 4) `check_column_stochastic`
- What it does: validates nonnegative entries and column sums = 1.
- Why it matters: ensures valid transition matrix under `x_{t+1}=Px_t` convention.

### 5) `stationary_distribution`
- What it does: computes eigenvector for eigenvalue 1 and normalizes to probability vector.
- Why it matters: defines equilibrium distribution `pi`.

### 6) `spectral_data`
- What it does: computes eigenvalues, second-largest modulus `mu=|lambda_2|`, spectral gap `g=1-mu`.
- Why it matters: theoretical predictor of convergence speed.

### 7) `simulate` + `total_variation`
- `simulate`: iterates `x_{t+1}=Px_t` from concentrated initial states.
- `total_variation`: computes `delta(t)=0.5*||x_t-pi||_1`.
- Why they matter: numerical verification of spectral prediction.

### 8) `first_step_below`
- What it does: first time `delta(t)` drops below tolerance.
- Why it matters: operational convergence-time metric for tables.

---

## Project 4 — Persistent Homology + Vicsek (`project4`)

Core idea: Task 2 extracts topological features vs scale; Task 3 measures order-disorder in collective motion.

### Part A: Persistent homology (`task2_persistent_homology.py`)

### 1) `build_vietoris_rips_complex`
- What it does: builds edges `(i,j)` when distance `< r`, and triangles `(i,j,k)` when all three pairwise edges exist.
- Why it matters: this is the data->complex conversion.

### 2) `boundary_matrix_1` and `boundary_matrix_2`
- `boundary_matrix_1`: maps edges to endpoint incidence in `C1 -> C0`.
- `boundary_matrix_2`: maps triangles to edge incidence in `C2 -> C1`.
- Why they matter: these encode topology algebraically.

### 3) `rank_mod2`
- What it does: Gaussian elimination over `Z2` using XOR row operations.
- Why it matters: computes ranks needed for Betti numbers.

### 4) `compute_betti_numbers`
- What it does: computes `beta0 = dim(C0)-rank(d1)` and `beta1 = dim(C1)-rank(d1)-rank(d2)`.
- Why it matters: outputs connected components and 1D holes at each `r`.

### Part B: Vicsek (`task3_vicsek_model.py`)

### 5) `simulate_vicsek`
- What it does:
  - computes periodic neighbor distances (minimum-image convention),
  - averages neighbor directions via vector sums,
  - adds noise `eta * U[-1/2,1/2]`,
  - updates heading and position with periodic wrap.
- Why it matters: this is the full dynamical model implementation.

### 6) `polarization`
- What it does: computes global order parameter `Phi(t)=|mean direction vector|`.
- Why it matters: key metric of alignment vs disorder.

---

## Fast Oral Defense Pattern (works for any function)
Use this 4-line structure:
1. “Input/Output”: what goes in and what returns.
2. “Rule”: the exact update or formula it applies.
3. “Role”: where this sits in the full pipeline.
4. “Why important”: what breaks if this is wrong.

Example (P3 `spectral_data`):
- Input/Output: takes `P`, returns eigenvalues, `mu`, `gap`.
- Rule: sorts eigenvalues by modulus, sets `mu=|lambda_2|`, `gap=1-mu`.
- Role: theoretical convergence predictor.
- Why important: wrong `mu` means wrong speed interpretation and wrong conclusions.

## 10 High-Probability Viva Questions
1. Why discard transients before analyzing long-term behavior?
2. Why do we need many realizations in Watts-Strogatz?
3. Why does `mu` control Markov convergence speed?
4. Why compute diameter/path on largest component if disconnected?
5. How do you ensure a period-3 point is not period-1 or period-2?
6. Why are GHCA periods detectable by state hashing?
7. Why are boundary matrices over `Z2` enough for these Betti computations?
8. Why does `arg(sum e^{i theta_j})` represent an average direction?
9. Why does increasing Vicsek noise reduce polarization?
10. Which function is the “single source of truth” in each project?
