
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PlantConfig:
    Kp: float = 2.0
    T: float = 0.2

@dataclass
class LeadConfig:
    Kc: float = 1.6
    Td: float = 0.12
    beta: float = 0.35
    enabled: bool = True

@dataclass
class LagConfig:
    Tlag: float = 0.8
    gamma: float = 3.0
    enabled: bool = True

@dataclass
class SimConfig:
    wmin: float = 0.1
    wmax: float = 10.0
    npts: int = 600
    t_end: float = 8.0
    n_time: int = 2000
    disturb_step: float = -0.2

def tf_from_cfg(plant_cfg, lead_cfg, lag_cfg):
    num_G = np.array([plant_cfg.Kp])
    den_G = np.polymul([1.0, 0.0], [plant_cfg.T, 1.0])
    if lead_cfg.enabled:
        num_lead = lead_cfg.Kc * np.array([lead_cfg.Td, 1.0])
        den_lead = np.array([lead_cfg.beta * lead_cfg.Td, 1.0])
    else:
        num_lead = np.array([1.0]); den_lead = np.array([1.0])
    if lag_cfg.enabled:
        num_lag = np.array([lag_cfg.Tlag, 1.0])
        den_lag = np.array([lag_cfg.gamma * lag_cfg.Tlag, 1.0])
    else:
        num_lag = np.array([1.0]); den_lag = np.array([1.0])
    num_C = np.polymul(num_lead, num_lag)
    den_C = np.polymul(den_lead, den_lag)
    num_Go = np.polymul(num_C, num_G)
    den_Go = np.polymul(den_C, den_G)
    num_Gc = num_Go
    den_Gc = np.polyadd(den_Go, num_Go)
    num_S  = den_Go
    den_S  = den_Gc
    Go = signal.TransferFunction(num_Go, den_Go)
    Gc = signal.TransferFunction(num_Gc, den_Gc)
    S  = signal.TransferFunction(num_S,  den_S)
    return Go, Gc, S

def bode_margins(Go, sim_cfg):
    w = np.logspace(np.log10(sim_cfg.wmin), np.log10(sim_cfg.wmax), sim_cfg.npts)
    w, H = signal.freqs(Go.num, Go.den, worN=w)
    mag = np.abs(H); phase = np.unwrap(np.angle(H))
    idx = np.where(mag[:-1] >= 1.0)[0]
    wc = np.nan; phi_m = np.nan
    if len(idx) > 0:
        i = idx[-1]
        m1, m2 = mag[i], mag[i+1]; w1, w2 = w[i], w[i+1]
        wc = np.exp(np.interp(np.log(1.0), np.log([m1, m2]), np.log([w1, w2]))) if m1!=m2 else w1
        ph_wc = np.interp(np.log(wc), np.log(w), phase)
        phi_m = np.degrees(np.pi + ph_wc)
    cross_idx = np.where(np.diff(np.sign(phase + np.pi)) != 0)[0]
    Am = np.nan
    if len(cross_idx) > 0:
        j = cross_idx[0]; w1, w2 = w[j], w[j+1]; ph1, ph2 = phase[j], phase[j+1]
        wp = np.exp(np.interp(-np.pi, [ph1, ph2], np.log([w1, w2]))) if ph1!=ph2 else w1
        mag_wp = np.abs(np.interp(np.log(wp), np.log(w), H))
        Am = 1.0 / mag_wp
    return w, mag, phase, wc, phi_m, Am

def step_metrics(t, y, ref=1.0):
    y_end = y[-1]
    overshoot = max(0.0, (np.max(y) - ref) / max(ref, 1e-12) * 100.0)
    try:
        t10 = t[np.where(y >= 0.1*ref)[0][0]]
        t90 = t[np.where(y >= 0.9*ref)[0][0]]
        rise = t90 - t10
    except IndexError:
        rise = np.nan
    tol = 0.02 * max(ref, 1e-12)
    within = np.where(np.abs(y - ref) <= tol)[0]
    settle = np.nan
    if len(within) > 0:
        last_out = np.max(np.where(np.abs(y - ref) > tol)[0]) if np.any(np.abs(y - ref) > tol) else -1
        after = np.arange(last_out + 1, len(t))
        if len(after) > 0:
            settle = t[after[0]]
    steady_err = ref - y_end
    return dict(rise_time=rise, overshoot_pct=overshoot, settling_time=settle, steady_state_error=steady_err)

def simulate_and_plot():
    plant_cfg = PlantConfig(); lead_cfg = LeadConfig(); lag_cfg = LagConfig(); sim_cfg = SimConfig()
    Go, Gc, S = tf_from_cfg(plant_cfg, lead_cfg, lag_cfg)
    w, mag, phase, wc, phi_m, Am = bode_margins(Go, sim_cfg)

    plt.figure(); plt.semilogx(w, 20*np.log10(mag)); plt.axhline(0, linestyle='--')
    if not np.isnan(wc): plt.axvline(wc, linestyle='--')
    plt.xlabel('Frequency (rad/s)'); plt.ylabel('Magnitude (dB)'); plt.title('Open-loop Bode |Go|'); plt.show()

    plt.figure(); plt.semilogx(w, np.degrees(phase)); plt.axhline(-180, linestyle='--')
    if not np.isnan(wc): plt.axvline(wc, linestyle='--')
    plt.xlabel('Frequency (rad/s)'); plt.ylabel('Phase (deg)'); plt.title('Open-loop phase ∠Go'); plt.show()

    t = np.linspace(0, sim_cfg.t_end, sim_cfg.n_time)
    t1, y_r = signal.step(Gc, T=t); t2, y_s = signal.step(S, T=t)
    y_total = y_r + sim_cfg.disturb_step * y_s

    m = step_metrics(t1, y_r, ref=1.0)

    plt.figure(); plt.plot(t1, y_r); plt.xlabel('Time (s)'); plt.ylabel('y'); plt.title('Reference step (1.0)'); plt.show()
    plt.figure(); plt.plot(t, y_total); plt.xlabel('Time (s)'); plt.ylabel('y'); plt.title(f'Ref + disturbance (v={sim_cfg.disturb_step})'); plt.show()

    print("=== Loop-shaping summary ===")
    print(f"wc: {wc:.3f} rad/s" if not np.isnan(wc) else "wc: n/a")
    print(f"Phase margin: {phi_m:.1f} deg" if not np.isnan(phi_m) else "phi_m: n/a")
    print(f"Gain margin (ratio): {Am:.2f}" if not np.isnan(Am) else "Am: n/a")
    print("--- Step metrics ---")
    for k,v in m.items():
        print(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    simulate_and_plot()
