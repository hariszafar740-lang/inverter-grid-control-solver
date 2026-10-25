# Grid-Forming Inverter Control & Thévenin Network Dynamic Solver

[![CI](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Physics-based Python simulation and verification framework for **Grid-Forming (GFM) Inverter Control**:

- **Closed-Loop Coupled Thévenin AC Network (`src/thevenin_network.py`):** Couples a Virtual Synchronous Generator (`VSG`) swing equation, outer $Q\text{--}V$ voltage droop (`src/droop_control.py`), stationary $\alpha\beta$ virtual impedance $\mathbf{Z}_v(\omega) = R_v + j\omega L_v$ (`src/virtual_impedance.py`), and radial peak current limiting (`src/current_control.py`) to an AC Thévenin equivalent grid ($\text{SCR}$, $X_g/R_g$), cross-validated against `scipy.integrate.solve_ivp`.
- **Fail-Closed Voltage-Sag / FRT Benchmark (`src/benchmark.py`):** Positive-sequence discrete-time sag response with exact half-open fault windowing, independent $\alpha\beta$ power-residual verification, windowed RoCoF separation, and post-fault recovery checks.

![Closed-Loop Grid-Forming VSG & Thévenin Network Response](results/inverter_transient_response.png)

## Quick Start

Python `3.10–3.13` is supported and tested in CI:

```bash
python -m pip install -r requirements-dev.txt
python main.py
python -m src.benchmark
python -m pytest -v
```

The historical script entry point `python src/benchmark.py` is also preserved.

## Governing Equations

### 1. Closed-Loop Coupled VSG + Virtual Impedance + Thévenin Grid (`src/thevenin_network.py`)

Using amplitude-invariant Clarke ($\alpha\beta$) peak phase quantities ($P = \frac{3}{2}(v_\alpha i_\alpha + v_\beta i_\beta)$, $Q = \frac{3}{2}(v_\beta i_\alpha - v_\alpha i_\beta)$), the 4-state electromechanical GFM model integrates rotor angle $\delta$, virtual angular velocity $\omega$, and low-pass filtered powers $(P_{\text{filt}}, Q_{\text{filt}})$:

$$J \frac{d\omega}{dt} = \frac{P_{\text{ref}}(t) - P_{\text{filt}}}{\omega_0} - D_p (\omega - \omega_0), \qquad \frac{d\delta}{dt} = \omega - \omega_g(t)$$

$$E^*(t) = \operatorname{clamp}\!\left(E_0 - n_q (Q_{\text{filt}} - Q_{\text{ref}}),\; 0.75 E_0,\; 1.20 E_0\right)$$

With internal EMF $\mathbf{e}_{\alpha\beta} = E^* [\cos\delta,\, \sin\delta]^\top$ and stiff/weak grid bus $\mathbf{v}_{g,\alpha\beta} = [V_g,\, 0]^\top$, total series impedance $\mathbf{Z}_{\text{tot}}(\omega) = (R_v + R_g) + j(\omega L_v + \omega_g L_g)$ yields the unsaturated network current before radial current limiting ($|\mathbf{i}_{\alpha\beta}| \le I_{\max}$). Under a $-0.20\text{ Hz}$ grid frequency drop ($49.80\text{ Hz}$), steady-state synchronizing lock satisfies $\omega_{\infty} = \omega_g$ and delivers exact primary frequency droop support $\Delta P_{\infty} = D_p \omega_0 (\omega_0 - \omega_g)$.

### 2. Reduced-Order Prescribed-Sag FRT Benchmark (`src/benchmark.py`)

`src/benchmark.py` evaluates positive-sequence algebraic P/Q current commands under a prescribed `0.5 pu` voltage sag on `[0.300, 0.600) s` with radial peak current limiting (`30 A` peak), verifying simultaneous $V\text{--}I$ power consistency, fault-windowed RoCoF (`0.5139 Hz/s`), and post-clearing recovery (`0.079 s`).

## Scope & Engineering Limitations

This repository models positive-sequence fundamental-frequency / averaged dynamic control blocks and Thévenin network equivalents. In Module 1 (`v0.2.0`), PWM switching harmonics, semiconductor thermal limits, DC-link capacitor dynamics, negative/zero-sequence unbalanced faults, and electromagnetic $LCL$ differential filter states are not yet modeled (addressed progressively in subsequent modules). Passing tests and benchmarks verify **mathematical and software consistency for the documented models only**—they are not evidence of real-hardware performance, grid-code compliance, or equipment certification.

## Citation

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff) under the [MIT License](LICENSE).
