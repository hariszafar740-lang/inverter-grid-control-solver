# Grid-Forming Inverter Control & Thévenin Network Dynamic Solver

[![CI](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Physics-based Python simulation and verification framework for **Grid-Forming (GFM) Inverter Control**:

- **14-State Electromagnetic Synchronous-$dq$ $LCL$ Plant & Cascaded Dual-Loop PI Control (`src/dq_frames.py`, `src/lcl_plant.py`, `src/cascaded_control.py`):** Couples the 6-state electromagnetic $LCL$ output filter ($L_1, C_f, R_d, L_2$) and Thévenin AC grid ($R_g, L_g$) with synchronous-$dq$ capacitor-voltage and bridge-current PI loops, $\omega C_f$ / $\omega L_1$ cross-coupling decoupling, Qoria partial load-current feedforward ($F_i = 0.75$), SVPWM linear modulation clamping, radial peak current saturation, and conditional anti-windup.
- **4-State Quasi-Stationary Coupled Thévenin GFM Model (`src/thevenin_network.py`):** Couples a Virtual Synchronous Generator (`VSG`) swing equation, outer $Q\text{--}V$ droop (`src/droop_control.py`), and stationary $\alpha\beta$ virtual impedance (`src/virtual_impedance.py`), cross-validated against `scipy.integrate.solve_ivp` and shown to match the 14-state electromagnetic model on macro-timescales.
- **Fail-Closed Voltage-Sag / FRT Benchmark (`src/benchmark.py`):** Positive-sequence discrete-time sag response with exact half-open fault windowing, independent $\alpha\beta$ power-residual verification, windowed RoCoF separation, and post-fault recovery checks.

![Closed-Loop Grid-Forming VSG & Thévenin Network Response](results/inverter_transient_response.png)

![14-State Electromagnetic dq LCL Plant & Cascaded Dual-Loop PI Response](results/lcl_cascaded_electromagnetic_response.png)

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

### 1. 14-State Synchronous-$dq$ Electromagnetic $LCL$ + Cascaded PI GFM (`src/lcl_plant.py`, `src/cascaded_control.py`)

In the inverter's synchronous Park $dq$ reference frame rotating at virtual rotor speed $\omega(t)$ ($\mathbf{J} = \begin{bmatrix}0 & -1 \\ 1 & 0\end{bmatrix}$), the 6 electromagnetic $LCL$ + Thévenin network states $\mathbf{x}_{\text{lcl}} = [i_{1d}, i_{1q}, v_{cd}, v_{cq}, i_{2d}, i_{2q}]^\top$ satisfy:

$$\mathbf{v}_{f,dq} = \mathbf{v}_{c,dq} + R_d (\mathbf{i}_{1,dq} - \mathbf{i}_{2,dq})$$

$$L_1 \frac{d\mathbf{i}_{1,dq}}{dt} = \mathbf{v}_{\text{inv},dq} - \mathbf{v}_{f,dq} - R_1 \mathbf{i}_{1,dq} - \omega \mathbf{J} L_1 \mathbf{i}_{1,dq}, \qquad C_f \frac{d\mathbf{v}_{c,dq}}{dt} = (\mathbf{i}_{1,dq} - \mathbf{i}_{2,dq}) - \omega \mathbf{J} C_f \mathbf{v}_{c,dq}$$

$$(L_2 + L_g) \frac{d\mathbf{i}_{2,dq}}{dt} = \mathbf{v}_{f,dq} - \mathbf{v}_{g,dq}(\delta) - (R_2 + R_g) \mathbf{i}_{2,dq} - \omega \mathbf{J} (L_2 + L_g) \mathbf{i}_{2,dq}$$

Because $\mathbf{x}^\top \mathbf{J} \mathbf{x} \equiv 0$, rotational cross-coupling performs zero net work and total instantaneous three-phase electromagnetic energy $\mathcal{E} = \frac{3}{2}\left[\frac{1}{2}L_1|\mathbf{i}_1|^2 + \frac{1}{2}C_f|\mathbf{v}_c|^2 + \frac{1}{2}(L_2+L_g)|\mathbf{i}_2|^2\right]$ obeys exact passivity balance $\dot{\mathcal{E}} + P_{\text{loss}} = P_{\text{inv}} - P_{\text{grid}}$. Timescale-separated IMC dual-loop PI controllers ($f_{bw,i} = 1.0\text{ kHz}$, $f_{bw,v} = 180\text{ Hz}$) with partial load-current feedforward ($F_i = 0.75$, following Qoria et al., *IEEE TPEL*, 2020) maintain strict small-signal Hurwitz stability ($\max_k \operatorname{Re}(\lambda_k) < -4.2\text{ s}^{-1}$), whereas unity feedforward ($F_i = 1.0$) destabilizes the sub-synchronous electromagnetic flux mode.

### 2. 4-State Quasi-Stationary VSG + Thévenin Grid (`src/thevenin_network.py`)

$$J \frac{d\omega}{dt} = \frac{P_{\text{ref}}(t) - P_{\text{filt}}}{\omega_0} - D_p (\omega - \omega_0), \qquad \frac{d\delta}{dt} = \omega - \omega_g(t)$$

$$E^*(t) = \operatorname{clamp}\!\left(E_0 - n_q (Q_{\text{filt}} - Q_{\text{ref}}),\; 0.75 E_0,\; 1.20 E_0\right)$$

Under a $10\text{ kW}$ step followed by a $-0.20\text{ Hz}$ grid frequency drop ($49.80\text{ Hz}$), both the 4-state quasi-static solver and the 14-state electromagnetic $LCL$ solver converge to the primary droop equilibrium ($f_\infty = 49.80\text{ Hz}$, $P_\infty \approx 17.88\text{--}17.90\text{ kW}$, nadir $49.748\text{ Hz}$).

### 3. Reduced-Order Prescribed-Sag FRT Benchmark (`src/benchmark.py`)

`src/benchmark.py` evaluates positive-sequence algebraic P/Q current commands under a prescribed `0.5 pu` voltage sag on `[0.300, 0.600) s` with radial peak current limiting (`30 A` peak), verifying simultaneous $V\text{--}I$ power consistency, fault-windowed RoCoF (`0.5139 Hz/s`), and post-clearing recovery (`0.079 s`).

## Scope & Engineering Limitations

This repository models balanced three-phase (positive-sequence) averaged inverter bridge modulation, $LCL$ filter electromagnetics, cascaded $dq$ control loops, and Thévenin AC grid equivalents. In Module 2 (`v0.3.0`), discrete PWM switching ripple, dead-time distortion, semiconductor junction thermal limits, DC-link capacitor energy balance, and unbalanced negative/zero-sequence faults are not modeled. Passing tests and benchmarks verify **mathematical and software consistency for the documented models only**—they are not evidence of real-hardware performance, grid-code compliance, or equipment certification.

## Citation

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff) under the [MIT License](LICENSE).
