# Grid-Forming Inverter Control & Thévenin Network Dynamic Solver

[![CI](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Physics-based Python simulation and verification framework for **Grid-Forming (GFM) Inverter Control**:

- **Comparative GFM Control Triad (`src/gfm_triad.py`, `src/dvoc_control.py`):** Evaluates three canonical GFM primary synchronizing laws side-by-side on the identical 6-state electromagnetic $dq$ $LCL$ plant + cascaded dual-loop PI architecture: **13-State Multi-Loop Droop**, **14-State Virtual Synchronous Generator (`VSG`)**, and **12-State Nonlinear Andronov-Hopf Dispatchable Virtual Oscillator Control (`dVOC`)** with black-start limit-cycle self-excitation.
- **14-State Electromagnetic Synchronous-$dq$ $LCL$ Plant & Cascaded Dual-Loop PI Control (`src/dq_frames.py`, `src/lcl_plant.py`, `src/cascaded_control.py`):** Couples the 6-state electromagnetic $LCL$ output filter ($L_1, C_f, R_d, L_2$) and Thévenin AC grid ($R_g, L_g$) with synchronous-$dq$ capacitor-voltage and bridge-current PI loops, $\omega C_f$ / $\omega L_1$ cross-coupling decoupling, Qoria partial load-current feedforward ($F_i = 0.75$), SVPWM linear modulation clamping, radial peak current saturation, and conditional anti-windup.
- **4-State Quasi-Stationary Coupled Thévenin GFM Model (`src/thevenin_network.py`):** Couples a `VSG` swing equation, outer $Q\text{--}V$ droop (`src/droop_control.py`), and stationary $\alpha\beta$ virtual impedance (`src/virtual_impedance.py`), cross-validated against `scipy.integrate.solve_ivp` and shown to match the 14-state electromagnetic model on macro-timescales.
- **Fail-Closed Voltage-Sag / FRT Benchmark (`src/benchmark.py`):** Positive-sequence discrete-time sag response with exact half-open fault windowing, independent $\alpha\beta$ power-residual verification, windowed RoCoF separation, and post-fault recovery checks.

![Comparative Grid-Forming Control Triad Response](results/gfm_triad_comparative_response.png)

![14-State Electromagnetic dq LCL Plant & Cascaded Dual-Loop PI Response](results/lcl_cascaded_electromagnetic_response.png)

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

## Comparative GFM Control Triad Summary (`20 kVA`, `SCR = 6.0`, `X/R = 7.0`)

Under an identical $+10\text{ kW}$ active-power dispatch step ($t = 0.25\text{ s}$) followed by a $-0.20\text{ Hz}$ grid frequency drop ($t = 1.15\text{ s}$) on the 6-state $dq$ $LCL$ + cascaded PI plant ($dt = 80\text{ }\mu\text{s}$):

| GFM Architecture | Total ODE States | Power LPF | Step Rise $t_{90}$ | Grid-Drop Peak $P_{\text{pcc}}$ | Frequency Nadir | Steady $P_{\text{pcc},\infty}$ | Max $\operatorname{Re}(\lambda)$ (`SCR=6` / `SCR=2`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multi-Loop Droop** | `13` | `15 Hz` | `99.0 ms` | `18.44 kW` | `49.788 Hz` | `17.89 kW` | `-4.10` / `-5.37 s^-1` |
| **Virtual Sync. Gen. (`VSG`)** | `14` | `15 Hz` | `189.0 ms` | `21.52 kW` | `49.748 Hz` | `17.88 kW` | `-4.24` / `-4.65 s^-1` |
| **Andronov-Hopf `dVOC`** | `12` | None | `214.0 ms` | `17.97 kW` | `49.791 Hz` | `17.59 kW` | `-3.03` / `-5.06 s^-1` |

## Governing Equations

### 1. Nonlinear Andronov-Hopf `dVOC` (`src/dvoc_control.py`, `src/gfm_triad.py`)

In three-phase amplitude-invariant Clarke $\alpha\beta$ coordinates with inductive rotation $\mathbf{R}(\pi/2) = \mathbf{J}$, the Dispatchable Virtual Oscillator evolves according to:

$$\dot{\mathbf{v}}_{\alpha\beta} = \omega_0 \mathbf{J}\mathbf{v}_{\alpha\beta} + \eta \left(\mathbf{K}(P_{\text{ref}}, Q_{\text{ref}})\,\mathbf{v}_{\alpha\beta} - \frac{3}{2}\mathbf{J}\mathbf{i}_{\alpha\beta}\right) + \eta \alpha_{\text{osc}}\frac{V_0^2 - \|\mathbf{v}_{\alpha\beta}\|^2}{V_0^2}\mathbf{v}_{\alpha\beta}$$

Matching nominal small-signal slopes ($\eta = m_p V_0^2 = 16.84\text{ V}^2/(\text{W}\cdot\text{s})$, $\alpha_{\text{osc}} = 1/(2 n_q V_0) = 1.281\text{ W/V}^2$) yields exact polar kinetics $(\dot{\theta}, \dot{\hat{V}})$ without power-measurement low-pass filters, global unloaded Bernoulli limit-cycle self-excitation from arbitrary non-zero black-start seeds, and steady-state PCC power $P_{\text{pcc},\infty} = (\hat{V}/V_0)^2 P_{\text{droop,nom}} - \frac{3}{2}(R_v + R_2)|\mathbf{i}_2|^2$.

### 2. 14-State Synchronous-$dq$ Electromagnetic $LCL$ + Cascaded PI GFM (`src/lcl_plant.py`, `src/cascaded_control.py`)

In the inverter's synchronous Park $dq$ reference frame rotating at virtual speed $\omega(t)$ ($\mathbf{J} = \begin{bmatrix}0 & -1 \\ 1 & 0\end{bmatrix}$), the 6 electromagnetic $LCL$ + Thévenin network states $\mathbf{x}_{\text{lcl}} = [i_{1d}, i_{1q}, v_{cd}, v_{cq}, i_{2d}, i_{2q}]^\top$ satisfy:

$$\mathbf{v}_{f,dq} = \mathbf{v}_{c,dq} + R_d (\mathbf{i}_{1,dq} - \mathbf{i}_{2,dq})$$

$$L_1 \frac{d\mathbf{i}_{1,dq}}{dt} = \mathbf{v}_{\text{inv},dq} - \mathbf{v}_{f,dq} - R_1 \mathbf{i}_{1,dq} - \omega \mathbf{J} L_1 \mathbf{i}_{1,dq}, \qquad C_f \frac{d\mathbf{v}_{c,dq}}{dt} = (\mathbf{i}_{1,dq} - \mathbf{i}_{2,dq}) - \omega \mathbf{J} C_f \mathbf{v}_{c,dq}$$

$$(L_2 + L_g) \frac{d\mathbf{i}_{2,dq}}{dt} = \mathbf{v}_{f,dq} - \mathbf{v}_{g,dq}(\delta) - (R_2 + R_g) \mathbf{i}_{2,dq} - \omega \mathbf{J} (L_2 + L_g) \mathbf{i}_{2,dq}$$

Because $\mathbf{x}^\top \mathbf{J} \mathbf{x} \equiv 0$, rotational cross-coupling performs zero net work and total three-phase electromagnetic energy obeys exact passivity balance $\dot{\mathcal{E}} + P_{\text{loss}} = P_{\text{inv}} - P_{\text{grid}}$. Timescale-separated IMC dual-loop PI controllers ($f_{bw,i} = 1.0\text{ kHz}$, $f_{bw,v} = 180\text{ Hz}$) with Qoria partial load-current feedforward ($F_i = 0.75$) maintain strict small-signal Hurwitz stability across all three GFM laws (`Droop`, `VSG`, and `dVOC`).

### 3. 4-State Quasi-Stationary VSG + Thévenin Grid & FRT Benchmark (`src/thevenin_network.py`, `src/benchmark.py`)

Module `src/thevenin_network.py` solves the 4-state electromechanical VSG swing + $Q\text{--}V$ droop + virtual impedance system, while `src/benchmark.py` evaluates positive-sequence algebraic P/Q current commands under a prescribed `0.5 pu` voltage sag on `[0.300, 0.600) s` with radial peak current limiting (`30 A` peak), fault-windowed RoCoF (`0.5139 Hz/s`), and post-clearing recovery (`0.079 s`).

## Scope & Engineering Limitations

This repository models balanced three-phase (positive-sequence) averaged inverter bridge modulation, $LCL$ filter electromagnetics, cascaded $dq$ control loops, and Thévenin AC grid equivalents. In Module 3 (`v0.4.0`), discrete PWM switching ripple, dead-time distortion, semiconductor junction thermal limits, DC-link capacitor energy balance, and unbalanced negative/zero-sequence faults are not modeled. Passing tests and benchmarks verify **mathematical and software consistency for the documented models only**—they are not evidence of real-hardware performance, grid-code compliance, or equipment certification.

## Citation

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff) under the [MIT License](LICENSE).
