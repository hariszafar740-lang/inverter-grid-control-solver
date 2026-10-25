# Grid-Forming Inverter Control & Thévenin Network Dynamic Solver

[![CI](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Physics-based Python simulation and verification framework for **Grid-Forming (GFM) and Grid-Following (GFL) Inverter Control**:

- **Weak-Grid `SCR` Collapse Benchmark — `GFL (SRF-PLL)` vs. `GFM` (`src/gfl_pll.py`, `src/weak_grid_benchmark.py`):** Couples an 11-state Synchronous Reference Frame Phase-Locked Loop (`SRF-PLL`) + inner $dq$ current PI Grid-Following model to the identical 6-state electromagnetic $dq$ $LCL$ + Thévenin network plant, quantifying both **bandwidth-dependent PLL–impedance Hopf bifurcation** ($\text{SCR}_{\text{crit}}(f_{bw,\text{pll}})$) and **static nose-curve saddle-node Loss of Synchronism (`LOS`)** during parallel-line opening (`SCR: 4.0 -> 1.22`).
- **Comparative GFM Control Triad (`src/gfm_triad.py`, `src/dvoc_control.py`):** Evaluates three canonical GFM primary synchronizing laws side-by-side on the identical 6-state electromagnetic $dq$ $LCL$ plant + cascaded dual-loop PI architecture: **13-State Multi-Loop Droop**, **14-State Virtual Synchronous Generator (`VSG`)**, and **12-State Nonlinear Andronov-Hopf Dispatchable Virtual Oscillator Control (`dVOC`)** with black-start limit-cycle self-excitation.
- **14-State Electromagnetic Synchronous-$dq$ $LCL$ Plant & Cascaded Dual-Loop PI Control (`src/dq_frames.py`, `src/lcl_plant.py`, `src/cascaded_control.py`):** Couples the 6-state electromagnetic $LCL$ output filter ($L_1, C_f, R_d, L_2$) and Thévenin AC grid ($R_g, L_g$) with synchronous-$dq$ capacitor-voltage and bridge-current PI loops, $\omega C_f$ / $\omega L_1$ cross-coupling decoupling, Qoria partial load-current feedforward ($F_i = 0.75$), SVPWM linear modulation clamping, radial peak current saturation, and conditional anti-windup.
- **4-State Quasi-Stationary Coupled Thévenin GFM Model & Fail-Closed FRT Benchmark (`src/thevenin_network.py`, `src/benchmark.py`):** Couples a `VSG` swing equation, outer $Q\text{--}V$ droop (`src/droop_control.py`), and stationary $\alpha\beta$ virtual impedance (`src/virtual_impedance.py`), alongside positive-sequence voltage-sag Fault Ride-Through (`FRT`) verification.

![Weak-Grid SCR Stability & Line-Opening Collapse: GFL vs GFM](results/gfl_vs_gfm_weak_grid_collapse.png)

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

## Weak-Grid `SCR` Stability & Line-Opening Collapse (`20 kVA`, $P_{\text{ref}} = 15\text{ kW} = 0.75\text{ pu}$, `X/R = 7.0`)

Sweeping grid Short-Circuit Ratio from `SCR = 6.0` down to `1.15` on the identical 6-state electromagnetic $dq$ $LCL$ filter + Thévenin network plant, followed by a sudden parallel-line opening contingency (`SCR: 4.0 -> 1.22` at $t = 0.30\text{ s}$, $dt = 80\text{ }\mu\text{s}$):

| Control Architecture | ODE States | Critical `SCR_crit` (Hopf) | Equilibrium @ `SCR = 1.15` | Line-Trip `SCR 4.0 -> 1.22` | Min $|\mathbf{v}_{\text{pcc}}|$ | Max $|\delta(t)|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GFL `SRF-PLL` (`55 Hz` Fast)** | `11` | `2.99` | None (`SCR_static ~ 1.33`) | Unbounded Pole-Slipping (`LOS`) | `< 0.05 pu` | `> 3,500 deg` |
| **GFL `SRF-PLL` (`35 Hz` Nominal)** | `11` | `1.80` | None (`SCR_static ~ 1.33`) | Unbounded Pole-Slipping (`LOS`) | `0.048 pu` | `4,250.0 deg` |
| **GFL `SRF-PLL` (`10 Hz` Slow)** | `11` | `1.67` | None (`SCR_static ~ 1.33`) | Unbounded Pole-Slipping (`LOS`) | `0.008 pu` | `4,536.3 deg` |
| **GFM `VSG` (`H = 4.93 s`)** | `14` | `< 1.15` (`-3.97 s^-1`) | `0.954 pu`, `47.2 deg` | Synchronized Ride-Through | `0.937 pu` | `43.5 deg` |
| **GFM `dVOC` (Andronov-Hopf)** | `12` | `< 1.15` (`-2.65 s^-1`) | `0.953 pu`, `45.1 deg` | Synchronized Ride-Through | `0.924 pu` | `40.8 deg` |

## Comparative GFM Control Triad Summary (`20 kVA`, `SCR = 6.0`, `X/R = 7.0`)

Under an identical $+10\text{ kW}$ active-power dispatch step ($t = 0.25\text{ s}$) followed by a $-0.20\text{ Hz}$ grid frequency drop ($t = 1.15\text{ s}$) on the 6-state $dq$ $LCL$ + cascaded PI plant ($dt = 80\text{ }\mu\text{s}$):

| GFM Architecture | Total ODE States | Power LPF | Step Rise $t_{90}$ | Grid-Drop Peak $P_{\text{pcc}}$ | Frequency Nadir | Steady $P_{\text{pcc},\infty}$ | Max $\operatorname{Re}(\lambda)$ (`SCR=6` / `SCR=2`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multi-Loop Droop** | `13` | `15 Hz` | `99.0 ms` | `18.44 kW` | `49.788 Hz` | `17.89 kW` | `-4.10` / `-5.37 s^-1` |
| **Virtual Sync. Gen. (`VSG`)** | `14` | `15 Hz` | `189.0 ms` | `21.52 kW` | `49.748 Hz` | `17.88 kW` | `-4.24` / `-4.65 s^-1` |
| **Andronov-Hopf `dVOC`** | `12` | None | `214.0 ms` | `17.97 kW` | `49.791 Hz` | `17.59 kW` | `-3.03` / `-5.06 s^-1` |

## Governing Equations

### 1. Weak-Grid `SRF-PLL` Impedance Coupling & Pole-Slipping `LOS` (`src/gfl_pll.py`, `src/weak_grid_benchmark.py`)

In the PLL estimated rotating frame $\theta_{\text{pll}}(t)$ ($\delta_{\text{pll}} = \theta_{\text{pll}} - \theta_g$), the normalized Synchronous Reference Frame PLL and $q$-axis filter-capacitor terminal voltage satisfy:

$$\dot{\delta}_{\text{pll}} = \omega_{\text{pll}} - \omega_g, \qquad \omega_{\text{pll}} = \omega_0 + K_{p,\text{pll}}\frac{v_{fq}}{V_0} + K_{i,\text{pll}}\xi_{\text{pll}}, \qquad \dot{\xi}_{\text{pll}} = \frac{v_{fq}}{V_0}$$

$$v_{fq} = -V_g \sin(\delta_{\text{pll}}) + R_{\text{net}} i_{2q} + \omega_{\text{pll}} L_{\text{net}} i_{2d} + L_{\text{net}}\frac{di_{2q}}{dt}$$

In weak grids ($L_{\text{net}}$ large, `SCR` low) under high active current $i_{2d}$, perturbations in $\omega_{\text{pll}}$ feed back positively into $v_{fq}$ through $+\omega_{\text{pll}} L_{\text{net}} i_{2d}$ (Wen et al., *IEEE TPEL*, 2016; Dong et al., *IEEE TPWRS*, 2015), causing RHP Hopf instability at higher `SCR` as $f_{bw,\text{pll}}$ increases (`2.99` at `55 Hz` vs. `1.67` at `10 Hz`). Below the static nose-curve limit ($X_{\text{net}} i_{2d} > V_g$, $\text{SCR}_{\text{static}} \approx 1.33$ at `0.75 pu`), $v_{fq} > 0$ for all $\delta_{\text{pll}}$, driving unbounded PLL acceleration and cycle-slipping regardless of PLL tuning.

### 2. Nonlinear Andronov-Hopf `dVOC` (`src/dvoc_control.py`, `src/gfm_triad.py`)

In three-phase amplitude-invariant Clarke $\alpha\beta$ coordinates with inductive rotation $\mathbf{R}(\pi/2) = \mathbf{J}$, the Dispatchable Virtual Oscillator evolves according to:

$$\dot{\mathbf{v}}_{\alpha\beta} = \omega_0 \mathbf{J}\mathbf{v}_{\alpha\beta} + \eta \left(\mathbf{K}(P_{\text{ref}}, Q_{\text{ref}})\,\mathbf{v}_{\alpha\beta} - \frac{3}{2}\mathbf{J}\mathbf{i}_{\alpha\beta}\right) + \eta \alpha_{\text{osc}}\frac{V_0^2 - \|\mathbf{v}_{\alpha\beta}\|^2}{V_0^2}\mathbf{v}_{\alpha\beta}$$

Matching nominal small-signal slopes ($\eta = m_p V_0^2 = 16.84\text{ V}^2/(\text{W}\cdot\text{s})$, $\alpha_{\text{osc}} = 1/(2 n_q V_0) = 1.281\text{ W/V}^2$) yields exact polar kinetics $(\dot{\theta}, \dot{\hat{V}})$ without power-measurement low-pass filters and global unloaded Bernoulli limit-cycle self-excitation from non-zero black-start seeds.

### 3. 14-State Synchronous-$dq$ Electromagnetic $LCL$ + Cascaded PI GFM (`src/lcl_plant.py`, `src/cascaded_control.py`)

In the inverter's synchronous Park $dq$ reference frame rotating at virtual speed $\omega(t)$, the 6 electromagnetic $LCL$ + Thévenin network states $\mathbf{x}_{\text{lcl}} = [i_{1d}, i_{1q}, v_{cd}, v_{cq}, i_{2d}, i_{2q}]^\top$ obey exact rotational passivity ($\mathbf{x}^\top \mathbf{J} \mathbf{x} \equiv 0$). Timescale-separated IMC dual-loop PI controllers ($f_{bw,i} = 1.0\text{ kHz}$, $f_{bw,v} = 180\text{ Hz}$) with Qoria partial load-current feedforward ($F_i = 0.75$) maintain strict small-signal Hurwitz stability across `Droop`, `VSG`, and `dVOC` down to ultra-weak `SCR = 1.15`.

### 4. 4-State Quasi-Stationary VSG + Thévenin Grid & FRT Benchmark (`src/thevenin_network.py`, `src/benchmark.py`)

Module `src/thevenin_network.py` solves the 4-state electromechanical VSG swing + $Q\text{--}V$ droop + virtual impedance system, while `src/benchmark.py` evaluates positive-sequence algebraic P/Q current commands under a prescribed `0.5 pu` voltage sag on `[0.300, 0.600) s` with radial peak current limiting (`30 A` peak), fault-windowed RoCoF (`0.5139 Hz/s`), and post-clearing recovery (`0.079 s`).

## Scope & Engineering Limitations

This repository models balanced three-phase (positive-sequence) averaged inverter bridge modulation, $LCL$ filter electromagnetics, cascaded $dq$ / `SRF-PLL` control loops, and Thévenin AC grid equivalents. In Module 4 (`v0.5.0`), discrete PWM switching ripple, dead-time distortion, semiconductor junction thermal limits, DC-link capacitor energy balance, and unbalanced negative/zero-sequence faults are not modeled. Passing tests and benchmarks verify **mathematical and software consistency for the documented models only**—they are not evidence of real-hardware performance, grid-code compliance, or equipment certification.

## Citation

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff) under the [MIT License](LICENSE).
