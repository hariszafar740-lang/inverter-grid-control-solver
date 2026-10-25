# Grid-Forming Inverter Control & Thévenin Network Dynamic Solver

[![CI](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Physics-based Python simulation and verification framework for **Grid-Forming (GFM) and Grid-Following (GFL) Inverter Control**:

- **14-State Electromagnetic Fault Ride-Through (`FRT`), `-45 deg` Phase Jump, Threshold Transient Virtual Impedance (`TVI`) & Power Anti-Windup (`src/transient_virtual_impedance.py`, `src/em_frt_benchmark.py`):** Contrasts unprotected overcurrent spikes (`160.2 A = 3.91 pu`), Qoria circular current-limiter latch-up & virtual-rotor windup (`440.7 deg` under a `0.15 pu` sag; `344.7 deg` under a `-45 deg` phase jump), and coordinated **`TVI + Synchronizing Power-Loop Anti-Windup` (`PAW`)** (`|i_{1,dq}| <= 47.95 A <= I_{max}`, `0.1%` voltage-loop saturation, `max delta = 27.5 deg`, zero cycle slips).
- **Weak-Grid `SCR` Collapse Benchmark — `GFL (SRF-PLL)` vs. `GFM` (`src/gfl_pll.py`, `src/weak_grid_benchmark.py`):** Couples an 11-state Synchronous Reference Frame Phase-Locked Loop (`SRF-PLL`) + inner $dq$ current PI Grid-Following model to the identical 6-state electromagnetic $dq$ $LCL$ + Thévenin network plant, quantifying both **bandwidth-dependent PLL–impedance Hopf bifurcation** ($\text{SCR}_{\text{crit}}(f_{bw,\text{pll}})$) and **static nose-curve saddle-node Loss of Synchronism (`LOS`)** during parallel-line opening (`SCR: 4.0 -> 1.22`).
- **Comparative GFM Control Triad (`src/gfm_triad.py`, `src/dvoc_control.py`):** Evaluates three canonical GFM primary synchronizing laws side-by-side on the identical 6-state electromagnetic $dq$ $LCL$ plant + cascaded dual-loop PI architecture: **13-State Multi-Loop Droop**, **14-State Virtual Synchronous Generator (`VSG`)**, and **12-State Nonlinear Andronov-Hopf Dispatchable Virtual Oscillator Control (`dVOC`)** with black-start limit-cycle self-excitation.
- **14-State Electromagnetic Synchronous-$dq$ $LCL$ Plant, 4-State Thévenin Baseline & Fail-Closed Algebraic FRT CLI (`src/lcl_plant.py`, `src/cascaded_control.py`, `src/thevenin_network.py`, `src/benchmark.py`):** Full electromagnetic $LCL$ + Thévenin network passivity verification, IMC dual-loop PI tuning with Qoria partial feedforward ($F_i = 0.75$), and discrete-time positive-sequence sag checks.

![14-State Electromagnetic FRT, TVI, Anti-Windup & Phase-Jump Response](results/em_frt_tvi_phase_jump_response.png)

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

## 14-State Electromagnetic `FRT` & `-45 deg` Grid Phase-Jump Summary (`20 kVA`, `SCR = 4.0`, $I_{\max} = 48.0\text{ A}$)

Evaluating deep symmetrical three-phase voltage fault ride-through ($V_g: 1.00 \to 0.15\text{ pu}$ on $[0.20, 0.65)\text{ s}$ at $P_{\text{ref}} = 17\text{ kW} = 0.85\text{ pu}$) and an instantaneous $-45^\circ$ grid phase jump ($t = 0.20\text{ s}$ at $P_{\text{ref}} = 12\text{ kW}$) on the 14-state electromagnetic $dq$ $LCL$ `VSG` plant ($H = 1.97\text{ s}$, $I_{\text{th}} = 39.0\text{ A}$, $\sigma_{XR} = 5.0$, $dt = 80\text{ }\mu\text{s}$):

| Contingency & Protection Mode | Peak Bridge $|\mathbf{i}_{1,dq}|$ | Max $|\delta_{\text{rel}}(t)|$ | Voltage-Loop Sat. Duty | Fault / Event $\bar{Q}_{\text{pcc}}$ | Post-Event $P_{\text{pcc}}$ | Ride-Through |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.15 pu` Sag — Unprotected** | `160.23 A` (`3.91 pu`) | `71.6 deg` | `0.0%` | `+26.53 kvar` | `16.69 kW` | Fail (Overcurrent) |
| **`0.15 pu` Sag — Hard Clamp Only** | `52.01 A` | `440.7 deg` | `81.7%` | `+4.87 kvar` | `-17.87 kW` | Fail (Windup `LOS`) |
| **`0.15 pu` Sag — `TVI` Only** | `48.00 A` | `386.5 deg` | `40.6%` | `+6.75 kvar` | `19.08 kW` | Fail (`1` Cycle Slip) |
| **`0.15 pu` Sag — `TVI + Power AW`** | **`47.95 A`** (`<= 48 A`) | **`27.5 deg`** | **`0.1%`** | **`+8.28 kvar`** | **`16.98 kW`** | **Pass (Zero Slips)** |
| **`-45 deg` Jump — Unprotected** | `142.76 A` (`3.48 pu`) | `58.3 deg` | `0.0%` | `+2.33 kvar` | `12.00 kW` | Fail (Overcurrent) |
| **`-45 deg` Jump — Hard Clamp Only** | `51.40 A` | `344.7 deg` | `81.5%` | `-16.09 kvar` | `-15.62 kW` | Fail (Qoria Latch-Up) |
| **`-45 deg` Jump — `TVI + Power AW`** | **`46.86 A`** (`<= 48 A`) | **`58.2 deg`** | **`0.0%`** | `-7.37 kvar` | **`12.21 kW`** | **Pass (Zero Slips)** |

## Weak-Grid `SCR` Stability & Line-Opening Collapse (`20 kVA`, $P_{\text{ref}} = 15\text{ kW} = 0.75\text{ pu}$, `X/R = 7.0`)

| Control Architecture | ODE States | Critical `SCR_crit` (Hopf) | Equilibrium @ `SCR = 1.15` | Line-Trip `SCR 4.0 -> 1.22` | Min $|\mathbf{v}_{\text{pcc}}|$ | Max $|\delta(t)|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GFL `SRF-PLL` (`55 / 35 / 10 Hz`)** | `11` | `2.99 / 1.80 / 1.67` | None (`SCR_static ~ 1.33`) | Unbounded Pole-Slipping (`LOS`) | `<= 0.048 pu` | `> 4,200 deg` |
| **GFM `VSG` (`H = 4.93 s`)** | `14` | `< 1.15` (`-3.97 s^-1`) | `0.954 pu`, `47.2 deg` | Synchronized Ride-Through | `0.937 pu` | `43.5 deg` |
| **GFM `dVOC` (Andronov-Hopf)** | `12` | `< 1.15` (`-2.65 s^-1`) | `0.953 pu`, `45.1 deg` | Synchronized Ride-Through | `0.924 pu` | `40.8 deg` |

## Comparative GFM Control Triad Summary (`20 kVA`, `SCR = 6.0`, `X/R = 7.0`)

| GFM Architecture | Total ODE States | Power LPF | Step Rise $t_{90}$ | Grid-Drop Peak $P_{\text{pcc}}$ | Frequency Nadir | Steady $P_{\text{pcc},\infty}$ | Max $\operatorname{Re}(\lambda)$ (`SCR=6` / `SCR=2`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multi-Loop Droop** | `13` | `15 Hz` | `99.0 ms` | `18.44 kW` | `49.788 Hz` | `17.89 kW` | `-4.10` / `-5.37 s^-1` |
| **Virtual Sync. Gen. (`VSG`)** | `14` | `15 Hz` | `189.0 ms` | `21.52 kW` | `49.748 Hz` | `17.88 kW` | `-4.24` / `-4.65 s^-1` |
| **Andronov-Hopf `dVOC`** | `12` | None | `214.0 ms` | `17.97 kW` | `49.791 Hz` | `17.59 kW` | `-3.03` / `-5.06 s^-1` |

## Governing Equations

### 1. Threshold Transient Virtual Impedance (`TVI`) & Synchronizing Power Anti-Windup (`src/transient_virtual_impedance.py`, `src/em_frt_benchmark.py`)

Above sensed peak current $I_{\text{th}}$, `TVI` augments base virtual impedance $Z_v = R_v + j\omega L_v$ while `PAW` derates accelerating swing power during deep PCC voltage depressions (Paquette & Divan, *IEEE TIA*, 2015; Qoria et al., *IEEE TPEL*, 2020; Taul et al., *IEEE JESTPE*, 2020; Rokrok et al., *IEEE TPWRS*, 2022):

$$\Delta R_v(|\mathbf{i}|) = k_{\text{tvi}}\max(0,\, |\mathbf{i}| - I_{\text{th}}), \qquad \Delta X_v(|\mathbf{i}|, \omega) = \sigma_{XR}\,\frac{\omega}{\omega_0}\,\Delta R_v(|\mathbf{i}|), \qquad k_{\text{tvi}} = \frac{c_m V_0 / I_{\max} - |Z_{v,0}|}{(I_{\max} - I_{\text{th}})\sqrt{1 + \sigma_{XR}^2}}$$

$$P_{\text{ref,eff}} = P_{\text{ref}}\cdot \begin{cases}(|\mathbf{v}_{\text{pcc}}|/V_0)^{\gamma_v}, & |\mathbf{v}_{\text{pcc}}|/V_0 < V_{\text{knee}} \\ 1, & \text{otherwise}\end{cases} \;-\; K_{\text{aw}}\max(0,\, |\mathbf{i}| - I_{\text{th}})$$

Preserving unsaturated outer voltage-loop regulation prevents circular current-clamp latch-up under severe grid phase jumps while bounding $|\mathbf{i}_{1,dq}| \le I_{\max}$ and eliminating virtual-rotor angle windup during prolonged deep faults.

### 2. Weak-Grid `SRF-PLL` Impedance Coupling & Pole-Slipping `LOS` (`src/gfl_pll.py`, `src/weak_grid_benchmark.py`)

In weak inductive grids ($L_{\text{net}}$ large, `SCR` low), $v_{fq} = -V_g \sin(\delta_{\text{pll}}) + R_{\text{net}} i_{2q} + \omega_{\text{pll}} L_{\text{net}} i_{2d} + L_{\text{net}}\,di_{2q}/dt$ creates positive feedback $+\omega_{\text{pll}} L_{\text{net}} i_{2d}$ into $\omega_{\text{pll}}$, triggering bandwidth-dependent Hopf instability (`SCR_crit = 2.99 / 1.80 / 1.67` at `55 / 35 / 10 Hz`) above the static nose-curve collapse (`SCR_static ~ 1.33`).

### 3. Nonlinear Andronov-Hopf `dVOC` & 14-State Synchronous-$dq$ $LCL$ Plant (`src/dvoc_control.py`, `src/lcl_plant.py`, `src/cascaded_control.py`)

The shared 6-state electromagnetic $LCL$ + Thévenin network plant $\mathbf{x}_{\text{lcl}} = [i_{1d}, i_{1q}, v_{cd}, v_{cq}, i_{2d}, i_{2q}]^\top$ obeys exact rotational passivity ($\mathbf{x}^\top \mathbf{J} \mathbf{x} \equiv 0$) and couples cleanly to `Droop`, `VSG`, and `dVOC` ($\dot{\mathbf{v}}_{\alpha\beta}$ Andronov-Hopf oscillator with global unloaded Bernoulli black-start convergence).

## Scope & Engineering Limitations

This repository models balanced three-phase (positive-sequence) averaged inverter bridge modulation, $LCL$ filter electromagnetics, cascaded $dq$ / `SRF-PLL` / `TVI` control loops, and Thévenin AC grid equivalents. In Module 5 (`v0.6.0`), discrete PWM switching ripple, dead-time distortion, semiconductor junction thermal limits, DC-link capacitor energy balance, and unbalanced negative/zero-sequence faults are not modeled. Passing tests and benchmarks verify **mathematical and software consistency for the documented models only**—they are not evidence of real-hardware performance, grid-code compliance, or equipment certification.

## Citation

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff) under the [MIT License](LICENSE).
