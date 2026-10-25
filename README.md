# Grid-Forming Inverter Control & Thévenin Network Dynamic Solver

[![CI](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/hariszafar740-lang/inverter-grid-control-solver/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![Interactive Workbench](https://img.shields.io/badge/Interactive_Workbench-GitHub_Pages-10b981?logo=github)](https://hariszafar740-lang.github.io/inverter-grid-control-solver/)
![Release](https://img.shields.io/badge/release-v1.0.0-0284c7)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Physics-based Python simulation, modal participation, synchronous-$dq$ impedance passivity verification framework (`gfm_solver` / `gfm-sim`), and **[Zero-CDN Interactive HTML5/Canvas Workbench](https://hariszafar740-lang.github.io/inverter-grid-control-solver/)** for **Grid-Forming (GFM) and Grid-Following (GFL) Inverter Control**:

- **Zero-CDN Interactive HTML5/Canvas GFM & Weak-Grid Workbench (`src/web_dashboard.py`, `gfm-sim web`):** Self-contained offline/GitHub Pages engineering workbench combining a real-time browser 4th-order Runge-Kutta (`dt = 250 us`) electromagnetic fault & weak-grid simulator (`VSG`, `Droop`, `dVOC`, `SRF-PLL`; `TVI + PAW`, `Hard Clamp`, `Unprotected`) with live Python state-space root-locus, passivity $\nu(f)$, Generalized Nyquist $Z = P + N$, and one-click CSV trace export.
- **Small-Signal Eigenvalue Participation, Synchronous-$dq$ Admittance Passivity & Generalized Nyquist (`src/small_signal_impedance.py`, `src/cli.py`):** Decomposes right/left state-space eigenvectors into exact normalized participation matrices $P_{ki}$, derives the exact $2\times 2$ synchronous-$dq$ PCC input admittance $\mathbf{Y}_{\text{inv},dq}(s)$ satisfying the Schur-complement pole identity $\det(\mathbf{I}_2 + \mathbf{Z}_{g,dq}(\lambda_i)\mathbf{Y}_{\text{inv},dq}(\lambda_i)) = 0$, proves why Qoria partial feedforward ($F_i = 0.75$) maintains medium/high-frequency passivity ($\nu(f) > 0$ across $[25, 6000]\text{ Hz}$) whereas naive $F_i = 1.00$ induces a non-passive dip ($\nu = -1.04\text{ S}$ on $[25.1, 93.3]\text{ Hz}$), and verifies exact Generalized Nyquist RHP pole counts $Z = P + N \equiv Z_{\text{ss}}$ across `VSG`, `Droop`, `dVOC`, and `GFL`.
- **14-State Electromagnetic Fault Ride-Through (`FRT`), `-45 deg` Phase Jump, Threshold Transient Virtual Impedance (`TVI`) & Power Anti-Windup (`src/transient_virtual_impedance.py`, `src/em_frt_benchmark.py`):** Contrasts unprotected overcurrent spikes (`160.2 A = 3.91 pu`), Qoria circular current-limiter latch-up & virtual-rotor windup (`440.7 deg` under a `0.15 pu` sag; `344.7 deg` under a `-45 deg` phase jump), and coordinated **`TVI + Synchronizing Power-Loop Anti-Windup` (`PAW`)** (`|i_{1,dq}| <= 47.95 A <= I_{max}`, `0.1%` voltage-loop saturation, `max delta = 27.5 deg`, zero cycle slips).
- **Weak-Grid `SCR` Collapse Benchmark — `GFL (SRF-PLL)` vs. `GFM` (`src/gfl_pll.py`, `src/weak_grid_benchmark.py`):** Quantifies bandwidth-dependent PLL–impedance Hopf bifurcation ($\text{SCR}_{\text{crit}}(f_{bw,\text{pll}}) = 2.99 / 1.80 / 1.67$) and static nose-curve saddle-node Loss of Synchronism (`LOS`) during parallel-line opening (`SCR: 4.0 -> 1.22`).
- **Comparative GFM Control Triad & 14-State Synchronous-$dq$ $LCL$ Plant (`src/gfm_triad.py`, `src/dvoc_control.py`, `src/lcl_plant.py`, `src/cascaded_control.py`):** Evaluates **13-State Multi-Loop Droop**, **14-State `VSG`**, and **12-State Nonlinear Andronov-Hopf `dVOC`** with black-start limit-cycle self-excitation alongside the 4-state Thévenin baseline and fail-closed algebraic FRT CLI.

![Small-Signal Modal Participation, dq Admittance Passivity & Generalized Nyquist](results/small_signal_eigenvalue_impedance_passivity.png)

![14-State Electromagnetic FRT, TVI, Anti-Windup & Phase-Jump Response](results/em_frt_tvi_phase_jump_response.png)

![Weak-Grid SCR Stability & Line-Opening Collapse: GFL vs GFM](results/gfl_vs_gfm_weak_grid_collapse.png)

![Comparative Grid-Forming Control Triad Response](results/gfm_triad_comparative_response.png)

![14-State Electromagnetic dq LCL Plant & Cascaded Dual-Loop PI Response](results/lcl_cascaded_electromagnetic_response.png)

![Closed-Loop Grid-Forming VSG & Thévenin Network Response](results/inverter_transient_response.png)

## Quick Start & `gfm-sim` CLI

Python `3.10–3.13` is supported and tested in CI as a PEP 621 / PEP 561 typed package (`gfm_solver`):

```bash
python -m pip install -e ".[dev]"
gfm-sim info --json
gfm-sim impedance --scr 1.65 --p-kw 15.0
gfm-sim web --output-dir site --json
python main.py
python -m src.benchmark
python -m pytest -v
```

All CLI subcommands (`gfm-sim info`, `gfm-sim triad`, `gfm-sim weak-grid`, `gfm-sim frt`, `gfm-sim impedance`, `gfm-sim web`, or `python -m gfm_solver <subcommand>`) support `--json` structured output.

## Small-Signal `dq` Admittance Passivity & Generalized Nyquist Summary (`SCR = 1.65`, $P_{\text{ref}} = 15\text{ kW}$)

Cutting the coupled converter–network loop at the PCC in the synchronous grid $dq$ frame ($\Delta \mathbf{i}_{\text{in},dq}^g = \mathbf{Y}_{\text{inv},dq}(s)\,\Delta \mathbf{v}_{\text{pcc},dq}^g$) yields exact equivalence between frequency-domain Cauchy winding ($Z_{\text{nyq}} = P + N$) and coupled state-space eigenvalues ($Z_{\text{ss}}$):

| Converter Architecture | Passivity $\min_{f \ge 25\text{ Hz}} \nu(f)$ | Non-Passive Band ($\ge 25\text{ Hz}$) | Negative $\operatorname{Re}\{Y_{qq}\}$ Band | Stiff-Bus RHP $P$ | Nyquist Winding $N$ | Closed-Loop RHP $Z = P + N$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GFM VSG` (Qoria $F_i = 0.75$)** | **`+0.0019 S`** | **None (Passive)** | `[1.0, 3.3] Hz` | `2` | `-2` | **`0` (Hurwitz Stable)** |
| **`GFM VSG` (Naive $F_i = 1.00$)** | `-1.0415 S` | `[25.1, 93.3] Hz` | `[1.0, 89.6] Hz` | `2` | `0` | `2` (`42 Hz` Unstable!) |
| **`GFM dVOC` (Andronov-Hopf)** | **`+0.0019 S`** | **None (Passive)** | `[1.0, 6.5] Hz` | `2` | `-2` | **`0` (Hurwitz Stable)** |
| **`GFM Droop` (`13`-State)** | **`+0.0019 S`** | **None (Passive)** | `[1.0, 7.2] Hz` | `2` | `-2` | **`0` (Hurwitz Stable)** |
| **`GFL SRF-PLL` (`10 Hz`)** | `-0.0168 S` | `[25.1, 479.6] Hz` | `[1.0, 19.7] Hz` | `0` | `+2` | `2` (Unstable Hopf) |
| **`GFL SRF-PLL` (`35 Hz`)** | `-0.1243 S` | `[25.1, 479.6] Hz` | `[1.0, 283.6] Hz` | `0` | `+2` | `2` (Unstable Hopf) |
| **`GFL SRF-PLL` (`55 Hz`)** | `-0.1259 S` | `[25.1, 479.6] Hz` | `[1.0, 361.4] Hz` | `0` | `+4` | `4` (2 Unstable Pairs) |

## 14-State Electromagnetic `FRT` & `-45 deg` Grid Phase-Jump Summary (`20 kVA`, `SCR = 4.0`, $I_{\max} = 48.0\text{ A}$)

| Contingency & Protection Mode | Peak Bridge $|\mathbf{i}_{1,dq}|$ | Max $|\delta_{\text{rel}}(t)|$ | Voltage-Loop Sat. Duty | Fault / Event $\bar{Q}_{\text{pcc}}$ | Post-Event $P_{\text{pcc}}$ | Ride-Through |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.15 pu` Sag — Unprotected** | `160.23 A` (`3.91 pu`) | `71.6 deg` | `0.0%` | `+26.53 kvar` | `16.69 kW` | Fail (Overcurrent) |
| **`0.15 pu` Sag — Hard Clamp Only** | `52.01 A` | `440.7 deg` | `81.7%` | `+4.87 kvar` | `-17.87 kW` | Fail (Windup `LOS`) |
| **`0.15 pu` Sag — `TVI` Only** | `48.00 A` | `386.5 deg` | `40.6%` | `+6.75 kvar` | `19.08 kW` | Fail (`1` Cycle Slip) |
| **`0.15 pu` Sag — `TVI + Power AW`** | **`47.95 A`** (`<= 48 A`) | **`27.5 deg`** | **`0.1%`** | **`+8.28 kvar`** | **`16.98 kW`** | **Pass (Zero Slips)** |
| **`-45 deg` Jump — Hard Clamp Only** | `51.40 A` | `344.7 deg` | `81.5%` | `-16.09 kvar` | `-15.62 kW` | Fail (Qoria Latch-Up) |
| **`-45 deg` Jump — `TVI + Power AW`** | **`46.86 A`** (`<= 48 A`) | **`58.2 deg`** | **`0.0%`** | `-7.37 kvar` | **`12.21 kW`** | **Pass (Zero Slips)** |

## Weak-Grid `SCR` Collapse & GFM Triad Summary (`20 kVA`, `X/R = 7.0`)

- **Weak-Grid `GFL` vs. `GFM` ($P_{\text{ref}} = 15\text{ kW}$):** `GFL SRF-PLL` (`55 / 35 / 10 Hz`) exhibits Hopf instability at $\text{SCR}_{\text{crit}} = 2.99 / 1.80 / 1.67$ and static nose collapse below $\text{SCR} \approx 1.33$, whereas `VSG` ($\max\operatorname{Re}(\lambda) = -3.97\text{ s}^{-1}$) and `dVOC` ($-2.65\text{ s}^{-1}$) remain Hurwitz down to $\text{SCR} = 1.15$ and ride through `SCR 4.0 -> 1.22` line opening without `LOS`.
- **Comparative GFM Control Triad (`SCR = 6.0`):** `Droop` (`13` states, $t_{90} = 99.0\text{ ms}$, peak $18.44\text{ kW}$), `VSG` (`14` states, $t_{90} = 189.0\text{ ms}$, peak $21.52\text{ kW}$, nadir $49.748\text{ Hz}$), and `dVOC` (`12` states, $t_{90} = 214.0\text{ ms}$, peak $17.97\text{ kW}$, exact Bernoulli black-start self-excitation).

## Governing Equations

### 1. Modal Participation, Synchronous-$dq$ Input Admittance & Generalized Nyquist (`src/small_signal_impedance.py`)

For right/left eigenvectors $\mathbf{A}\mathbf{v}_i = \lambda_i\mathbf{v}_i$ and $\mathbf{w}_i^\top \mathbf{A} = \lambda_i \mathbf{w}_i^\top$, state $k$'s participation in mode $i$ is $P_{ki} = |w_{ki}v_{ki}| / \sum_m |w_{mi}v_{mi}|$. Linearizing the stiff-PCC converter realization $(\mathbf{A}_0, \mathbf{B}_0, \mathbf{C}_0)$ around the loaded PCC operating point gives:

$$\mathbf{Y}_{\text{inv},dq}(s) = \mathbf{C}_0(s\mathbf{I} - \mathbf{A}_0)^{-1}\mathbf{B}_0, \qquad \nu(\omega) = \lambda_{\min}\!\left(\tfrac{1}{2}\bigl[\mathbf{Y}_{\text{inv},dq}(j\omega) + \mathbf{Y}_{\text{inv},dq}^H(j\omega)\bigr]\right), \qquad \det\!\left(\mathbf{I}_2 + \mathbf{Z}_{g,dq}(\lambda_i)\mathbf{Y}_{\text{inv},dq}(\lambda_i)\right) = 0$$

### 2. Threshold `TVI` & Synchronizing Power Anti-Windup (`src/transient_virtual_impedance.py`)

$$\Delta R_v(|\mathbf{i}|) = k_{\text{tvi}}\max(0,\, |\mathbf{i}| - I_{\text{th}}), \qquad \Delta X_v(|\mathbf{i}|, \omega) = \sigma_{XR}\,\frac{\omega}{\omega_0}\,\Delta R_v(|\mathbf{i}|), \qquad P_{\text{ref,eff}} = P_{\text{ref}}\cdot \Phi_v(V_{\text{pcc}}) - K_{\text{aw}}\max(0,\, |\mathbf{i}| - I_{\text{th}})$$

## Scope & Engineering Limitations

This repository models balanced three-phase (positive-sequence) averaged inverter bridge modulation, $LCL$ filter electromagnetics, cascaded $dq$ / `SRF-PLL` / `TVI` control loops, small-signal $dq$ terminal admittances, and Thévenin AC grid equivalents. In `v1.0.0`, discrete PWM switching ripple, dead-time distortion, semiconductor junction thermal limits, DC-link capacitor energy balance, and unbalanced negative/zero-sequence faults are not modeled. Passing tests and benchmarks verify **mathematical and software consistency for the documented models only**—they are not evidence of real-hardware performance, grid-code compliance, or equipment certification.

## Citation

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff) under the [MIT License](LICENSE).
