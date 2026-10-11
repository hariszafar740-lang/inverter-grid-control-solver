# Inverter Grid Control Solver

A small Python sandbox for inverter-control blocks: P–ω and Q–V droop, a
virtual-synchronous-generator (VSG) swing equation, alpha-beta power
measurement, virtual impedance, plotting, and a reproducible voltage-sag
response benchmark.

## Install and run

Python 3.10 or later is required. From a clean checkout:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.benchmark
```

The FRT benchmark's historical direct-script entry point also works from the
repository root:

```bash
python src/benchmark.py
```

The test and benchmark commands return a nonzero exit code on failed
acceptance checks. The benchmark validates that a sag was actually applied,
that event samples exist, that reported P/Q agree with simultaneous voltage
and current, that the current limit is respected and exercised for its default
severe-sag case, that event RoCoF is computed from the recorded VSG frequency,
and that the declared post-fault settling window is met.

## Model and equations

This repository is a **simplified discrete-time dynamic control-block model**
with algebraic, positive-sequence alpha-beta voltage/current vectors. It has
no switching waveform or PWM model and does not solve the converter's complete
averaged electrical state-space model or an AC network. It is therefore not an
EMT simulation and not a validated transient-stability tool.

The alpha-beta components use an amplitude-invariant, peak phase-equivalent
convention (zero sequence omitted). Three-phase instantaneous power is
calculated from the actual simultaneous samples:

```text
P = 3/2 (v_alpha i_alpha + v_beta i_beta)
Q = 3/2 (v_beta i_alpha - v_alpha i_beta)
```

The benchmark's algebraic P/Q current reference is

```text
i_alpha* = (v_alpha P_ref + v_beta Q_ref) / (3/2 |V|^2)
i_beta*  = (v_beta P_ref - v_alpha Q_ref) / (3/2 |V|^2)
```

A zero-voltage sample has no defined voltage-aligned P/Q command; in this
reduced model it commands zero current and hence zero transferred P/Q. The
current limiter radially scales both alpha-beta components to the declared
peak-current limit. When it operates, it scales active and reactive power
together; there is no active/reactive priority policy. Output powers are
recomputed from the limited currents and the contemporaneous grid voltage.
Constant-power tracking is therefore only achieved when the corresponding
current reference is within the declared current capability.

The filtered active power drives the VSG frequency state:

```text
dω/dt = ((P_ref - P_meas)/ω0 - Dp(ω - ω0)) / J
f = ω / (2π)
```

`P_meas` is the first-order-filtered output power. The filter uses forward
Euler with `2π f_c dt <= 1`; the benchmark uses `f_c = 10 Hz` and `dt = 1 ms`.
The explicit-Euler VSG damping update is checked for its linear stability
condition. These are numerical checks for the implemented blocks, not
validation of a physical inverter.

## Default sag scenario and measured windows

The default case uses a 230 V RMS phase-equivalent nominal grid voltage
(325.27 V peak), `P_ref = 10 kW`, `Q_ref = 0`, a 30 A peak current limit, a
50% voltage-magnitude sag from **0.300 s inclusive to 0.600 s exclusive**, and
clearing/recovery through 1.5 s. Every event boundary is aligned to the 1 ms
sample grid. The power filter and VSG start at the unsaturated pre-fault
operating point; startup is not counted as fault response.

At nominal voltage, the 10 kW command needs about 20.49 A peak. At 0.5 pu
voltage it requests about 40.99 A, so the radial limiter sets actual current to
30 A. The resulting active output is about
`1.5 * 0.5 * 325.27 * 30 = 7.32 kW`, not the pre-fault 10 kW. The benchmark
records voltage, alpha-beta and voltage-aligned active/reactive current
references and outputs, current-limit state, instantaneous and filtered P/Q,
VSG frequency and angle, event times, fault-window RoCoF, and recovery.

RoCoF is the finite difference of successive VSG frequency samples, assigned
to each interval's right endpoint. Pre-event, fault, and post-event intervals
are reported separately; only intervals ending in `[fault_start, fault_clear)`
contribute to `max_rocof`. The interval ending at fault application is included
in the event measure, while the interval ending at fault clearing is assigned
to recovery. Startup behavior cannot dominate the fault metric. Recovery
settling means P/Q are within 2% of their references (with a 1 W/var minimum
tolerance) and VSG frequency is within 0.05 Hz (0.1% of nominal) continuously
for at least 0.1 s, within the declared post-event measurement window. These
are model-specific numerical acceptance tolerances, not grid-code thresholds.

## Important limitations and claims

The benchmark's nominal grid voltage is prescribed; there is no network or
fault-impedance solution. The current reference is an ideal algebraic P/Q
tracking law with uniform vector limiting, not an implemented inner current
controller. The default case has no reactive-current boost law. The VSG angle
is a virtual rotor state; the prescribed voltage vector does not rotate or
couple back to an electrical network. The model omits output-filter dynamics,
modulation, semiconductor limits/thermal behavior, DC-link dynamics,
sequence/unbalance behavior, protection, control-mode transitions, and
controller integrators/anti-windup. The separate virtual-impedance block
changes a voltage reference algebraically but is not part of the benchmark's
network solution.

Accordingly, a passing benchmark demonstrates **software and equation
consistency for the declared reduced-order simulation only**. It does not
establish physical-model validation, real-hardware performance, transient
stability, IEEE/IEC/national grid-code compliance, or any certification. No
particular standard's requirements or test procedure are claimed here.

## CI and quality checks

GitHub Actions installs the declared development dependencies, runs the test
suite, checks source/test lint, compiles the Python modules, and executes the
FRT acceptance benchmark as a required command (a failed check fails the job).
The project does not currently configure a type checker or a packaging/build
backend.
