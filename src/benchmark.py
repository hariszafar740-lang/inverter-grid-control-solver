"""Reproducible reduced-order voltage-sag/FRT response benchmark.

This benchmark is a simplified positive-sequence dynamic control-block study,
not an EMT or grid-code compliance test.  Its current reference and radial current
limit are explicit algebraic assumptions documented in README.md.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Support standalone wheel imports (``gfm_solver``), module execution
# (``python -m src.benchmark`` / ``python -m gfm_solver.benchmark``), and direct
# historical script execution (``python src/benchmark.py``).
if __package__ not in (None, ""):
    from .current_control import apply_current_limit, current_reference_from_power
    from .power_measurement import PowerMeasurementLPF
    from .virtual_inertia import VirtualInertiaEngine
else:
    _module_dir = str(Path(__file__).resolve().parent)
    if _module_dir not in sys.path:
        sys.path.insert(0, _module_dir)
    from current_control import apply_current_limit, current_reference_from_power  # type: ignore[no-redef]
    from power_measurement import PowerMeasurementLPF  # type: ignore[no-redef]
    from virtual_inertia import VirtualInertiaEngine  # type: ignore[no-redef]

NOMINAL_FREQUENCY_HZ = 50.0
NOMINAL_VOLTAGE_RMS_PHASE_V = 230.0
NOMINAL_VOLTAGE_PEAK_PHASE_V = NOMINAL_VOLTAGE_RMS_PHASE_V * math.sqrt(2.0)
DEFAULT_DT_S = 1e-3
DEFAULT_SIMULATION_END_S = 1.5
DEFAULT_FAULT_START_S = 0.3
DEFAULT_FAULT_DURATION_S = 0.3
DEFAULT_SAG_DEPTH_PU = 0.5
DEFAULT_P_REF_W = 10_000.0
DEFAULT_Q_REF_VAR = 0.0
DEFAULT_CURRENT_LIMIT_PEAK_A = 30.0
DEFAULT_POWER_CUTOFF_HZ = 10.0
DEFAULT_INERTIA_KGM2 = 2.0
DEFAULT_DAMPING_NMS_PER_RAD = 15.0
DEFAULT_PRE_EVENT_WINDOW_S = 0.1
DEFAULT_POST_EVENT_WINDOW_S = 0.5
DEFAULT_SETTLING_HOLD_S = 0.1


class BenchmarkAcceptanceError(RuntimeError):
    """Raised when the mandatory fault-response evidence is absent or invalid."""


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _sample_index(seconds: float, dt: float, name: str) -> int:
    index = round(seconds / dt)
    represented_seconds = index * dt
    tolerance = max(1e-12, 1e-9 * max(abs(seconds), dt))
    if abs(represented_seconds - seconds) > tolerance:
        raise ValueError(f"{name} must be an integer multiple of dt")
    return int(index)


def _max_abs_or_none(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.max(np.abs(values[mask])))


def calculate_rocof_windows(
    time_s: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    fault_start_s: float,
    fault_clear_s: float,
    pre_event_start_s: float,
    post_event_end_s: float,
) -> dict[str, Any]:
    """Differentiate the VSG frequency samples and report explicit time windows.

    Each finite difference is timestamped at its right endpoint.  The interval
    ending at fault application is included in the fault metric; the interval
    ending at clearing is assigned to recovery.  The pre-event metric is
    reported separately and never contributes to ``fault_max_rocof_hz_per_s``.
    """
    time_s = np.asarray(time_s, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if time_s.ndim != 1 or frequency_hz.shape != time_s.shape or time_s.size < 2:
        raise ValueError("time and frequency must be equal-length one-dimensional samples")
    if not (np.all(np.isfinite(time_s)) and np.all(np.isfinite(frequency_hz))):
        raise ValueError("time and frequency samples must be finite")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time samples must be strictly increasing")
    if not (pre_event_start_s < fault_start_s < fault_clear_s <= post_event_end_s):
        raise ValueError("RoCoF measurement windows are not ordered")

    rocof_hz_per_s = np.diff(frequency_hz) / np.diff(time_s)
    interval_end_s = time_s[1:]
    pre_mask = (interval_end_s > pre_event_start_s) & (interval_end_s < fault_start_s)
    fault_mask = (interval_end_s >= fault_start_s) & (interval_end_s < fault_clear_s)
    post_mask = (interval_end_s >= fault_clear_s) & (interval_end_s <= post_event_end_s + 1e-12)
    if not np.any(fault_mask):
        raise ValueError("fault window contains no frequency-difference samples")

    return {
        "rocof_interval_end_s": interval_end_s,
        "rocof_hz_per_s": rocof_hz_per_s,
        "pre_event_max_rocof_hz_per_s": _max_abs_or_none(rocof_hz_per_s, pre_mask),
        "fault_max_rocof_hz_per_s": _max_abs_or_none(rocof_hz_per_s, fault_mask),
        "post_event_max_rocof_hz_per_s": _max_abs_or_none(rocof_hz_per_s, post_mask),
    }


def _settling_time(
    time_s: np.ndarray,
    frequency_hz: np.ndarray,
    p_inst_w: np.ndarray,
    q_inst_var: np.ndarray,
    clear_index: int,
    recovery_end_index: int,
    dt_s: float,
    p_ref_w: float,
    q_ref_var: float,
    hold_s: float,
    frequency_tolerance_hz: float,
    power_tolerance_fraction: float,
) -> float | None:
    hold_steps = _sample_index(hold_s, dt_s, "settling_hold_s")
    p_tolerance = max(1.0, power_tolerance_fraction * abs(p_ref_w))
    q_tolerance = max(1.0, power_tolerance_fraction * abs(q_ref_var))
    settled = (
        (np.abs(frequency_hz - NOMINAL_FREQUENCY_HZ) <= frequency_tolerance_hz)
        & (np.abs(p_inst_w - p_ref_w) <= p_tolerance)
        & (np.abs(q_inst_var - q_ref_var) <= q_tolerance)
    )

    last_start = min(len(time_s) - hold_steps - 1, recovery_end_index - hold_steps)
    for index in range(clear_index, last_start + 1):
        if np.all(settled[index : index + hold_steps + 1]):
            return float(time_s[index] - time_s[clear_index])
    return None


def run_fault_ride_through_benchmark(
    *,
    dt_s: float = DEFAULT_DT_S,
    simulation_end_s: float = DEFAULT_SIMULATION_END_S,
    fault_start_s: float = DEFAULT_FAULT_START_S,
    fault_duration_s: float = DEFAULT_FAULT_DURATION_S,
    sag_depth_pu: float = DEFAULT_SAG_DEPTH_PU,
    p_ref_w: float = DEFAULT_P_REF_W,
    q_ref_var: float = DEFAULT_Q_REF_VAR,
    nominal_voltage_peak_v: float = NOMINAL_VOLTAGE_PEAK_PHASE_V,
    current_limit_peak_a: float = DEFAULT_CURRENT_LIMIT_PEAK_A,
    power_cutoff_hz: float = DEFAULT_POWER_CUTOFF_HZ,
    inertia_kgm2: float = DEFAULT_INERTIA_KGM2,
    damping_nms_per_rad: float = DEFAULT_DAMPING_NMS_PER_RAD,
    pre_event_window_s: float = DEFAULT_PRE_EVENT_WINDOW_S,
    post_event_window_s: float = DEFAULT_POST_EVENT_WINDOW_S,
    settling_hold_s: float = DEFAULT_SETTLING_HOLD_S,
    frequency_settling_tolerance_hz: float = 0.05,
    power_settling_tolerance_fraction: float = 0.02,
) -> dict[str, Any]:
    """Simulate pre-fault, sag, clearing, and recovery at fixed sample intervals.

    ``sag_depth_pu`` is a fractional drop (0.5 means the remaining voltage is
    0.5 pu).  The event is active on ``[fault_start_s, fault_clear_s)``.  Event
    edges and the simulation end must lie exactly on the requested sample grid.
    The LPF and VSG are initialized at the unsaturated pre-fault operating
    point, so startup behavior is not included in event RoCoF.
    """
    dt_s = _finite_float("dt_s", dt_s)
    simulation_end_s = _finite_float("simulation_end_s", simulation_end_s)
    fault_start_s = _finite_float("fault_start_s", fault_start_s)
    fault_duration_s = _finite_float("fault_duration_s", fault_duration_s)
    sag_depth_pu = _finite_float("sag_depth_pu", sag_depth_pu)
    p_ref_w = _finite_float("p_ref_w", p_ref_w)
    q_ref_var = _finite_float("q_ref_var", q_ref_var)
    nominal_voltage_peak_v = _finite_float(
        "nominal_voltage_peak_v", nominal_voltage_peak_v
    )
    current_limit_peak_a = _finite_float("current_limit_peak_a", current_limit_peak_a)
    pre_event_window_s = _finite_float("pre_event_window_s", pre_event_window_s)
    post_event_window_s = _finite_float("post_event_window_s", post_event_window_s)
    settling_hold_s = _finite_float("settling_hold_s", settling_hold_s)
    frequency_settling_tolerance_hz = _finite_float(
        "frequency_settling_tolerance_hz", frequency_settling_tolerance_hz
    )
    power_settling_tolerance_fraction = _finite_float(
        "power_settling_tolerance_fraction", power_settling_tolerance_fraction
    )

    if dt_s <= 0.0 or simulation_end_s <= 0.0:
        raise ValueError("dt_s and simulation_end_s must be greater than zero")
    if fault_start_s <= 0.0 or fault_duration_s <= 0.0:
        raise ValueError("fault_start_s and fault_duration_s must be greater than zero")
    if not 0.0 <= sag_depth_pu <= 1.0:
        raise ValueError("sag_depth_pu must be between zero and one")
    if nominal_voltage_peak_v <= 0.0:
        raise ValueError("nominal_voltage_peak_v must be greater than zero")
    if current_limit_peak_a <= 0.0:
        raise ValueError("current_limit_peak_a must be greater than zero")
    if pre_event_window_s <= 0.0 or post_event_window_s <= 0.0:
        raise ValueError("pre- and post-event measurement windows must be positive")
    if settling_hold_s <= 0.0:
        raise ValueError("settling_hold_s must be greater than zero")
    if frequency_settling_tolerance_hz <= 0.0:
        raise ValueError("frequency_settling_tolerance_hz must be greater than zero")
    if power_settling_tolerance_fraction <= 0.0:
        raise ValueError("power_settling_tolerance_fraction must be greater than zero")

    end_index = _sample_index(simulation_end_s, dt_s, "simulation_end_s")
    start_index = _sample_index(fault_start_s, dt_s, "fault_start_s")
    duration_steps = _sample_index(fault_duration_s, dt_s, "fault_duration_s")
    if start_index <= 0:
        raise ValueError("fault_start_s must leave at least one pre-fault sample")
    if duration_steps <= 0:
        raise ValueError("fault_duration_s must span at least one simulation step")
    clear_index = start_index + duration_steps
    fault_clear_s = clear_index * dt_s
    if end_index <= clear_index:
        raise ValueError("simulation_end_s must be later than fault clearing")
    _sample_index(pre_event_window_s, dt_s, "pre_event_window_s")
    post_window_steps = _sample_index(post_event_window_s, dt_s, "post_event_window_s")
    _sample_index(settling_hold_s, dt_s, "settling_hold_s")
    if post_window_steps <= 0 or _sample_index(settling_hold_s, dt_s, "settling_hold_s") <= 0:
        raise ValueError("post-event and settling windows must span at least one step")
    if post_window_steps < _sample_index(settling_hold_s, dt_s, "settling_hold_s"):
        raise ValueError("post_event_window_s must be at least settling_hold_s")
    post_end_index = min(end_index, clear_index + post_window_steps)
    post_end_s = post_end_index * dt_s

    # Establish a consistent pre-fault equilibrium before recording any event.
    i_alpha_pre, i_beta_pre = current_reference_from_power(
        nominal_voltage_peak_v, 0.0, p_ref_w, q_ref_var
    )
    i_alpha_pre_actual, i_beta_pre_actual, pre_fault_limited = apply_current_limit(
        i_alpha_pre, i_beta_pre, current_limit_peak_a
    )
    if pre_fault_limited:
        raise ValueError(
            "pre-fault P/Q reference exceeds the current limit; no unsaturated "
            "pre-fault operating point exists for this scenario"
        )
    p_pre_w, q_pre_var = PowerMeasurementLPF(
        cutoff_freq_hz=power_cutoff_hz,
        dt=dt_s,
    ).compute_instantaneous_power(
        nominal_voltage_peak_v, 0.0, i_alpha_pre_actual, i_beta_pre_actual
    )

    time_s = np.arange(end_index + 1, dtype=float) * dt_s
    fault_mask = np.zeros(time_s.size, dtype=bool)
    fault_mask[start_index:clear_index] = True
    fault_start_s = start_index * dt_s
    pre_start_s = max(0.0, fault_start_s - pre_event_window_s)
    pre_event_mask = (time_s >= pre_start_s) & (time_s < fault_start_s)
    post_event_mask = (time_s >= fault_clear_s) & (time_s <= post_end_s + 1e-12)

    voltage_magnitude_v = np.full(time_s.size, nominal_voltage_peak_v, dtype=float)
    voltage_magnitude_v[fault_mask] *= 1.0 - sag_depth_pu
    v_alpha_v = voltage_magnitude_v.copy()
    v_beta_v = np.zeros(time_s.size, dtype=float)

    power_meter = PowerMeasurementLPF(
        cutoff_freq_hz=power_cutoff_hz,
        dt=dt_s,
        initial_p_w=p_pre_w,
        initial_q_var=q_pre_var,
    )
    vsm = VirtualInertiaEngine(
        j_kgm2=inertia_kgm2,
        dp=damping_nms_per_rad,
        w0=2.0 * math.pi * NOMINAL_FREQUENCY_HZ,
        dt=dt_s,
    )

    i_alpha_ref_a = np.zeros(time_s.size, dtype=float)
    i_beta_ref_a = np.zeros(time_s.size, dtype=float)
    i_alpha_a = np.zeros(time_s.size, dtype=float)
    i_beta_a = np.zeros(time_s.size, dtype=float)
    i_active_ref_a = np.zeros(time_s.size, dtype=float)
    i_reactive_ref_a = np.zeros(time_s.size, dtype=float)
    i_active_a = np.zeros(time_s.size, dtype=float)
    i_reactive_a = np.zeros(time_s.size, dtype=float)
    current_magnitude_a = np.zeros(time_s.size, dtype=float)
    current_reference_magnitude_a = np.zeros(time_s.size, dtype=float)
    current_limit_active = np.zeros(time_s.size, dtype=bool)
    p_inst_w = np.zeros(time_s.size, dtype=float)
    q_inst_var = np.zeros(time_s.size, dtype=float)
    p_filtered_w = np.zeros(time_s.size, dtype=float)
    q_filtered_var = np.zeros(time_s.size, dtype=float)
    omega_rad_s = np.full(time_s.size, vsm.w0, dtype=float)
    frequency_hz = np.full(time_s.size, NOMINAL_FREQUENCY_HZ, dtype=float)
    theta_rad = np.zeros(time_s.size, dtype=float)

    for index in range(time_s.size):
        voltage_alpha = v_alpha_v[index]
        voltage_beta = v_beta_v[index]
        i_alpha_ref, i_beta_ref = current_reference_from_power(
            voltage_alpha, voltage_beta, p_ref_w, q_ref_var
        )
        i_alpha, i_beta, limited = apply_current_limit(
            i_alpha_ref, i_beta_ref, current_limit_peak_a
        )
        p_instantaneous, q_instantaneous = power_meter.compute_instantaneous_power(
            voltage_alpha, voltage_beta, i_alpha, i_beta
        )

        if index == 0:
            # The event begins later, so the first sample is exactly the
            # initialized pre-fault equilibrium and needs no integration step.
            p_filtered = p_pre_w
            q_filtered = q_pre_var
        else:
            p_filtered, q_filtered = power_meter.update(
                voltage_alpha, voltage_beta, i_alpha, i_beta
            )
            omega, frequency, theta = vsm.step(p_ref_w, p_filtered)
            omega_rad_s[index] = omega
            frequency_hz[index] = frequency
            theta_rad[index] = theta

        voltage_magnitude = math.hypot(voltage_alpha, voltage_beta)
        if voltage_magnitude > 0.0:
            i_active_reference = (
                voltage_alpha * i_alpha_ref + voltage_beta * i_beta_ref
            ) / voltage_magnitude
            i_reactive_reference = (
                voltage_beta * i_alpha_ref - voltage_alpha * i_beta_ref
            ) / voltage_magnitude
            i_active = (voltage_alpha * i_alpha + voltage_beta * i_beta) / voltage_magnitude
            i_reactive = (
                voltage_beta * i_alpha - voltage_alpha * i_beta
            ) / voltage_magnitude
        else:
            i_active_reference = 0.0
            i_reactive_reference = 0.0
            i_active = 0.0
            i_reactive = 0.0

        i_alpha_ref_a[index] = i_alpha_ref
        i_beta_ref_a[index] = i_beta_ref
        i_alpha_a[index] = i_alpha
        i_beta_a[index] = i_beta
        i_active_ref_a[index] = i_active_reference
        i_reactive_ref_a[index] = i_reactive_reference
        i_active_a[index] = i_active
        i_reactive_a[index] = i_reactive
        current_reference_magnitude_a[index] = math.hypot(i_alpha_ref, i_beta_ref)
        current_magnitude_a[index] = math.hypot(i_alpha, i_beta)
        current_limit_active[index] = limited
        p_inst_w[index] = p_instantaneous
        q_inst_var[index] = q_instantaneous
        p_filtered_w[index] = p_filtered
        q_filtered_var[index] = q_filtered

    arrays_to_check = (
        voltage_magnitude_v,
        i_alpha_ref_a,
        i_beta_ref_a,
        i_alpha_a,
        i_beta_a,
        i_active_ref_a,
        i_reactive_ref_a,
        i_active_a,
        i_reactive_a,
        current_magnitude_a,
        current_reference_magnitude_a,
        p_inst_w,
        q_inst_var,
        p_filtered_w,
        q_filtered_var,
        omega_rad_s,
        frequency_hz,
        theta_rad,
    )
    if not all(np.all(np.isfinite(values)) for values in arrays_to_check):
        raise FloatingPointError("simulation produced non-finite values")

    rocof_results = calculate_rocof_windows(
        time_s,
        frequency_hz,
        fault_start_s=fault_start_s,
        fault_clear_s=fault_clear_s,
        pre_event_start_s=pre_start_s,
        post_event_end_s=post_end_s,
    )

    frequency_settling_s = _settling_time(
        time_s=time_s,
        frequency_hz=frequency_hz,
        p_inst_w=p_inst_w,
        q_inst_var=q_inst_var,
        clear_index=clear_index,
        recovery_end_index=post_end_index,
        dt_s=dt_s,
        p_ref_w=p_ref_w,
        q_ref_var=q_ref_var,
        hold_s=settling_hold_s,
        frequency_tolerance_hz=frequency_settling_tolerance_hz,
        power_tolerance_fraction=power_settling_tolerance_fraction,
    )

    fault_frequency_hz = frequency_hz[fault_mask]
    fault_current_a = current_magnitude_a[fault_mask]
    fault_p_w = p_inst_w[fault_mask]
    fault_q_var = q_inst_var[fault_mask]
    result: dict[str, Any] = {
        "time_s": time_s,
        "dt_s": dt_s,
        "simulation_end_s": simulation_end_s,
        "fault_start_s": fault_start_s,
        "fault_clear_s": fault_clear_s,
        "fault_duration_s": fault_clear_s - fault_start_s,
        "sag_depth_pu": sag_depth_pu,
        "nominal_frequency_hz": NOMINAL_FREQUENCY_HZ,
        "nominal_voltage_peak_v": nominal_voltage_peak_v,
        "current_limit_peak_a": current_limit_peak_a,
        "p_ref_w": p_ref_w,
        "q_ref_var": q_ref_var,
        "power_cutoff_hz": power_cutoff_hz,
        "inertia_kgm2": inertia_kgm2,
        "damping_nms_per_rad": damping_nms_per_rad,
        "frequency_settling_tolerance_hz": frequency_settling_tolerance_hz,
        "power_settling_tolerance_fraction": power_settling_tolerance_fraction,
        "settling_hold_s": settling_hold_s,
        "fault_window": (fault_start_s, fault_clear_s),
        "pre_event_window": (pre_start_s, fault_start_s),
        "post_event_window": (fault_clear_s, post_end_s),
        "pre_event_mask": pre_event_mask,
        "fault_mask": fault_mask,
        "post_event_mask": post_event_mask,
        "fault_window_samples": int(np.count_nonzero(fault_mask)),
        "fault_applied_samples": int(
            np.count_nonzero(fault_mask & (voltage_magnitude_v < nominal_voltage_peak_v))
        ),
        "fault_applied": bool(
            np.any(fault_mask)
            and np.any(voltage_magnitude_v[fault_mask] < nominal_voltage_peak_v)
        ),
        "voltage_magnitude_v": voltage_magnitude_v,
        "voltage_magnitude_pu": voltage_magnitude_v / nominal_voltage_peak_v,
        "v_alpha_v": v_alpha_v,
        "v_beta_v": v_beta_v,
        "i_alpha_reference_a": i_alpha_ref_a,
        "i_beta_reference_a": i_beta_ref_a,
        "i_alpha_a": i_alpha_a,
        "i_beta_a": i_beta_a,
        "i_active_reference_a": i_active_ref_a,
        "i_reactive_reference_a": i_reactive_ref_a,
        "i_active_a": i_active_a,
        "i_reactive_a": i_reactive_a,
        "current_reference_magnitude_a": current_reference_magnitude_a,
        "current_magnitude_a": current_magnitude_a,
        "current_limit_active": current_limit_active,
        "control_mode": np.where(
            voltage_magnitude_v == 0.0,
            "ZERO_VOLTAGE_NO_POWER_TRANSFER",
            np.where(current_limit_active, "CURRENT_LIMITED", "P_Q_REFERENCE_TRACKING"),
        ),
        "p_inst_w": p_inst_w,
        "q_inst_var": q_inst_var,
        "p_filtered_w": p_filtered_w,
        "q_filtered_var": q_filtered_var,
        "omega_rad_s": omega_rad_s,
        "frequency_hz": frequency_hz,
        "theta_rad": theta_rad,
        **rocof_results,
        "fault_min_frequency_hz": float(np.min(fault_frequency_hz)),
        "fault_max_frequency_hz": float(np.max(fault_frequency_hz)),
        "fault_max_frequency_deviation_hz": float(
            np.max(np.abs(fault_frequency_hz - NOMINAL_FREQUENCY_HZ))
        ),
        "fault_max_current_a": float(np.max(fault_current_a)),
        "fault_mean_active_power_w": float(np.mean(fault_p_w)),
        "fault_mean_reactive_power_var": float(np.mean(fault_q_var)),
        "pre_event_mean_active_power_w": float(np.mean(p_inst_w[pre_event_mask])),
        "pre_event_mean_reactive_power_var": float(np.mean(q_inst_var[pre_event_mask])),
        "post_event_mean_active_power_w": float(np.mean(p_inst_w[post_event_mask])),
        "post_event_mean_reactive_power_var": float(np.mean(q_inst_var[post_event_mask])),
        "current_limit_active_samples": int(
            np.count_nonzero(current_limit_active[fault_mask])
        ),
        "recovery_frequency_final_hz": float(frequency_hz[-1]),
        "recovery_active_power_final_w": float(p_inst_w[-1]),
        "recovery_reactive_power_final_var": float(q_inst_var[-1]),
        "recovery_settling_time_s": frequency_settling_s,
        # Compatibility names from the original benchmark; now deliberately
        # scoped to the active fault interval instead of startup/global extrema.
        "max_rocof": rocof_results["fault_max_rocof_hz_per_s"],
        "min_freq": float(np.min(fault_frequency_hz)),
        "max_current": float(np.max(fault_current_a)),
        "f_final": float(frequency_hz[-1]),
    }
    return result


def validate_frt_benchmark_results(
    results: dict[str, Any], *, require_current_limit: bool = True
) -> None:
    """Fail closed unless the requested disturbance and its output are present.

    These are software/model-consistency checks for the declared scenario, not
    grid-code acceptance thresholds.  The independent power residual uses the
    documented 3/2 alpha-beta equations and validates every time sample.
    """
    sample_array_keys = (
        "voltage_magnitude_pu",
        "v_alpha_v",
        "v_beta_v",
        "i_alpha_reference_a",
        "i_beta_reference_a",
        "i_alpha_a",
        "i_beta_a",
        "i_active_reference_a",
        "i_reactive_reference_a",
        "i_active_a",
        "i_reactive_a",
        "current_reference_magnitude_a",
        "current_magnitude_a",
        "current_limit_active",
        "p_inst_w",
        "q_inst_var",
        "p_filtered_w",
        "q_filtered_var",
        "omega_rad_s",
        "frequency_hz",
        "theta_rad",
    )
    mask_keys = ("pre_event_mask", "fault_mask", "post_event_mask")
    required_keys = ("time_s", "rocof_hz_per_s", *sample_array_keys, *mask_keys)
    missing = [key for key in required_keys if key not in results]
    if missing:
        raise BenchmarkAcceptanceError(f"missing required result arrays: {', '.join(missing)}")

    time_s = np.asarray(results["time_s"], dtype=float)
    if time_s.ndim != 1 or time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise BenchmarkAcceptanceError("invalid simulation timestamps")
    if np.any(np.diff(time_s) <= 0.0):
        raise BenchmarkAcceptanceError("simulation timestamps are not strictly increasing")

    arrays = {key: np.asarray(results[key]) for key in sample_array_keys}
    for key, values in arrays.items():
        if values.shape != time_s.shape:
            raise BenchmarkAcceptanceError(f"{key} has the wrong sample count")
        if not np.all(np.isfinite(values.astype(float))):
            raise BenchmarkAcceptanceError(f"{key} contains non-finite values")
    masks = {key: np.asarray(results[key], dtype=bool) for key in mask_keys}
    for key, mask in masks.items():
        if mask.shape != time_s.shape:
            raise BenchmarkAcceptanceError(f"{key} has the wrong sample count")

    fault_mask = masks["fault_mask"]
    if not np.any(fault_mask):
        raise BenchmarkAcceptanceError("fault window is missing or has invalid timestamps")
    if not bool(results.get("fault_applied", False)):
        raise BenchmarkAcceptanceError("the requested voltage sag was not applied")

    expected_sample_count = int(results.get("fault_window_samples", 0))
    applied_sample_count = int(results.get("fault_applied_samples", 0))
    if expected_sample_count != int(np.count_nonzero(fault_mask)):
        raise BenchmarkAcceptanceError("fault sample count does not match the measurement mask")
    if applied_sample_count <= 0:
        raise BenchmarkAcceptanceError("no voltage-sag samples were recorded")

    fault_start_s = _finite_float("fault_start_s", results.get("fault_start_s"))
    fault_clear_s = _finite_float("fault_clear_s", results.get("fault_clear_s"))
    pre_start_s = _finite_float("pre_event_window_start_s", results.get("pre_event_window", (None, None))[0])
    post_end_s = _finite_float("post_event_window_end_s", results.get("post_event_window", (None, None))[1])
    expected_fault_mask = (time_s >= fault_start_s - 1e-12) & (time_s < fault_clear_s - 1e-12)
    expected_pre_mask = (time_s >= pre_start_s - 1e-12) & (time_s < fault_start_s - 1e-12)
    expected_post_mask = (time_s >= fault_clear_s - 1e-12) & (time_s <= post_end_s + 1e-12)
    if not np.array_equal(fault_mask, expected_fault_mask):
        raise BenchmarkAcceptanceError("fault mask does not match the declared event interval")
    if not np.array_equal(masks["pre_event_mask"], expected_pre_mask):
        raise BenchmarkAcceptanceError("pre-event mask does not match the declared measurement window")
    if not np.array_equal(masks["post_event_mask"], expected_post_mask):
        raise BenchmarkAcceptanceError("post-event mask does not match the declared measurement window")

    rocof_hz_per_s = np.asarray(results["rocof_hz_per_s"], dtype=float)
    if rocof_hz_per_s.shape != (time_s.size - 1,) or not np.all(np.isfinite(rocof_hz_per_s)):
        raise BenchmarkAcceptanceError("RoCoF samples are missing or non-finite")

    sag_depth_pu = _finite_float("sag_depth_pu", results.get("sag_depth_pu"))
    if not 0.0 <= sag_depth_pu <= 1.0:
        raise BenchmarkAcceptanceError("declared sag depth is outside [0, 1] pu")
    nominal_voltage_peak_v = _finite_float(
        "nominal_voltage_peak_v", results.get("nominal_voltage_peak_v")
    )
    if nominal_voltage_peak_v <= 0.0:
        raise BenchmarkAcceptanceError("nominal voltage must be greater than zero")
    voltage_magnitude_from_components = np.hypot(
        arrays["v_alpha_v"].astype(float), arrays["v_beta_v"].astype(float)
    )
    expected_voltage_pu = voltage_magnitude_from_components / nominal_voltage_peak_v
    if not np.allclose(
        arrays["voltage_magnitude_pu"].astype(float), expected_voltage_pu, rtol=1e-10, atol=1e-12
    ):
        raise BenchmarkAcceptanceError("normalized voltage does not match alpha-beta voltage components")

    expected_fault_voltage_pu = 1.0 - sag_depth_pu
    measured_fault_voltage_pu = arrays["voltage_magnitude_pu"][fault_mask].astype(float)
    if np.any(np.abs(measured_fault_voltage_pu - expected_fault_voltage_pu) > 1e-10):
        raise BenchmarkAcceptanceError("voltage samples do not match the declared sag depth")
    if np.allclose(measured_fault_voltage_pu, 1.0, rtol=0.0, atol=1e-12):
        raise BenchmarkAcceptanceError("the benchmark did not measure a voltage disturbance")
    recorded_sag_samples = int(
        np.count_nonzero(fault_mask & (expected_voltage_pu < 1.0 - 1e-12))
    )
    if applied_sample_count != recorded_sag_samples:
        raise BenchmarkAcceptanceError("recorded fault-application count does not match voltage samples")

    current_limit_peak_a = _finite_float(
        "current_limit_peak_a", results.get("current_limit_peak_a")
    )
    if current_limit_peak_a <= 0.0:
        raise BenchmarkAcceptanceError("current limit must be greater than zero")
    current_magnitude_a = arrays["current_magnitude_a"].astype(float)
    if np.any(current_magnitude_a > current_limit_peak_a + 1e-9):
        raise BenchmarkAcceptanceError("actual current exceeds the declared current limit")
    if require_current_limit and not np.any(
        arrays["current_limit_active"].astype(bool)[fault_mask]
    ):
        raise BenchmarkAcceptanceError("severe-sag benchmark did not exercise current limiting")

    v_alpha = arrays["v_alpha_v"].astype(float)
    v_beta = arrays["v_beta_v"].astype(float)
    i_alpha_ref = arrays["i_alpha_reference_a"].astype(float)
    i_beta_ref = arrays["i_beta_reference_a"].astype(float)
    i_alpha = arrays["i_alpha_a"].astype(float)
    i_beta = arrays["i_beta_a"].astype(float)
    reference_magnitude = np.hypot(i_alpha_ref, i_beta_ref)
    actual_magnitude = np.hypot(i_alpha, i_beta)
    if not np.allclose(
        arrays["current_reference_magnitude_a"].astype(float),
        reference_magnitude,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise BenchmarkAcceptanceError("current-reference magnitude is inconsistent with its components")
    if not np.allclose(
        arrays["current_magnitude_a"].astype(float),
        actual_magnitude,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise BenchmarkAcceptanceError("actual current magnitude is inconsistent with its components")

    expected_scale = np.ones_like(reference_magnitude)
    limited_reference = reference_magnitude > current_limit_peak_a
    expected_scale[limited_reference] = (
        current_limit_peak_a / reference_magnitude[limited_reference]
    )
    if not np.allclose(i_alpha, i_alpha_ref * expected_scale, rtol=1e-10, atol=1e-10):
        raise BenchmarkAcceptanceError("actual alpha current does not follow the reference and limiter")
    if not np.allclose(i_beta, i_beta_ref * expected_scale, rtol=1e-10, atol=1e-10):
        raise BenchmarkAcceptanceError("actual beta current does not follow the reference and limiter")
    if not np.array_equal(arrays["current_limit_active"].astype(bool), limited_reference):
        raise BenchmarkAcceptanceError("current-limit activation flag does not match the requested current")

    p_ref_w = _finite_float("p_ref_w", results.get("p_ref_w"))
    q_ref_var = _finite_float("q_ref_var", results.get("q_ref_var"))
    voltage_magnitude = np.hypot(v_alpha, v_beta)
    expected_i_alpha_ref = np.zeros_like(voltage_magnitude)
    expected_i_beta_ref = np.zeros_like(voltage_magnitude)
    nonzero_voltage = voltage_magnitude > 0.0
    scale = PowerMeasurementLPF.POWER_SCALE * voltage_magnitude[nonzero_voltage]
    v_alpha_unit = v_alpha[nonzero_voltage] / voltage_magnitude[nonzero_voltage]
    v_beta_unit = v_beta[nonzero_voltage] / voltage_magnitude[nonzero_voltage]
    expected_i_alpha_ref[nonzero_voltage] = (
        v_alpha_unit * p_ref_w + v_beta_unit * q_ref_var
    ) / scale
    expected_i_beta_ref[nonzero_voltage] = (
        v_beta_unit * p_ref_w - v_alpha_unit * q_ref_var
    ) / scale
    if not np.allclose(i_alpha_ref, expected_i_alpha_ref, rtol=1e-10, atol=1e-10):
        raise BenchmarkAcceptanceError("alpha current reference is inconsistent with P/Q and voltage")
    if not np.allclose(i_beta_ref, expected_i_beta_ref, rtol=1e-10, atol=1e-10):
        raise BenchmarkAcceptanceError("beta current reference is inconsistent with P/Q and voltage")
    p_recomputed = PowerMeasurementLPF.POWER_SCALE * (
        v_alpha * i_alpha + v_beta * i_beta
    )
    q_recomputed = PowerMeasurementLPF.POWER_SCALE * (
        v_beta * i_alpha - v_alpha * i_beta
    )
    p_measured = arrays["p_inst_w"].astype(float)
    q_measured = arrays["q_inst_var"].astype(float)
    p_tolerance = 1e-8 + 1e-10 * np.maximum(np.abs(p_recomputed), 1.0)
    q_tolerance = 1e-8 + 1e-10 * np.maximum(np.abs(q_recomputed), 1.0)
    if np.any(np.abs(p_measured - p_recomputed) > p_tolerance):
        raise BenchmarkAcceptanceError("active power is inconsistent with simultaneous V and I")
    if np.any(np.abs(q_measured - q_recomputed) > q_tolerance):
        raise BenchmarkAcceptanceError("reactive power is inconsistent with simultaneous V and I")

    expected_i_active = np.zeros_like(voltage_magnitude)
    expected_i_reactive = np.zeros_like(voltage_magnitude)
    expected_i_active_ref = np.zeros_like(voltage_magnitude)
    expected_i_reactive_ref = np.zeros_like(voltage_magnitude)
    expected_i_active[nonzero_voltage] = (
        v_alpha[nonzero_voltage] * i_alpha[nonzero_voltage]
        + v_beta[nonzero_voltage] * i_beta[nonzero_voltage]
    ) / voltage_magnitude[nonzero_voltage]
    expected_i_reactive[nonzero_voltage] = (
        v_beta[nonzero_voltage] * i_alpha[nonzero_voltage]
        - v_alpha[nonzero_voltage] * i_beta[nonzero_voltage]
    ) / voltage_magnitude[nonzero_voltage]
    expected_i_active_ref[nonzero_voltage] = (
        v_alpha[nonzero_voltage] * i_alpha_ref[nonzero_voltage]
        + v_beta[nonzero_voltage] * i_beta_ref[nonzero_voltage]
    ) / voltage_magnitude[nonzero_voltage]
    expected_i_reactive_ref[nonzero_voltage] = (
        v_beta[nonzero_voltage] * i_alpha_ref[nonzero_voltage]
        - v_alpha[nonzero_voltage] * i_beta_ref[nonzero_voltage]
    ) / voltage_magnitude[nonzero_voltage]
    for key, expected in (
        ("i_active_a", expected_i_active),
        ("i_reactive_a", expected_i_reactive),
        ("i_active_reference_a", expected_i_active_ref),
        ("i_reactive_reference_a", expected_i_reactive_ref),
    ):
        if not np.allclose(arrays[key].astype(float), expected, rtol=1e-10, atol=1e-10):
            raise BenchmarkAcceptanceError(f"{key} does not match its voltage-aligned components")

    pre_event_mask = masks["pre_event_mask"]
    if not np.any(pre_event_mask):
        raise BenchmarkAcceptanceError("pre-fault steady-state measurement window is empty")
    p_ref_tolerance = max(1e-6, 1e-10 * abs(p_ref_w))
    q_ref_tolerance = max(1e-6, 1e-10 * abs(q_ref_var))
    if np.any(np.abs(p_measured[pre_event_mask] - p_ref_w) > p_ref_tolerance):
        raise BenchmarkAcceptanceError("pre-fault active power is not at its initialized reference")
    if np.any(np.abs(q_measured[pre_event_mask] - q_ref_var) > q_ref_tolerance):
        raise BenchmarkAcceptanceError("pre-fault reactive power is not at its initialized reference")
    if np.any(np.abs(arrays["frequency_hz"].astype(float)[pre_event_mask] - NOMINAL_FREQUENCY_HZ) > 1e-8):
        raise BenchmarkAcceptanceError("pre-fault VSG frequency is not at its initialized operating point")

    frequency_hz = arrays["frequency_hz"].astype(float)
    recomputed_rocof = np.diff(frequency_hz) / np.diff(time_s)
    if not np.allclose(rocof_hz_per_s, recomputed_rocof, rtol=1e-10, atol=1e-12):
        raise BenchmarkAcceptanceError("RoCoF does not match finite differences of VSG frequency")
    interval_end_s = time_s[1:]
    fault_rocof_mask = (interval_end_s >= fault_start_s) & (interval_end_s < fault_clear_s)
    if not np.any(fault_rocof_mask):
        raise BenchmarkAcceptanceError("fault interval has no RoCoF measurement samples")
    measured_fault_rocof = float(np.max(np.abs(recomputed_rocof[fault_rocof_mask])))
    rocof_window_metrics = calculate_rocof_windows(
        time_s,
        frequency_hz,
        fault_start_s=fault_start_s,
        fault_clear_s=fault_clear_s,
        pre_event_start_s=pre_start_s,
        post_event_end_s=post_end_s,
    )
    for key in (
        "pre_event_max_rocof_hz_per_s",
        "fault_max_rocof_hz_per_s",
        "post_event_max_rocof_hz_per_s",
    ):
        expected_metric = rocof_window_metrics[key]
        observed_metric = results.get(key)
        if expected_metric is None:
            if observed_metric is not None:
                raise BenchmarkAcceptanceError(f"{key} was reported for an empty window")
        elif observed_metric is None or not math.isclose(
            float(observed_metric), expected_metric, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise BenchmarkAcceptanceError(f"{key} does not match its declared time window")
    fault_rocof = results.get("fault_max_rocof_hz_per_s")
    if fault_rocof is None or not math.isfinite(float(fault_rocof)):
        raise BenchmarkAcceptanceError("fault-window RoCoF data are missing or non-finite")
    if not math.isclose(float(fault_rocof), measured_fault_rocof, rel_tol=1e-10, abs_tol=1e-12):
        raise BenchmarkAcceptanceError("reported RoCoF is not scoped to the fault measurement window")

    settling_time = results.get("recovery_settling_time_s")
    if settling_time is None or not math.isfinite(float(settling_time)):
        raise BenchmarkAcceptanceError("frequency and P/Q did not meet the declared recovery window")
    if not 0.0 <= float(settling_time) <= post_end_s - fault_clear_s:
        raise BenchmarkAcceptanceError("reported recovery settling time is outside the post-event window")
    dt_s = _finite_float("dt_s", results.get("dt_s"))
    settling_hold_s = _finite_float("settling_hold_s", results.get("settling_hold_s"))
    frequency_tolerance_hz = _finite_float(
        "frequency_settling_tolerance_hz", results.get("frequency_settling_tolerance_hz")
    )
    power_tolerance_fraction = _finite_float(
        "power_settling_tolerance_fraction", results.get("power_settling_tolerance_fraction")
    )
    clear_index = int(np.searchsorted(time_s, fault_clear_s))
    post_end_index = int(np.searchsorted(time_s, post_end_s))
    recomputed_settling_time = _settling_time(
        time_s=time_s,
        frequency_hz=frequency_hz,
        p_inst_w=p_measured,
        q_inst_var=q_measured,
        clear_index=clear_index,
        recovery_end_index=post_end_index,
        dt_s=dt_s,
        p_ref_w=p_ref_w,
        q_ref_var=q_ref_var,
        hold_s=settling_hold_s,
        frequency_tolerance_hz=frequency_tolerance_hz,
        power_tolerance_fraction=power_tolerance_fraction,
    )
    if recomputed_settling_time is None or not math.isclose(
        float(settling_time), recomputed_settling_time, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise BenchmarkAcceptanceError("recovery settling metric does not match the recorded response")
    final_frequency = _finite_float(
        "recovery_frequency_final_hz", results.get("recovery_frequency_final_hz")
    )
    if not math.isclose(final_frequency, frequency_hz[-1], rel_tol=1e-10, abs_tol=1e-12):
        raise BenchmarkAcceptanceError("reported final frequency does not match the recorded signal")
    if abs(final_frequency - NOMINAL_FREQUENCY_HZ) > frequency_tolerance_hz:
        raise BenchmarkAcceptanceError("final frequency is outside the declared recovery tolerance")
    final_active_power = _finite_float(
        "recovery_active_power_final_w", results.get("recovery_active_power_final_w")
    )
    final_reactive_power = _finite_float(
        "recovery_reactive_power_final_var", results.get("recovery_reactive_power_final_var")
    )
    if abs(final_active_power - p_ref_w) > p_ref_tolerance:
        raise BenchmarkAcceptanceError("final active power did not recover to its reference")
    if abs(final_reactive_power - q_ref_var) > q_ref_tolerance:
        raise BenchmarkAcceptanceError("final reactive power did not recover to its reference")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        results = run_fault_ride_through_benchmark()
        validate_frt_benchmark_results(results, require_current_limit=True)
    except (BenchmarkAcceptanceError, ValueError, FloatingPointError) as exc:
        print(f"FRT benchmark FAILED: {exc}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("REDUCED-ORDER VOLTAGE-SAG / FRT RESPONSE BENCHMARK")
    print("This simulation is not a grid-code compliance or hardware certification test.")
    print("=" * 78)
    print(
        f"Fault window:           [{results['fault_start_s']:.3f}, "
        f"{results['fault_clear_s']:.3f}) s"
    )
    print(f"Voltage during fault:   {1.0 - results['sag_depth_pu']:.3f} pu")
    print(
        f"Mean fault P / Q:       {results['fault_mean_active_power_w']:.1f} W / "
        f"{results['fault_mean_reactive_power_var']:.1f} var"
    )
    print(
        f"Peak fault current:     {results['fault_max_current_a']:.2f} A "
        f"(limit {results['current_limit_peak_a']:.2f} A)"
    )
    print(f"VSG-state max fault RoCoF: {results['max_rocof']:.4f} Hz/s")
    print(
        f"Fault-window frequency: {results['fault_min_frequency_hz']:.4f} to "
        f"{results['fault_max_frequency_hz']:.4f} Hz"
    )
    print(f"Final frequency:        {results['f_final']:.4f} Hz")
    print(
        f"Recovery settling time: {results['recovery_settling_time_s']:.3f} s "
        "after clearing"
    )
    print("Acceptance checks:      PASS (model consistency only)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
