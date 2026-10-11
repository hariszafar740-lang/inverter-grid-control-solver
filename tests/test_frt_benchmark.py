import math

import numpy as np
import pytest

from src import benchmark
from src.benchmark import (
    BenchmarkAcceptanceError,
    calculate_rocof_windows,
    run_fault_ride_through_benchmark,
    validate_frt_benchmark_results,
)


def test_default_scenario_passes_fail_closed_model_consistency_checks():
    result = run_fault_ride_through_benchmark()
    assert validate_frt_benchmark_results(result) is None


def test_prefault_operating_point_is_initialized_and_stable():
    result = run_fault_ride_through_benchmark()
    pre = result["pre_event_mask"]

    assert np.any(pre)
    assert result["pre_event_mean_active_power_w"] == pytest.approx(10_000.0, abs=1e-8)
    assert result["pre_event_mean_reactive_power_var"] == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(result["frequency_hz"][pre], 50.0, rtol=0.0, atol=1e-10)
    assert result["pre_event_max_rocof_hz_per_s"] == pytest.approx(0.0, abs=1e-9)


def test_fault_sag_uses_exact_declared_half_open_window():
    result = run_fault_ride_through_benchmark()
    time_s = result["time_s"]
    fault = result["fault_mask"]
    clear_index = int(round(result["fault_clear_s"] / result["dt_s"]))

    assert result["fault_window"] == pytest.approx((0.3, 0.6))
    assert result["fault_window_samples"] == 300
    assert result["fault_applied_samples"] == 300
    assert time_s[fault][0] == pytest.approx(0.3, abs=1e-12)
    assert time_s[fault][-1] == pytest.approx(0.599, abs=1e-12)
    assert np.allclose(result["voltage_magnitude_pu"][fault], 0.5, atol=1e-12)
    assert result["voltage_magnitude_pu"][clear_index] == pytest.approx(1.0)
    assert not result["fault_mask"][clear_index]


def test_power_is_recomputed_from_simultaneous_actual_voltage_and_current():
    result = run_fault_ride_through_benchmark()
    p_independent = 1.5 * (
        result["v_alpha_v"] * result["i_alpha_a"]
        + result["v_beta_v"] * result["i_beta_a"]
    )
    q_independent = 1.5 * (
        result["v_beta_v"] * result["i_alpha_a"]
        - result["v_alpha_v"] * result["i_beta_a"]
    )

    assert np.allclose(result["p_inst_w"], p_independent, rtol=1e-12, atol=1e-8)
    assert np.allclose(result["q_inst_var"], q_independent, rtol=1e-12, atol=1e-8)
    assert result["fault_mean_active_power_w"] < result["p_ref_w"]
    assert result["fault_mean_active_power_w"] == pytest.approx(
        result["nominal_voltage_peak_v"] * 0.5 * 30.0 * 1.5,
        rel=1e-12,
    )


def test_lpf_and_vsg_states_follow_the_documented_discrete_equations():
    result = run_fault_ride_through_benchmark()
    time_s = result["time_s"]
    dt_s = result["dt_s"]
    alpha = 2.0 * math.pi * result["power_cutoff_hz"] * dt_s

    p_filtered_reconstructed = np.empty_like(result["p_filtered_w"])
    q_filtered_reconstructed = np.empty_like(result["q_filtered_var"])
    p_filtered_reconstructed[0] = result["p_inst_w"][0]
    q_filtered_reconstructed[0] = result["q_inst_var"][0]
    for index in range(1, time_s.size):
        p_filtered_reconstructed[index] = p_filtered_reconstructed[index - 1] + alpha * (
            result["p_inst_w"][index] - p_filtered_reconstructed[index - 1]
        )
        q_filtered_reconstructed[index] = q_filtered_reconstructed[index - 1] + alpha * (
            result["q_inst_var"][index] - q_filtered_reconstructed[index - 1]
        )
    assert np.allclose(result["p_filtered_w"], p_filtered_reconstructed, rtol=1e-12, atol=1e-9)
    assert np.allclose(result["q_filtered_var"], q_filtered_reconstructed, rtol=1e-12, atol=1e-9)

    w0 = 2.0 * math.pi * result["nominal_frequency_hz"]
    omega_reconstructed = np.empty_like(result["omega_rad_s"])
    omega_reconstructed[0] = w0
    for index in range(1, time_s.size):
        torque_error = (result["p_ref_w"] - result["p_filtered_w"][index]) / w0
        damping_torque = result["damping_nms_per_rad"] * (
            omega_reconstructed[index - 1] - w0
        )
        omega_reconstructed[index] = omega_reconstructed[index - 1] + (
            (torque_error - damping_torque) / result["inertia_kgm2"]
        ) * dt_s
    assert np.allclose(result["omega_rad_s"], omega_reconstructed, rtol=1e-12, atol=1e-10)
    assert np.allclose(result["frequency_hz"], omega_reconstructed / (2.0 * math.pi))


def test_severe_sag_exercises_limit_without_exceeding_it():
    result = run_fault_ride_through_benchmark()
    fault = result["fault_mask"]

    assert result["current_limit_active_samples"] == result["fault_window_samples"]
    assert np.all(result["current_reference_magnitude_a"][fault] > 30.0)
    assert np.max(result["current_magnitude_a"]) <= 30.0 + 1e-12
    assert np.allclose(result["current_magnitude_a"][fault], 30.0, atol=1e-12)
    assert set(result["control_mode"][fault]) == {"CURRENT_LIMITED"}


def test_fault_severity_changes_power_consistently_with_available_current():
    shallow = run_fault_ride_through_benchmark(sag_depth_pu=0.2, q_ref_var=2_000.0)
    medium = run_fault_ride_through_benchmark(sag_depth_pu=0.5, q_ref_var=2_000.0)
    deep = run_fault_ride_through_benchmark(sag_depth_pu=0.8, q_ref_var=2_000.0)

    assert shallow["current_limit_active_samples"] == 0
    assert medium["current_limit_active_samples"] > 0
    assert deep["current_limit_active_samples"] > 0
    assert shallow["fault_mean_active_power_w"] == pytest.approx(10_000.0, rel=1e-12)
    assert (
        shallow["fault_mean_active_power_w"]
        > medium["fault_mean_active_power_w"]
        > deep["fault_mean_active_power_w"]
    )
    for result in (shallow, medium, deep):
        assert np.max(result["current_magnitude_a"]) <= result["current_limit_peak_a"] + 1e-12


def test_longer_fault_produces_a_larger_model_frequency_deviation():
    short = run_fault_ride_through_benchmark(fault_duration_s=0.1)
    long = run_fault_ride_through_benchmark(fault_duration_s=0.3)
    short_deviation = np.max(np.abs(short["frequency_hz"][short["fault_mask"]] - 50.0))
    long_deviation = np.max(np.abs(long["frequency_hz"][long["fault_mask"]] - 50.0))
    assert long_deviation > short_deviation > 0.0


def test_zero_voltage_fault_uses_declared_blocking_convention_and_recovers():
    result = run_fault_ride_through_benchmark(sag_depth_pu=1.0)
    fault = result["fault_mask"]

    assert np.all(result["voltage_magnitude_pu"][fault] == 0.0)
    assert np.all(result["current_magnitude_a"][fault] == 0.0)
    assert np.all(result["p_inst_w"][fault] == 0.0)
    assert np.all(result["q_inst_var"][fault] == 0.0)
    assert validate_frt_benchmark_results(result, require_current_limit=False) is None


def test_fault_clearing_restores_voltage_and_power_and_recovery_is_measured():
    result = run_fault_ride_through_benchmark()
    clear_index = int(round(result["fault_clear_s"] / result["dt_s"]))

    assert result["voltage_magnitude_pu"][clear_index] == pytest.approx(1.0)
    assert result["p_inst_w"][clear_index] == pytest.approx(result["p_ref_w"], abs=1e-8)
    assert result["recovery_active_power_final_w"] == pytest.approx(10_000.0, abs=1e-8)
    assert result["recovery_frequency_final_hz"] == pytest.approx(50.0, abs=1e-3)
    assert result["recovery_settling_time_s"] is not None


def test_fault_rocof_is_finite_difference_in_event_window_not_startup_global():
    result = run_fault_ride_through_benchmark()
    rates = np.diff(result["frequency_hz"]) / np.diff(result["time_s"])
    right_endpoint = result["time_s"][1:]
    event_intervals = (right_endpoint >= result["fault_start_s"]) & (
        right_endpoint < result["fault_clear_s"]
    )

    assert result["max_rocof"] == pytest.approx(np.max(np.abs(rates[event_intervals])))
    assert result["max_rocof"] > 0.0
    assert result["pre_event_max_rocof_hz_per_s"] == pytest.approx(0.0, abs=1e-9)


def test_windowed_rocof_does_not_mistake_a_pre_event_startup_spike_for_fault_rocof():
    time_s = np.arange(8, dtype=float)
    frequency_hz = np.array([50.0, 48.0, 50.0, 50.0, 51.0, 51.5, 52.0, 52.0])
    metrics = calculate_rocof_windows(
        time_s,
        frequency_hz,
        fault_start_s=3.0,
        fault_clear_s=6.0,
        pre_event_start_s=0.0,
        post_event_end_s=7.0,
    )

    assert metrics["pre_event_max_rocof_hz_per_s"] == pytest.approx(2.0)
    assert metrics["fault_max_rocof_hz_per_s"] == pytest.approx(1.0)
    assert metrics["post_event_max_rocof_hz_per_s"] == pytest.approx(0.5)


def test_benchmark_acceptance_fails_if_voltage_fault_was_not_applied():
    no_fault = run_fault_ride_through_benchmark(sag_depth_pu=0.0)
    with pytest.raises(BenchmarkAcceptanceError, match="not applied"):
        validate_frt_benchmark_results(no_fault)


def test_benchmark_acceptance_fails_if_power_or_rocof_measurements_are_invalid():
    result = run_fault_ride_through_benchmark()
    result["p_inst_w"] = result["p_inst_w"].copy()
    result["p_inst_w"][10] = np.nan
    with pytest.raises(BenchmarkAcceptanceError, match="non-finite"):
        validate_frt_benchmark_results(result)

    result = run_fault_ride_through_benchmark()
    result["fault_max_rocof_hz_per_s"] += 1.0
    with pytest.raises(BenchmarkAcceptanceError, match="time window"):
        validate_frt_benchmark_results(result)


def test_cli_returns_nonzero_when_mandatory_fault_acceptance_fails(monkeypatch, capsys):
    no_fault = run_fault_ride_through_benchmark(sag_depth_pu=0.0)
    monkeypatch.setattr(
        benchmark,
        "run_fault_ride_through_benchmark",
        lambda: no_fault,
    )

    assert benchmark.main([]) == 1
    assert "FRT benchmark FAILED" in capsys.readouterr().err


@pytest.mark.parametrize(
    "parameters",
    [
        {"dt_s": 0.0},
        {"sag_depth_pu": -0.1},
        {"sag_depth_pu": 1.1},
        {"fault_start_s": 0.3005},
        {"fault_duration_s": 0.0},
        {"simulation_end_s": 0.5},
        {"p_ref_w": np.nan},
        {"p_ref_w": 30_000.0},
        {"dt_s": 0.02},
    ],
)
def test_invalid_scenario_parameters_fail_explicitly(parameters):
    with pytest.raises((ValueError, FloatingPointError)):
        run_fault_ride_through_benchmark(**parameters)
