import numpy as np
import pytest

from src.power_measurement import PowerMeasurementLPF


def test_instantaneous_three_phase_power_uses_simultaneous_alpha_beta_values():
    meter = PowerMeasurementLPF(cutoff_freq_hz=10.0, dt=1e-3)
    p_inst, q_inst = meter.compute_instantaneous_power(
        v_alpha=400.0,
        v_beta=0.0,
        i_alpha=50.0,
        i_beta=-10.0,
    )
    assert p_inst == pytest.approx(30_000.0, abs=1e-10)
    assert q_inst == pytest.approx(6_000.0, abs=1e-10)


def test_lpf_transient_settles_to_measured_three_phase_power():
    meter = PowerMeasurementLPF(cutoff_freq_hz=10.0, dt=1e-3)
    for _ in range(1000):
        p_filtered, q_filtered = meter.update(
            v_alpha=400.0,
            v_beta=0.0,
            i_alpha=50.0,
            i_beta=0.0,
        )
    assert p_filtered == pytest.approx(30_000.0, abs=1e-6)
    assert q_filtered == pytest.approx(0.0, abs=1e-10)


def test_filter_can_start_at_a_declared_steady_operating_point():
    meter = PowerMeasurementLPF(
        cutoff_freq_hz=10.0,
        dt=1e-3,
        initial_p_w=30_000.0,
        initial_q_var=6_000.0,
    )
    p_filtered, q_filtered = meter.update(
        v_alpha=400.0,
        v_beta=0.0,
        i_alpha=50.0,
        i_beta=-10.0,
    )
    assert p_filtered == pytest.approx(30_000.0, abs=1e-10)
    assert q_filtered == pytest.approx(6_000.0, abs=1e-10)


def test_filter_rejects_invalid_or_numerically_unstable_parameters():
    with pytest.raises(ValueError, match="greater than zero"):
        PowerMeasurementLPF(cutoff_freq_hz=0.0)
    with pytest.raises(ValueError, match="greater than zero"):
        PowerMeasurementLPF(dt=0.0)
    with pytest.raises(ValueError, match="non-overshooting"):
        PowerMeasurementLPF(cutoff_freq_hz=200.0, dt=1e-3)
    with pytest.raises(ValueError, match="finite"):
        PowerMeasurementLPF(initial_p_w=np.nan)


def test_filter_rejects_nonfinite_measurement_samples():
    meter = PowerMeasurementLPF()
    with pytest.raises(ValueError, match="must be finite"):
        meter.update(v_alpha=np.inf, v_beta=0.0, i_alpha=1.0, i_beta=0.0)
