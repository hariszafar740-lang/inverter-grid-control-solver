import math

import numpy as np
import pytest

from src.current_control import apply_current_limit, current_reference_from_power
from src.power_measurement import PowerMeasurementLPF


def test_pq_references_reproduce_requested_three_phase_power():
    voltage_magnitude_v = 325.27
    voltage_angle_rad = 0.47
    v_alpha = voltage_magnitude_v * math.cos(voltage_angle_rad)
    v_beta = voltage_magnitude_v * math.sin(voltage_angle_rad)
    p_ref_w = 10_000.0
    q_ref_var = 2_500.0

    i_alpha, i_beta = current_reference_from_power(
        v_alpha, v_beta, p_ref_w, q_ref_var
    )
    p_actual, q_actual = PowerMeasurementLPF().compute_instantaneous_power(
        v_alpha, v_beta, i_alpha, i_beta
    )
    assert p_actual == pytest.approx(p_ref_w, rel=1e-12)
    assert q_actual == pytest.approx(q_ref_var, rel=1e-12)


def test_zero_voltage_uses_documented_zero_power_transfer_convention():
    assert current_reference_from_power(0.0, 0.0, 10_000.0, 2_000.0) == (0.0, 0.0)


def test_radial_limiter_caps_magnitude_and_preserves_vector_angle():
    i_alpha, i_beta, limited = apply_current_limit(40.0, -30.0, 25.0)
    assert limited
    assert math.hypot(i_alpha, i_beta) == pytest.approx(25.0, abs=1e-12)
    assert i_beta / i_alpha == pytest.approx(-30.0 / 40.0)


def test_current_below_limit_is_unchanged():
    result = apply_current_limit(3.0, -4.0, 5.0)
    assert result == (3.0, -4.0, False)


@pytest.mark.parametrize(
    "inputs",
    [
        (np.nan, 0.0, 1_000.0, 0.0),
        (1.0, np.inf, 1_000.0, 0.0),
    ],
)
def test_current_reference_rejects_nonfinite_inputs(inputs):
    with pytest.raises(ValueError, match="finite"):
        current_reference_from_power(*inputs)


def test_current_limiter_rejects_invalid_limits_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="greater than zero"):
        apply_current_limit(1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        apply_current_limit(np.inf, 0.0, 10.0)
