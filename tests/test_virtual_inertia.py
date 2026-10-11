import numpy as np
import pytest

from src.virtual_inertia import VirtualInertiaEngine


def test_nominal_vsm_steady_state():
    vsm = VirtualInertiaEngine(w0=2 * np.pi * 50.0, dt=1e-3)
    _, f_out, _ = vsm.step(p_ref=10_000.0, p_meas=10_000.0)
    assert f_out == pytest.approx(50.0, abs=1e-10)


def test_inertia_frequency_deceleration_under_load_step():
    vsm = VirtualInertiaEngine(j_kgm2=5.0, dp=20.0, w0=2 * np.pi * 50.0, dt=1e-3)
    for _ in range(100):
        _, f_vsm, _ = vsm.step(p_ref=0.0, p_meas=10_000.0)
    assert f_vsm < 50.0


def test_angle_remains_wrapped_and_frequency_is_finite():
    vsm = VirtualInertiaEngine(dt=1e-3)
    for _ in range(10_000):
        omega, frequency, theta = vsm.step(p_ref=1_000.0, p_meas=1_000.0)
    assert np.isfinite(omega)
    assert np.isfinite(frequency)
    assert 0.0 <= theta < 2 * np.pi


def test_vsm_rejects_invalid_or_unstable_parameters_and_inputs():
    with pytest.raises(ValueError, match="j_kgm2"):
        VirtualInertiaEngine(j_kgm2=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        VirtualInertiaEngine(dp=-1.0)
    with pytest.raises(ValueError, match="unstable"):
        VirtualInertiaEngine(j_kgm2=1.0, dp=10.0, dt=0.2)
    with pytest.raises(ValueError, match="finite"):
        VirtualInertiaEngine().step(p_ref=np.nan, p_meas=0.0)
