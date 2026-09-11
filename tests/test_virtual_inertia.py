import pytest
import numpy as np
from src.virtual_inertia import VirtualInertiaEngine

def test_nominal_vsm_steady_state():
    vsm = VirtualInertiaEngine(w0=2*np.pi*50.0, dt=1e-3)
    w_out, f_out, _ = vsm.step(p_ref=10000.0, p_meas=10000.0)
    assert pytest.approx(f_out, abs=1e-2) == 50.0

def test_inertia_frequency_deceleration_under_load_step():
    vsm = VirtualInertiaEngine(j_kgm2=5.0, dp=20.0, w0=2*np.pi*50.0, dt=1e-3)
    # Apply abrupt load step (P_meas > P_ref)
    for _ in range(100):
        _, f_vsm, _ = vsm.step(p_ref=0.0, p_meas=10000.0)
    # Expected: Virtual rotor decelerates, dropping output frequency below 50 Hz
    assert f_vsm < 50.0
