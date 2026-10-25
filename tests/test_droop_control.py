import numpy as np
import pytest

from src.droop_control import InverterDroopController


def test_nominal_droop_setpoints():
    controller = InverterDroopController(w0=2*np.pi*50.0, v0=400.0)
    _, f_out, v_out = controller.compute_setpoints(p_meas=0.0, q_meas=0.0)
    assert pytest.approx(f_out, abs=1e-2) == 50.0
    assert pytest.approx(v_out, abs=1e-2) == 400.0

def test_active_power_frequency_droop():
    controller = InverterDroopController(w0=2*np.pi*50.0, mp=1e-4)
    _, f_droop, _ = controller.compute_setpoints(p_meas=10000.0, q_meas=0.0)
    # Expected: f drops below 50 Hz under load
    assert f_droop < 50.0
