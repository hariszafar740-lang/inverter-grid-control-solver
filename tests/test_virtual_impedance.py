import numpy as np
import pytest

from src.virtual_impedance import VirtualImpedanceController


def test_zero_current_no_voltage_drop():
    v_imp = VirtualImpedanceController(r_v_ohm=0.2, l_v_h=3e-3)
    v_a_mod, v_b_mod = v_imp.apply_virtual_impedance(v_alpha_ref=325.27, v_beta_ref=0.0, i_alpha=0.0, i_beta=0.0)
    assert pytest.approx(v_a_mod, abs=1e-3) == 325.27
    assert pytest.approx(v_b_mod, abs=1e-3) == 0.0

def test_inductive_virtual_voltage_drop():
    v_imp = VirtualImpedanceController(r_v_ohm=0.0, l_v_h=2e-3, w0=2*np.pi*50.0)
    # Pure reactive current lag (i_beta = -50 A)
    v_a_mod, _ = v_imp.apply_virtual_impedance(v_alpha_ref=325.27, v_beta_ref=0.0, i_alpha=0.0, i_beta=-50.0)
    # Expected drop: - (0 - (2*pi*50 * 0.002) * -50) = -31.415 V
    assert v_a_mod < 325.27
