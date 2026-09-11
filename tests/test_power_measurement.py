import pytest
import numpy as np
from src.power_measurement import PowerMeasurementLPF

def test_instantaneous_power_calculation():
    p_meas = PowerMeasurementLPF(cutoff_freq_hz=10.0, dt=1e-3)
    p_inst, q_inst = p_meas.compute_instantaneous_power(v_alpha=400.0, v_beta=0.0, i_alpha=50.0, i_beta=0.0)
    assert pytest.approx(p_inst, abs=1e-1) == 20000.0
    assert pytest.approx(q_inst, abs=1e-1) == 0.0

def test_lpf_transient_settling():
    p_meas = PowerMeasurementLPF(cutoff_freq_hz=10.0, dt=1e-3)
    for _ in range(1000):
        p_filt, _ = p_meas.update(v_alpha=400.0, v_beta=0.0, i_alpha=50.0, i_beta=0.0)
    assert pytest.approx(p_filt, abs=10.0) == 20000.0
