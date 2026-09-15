import pytest
from src.benchmark import run_fault_ride_through_benchmark

def test_grid_code_compliance_limits():
    results = run_fault_ride_through_benchmark()
    
    # ENTSO-E / IEEE 1547 grid compliance assertions
    assert results["max_rocof"] < 2.5, "RoCoF exceeded 2.5 Hz/s compliance threshold"
    assert results["min_freq"] >= 48.5, "Frequency nadir dropped below 48.5 Hz trip limit"
    assert results["f_final"] == pytest.approx(50.0, abs=0.5), "Frequency failed to recover to quasi-steady state"
