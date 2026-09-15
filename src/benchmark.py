import numpy as np
from src.power_measurement import PowerMeasurementLPF
from src.virtual_inertia import VirtualInertiaEngine
from src.virtual_impedance import VirtualImpedanceController

def run_fault_ride_through_benchmark():
    """
    Simulates a 300 ms symmetric voltage sag (50% drop) to test 
    Grid-Forming Inverter Fault Ride-Through (FRT) compliance.
    """
    p_meter = PowerMeasurementLPF(cutoff_freq_hz=10.0, dt=1e-3)
    vsm = VirtualInertiaEngine(j_kgm2=2.0, dp=15.0, dt=1e-3)
    v_imp = VirtualImpedanceController(r_v_ohm=0.15, l_v_h=3.0e-3)
    
    total_steps = 1000
    time_array = np.linspace(0, 1.0, total_steps)
    
    f_vsm_hist = []
    i_peak_hist = []
    
    for t in time_array:
        # Simulate 50% voltage sag between t = 0.3s and t = 0.6s
        v_grid_mag = 162.63 if (0.3 <= t <= 0.6) else 325.27
        i_alpha = 60.0 if (0.3 <= t <= 0.6) else 30.0
        i_beta = -20.0 if (0.3 <= t <= 0.6) else -10.0
        
        i_mag = np.sqrt(i_alpha**2 + i_beta**2)
        i_peak_hist.append(i_mag)
        
        p_filt, _ = p_meter.update(v_alpha=v_grid_mag, v_beta=0.0, i_alpha=i_alpha, i_beta=i_beta)
        w_vsm, f_vsm, _ = vsm.step(p_ref=0.0, p_meas=p_filt)
        v_imp.apply_virtual_impedance(v_grid_mag, 0.0, i_alpha, i_beta, w_inst=w_vsm)
        
        f_vsm_hist.append(f_vsm)
        
    # Calculate maximum Rate of Change of Frequency (RoCoF in Hz/s)
    df_dt = np.abs(np.diff(f_vsm_hist) / 1e-3)
    max_rocof = np.max(df_dt)
    min_freq = min(f_vsm_hist)
    max_current = max(i_peak_hist)
    
    return {
        "max_rocof": max_rocof,
        "min_freq": min_freq,
        "max_current": max_current,
        "f_final": f_vsm_hist[-1]
    }

if __name__ == "__main__":
    results = run_fault_ride_through_benchmark()
    print("=" * 75)
    print("DAY 20: FAULT RIDE-THROUGH (FRT) COMPLIANCE BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Maximum RoCoF:          {results['max_rocof']:.3f} Hz/s")
    print(f"Frequency Nadir:        {results['min_freq']:.3f} Hz")
    print(f"Peak Fault Current:     {results['max_current']:.2f} A")
    print(f"Final Frequency:        {results['f_final']:.3f} Hz")
    print("=" * 75)
