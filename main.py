import numpy as np
from src.power_measurement import PowerMeasurementLPF
from src.virtual_inertia import VirtualInertiaEngine

def main():
    p_meter = PowerMeasurementLPF(cutoff_freq_hz=10.0, dt=1e-3)
    vsm = VirtualInertiaEngine(j_kgm2=2.0, dp=15.0, dt=1e-3)
    
    total_steps = 1000
    time_array = np.linspace(0, 1.0, total_steps)
    f_vsm_hist = []
    
    for t in time_array:
        i_load = 0.0 if t < 0.2 else 40.0
        p_filt, _ = p_meter.update(v_alpha=400.0, v_beta=0.0, i_alpha=i_load, i_beta=0.0)
        
        _, f_vsm, _ = vsm.step(p_ref=0.0, p_meas=p_filt)
        f_vsm_hist.append(f_vsm)
        
    print("=" * 75)
    print("DAY 17: GRID-FORMING INVERTER VIRTUAL INERTIA (VSG) DYNAMIC EVALUATION")
    print("=" * 75)
    print(f"Pre-Event Frequency  (t = 0.1s): {f_vsm_hist[100]:.3f} Hz")
    print(f"Transient Frequency  (t = 0.3s): {f_vsm_hist[300]:.3f} Hz")
    print(f"Post-Event Frequency (t = 1.0s): {f_vsm_hist[-1]:.3f} Hz")
    print(f"Maximum Frequency Deviation (Nadir): {50.0 - min(f_vsm_hist):.3f} Hz")
    print("=" * 75)

if __name__ == "__main__":
    main()
