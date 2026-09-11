import numpy as np
from src.droop_control import InverterDroopController
from src.power_measurement import PowerMeasurementLPF

def main():
    controller = InverterDroopController(w0=2*np.pi*50.0, v0=400.0, mp=1e-4, nq=1e-3)
    p_meter = PowerMeasurementLPF(cutoff_freq_hz=10.0, dt=1e-3)
    
    total_steps = 1000
    time_array = np.linspace(0, 1.0, total_steps)
    
    p_filt_hist = []
    f_target_hist = []
    
    for t in time_array:
        i_load = 0.0 if t < 0.5 else 50.0
        v_alpha, v_beta = 400.0, 0.0
        i_alpha, i_beta = i_load, 0.0
        
        p_filt, q_filt = p_meter.update(v_alpha, v_beta, i_alpha, i_beta)
        _, f_target, _ = controller.compute_setpoints(p_filt, q_filt)
        
        p_filt_hist.append(p_filt)
        f_target_hist.append(f_target)
        
    print("=" * 70)
    print("DAY 16: INVERTER POWER MEASUREMENT LPF & TRANSIENT DROOP EVALUATION")
    print("=" * 70)
    print(f"Pre-Step Frequency (t = 0.2s):  {f_target_hist[200]:.3f} Hz")
    print(f"Post-Step Frequency (t = 1.0s): {f_target_hist[-1]:.3f} Hz")
    print(f"Filtered Active Power Measured: {p_filt_hist[-1]:.2f} W")
    print("=" * 70)

if __name__ == "__main__":
    main()
