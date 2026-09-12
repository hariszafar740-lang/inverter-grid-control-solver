import numpy as np
from src.power_measurement import PowerMeasurementLPF
from src.virtual_inertia import VirtualInertiaEngine
from src.virtual_impedance import VirtualImpedanceController

def main():
    p_meter = PowerMeasurementLPF(cutoff_freq_hz=10.0, dt=1e-3)
    vsm = VirtualInertiaEngine(j_kgm2=2.0, dp=15.0, dt=1e-3)
    v_imp = VirtualImpedanceController(r_v_ohm=0.1, l_v_h=2.5e-3)
    
    total_steps = 1000
    time_array = np.linspace(0, 1.0, total_steps)
    v_ref_mod_mag = []
    
    for t in time_array:
        i_alpha = 0.0 if t < 0.3 else 40.0
        i_beta = 0.0 if t < 0.3 else -20.0
        
        p_filt, _ = p_meter.update(v_alpha=325.27, v_beta=0.0, i_alpha=i_alpha, i_beta=i_beta)
        w_vsm, f_vsm, _ = vsm.step(p_ref=0.0, p_meas=p_filt)
        
        v_a_mod, v_b_mod = v_imp.apply_virtual_impedance(325.27, 0.0, i_alpha, i_beta, w_inst=w_vsm)
        v_ref_mod_mag.append(np.sqrt(v_a_mod**2 + v_b_mod**2))
        
    print("=" * 75)
    print("DAY 18: GRID-FORMING INVERTER VIRTUAL IMPEDANCE CONTROL EVALUATION")
    print("=" * 75)
    print(f"No-Load Peak Reference Voltage:  {v_ref_mod_mag[0]:.2f} V")
    print(f"Loaded Peak Reference Voltage:   {v_ref_mod_mag[-1]:.2f} V")
    print(f"Virtual Impedance Drop Magnitude: {v_ref_mod_mag[0] - v_ref_mod_mag[-1]:.2f} V")
    print("=" * 75)

if __name__ == "__main__":
    main()
