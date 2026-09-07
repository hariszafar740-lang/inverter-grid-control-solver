import numpy as np

class InverterDroopController:
    """
    Grid-Forming Inverter P-w and Q-V Droop Controller Engine.
    Adjusts angular frequency (w) and voltage magnitude (V) based on 
    measured active (P) and reactive (Q) power deviations.
    """
    def __init__(self, w0=2*np.pi*50.0, v0=400.0, mp=1e-4, nq=1e-3, p_ref=0.0, q_ref=0.0):
        self.w0 = w0        # Nominal angular frequency (rad/s) for 50 Hz
        self.v0 = v0        # Nominal line-to-line RMS voltage (V)
        self.mp = mp        # Active power droop gain (rad/(s*W))
        self.nq = nq        # Reactive power droop gain (V/VAr)
        self.p_ref = p_ref  # Target active power setpoint (W)
        self.q_ref = q_ref  # Target reactive power setpoint (VAr)

    def compute_setpoints(self, p_meas, q_meas):
        """Calculates grid target frequency (Hz) and voltage amplitude (V)."""
        w_target = self.w0 - self.mp * (p_meas - self.p_ref)
        v_target = self.v0 - self.nq * (q_meas - self.q_ref)
        f_target = w_target / (2.0 * np.pi)
        return w_target, f_target, v_target
