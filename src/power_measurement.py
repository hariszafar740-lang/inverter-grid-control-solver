import numpy as np

class PowerMeasurementLPF:
    """
    Computes instantaneous active (P) and reactive (Q) power in the alpha-beta 
    stationary reference frame and applies a 1st-order low-pass filter (LPF) 
    to emulate inverter voltage/current transducer bandwidth limitations.
    """
    def __init__(self, cutoff_freq_hz=10.0, dt=1e-3):
        self.wc = 2.0 * np.pi * cutoff_freq_hz  # Cutoff frequency in rad/s
        self.dt = dt
        self.p_filtered = 0.0
        self.q_filtered = 0.0

    def compute_instantaneous_power(self, v_alpha, v_beta, i_alpha, i_beta):
        """Calculates instantaneous p and q using Clarke-transformed components."""
        p_inst = v_alpha * i_alpha + v_beta * i_beta
        q_inst = v_beta * i_alpha - v_alpha * i_beta
        return p_inst, q_inst

    def update(self, v_alpha, v_beta, i_alpha, i_beta):
        """Updates discrete 1st-order Euler filtered power estimates."""
        p_inst, q_inst = self.compute_instantaneous_power(v_alpha, v_beta, i_alpha, i_beta)
        self.p_filtered += self.wc * (p_inst - self.p_filtered) * self.dt
        self.q_filtered += self.wc * (q_inst - self.q_filtered) * self.dt
        return self.p_filtered, self.q_filtered
