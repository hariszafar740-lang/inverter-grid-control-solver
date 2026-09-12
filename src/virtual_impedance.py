import numpy as np

class VirtualImpedanceController:
    """
    Virtual Impedance Loop for Grid-Forming Inverters.
    Modifies reference voltage setpoints in stationary alpha-beta frame
    to enforce predominantly inductive output impedance (X_v >> R_v),
    decoupling P-Q power flow and shaping inverter dynamic response.
    """
    def __init__(self, r_v_ohm=0.1, l_v_h=2e-3, w0=2*np.pi*50.0):
        self.r_v = r_v_ohm   # Virtual resistance (Ohms)
        self.l_v = l_v_h     # Virtual inductance (Henries)
        self.w0 = w0         # Nominal angular frequency (rad/s)

    def apply_virtual_impedance(self, v_alpha_ref, v_beta_ref, i_alpha, i_beta, w_inst=None):
        """
        Calculates output voltage reference reduction due to virtual impedance drop:
        v_alpha_ref* = v_alpha_ref - (R_v * i_alpha - w * L_v * i_beta)
        v_beta_ref*  = v_beta_ref  - (R_v * i_beta  + w * L_v * i_alpha)
        """
        w = self.w0 if w_inst is None else w_inst
        x_v = w * self.l_v   # Virtual reactance (Ohms)
        
        v_drop_alpha = self.r_v * i_alpha - x_v * i_beta
        v_drop_beta  = self.r_v * i_beta  + x_v * i_alpha
        
        v_alpha_mod = v_alpha_ref - v_drop_alpha
        v_beta_mod  = v_beta_ref  - v_drop_beta
        
        return v_alpha_mod, v_beta_mod
