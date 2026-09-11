import numpy as np

class VirtualInertiaEngine:
    """
    Virtual Synchronous Generator (VSG) Swing Equation Module.
    Emulates physical rotor mass inertia (J) and damping (Dp) to provide 
    synthetic inertia frequency response during grid power disturbances.
    """
    def __init__(self, j_kgm2=5.0, dp=20.0, w0=2*np.pi*50.0, dt=1e-3):
        self.j = j_kgm2          # Virtual moment of inertia (kg*m^2)
        self.dp = dp            # Damping factor (N*m*s/rad)
        self.w0 = w0            # Nominal grid angular frequency (rad/s)
        self.dt = dt            # Simulation time step (s)
        self.w_vsm = w0         # Current VSM angular frequency (rad/s)
        self.theta_vsm = 0.0    # Virtual rotor angle (rad)

    def step(self, p_ref, p_meas):
        """
        Calculates acceleration torque and updates angular frequency & rotor angle:
        d(w_vsm)/dt = [(P_ref - P_meas) / w0 - Dp * (w_vsm - w0)] / J
        """
        t_mech = p_ref / self.w0
        t_elec = p_meas / self.w0
        t_damp = self.dp * (self.w_vsm - self.w0)
        
        dw_dt = (t_mech - t_elec - t_damp) / self.j
        self.w_vsm += dw_dt * self.dt
        self.theta_vsm += self.w_vsm * self.dt
        
        # Keep theta bounded within [0, 2*pi)
        self.theta_vsm = np.fmod(self.theta_vsm, 2.0 * np.pi)
        f_vsm = self.w_vsm / (2.0 * np.pi)
        
        return self.w_vsm, f_vsm, self.theta_vsm
