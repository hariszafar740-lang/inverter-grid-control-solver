"""Discrete-time virtual synchronous generator swing-equation block."""

from __future__ import annotations

import math


class VirtualInertiaEngine:
    """Integrate the reduced-order VSG swing equation.

    The modeled state is angular frequency and virtual rotor angle only:

        d(w)/dt = ((P_ref - P_meas)/w0 - Dp*(w - w0)) / J.

    Frequency is returned as ``w/(2*pi)``.  The update uses explicit Euler for
    frequency and the updated frequency for angle (semi-implicit phase update).
    This block contains no electrical network, current controller, or power
    saturation of its own; limiting is reflected only through measured P.
    """

    def __init__(
        self,
        j_kgm2: float = 5.0,
        dp: float = 20.0,
        w0: float = 2.0 * math.pi * 50.0,
        dt: float = 1e-3,
    ) -> None:
        values = (j_kgm2, dp, w0, dt)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("VSG parameters must be finite")

        self.j = float(j_kgm2)
        self.dp = float(dp)
        self.w0 = float(w0)
        self.dt = float(dt)
        if self.j <= 0.0:
            raise ValueError("j_kgm2 must be greater than zero")
        if self.dp < 0.0:
            raise ValueError("dp must be non-negative")
        if self.w0 <= 0.0:
            raise ValueError("w0 must be greater than zero")
        if self.dt <= 0.0:
            raise ValueError("dt must be greater than zero")
        if self.dp * self.dt / self.j >= 2.0:
            raise ValueError("explicit-Euler damping update is unstable for these parameters")

        self.w_vsm = self.w0
        self.theta_vsm = 0.0

    def step(self, p_ref: float, p_meas: float) -> tuple[float, float, float]:
        """Advance frequency and angle by one sample and return ``(w, f, theta)``."""
        if not (math.isfinite(float(p_ref)) and math.isfinite(float(p_meas))):
            raise ValueError("p_ref and p_meas must be finite")

        p_ref = float(p_ref)
        p_meas = float(p_meas)
        t_mech = p_ref / self.w0
        t_elec = p_meas / self.w0
        t_damp = self.dp * (self.w_vsm - self.w0)
        dw_dt = (t_mech - t_elec - t_damp) / self.j
        w_next = self.w_vsm + dw_dt * self.dt
        theta_unwrapped = self.theta_vsm + w_next * self.dt
        if not (math.isfinite(w_next) and math.isfinite(theta_unwrapped)):
            raise FloatingPointError("non-finite VSG state")

        self.w_vsm = w_next
        self.theta_vsm = theta_unwrapped % (2.0 * math.pi)
        f_vsm = self.w_vsm / (2.0 * math.pi)
        return self.w_vsm, f_vsm, self.theta_vsm
