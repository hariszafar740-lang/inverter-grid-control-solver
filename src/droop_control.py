"""Grid-Forming Inverter P-omega and Q-V Droop Controller Engine."""

from __future__ import annotations

import math


class InverterDroopController:
    """Grid-forming outer P-omega and Q-V droop law with input validation.

    Governing steady-state droop equations:
        omega* = omega_0 - m_p * (P_meas - P_ref)
        V*     = V_0     - n_q * (Q_meas - Q_ref)
    """

    def __init__(
        self,
        w0: float = 2.0 * math.pi * 50.0,
        v0: float = 400.0,
        mp: float = 1e-4,
        nq: float = 1e-3,
        p_ref: float = 0.0,
        q_ref: float = 0.0,
    ) -> None:
        vals = (w0, v0, mp, nq, p_ref, q_ref)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("Droop controller parameters must be finite numbers")
        if float(w0) <= 0.0 or float(v0) <= 0.0:
            raise ValueError("Nominal frequency w0 and nominal voltage v0 must be positive")
        if float(mp) < 0.0 or float(nq) < 0.0:
            raise ValueError("Droop slopes mp and nq must be non-negative")

        self.w0 = float(w0)
        self.v0 = float(v0)
        self.mp = float(mp)
        self.nq = float(nq)
        self.p_ref = float(p_ref)
        self.q_ref = float(q_ref)

    def compute_setpoints(
        self,
        p_meas: float,
        q_meas: float,
        *,
        v_min: float = 0.0,
        v_max: float = float("inf"),
    ) -> tuple[float, float, float]:
        """Return ``(w_target_rad_s, f_target_hz, v_target_v)`` for measured ``(P, Q)``."""
        if not (math.isfinite(float(p_meas)) and math.isfinite(float(q_meas))):
            raise ValueError("p_meas and q_meas must be finite")
        if not (math.isfinite(float(v_min)) and (math.isfinite(float(v_max)) or v_max == float("inf"))):
            raise ValueError("voltage clamp bounds must be valid numbers")
        if v_min > v_max:
            raise ValueError("v_min must not exceed v_max")

        p_val = float(p_meas)
        q_val = float(q_meas)
        w_target = self.w0 - self.mp * (p_val - self.p_ref)
        v_raw = self.v0 - self.nq * (q_val - self.q_ref)
        v_target = min(max(v_raw, float(v_min)), float(v_max))
        f_target = w_target / (2.0 * math.pi)
        return w_target, f_target, v_target
