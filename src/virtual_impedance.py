"""Stationary alpha-beta Virtual Impedance Loop for Grid-Forming Inverters."""

from __future__ import annotations

import math


class VirtualImpedanceController:
    """Algebraic stationary-frame virtual impedance ``Z_v(w) = R_v + j w L_v``.

    Modifies reference voltage setpoints in amplitude-invariant Clarke ``alpha-beta``
    coordinates to enforce inductive output impedance (``X_v >> R_v``), decoupling
    active/reactive power sharing and shaping fault current transients:

        v_alpha_ref* = v_alpha_ref - (R_v * i_alpha - w * L_v * i_beta)
        v_beta_ref*  = v_beta_ref  - (R_v * i_beta  + w * L_v * i_alpha)
    """

    def __init__(
        self,
        r_v_ohm: float = 0.1,
        l_v_h: float = 2e-3,
        w0: float = 2.0 * math.pi * 50.0,
    ) -> None:
        if not all(math.isfinite(float(v)) for v in (r_v_ohm, l_v_h, w0)):
            raise ValueError("Virtual impedance parameters must be finite")
        if float(r_v_ohm) < 0.0 or float(l_v_h) < 0.0:
            raise ValueError("r_v_ohm and l_v_h must be non-negative")
        if float(w0) <= 0.0:
            raise ValueError("w0 must be positive")

        self.r_v = float(r_v_ohm)
        self.l_v = float(l_v_h)
        self.w0 = float(w0)

    def apply_virtual_impedance(
        self,
        v_alpha_ref: float,
        v_beta_ref: float,
        i_alpha: float,
        i_beta: float,
        w_inst: float | None = None,
    ) -> tuple[float, float]:
        """Return ``(v_alpha_mod, v_beta_mod)`` after subtracting ``Z_v * i_alphabeta``."""
        w = self.w0 if w_inst is None else float(w_inst)
        if not all(math.isfinite(float(v)) for v in (v_alpha_ref, v_beta_ref, i_alpha, i_beta, w)):
            raise ValueError("Voltage, current, and angular frequency inputs must be finite")

        va, vb, ia, ib = float(v_alpha_ref), float(v_beta_ref), float(i_alpha), float(i_beta)
        x_v = w * self.l_v
        v_drop_alpha = self.r_v * ia - x_v * ib
        v_drop_beta = self.r_v * ib + x_v * ia
        return va - v_drop_alpha, vb - v_drop_beta
