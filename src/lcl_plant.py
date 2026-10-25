"""Electromagnetic 6-state synchronous-dq LCL filter + Thévenin AC grid plant model.

Models the three-phase inverter output LCL filter with series capacitor passive damping
resistor ``R_d`` coupled to a Thévenin grid impedance ``(R_g, L_g)`` in a synchronous
``dq`` reference frame rotating at instantaneous angular speed ``omega_frame``:

    State vector: x_lcl = [i_1d, i_1q, v_cd, v_cq, i_2d, i_2q]^T

    v_f,dq = v_c,dq + R_d * (i_1,dq - i_2,dq)
    L_1 * d(i_1,dq)/dt         = v_inv,dq - v_f,dq - R_1 * i_1,dq - omega * J * L_1 * i_1,dq
    C_f * d(v_c,dq)/dt         = (i_1,dq - i_2,dq) - omega * J * C_f * v_c,dq
    (L_2 + L_g) * d(i_2,dq)/dt = v_f,dq - v_g,dq - (R_2 + R_g) * i_2,dq
                                 - omega * J * (L_2 + L_g) * i_2,dq

where ``J = [[0, -1], [1, 0]]`` is the 90-degree synchronous cross-coupling operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dq_frames import compute_pq_dq
from .thevenin_network import TheveninGridSpec


@dataclass(frozen=True)
class LCLFilterSpec:
    """Three-phase LCL output filter physical parameters (SI units: H, F, Ohm)."""

    l1_h: float = 2.2e-3
    r1_ohm: float = 0.08
    cf_f: float = 15.0e-6
    rd_ohm: float = 1.80
    l2_h: float = 0.9e-3
    r2_ohm: float = 0.04
    v_dc_link_v: float = 750.0

    def __post_init__(self) -> None:
        vals = (
            self.l1_h,
            self.r1_ohm,
            self.cf_f,
            self.rd_ohm,
            self.l2_h,
            self.r2_ohm,
            self.v_dc_link_v,
        )
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("All LCLFilterSpec parameters must be finite")
        if self.l1_h <= 0.0 or self.cf_f <= 0.0 or self.l2_h <= 0.0 or self.v_dc_link_v <= 0.0:
            raise ValueError("l1_h, cf_f, l2_h, and v_dc_link_v must be strictly positive")
        if self.r1_ohm < 0.0 or self.rd_ohm < 0.0 or self.r2_ohm < 0.0:
            raise ValueError("Filter resistances r1_ohm, rd_ohm, and r2_ohm must be >= 0")

    @property
    def v_inv_max_peak_v(self) -> float:
        """Linear SVPWM peak phase modulation ceiling ``V_dc / sqrt(3)``."""
        return self.v_dc_link_v / math.sqrt(3.0)

    @property
    def stiff_resonance_hz(self) -> float:
        """Stiff-grid LCL resonance frequency ``1 / (2*pi*sqrt(L_eq * C_f))``."""
        l_eq = (self.l1_h * self.l2_h) / (self.l1_h + self.l2_h)
        return 1.0 / (2.0 * math.pi * math.sqrt(l_eq * self.cf_f))

    def coupled_resonance_hz(self, l_grid_h: float) -> float:
        """Effective LCL resonance frequency including Thévenin grid inductance ``l_grid_h``."""
        if not math.isfinite(float(l_grid_h)) or float(l_grid_h) < 0.0:
            raise ValueError("l_grid_h must be finite and non-negative")
        l_g_side = self.l2_h + float(l_grid_h)
        l_eq = (self.l1_h * l_g_side) / (self.l1_h + l_g_side)
        return 1.0 / (2.0 * math.pi * math.sqrt(l_eq * self.cf_f))


class LCLGridPlant:
    """6-state electromagnetic LCL + Thévenin AC grid differential model in synchronous ``dq``."""

    def __init__(
        self,
        lcl: LCLFilterSpec | None = None,
        grid: TheveninGridSpec | None = None,
    ) -> None:
        self.lcl = lcl if lcl is not None else LCLFilterSpec()
        self.grid = grid if grid is not None else TheveninGridSpec()

    @property
    def r_net_ohm(self) -> float:
        return self.lcl.r2_ohm + self.grid.r_g_ohm

    @property
    def l_net_h(self) -> float:
        return self.lcl.l2_h + self.grid.l_g_h

    def filter_bus_voltage(self, x_lcl: np.ndarray) -> tuple[float, float]:
        """Compute damped capacitor branch node voltage ``v_f,dq = v_c,dq + R_d*(i_1,dq - i_2,dq)``."""
        i1d, i1q, vcd, vcq, i2d, i2q = (float(v) for v in x_lcl)
        rd = self.lcl.rd_ohm
        return vcd + rd * (i1d - i2d), vcq + rd * (i1q - i2q)

    def pcc_observables(
        self,
        x_lcl: np.ndarray,
        vg_d: float,
        vg_q: float,
        w_frame_rad_s: float,
    ) -> tuple[float, float, float, float]:
        """Exact algebraic PCC voltage ``(vpcc_d, vpcc_q)`` and 3-phase ``(P_pcc, Q_pcc)``."""
        _, _, _, _, i2d, i2q = (float(v) for v in x_lcl)
        vfd, vfq = self.filter_bus_voltage(x_lcl)
        w = float(w_frame_rad_s)
        di2d = (vfd - float(vg_d) - self.r_net_ohm * i2d + w * self.l_net_h * i2q) / self.l_net_h
        di2q = (vfq - float(vg_q) - self.r_net_ohm * i2q - w * self.l_net_h * i2d) / self.l_net_h

        rg = self.grid.r_g_ohm
        lg = self.grid.l_g_h
        vpcc_d = float(vg_d) + rg * i2d + lg * di2d - w * lg * i2q
        vpcc_q = float(vg_q) + rg * i2q + lg * di2q + w * lg * i2d
        p_pcc, q_pcc = compute_pq_dq(vpcc_d, vpcc_q, i2d, i2q)
        return vpcc_d, vpcc_q, p_pcc, q_pcc

    def rhs(
        self,
        x_lcl: np.ndarray,
        v_inv_d: float,
        v_inv_q: float,
        vg_d: float,
        vg_q: float,
        w_frame_rad_s: float,
    ) -> np.ndarray:
        """Continuous-time derivative ``dx_lcl/dt`` for ``[i1d, i1q, vcd, vcq, i2d, i2q]``."""
        i1d, i1q, vcd, vcq, i2d, i2q = (float(v) for v in x_lcl)
        w = float(w_frame_rad_s)
        rd = self.lcl.rd_ohm
        vfd = vcd + rd * (i1d - i2d)
        vfq = vcq + rd * (i1q - i2q)

        l1, r1, cf = self.lcl.l1_h, self.lcl.r1_ohm, self.lcl.cf_f
        l_net, r_net = self.l_net_h, self.r_net_ohm

        di1d = (float(v_inv_d) - vfd - r1 * i1d + w * l1 * i1q) / l1
        di1q = (float(v_inv_q) - vfq - r1 * i1q - w * l1 * i1d) / l1

        dvcd = ((i1d - i2d) + w * cf * vcq) / cf
        dvcq = ((i1q - i2q) - w * cf * vcd) / cf

        di2d = (vfd - float(vg_d) - r_net * i2d + w * l_net * i2q) / l_net
        di2q = (vfq - float(vg_q) - r_net * i2q - w * l_net * i2d) / l_net
        return np.array([di1d, di1q, dvcd, dvcq, di2d, di2q], dtype=float)

    def stored_energy_j(self, x_lcl: np.ndarray) -> float:
        """Total 3-phase stored electromagnetic energy ``E_L1 + E_Cf + E_Lnet`` in Joules."""
        i1d, i1q, vcd, vcq, i2d, i2q = (float(v) for v in x_lcl)
        e_l1 = 0.5 * self.lcl.l1_h * (i1d * i1d + i1q * i1q)
        e_cf = 0.5 * self.lcl.cf_f * (vcd * vcd + vcq * vcq)
        e_l2 = 0.5 * self.l_net_h * (i2d * i2d + i2q * i2q)
        return 1.5 * (e_l1 + e_cf + e_l2)

    def ohmic_dissipation_w(self, x_lcl: np.ndarray) -> float:
        """Total 3-phase resistive power loss ``P_R1 + P_Rd + P_Rnet`` in Watts."""
        i1d, i1q, _, _, i2d, i2q = (float(v) for v in x_lcl)
        ic_d = i1d - i2d
        ic_q = i1q - i2q
        p_r1 = self.lcl.r1_ohm * (i1d * i1d + i1q * i1q)
        p_rd = self.lcl.rd_ohm * (ic_d * ic_d + ic_q * ic_q)
        p_r2 = self.r_net_ohm * (i2d * i2d + i2q * i2q)
        return 1.5 * (p_r1 + p_rd + p_r2)

    def steady_state_equilibrium(
        self,
        vf_d: float,
        vf_q: float,
        vg_d: float,
        vg_q: float,
        w_frame_rad_s: float,
    ) -> tuple[np.ndarray, float, float]:
        """Exact ``dx_lcl/dt = 0`` equilibrium state and required bridge voltage ``(v_inv_d, v_inv_q)``."""
        w = float(w_frame_rad_s)
        r_net, x_net = self.r_net_ohm, w * self.l_net_h
        denom = r_net * r_net + x_net * x_net
        dv_d = float(vf_d) - float(vg_d)
        dv_q = float(vf_q) - float(vg_q)
        i2d = (r_net * dv_d + x_net * dv_q) / denom
        i2q = (r_net * dv_q - x_net * dv_d) / denom

        alpha = self.lcl.rd_ohm * w * self.lcl.cf_f
        det_c = 1.0 + alpha * alpha
        vcd = (float(vf_d) + alpha * float(vf_q)) / det_c
        vcq = (float(vf_q) - alpha * float(vf_d)) / det_c

        ic_d = -w * self.lcl.cf_f * vcq
        ic_q = w * self.lcl.cf_f * vcd
        i1d = i2d + ic_d
        i1q = i2q + ic_q

        v_inv_d = float(vf_d) + self.lcl.r1_ohm * i1d - w * self.lcl.l1_h * i1q
        v_inv_q = float(vf_q) + self.lcl.r1_ohm * i1q + w * self.lcl.l1_h * i1d
        x0 = np.array([i1d, i1q, vcd, vcq, i2d, i2q], dtype=float)
        return x0, v_inv_d, v_inv_q


def simulate_lcl_damping_step(
    *,
    rd_ohm: float,
    grid: TheveninGridSpec | None = None,
    step_v: float = 15.0,
    duration_s: float = 0.015,
    dt_s: float = 5.0e-6,
    step_time_s: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate 6-state LCL capacitor voltage ``v_cd(t)`` response to a bridge d-axis voltage step."""
    if duration_s <= 0.0 or dt_s <= 0.0 or dt_s > 5.0e-5:
        raise ValueError("duration_s must be > 0 and dt_s must lie in (0, 5e-5] s")
    g_spec = grid if grid is not None else TheveninGridSpec()
    plant = LCLGridPlant(lcl=LCLFilterSpec(rd_ohm=rd_ohm), grid=g_spec)
    x, vinv_d0, vinv_q0 = plant.steady_state_equilibrium(
        g_spec.v_peak_nom_v, 0.0, g_spec.v_peak_nom_v, 0.0, g_spec.w0_rad_s
    )
    n_steps = int(round(duration_s / dt_s)) + 1
    t_arr = np.arange(n_steps, dtype=float) * dt_s
    vcd_arr = np.empty(n_steps, dtype=float)
    vg_nom = g_spec.v_peak_nom_v
    w0 = g_spec.w0_rad_s

    for k, t_now in enumerate(t_arr):
        vcd_arr[k] = float(x[2])
        if k < n_steps - 1:
            vd1 = vinv_d0 + (step_v if t_now >= step_time_s - 1e-12 else 0.0)
            vd2 = vinv_d0 + (step_v if (t_now + 0.5 * dt_s) >= step_time_s - 1e-12 else 0.0)
            vd4 = vinv_d0 + (step_v if (t_now + dt_s) >= step_time_s - 1e-12 else 0.0)
            k1 = plant.rhs(x, vd1, vinv_q0, vg_nom, 0.0, w0)
            k2 = plant.rhs(x + 0.5 * dt_s * k1, vd2, vinv_q0, vg_nom, 0.0, w0)
            k3 = plant.rhs(x + 0.5 * dt_s * k2, vd2, vinv_q0, vg_nom, 0.0, w0)
            k4 = plant.rhs(x + dt_s * k3, vd4, vinv_q0, vg_nom, 0.0, w0)
            x = x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t_arr, vcd_arr
