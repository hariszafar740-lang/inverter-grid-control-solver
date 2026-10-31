"""Threshold Transient Virtual Impedance (TVI) & Synchronizing Power-Loop Anti-Windup (PAW).

Implements coordinated current-limiting and angle-stabilizing protection for multi-loop
electromagnetic Grid-Forming (GFM) inverters during deep symmetrical AC faults and severe
grid phase jumps (Paquette & Divan, *IEEE Trans. Ind. Appl.*, 2015; Qoria et al., *IEEE
TPEL*, 2020; Taul et al., *IEEE JESTPE*, 2020; Rokrok et al., *IEEE TPWRS*, 2022):

  1. Threshold Transient Virtual Impedance (TVI):
     Above current threshold ``I_th``, adds proportional resistive-inductive virtual
     impedance ``Delta Z_v(|i|) = Delta R_v + j * (omega / omega_0) * sigma_XR * Delta R_v``
     ahead of the synchronous-dq voltage PI loop so the inverter remains an unsaturated
     Thevenin voltage source while bounding bridge peak current ``|i_1,dq| <= I_max``.
  2. Synchronizing Active-Power Anti-Windup (PAW):
     Derates outer swing-equation power reference ``P_ref,eff`` during deep PCC voltage
     depressions and overcurrent excursions above ``I_th`` to prevent virtual-rotor windup
     and post-fault pole-slipping Loss of Synchronism (LOS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .thevenin_network import TheveninGridSpec


@dataclass(frozen=True)
class TVIConfig:
    """Configuration for Threshold Transient Virtual Impedance and Power Anti-Windup."""

    i_thresh_peak_a: float = 39.0
    i_max_peak_a: float = 48.0
    k_tvi_ohm_per_a: float = 0.145
    sigma_xr: float = 5.0
    w0_rad_s: float = 100.0 * math.pi
    paw_voltage_knee_pu: float = 0.88
    paw_voltage_exponent: float = 1.5
    paw_current_gain_w_per_a: float = 350.0

    def __post_init__(self) -> None:
        vals = (
            self.i_thresh_peak_a,
            self.i_max_peak_a,
            self.k_tvi_ohm_per_a,
            self.sigma_xr,
            self.w0_rad_s,
            self.paw_voltage_knee_pu,
            self.paw_voltage_exponent,
            self.paw_current_gain_w_per_a,
        )
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("All TVIConfig parameters must be finite")
        if self.i_thresh_peak_a <= 0.0 or self.i_max_peak_a <= self.i_thresh_peak_a:
            raise ValueError("Current bounds must satisfy 0 < i_thresh_peak_a < i_max_peak_a")
        if self.k_tvi_ohm_per_a < 0.0 or self.sigma_xr <= 0.0 or self.w0_rad_s <= 0.0:
            raise ValueError("k_tvi_ohm_per_a >= 0, sigma_xr > 0, and w0_rad_s > 0 required")
        if not (0.0 < self.paw_voltage_knee_pu <= 1.0):
            raise ValueError("paw_voltage_knee_pu must lie in (0.0, 1.0]")
        if self.paw_voltage_exponent <= 0.0 or self.paw_current_gain_w_per_a < 0.0:
            raise ValueError("paw_voltage_exponent > 0 and paw_current_gain_w_per_a >= 0 required")

    @classmethod
    def from_overcurrent_envelope(
        cls,
        grid: TheveninGridSpec | None = None,
        *,
        i_thresh_peak_a: float = 39.0,
        sigma_xr: float = 5.0,
        bolted_fault_margin: float = 1.08,
        paw_voltage_knee_pu: float = 0.88,
        paw_voltage_exponent: float = 1.5,
        paw_current_gain_w_per_a: float = 350.0,
    ) -> TVIConfig:
        """Size ``k_tvi_ohm_per_a`` analytically so bolted-fault ``|Z_v,tot(I_max)| >= margin * V_0 / I_max``."""
        g = grid if grid is not None else TheveninGridSpec()
        if bolted_fault_margin < 1.0 or not math.isfinite(bolted_fault_margin):
            raise ValueError("bolted_fault_margin must be finite and >= 1.0")
        i_max = g.i_max_peak_a
        if i_thresh_peak_a <= 0.0 or i_thresh_peak_a >= i_max:
            raise ValueError("i_thresh_peak_a must satisfy 0 < i_thresh_peak_a < grid.i_max_peak_a")
        if sigma_xr <= 0.0 or not math.isfinite(sigma_xr):
            raise ValueError("sigma_xr must be finite and > 0")

        z_req = bolted_fault_margin * g.v_peak_nom_v / i_max
        z_base_v = math.hypot(g.r_v_ohm, g.w0_rad_s * g.l_v_h)
        dz_needed = max(0.0, z_req - z_base_v)
        k_r = dz_needed / ((i_max - i_thresh_peak_a) * math.hypot(1.0, sigma_xr))
        return cls(
            i_thresh_peak_a=float(i_thresh_peak_a),
            i_max_peak_a=float(i_max),
            k_tvi_ohm_per_a=float(k_r),
            sigma_xr=float(sigma_xr),
            w0_rad_s=g.w0_rad_s,
            paw_voltage_knee_pu=float(paw_voltage_knee_pu),
            paw_voltage_exponent=float(paw_voltage_exponent),
            paw_current_gain_w_per_a=float(paw_current_gain_w_per_a),
        )


class TransientVirtualImpedance:
    """Piecewise-smooth threshold virtual impedance & synchronizing power anti-windup block."""

    def __init__(self, config: TVIConfig | None = None) -> None:
        self.config = config if config is not None else TVIConfig()

    def compute_incremental_zv(
        self,
        i_sense_peak_a: float,
        w_inst_rad_s: float | None = None,
    ) -> tuple[float, float, bool]:
        """Return ``(delta_r_ohm, delta_x_ohm, is_active)`` at current magnitude ``i_sense_peak_a``."""
        cfg = self.config
        w = cfg.w0_rad_s if w_inst_rad_s is None else float(w_inst_rad_s)
        i_mag = float(i_sense_peak_a)
        if not (math.isfinite(i_mag) and math.isfinite(w)) or i_mag < 0.0 or w <= 0.0:
            raise ValueError("i_sense_peak_a >= 0 and w_inst_rad_s > 0 must be finite")

        excess = i_mag - cfg.i_thresh_peak_a
        if excess <= 0.0 or cfg.k_tvi_ohm_per_a == 0.0:
            return 0.0, 0.0, False
        dr = cfg.k_tvi_ohm_per_a * excess
        dx = cfg.sigma_xr * dr * (w / cfg.w0_rad_s)
        return dr, dx, True

    def apply_dq_voltage_drop(
        self,
        e_star_v: float,
        i_d_a: float,
        i_q_a: float,
        r_v_base_ohm: float,
        l_v_base_h: float,
        w_inst_rad_s: float,
        *,
        i_sense_peak_a: float | None = None,
        enable_tvi: bool = True,
    ) -> tuple[float, float, float, bool]:
        """Return ``(vf_d_ref_v, vf_q_ref_v, z_tvi_mag_ohm, tvi_active)`` in synchronous ``dq``."""
        i_mag = math.hypot(float(i_d_a), float(i_q_a)) if i_sense_peak_a is None else float(i_sense_peak_a)
        if enable_tvi:
            dr, dx, active = self.compute_incremental_zv(i_mag, w_inst_rad_s)
        else:
            dr, dx, active = 0.0, 0.0, False

        r_tot = float(r_v_base_ohm) + dr
        x_tot = float(w_inst_rad_s) * float(l_v_base_h) + dx
        vf_d_ref = float(e_star_v) - (r_tot * float(i_d_a) - x_tot * float(i_q_a))
        vf_q_ref = -(r_tot * float(i_q_a) + x_tot * float(i_d_a))
        return vf_d_ref, vf_q_ref, math.hypot(dr, dx), active

    def compute_anti_windup_power_ref(
        self,
        p_ref_w: float,
        v_pcc_pu: float,
        i_sense_peak_a: float,
    ) -> float:
        """Compute derated outer swing-equation power command ``P_ref,eff`` during sags/overcurrent."""
        cfg = self.config
        p_cmd = float(p_ref_w)
        v_pu = min(1.0, max(0.0, float(v_pcc_pu)))
        i_mag = max(0.0, float(i_sense_peak_a))
        if not (math.isfinite(p_cmd) and math.isfinite(v_pu) and math.isfinite(i_mag)):
            raise ValueError("p_ref_w, v_pcc_pu, and i_sense_peak_a must be finite")

        if v_pu < cfg.paw_voltage_knee_pu:
            p_derated = p_cmd * (v_pu ** cfg.paw_voltage_exponent)
        else:
            p_derated = p_cmd
        excess_i = max(0.0, i_mag - cfg.i_thresh_peak_a)
        return p_derated - cfg.paw_current_gain_w_per_a * excess_i
