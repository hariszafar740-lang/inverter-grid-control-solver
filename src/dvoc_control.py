"""Nonlinear Andronov-Hopf Dispatchable Virtual Oscillator Control (dVOC) law.

Implements the three-phase amplitude-invariant stationary ``alpha-beta`` and synchronous
polar ``(theta, v_hat)`` formulations of Dispatchable Virtual Oscillator Control
(Colombino et al., *IEEE Trans. Autom. Control*, 2019; Seo et al., *IEEE JESTPE*, 2019;
Lu et al., *IEEE Trans. Power Electron.*, 2020) for inductive AC networks (``phi = pi/2``):

    dv_ab/dt = w_0 * J * v_ab + eta * (K(P_ref, Q_ref) * v_ab - 1.5 * R(phi) * i_ab)
               + eta * alpha_osc * ((V_0^2 - ||v_ab||^2) / V_0^2) * v_ab

where ``J = [[0, -1], [1, 0]]``, ``R(phi)`` is the 2D rotation by network impedance
angle ``phi``, and ``||v_ab|| = v_hat`` is the instantaneous peak phase EMF amplitude.

For inductive grids (``phi = pi/2``, ``R(pi/2) = J``), the exact polar representation is:

    d(theta)/dt = w_0 + eta * (P_ref / V_0^2 - P_e / v_hat^2)
    d(v_hat)/dt = eta * alpha_osc * v_hat * (1 - v_hat^2 / V_0^2)
                  + eta * ((Q_ref / V_0^2) * v_hat - Q_e / v_hat)

Unlike Droop or VSG controllers, dVOC requires zero power-measurement low-pass filters
and exhibits global almost-everywhere asymptotic convergence to the nominal sinusoidal
limit cycle from arbitrary non-zero black-start initial conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .thevenin_network import TheveninGridSpec


@dataclass(frozen=True)
class DVOCConfig:
    """Physical and control-gain parameters for an Andronov-Hopf dVOC oscillator."""

    w0_rad_s: float = 100.0 * math.pi
    v0_peak_v: float = 230.0 * math.sqrt(2.0)
    eta_v2_per_ws: float = 16.838592995497673
    alpha_osc_w_per_v2: float = 1.2809905633315946
    rotation_angle_rad: float = 0.5 * math.pi

    def __post_init__(self) -> None:
        vals = (
            self.w0_rad_s,
            self.v0_peak_v,
            self.eta_v2_per_ws,
            self.alpha_osc_w_per_v2,
            self.rotation_angle_rad,
        )
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("All DVOCConfig parameters must be finite")
        if (
            self.w0_rad_s <= 0.0
            or self.v0_peak_v <= 0.0
            or self.eta_v2_per_ws <= 0.0
            or self.alpha_osc_w_per_v2 <= 0.0
        ):
            raise ValueError("w0_rad_s, v0_peak_v, eta_v2_per_ws, and alpha_osc_w_per_v2 must be > 0")
        if not (0.0 <= self.rotation_angle_rad <= math.pi):
            raise ValueError("rotation_angle_rad must lie in [0, pi]")

    @property
    def mu_limit_cycle_s_inv(self) -> float:
        """Unloaded Andronov-Hopf radial limit-cycle attraction rate ``eta * alpha_osc`` (s^-1)."""
        return self.eta_v2_per_ws * self.alpha_osc_w_per_v2

    @property
    def mp_eq_rad_per_ws(self) -> float:
        """Nominal small-signal P-omega droop slope ``eta / V_0^2`` at ``v_hat = V_0``."""
        return self.eta_v2_per_ws / (self.v0_peak_v * self.v0_peak_v)

    @property
    def nq_eq_v_per_var(self) -> float:
        """Nominal small-signal Q-V droop slope ``1 / (2 * alpha_osc * V_0)`` at ``v_hat = V_0``."""
        return 1.0 / (2.0 * self.alpha_osc_w_per_v2 * self.v0_peak_v)

    @classmethod
    def from_grid_spec(
        cls,
        grid: TheveninGridSpec | None = None,
        *,
        rotation_angle_rad: float = 0.5 * math.pi,
    ) -> DVOCConfig:
        """Match nominal small-signal P-omega and Q-V slopes to ``TheveninGridSpec``."""
        g = grid if grid is not None else TheveninGridSpec()
        mp = 1.0 / (g.dp_nms_rad * g.w0_rad_s)
        v0 = g.v_peak_nom_v
        eta = mp * v0 * v0
        nq = max(g.nq_v_per_var, 1.0e-6)
        alpha_osc = 1.0 / (2.0 * nq * v0)
        return cls(
            w0_rad_s=g.w0_rad_s,
            v0_peak_v=v0,
            eta_v2_per_ws=eta,
            alpha_osc_w_per_v2=alpha_osc,
            rotation_angle_rad=rotation_angle_rad,
        )


@dataclass(frozen=True)
class BlackStartExcitationResult:
    """Open-circuit stationary ``alpha-beta`` limit-cycle excitation trajectory."""

    config: DVOCConfig
    time_s: np.ndarray
    v_alpha_v: np.ndarray
    v_beta_v: np.ndarray
    v_hat_v: np.ndarray
    v_hat_analytical_v: np.ndarray
    theta_rad: np.ndarray


class DispatchableVirtualOscillator:
    """Cartesian ``alpha-beta`` and polar ``(theta, v_hat)`` Andronov-Hopf dVOC engine."""

    def __init__(self, config: DVOCConfig | None = None) -> None:
        self.config = config if config is not None else DVOCConfig.from_grid_spec()

    def cartesian_rhs(
        self,
        v_alpha_v: float,
        v_beta_v: float,
        i_alpha_a: float,
        i_beta_a: float,
        p_ref_w: float = 0.0,
        q_ref_var: float = 0.0,
    ) -> tuple[float, float]:
        """Evaluate ``(dv_alpha/dt, dv_beta/dt)`` in stationary Clarke ``alpha-beta`` coordinates."""
        va = float(v_alpha_v)
        vb = float(v_beta_v)
        ia = float(i_alpha_a)
        ib = float(i_beta_a)
        pr = float(p_ref_w)
        qr = float(q_ref_var)
        if not all(math.isfinite(x) for x in (va, vb, ia, ib, pr, qr)):
            raise ValueError("All cartesian_rhs inputs must be finite")

        cfg = self.config
        c_phi = math.cos(cfg.rotation_angle_rad)
        s_phi = math.sin(cfg.rotation_angle_rad)
        v0_sq = cfg.v0_peak_v * cfg.v0_peak_v
        v_sq = va * va + vb * vb

        k_diag = (pr * c_phi + qr * s_phi) / v0_sq
        k_skew = (pr * s_phi - qr * c_phi) / v0_sq
        rot_ia = c_phi * ia - s_phi * ib
        rot_ib = s_phi * ia + c_phi * ib

        hopf = cfg.eta_v2_per_ws * cfg.alpha_osc_w_per_v2 * ((v0_sq - v_sq) / v0_sq)
        dva = -cfg.w0_rad_s * vb + cfg.eta_v2_per_ws * (k_diag * va - k_skew * vb - 1.5 * rot_ia) + hopf * va
        dvb = cfg.w0_rad_s * va + cfg.eta_v2_per_ws * (k_skew * va + k_diag * vb - 1.5 * rot_ib) + hopf * vb
        return dva, dvb

    def polar_kinetics(
        self,
        v_hat_v: float,
        p_elec_w: float,
        q_elec_var: float,
        p_ref_w: float = 0.0,
        q_ref_var: float = 0.0,
    ) -> tuple[float, float]:
        """Evaluate synchronous polar ``(omega_inst_rad_s, dv_hat_dt_v_s)`` with ``v_eff`` floor."""
        vh = float(v_hat_v)
        pe = float(p_elec_w)
        qe = float(q_elec_var)
        pr = float(p_ref_w)
        qr = float(q_ref_var)
        if not all(math.isfinite(x) for x in (vh, pe, qe, pr, qr)) or vh < 0.0:
            raise ValueError("polar_kinetics requires finite inputs with v_hat_v >= 0")

        cfg = self.config
        v0 = cfg.v0_peak_v
        v0_sq = v0 * v0
        v_eff = max(vh, 0.05 * v0)
        c_phi = math.cos(cfg.rotation_angle_rad)
        s_phi = math.sin(cfg.rotation_angle_rad)

        p_ref_rot = pr * s_phi - qr * c_phi
        q_ref_rot = pr * c_phi + qr * s_phi
        p_el_rot = pe * s_phi - qe * c_phi
        q_el_rot = pe * c_phi + qe * s_phi

        eta = cfg.eta_v2_per_ws
        omega_inst = cfg.w0_rad_s + eta * (p_ref_rot / v0_sq - p_el_rot / (v_eff * v_eff))
        hopf_v = eta * cfg.alpha_osc_w_per_v2 * vh * ((v0_sq - vh * vh) / v0_sq)
        dv_hat = hopf_v + eta * ((q_ref_rot / v0_sq) * vh - q_el_rot / v_eff)
        return omega_inst, dv_hat

    def analytical_unloaded_amplitude(
        self,
        time_s: np.ndarray | float,
        v_init_v: float,
    ) -> np.ndarray | float:
        """Exact closed-form Bernoulli solution ``v_hat(t)`` for unloaded black-start excitation."""
        v_init = float(v_init_v)
        if not math.isfinite(v_init) or v_init < 0.0:
            raise ValueError("v_init_v must be finite and non-negative")
        v0 = self.config.v0_peak_v
        mu = self.config.mu_limit_cycle_s_inv

        if isinstance(time_s, np.ndarray):
            t_arr = np.asarray(time_s, dtype=float)
            if np.any(~np.isfinite(t_arr)) or np.any(t_arr < 0.0):
                raise ValueError("time_s entries must be finite and >= 0")
            if v_init == 0.0:
                return np.zeros_like(t_arr)
            denom = np.sqrt(v_init * v_init + (v0 * v0 - v_init * v_init) * np.exp(-2.0 * mu * t_arr))
            return (v0 * v_init) / denom

        t_val = float(time_s)
        if not math.isfinite(t_val) or t_val < 0.0:
            raise ValueError("time_s must be finite and >= 0")
        if v_init == 0.0:
            return 0.0
        denom_f = math.sqrt(v_init * v_init + (v0 * v0 - v_init * v_init) * math.exp(-2.0 * mu * t_val))
        return (v0 * v_init) / denom_f

    def simulate_black_start_excitation(
        self,
        *,
        v_init_pu: float = 0.10,
        theta_init_rad: float = 0.0,
        duration_s: float = 0.35,
        dt_s: float = 40.0e-6,
    ) -> BlackStartExcitationResult:
        """Simulate open-circuit Cartesian ``alpha-beta`` limit-cycle self-excitation via RK4."""
        if not (0.0 < v_init_pu <= 2.0) or not math.isfinite(theta_init_rad):
            raise ValueError("v_init_pu must lie in (0, 2.0] and theta_init_rad must be finite")
        if duration_s <= 0.0 or dt_s <= 0.0 or dt_s > 5.0e-4:
            raise ValueError("duration_s must be > 0 and dt_s must lie in (0, 5e-4] s")

        n_steps = int(round(duration_s / dt_s)) + 1
        time_s = np.linspace(0.0, float(duration_s), n_steps)
        v_alpha = np.empty(n_steps, dtype=float)
        v_beta = np.empty(n_steps, dtype=float)

        v_init_v = float(v_init_pu) * self.config.v0_peak_v
        va = v_init_v * math.cos(float(theta_init_rad))
        vb = v_init_v * math.sin(float(theta_init_rad))
        rhs = self.cartesian_rhs

        for k in range(n_steps):
            v_alpha[k] = va
            v_beta[k] = vb
            if k < n_steps - 1:
                k1a, k1b = rhs(va, vb, 0.0, 0.0, 0.0, 0.0)
                k2a, k2b = rhs(va + 0.5 * dt_s * k1a, vb + 0.5 * dt_s * k1b, 0.0, 0.0, 0.0, 0.0)
                k3a, k3b = rhs(va + 0.5 * dt_s * k2a, vb + 0.5 * dt_s * k2b, 0.0, 0.0, 0.0, 0.0)
                k4a, k4b = rhs(va + dt_s * k3a, vb + dt_s * k3b, 0.0, 0.0, 0.0, 0.0)
                va += (dt_s / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
                vb += (dt_s / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)

        v_hat = np.hypot(v_alpha, v_beta)
        v_ana = np.asarray(self.analytical_unloaded_amplitude(time_s, v_init_v), dtype=float)
        theta = np.unwrap(np.arctan2(v_beta, v_alpha))
        return BlackStartExcitationResult(
            config=self.config,
            time_s=time_s,
            v_alpha_v=v_alpha,
            v_beta_v=v_beta,
            v_hat_v=v_hat,
            v_hat_analytical_v=v_ana,
            theta_rad=theta,
        )
