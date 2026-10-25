"""14-State Electromagnetic Fault Ride-Through (FRT) & Severe Grid Phase-Jump Benchmark.

Evaluates four protection architectures on the 14-state electromagnetic ``dq`` LCL +
cascaded dual-loop PI Virtual Synchronous Generator (VSG) plant:

  - ``unprotected``:     No inner current clamp and no TVI (exposes destructive >3.5 pu
                         electromagnetic bridge overcurrent spikes).
  - ``hard_clamp_only``: Circular inner current-reference clamp ``|i_1,dq*| <= I_max`` only
                         (exposes voltage-loop open-loop saturation, virtual-rotor angle
                         windup during deep sags, and Qoria latch-up LOS under phase jumps).
  - ``tvi_only``:        Threshold Transient Virtual Impedance + backup inner clamp without
                         outer synchronizing power anti-windup (bounds bridge current <= I_max
                         and rides through -45 deg phase jumps, yet slips a pole under
                         prolonged 0.15 pu voltage faults at 0.85 pu pre-fault loading).
  - ``tvi_and_paw``:     Coordinated Threshold TVI + Synchronizing Power-Loop Anti-Windup
                         (strictly bounds ``|i_1,dq| <= I_max``, keeps the outer voltage PI
                         loop unsaturated, eliminates rotor windup, and achieves zero-slip
                         ride-through across both deep 0.15 pu faults and -45 deg phase jumps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import root

from .cascaded_control import CascadedPIConfig
from .lcl_plant import LCLFilterSpec, LCLGridPlant
from .thevenin_network import TheveninGridSpec
from .transient_virtual_impedance import TransientVirtualImpedance, TVIConfig

ProtectionMode = Literal["unprotected", "hard_clamp_only", "tvi_only", "tvi_and_paw"]

PROTECTION_LABELS: dict[ProtectionMode, str] = {
    "unprotected": "Unprotected Voltage Source",
    "hard_clamp_only": "Hard Circular Current Clamp Only",
    "tvi_only": "Threshold TVI Only (No Power AW)",
    "tvi_and_paw": "Coordinated TVI + Power Anti-Windup",
}


@dataclass(frozen=True)
class ElectromagneticFRTTrajectory:
    """Time-domain trajectory and protection metrics for a 14-state EM FRT / phase-jump run."""

    mode: ProtectionMode
    label: str
    scenario: Literal["voltage_sag", "phase_jump"]
    p_dispatch_w: float
    event_start_s: float
    event_end_s: float
    time_s: np.ndarray
    f_vsg_hz: np.ndarray
    delta_rel_deg: np.ndarray
    i1_peak_a: np.ndarray
    i1_cmd_unsat_a: np.ndarray
    p_pcc_kw: np.ndarray
    q_pcc_kvar: np.ndarray
    v_pcc_pu: np.ndarray
    z_tvi_ohm: np.ndarray
    current_limited: np.ndarray
    tvi_active: np.ndarray
    peak_bridge_current_a: float
    max_abs_delta_deg: float
    saturation_duty_pct: float
    fault_mean_q_kvar: float
    post_event_power_kw: float
    rode_through: bool


@dataclass(frozen=True)
class ElectromagneticFRTBenchmarkResult:
    """Complete comparative 14-state EM voltage-sag & grid phase-jump benchmark bundle."""

    grid: TheveninGridSpec
    lcl: LCLFilterSpec
    tvi_cfg: TVIConfig
    sag_residual_pu: float
    sag_dispatch_w: float
    phase_jump_deg: float
    phase_jump_dispatch_w: float
    sag_unprotected: ElectromagneticFRTTrajectory
    sag_hard_clamp: ElectromagneticFRTTrajectory
    sag_tvi_only: ElectromagneticFRTTrajectory
    sag_tvi_paw: ElectromagneticFRTTrajectory
    jump_unprotected: ElectromagneticFRTTrajectory
    jump_hard_clamp: ElectromagneticFRTTrajectory
    jump_tvi_paw: ElectromagneticFRTTrajectory


class ElectromagneticFRTVSGModel:
    """14-state electromagnetic VSG model with selectable TVI, circular clamp, and PAW."""

    STATE_DIM = 14

    def __init__(
        self,
        grid: TheveninGridSpec | None = None,
        lcl: LCLFilterSpec | None = None,
        pi_cfg: CascadedPIConfig | None = None,
        tvi_cfg: TVIConfig | None = None,
        *,
        mode: ProtectionMode = "tvi_and_paw",
    ) -> None:
        if mode not in PROTECTION_LABELS:
            raise ValueError(f"Unsupported ProtectionMode: {mode!r}")
        self.grid = (
            grid
            if grid is not None
            else TheveninGridSpec(scr=4.0, xr_ratio=7.0, j_kgm2=0.8, dp_nms_rad=12.0, i_max_peak_a=48.0)
        )
        self.lcl = lcl if lcl is not None else LCLFilterSpec()
        self.pi_cfg = pi_cfg if pi_cfg is not None else CascadedPIConfig.from_bandwidths(self.lcl)
        self.tvi_cfg = (
            tvi_cfg
            if tvi_cfg is not None
            else TVIConfig.from_overcurrent_envelope(self.grid, i_thresh_peak_a=39.0)
        )
        self.tvi = TransientVirtualImpedance(self.tvi_cfg)
        self.plant = LCLGridPlant(lcl=self.lcl, grid=self.grid)
        self.mode: ProtectionMode = mode

    def steady_initial_state(self) -> np.ndarray:
        """Exact zero-dispatch nominal equilibrium across all 14 states."""
        vg_d, vg_q = self.grid.v_peak_nom_v, 0.0
        x_lcl0, v_inv_d0, v_inv_q0 = self.plant.steady_state_equilibrium(
            vf_d=vg_d,
            vf_q=vg_q,
            vg_d=vg_d,
            vg_q=vg_q,
            w_frame_rad_s=self.grid.w0_rad_s,
        )
        i1d0, i1q0, _, _, i2d0, i2q0 = (float(v) for v in x_lcl0)
        vfd0, vfq0 = self.plant.filter_bus_voltage(x_lcl0)
        w0 = self.grid.w0_rad_s

        ff = self.pi_cfg.feedforward_current_gain
        i1d_fb = i1d0 - ff * i2d0 + w0 * self.lcl.cf_f * vfq0
        i1q_fb = i1q0 - ff * i2q0 - w0 * self.lcl.cf_f * vfd0
        xi_vd0 = i1d_fb / self.pi_cfg.ki_v if self.pi_cfg.ki_v > 0.0 else 0.0
        xi_vq0 = i1q_fb / self.pi_cfg.ki_v if self.pi_cfg.ki_v > 0.0 else 0.0

        vinv_d_fb = v_inv_d0 - vfd0 + w0 * self.lcl.l1_h * i1q0
        vinv_q_fb = v_inv_q0 - vfq0 - w0 * self.lcl.l1_h * i1d0
        xi_id0 = vinv_d_fb / self.pi_cfg.ki_i if self.pi_cfg.ki_i > 0.0 else 0.0
        xi_iq0 = vinv_q_fb / self.pi_cfg.ki_i if self.pi_cfg.ki_i > 0.0 else 0.0

        y0 = np.zeros(self.STATE_DIM, dtype=float)
        y0[1] = w0
        y0[4:8] = [xi_vd0, xi_vq0, xi_id0, xi_iq0]
        y0[8:14] = x_lcl0
        return y0

    def solve_pre_event_equilibrium(self, p_dispatch_w: float, q_ref_var: float = 0.0) -> np.ndarray:
        """Solve loaded pre-disturbance equilibrium ``rhs = 0`` at nominal voltage & phase."""
        v0 = self.grid.v_peak_nom_v
        w0 = self.grid.w0_rad_s
        y_seed = self.steady_initial_state()
        y_seed[0] = min(0.65, max(0.0, float(p_dispatch_w) / (self.grid.s_rated_va * self.grid.scr)))
        sol = root(
            lambda y: self.rhs_continuous(y, float(p_dispatch_w), float(q_ref_var), v0, 0.0, w0),
            y_seed,
            tol=1.0e-12,
        )
        res_norm = float(
            np.linalg.norm(self.rhs_continuous(sol.x, float(p_dispatch_w), float(q_ref_var), v0, 0.0, w0))
        )
        if not sol.success or res_norm > 1.0e-7:
            raise ValueError(f"Failed to converge pre-event GFM equilibrium at P={p_dispatch_w} W")
        return np.asarray(sol.x, dtype=float)

    def _algebraic_tuple(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
        theta_grid_offset_rad: float,
    ) -> tuple[float, ...]:
        delta = float(state[0])
        omega = float(state[1])
        q_filt = float(state[3])
        xi_vd = float(state[4])
        xi_vq = float(state[5])
        xi_id = float(state[6])
        xi_iq = float(state[7])
        i1d = float(state[8])
        i1q = float(state[9])
        vcd = float(state[10])
        vcq = float(state[11])
        i2d = float(state[12])
        i2q = float(state[13])

        delta_rel = delta - float(theta_grid_offset_rad)
        vg = float(v_grid_peak_v)
        vg_d = vg * math.cos(delta_rel)
        vg_q = -vg * math.sin(delta_rel)

        v0 = self.grid.v_peak_nom_v
        e_raw = v0 - self.grid.nq_v_per_var * (q_filt - float(q_ref_var))
        e_star = min(max(e_raw, 0.75 * v0), 1.20 * v0)

        i1_mag = math.hypot(i1d, i1q)
        i2_mag = math.hypot(i2d, i2q)
        i_sense = max(i1_mag, i2_mag)
        use_tvi = self.mode in ("tvi_only", "tvi_and_paw")

        cfg_tvi = self.tvi_cfg
        if use_tvi and i_sense > cfg_tvi.i_thresh_peak_a:
            dr = cfg_tvi.k_tvi_ohm_per_a * (i_sense - cfg_tvi.i_thresh_peak_a)
            dx = cfg_tvi.sigma_xr * dr * (omega / cfg_tvi.w0_rad_s)
            z_tvi = math.hypot(dr, dx)
            tvi_f = 1.0
        else:
            dr = 0.0
            dx = 0.0
            z_tvi = 0.0
            tvi_f = 0.0

        rv_tot = self.grid.r_v_ohm + dr
        xv_tot = omega * self.grid.l_v_h + dx
        vf_d_ref = e_star - (rv_tot * i2d - xv_tot * i2q)
        vf_q_ref = -(rv_tot * i2q + xv_tot * i2d)

        rd = self.lcl.rd_ohm
        vfd = vcd + rd * (i1d - i2d)
        vfq = vcq + rd * (i1q - i2q)

        err_vd = vf_d_ref - vfd
        err_vq = vf_q_ref - vfq
        ff = self.pi_cfg.feedforward_current_gain
        cf = self.lcl.cf_f
        i1d_u = self.pi_cfg.kp_v * err_vd + self.pi_cfg.ki_v * xi_vd + ff * i2d - omega * cf * vfq
        i1q_u = self.pi_cfg.kp_v * err_vq + self.pi_cfg.ki_v * xi_vq + ff * i2q + omega * cf * vfd
        iu_mag = math.hypot(i1d_u, i1q_u)

        i_max = self.grid.i_max_peak_a
        if self.mode != "unprotected" and iu_mag > i_max:
            sc_i = i_max / iu_mag
            i1d_r = i1d_u * sc_i
            i1q_r = i1q_u * sc_i
            cur_lim_f = 1.0
        else:
            i1d_r = i1d_u
            i1q_r = i1q_u
            cur_lim_f = 0.0

        err_id = i1d_r - i1d
        err_iq = i1q_r - i1q
        l1 = self.lcl.l1_h
        vid_u = self.pi_cfg.kp_i * err_id + self.pi_cfg.ki_i * xi_id + vfd - omega * l1 * i1q
        viq_u = self.pi_cfg.kp_i * err_iq + self.pi_cfg.ki_i * xi_iq + vfq + omega * l1 * i1d
        vm = math.hypot(vid_u, viq_u)
        vmax = self.lcl.v_inv_max_peak_v
        if vm > vmax:
            sc_v = vmax / vm
            vid = vid_u * sc_v
            viq = viq_u * sc_v
            v_lim_f = 1.0
        else:
            vid = vid_u
            viq = viq_u
            v_lim_f = 0.0

        r_net = self.plant.r_net_ohm
        l_net = self.plant.l_net_h
        di2d = (vfd - vg_d - r_net * i2d + omega * l_net * i2q) / l_net
        di2q = (vfq - vg_q - r_net * i2q - omega * l_net * i2d) / l_net

        rg = self.grid.r_g_ohm
        lg = self.grid.l_g_h
        vpcc_d = vg_d + rg * i2d + lg * di2d - omega * lg * i2q
        vpcc_q = vg_q + rg * i2q + lg * di2q + omega * lg * i2d
        p_pcc = 1.5 * (vpcc_d * i2d + vpcc_q * i2q)
        q_pcc = 1.5 * (vpcc_q * i2d - vpcc_d * i2q)
        vpcc_mag = math.hypot(vpcc_d, vpcc_q)
        vpcc_pu = vpcc_mag / v0

        if self.mode == "tvi_and_paw":
            v_clamped = min(1.0, max(0.0, vpcc_pu))
            p_derated = (
                float(p_ref_w) * (v_clamped ** cfg_tvi.paw_voltage_exponent)
                if v_clamped < cfg_tvi.paw_voltage_knee_pu
                else float(p_ref_w)
            )
            p_ref_eff = p_derated - cfg_tvi.paw_current_gain_w_per_a * max(0.0, i_sense - cfg_tvi.i_thresh_peak_a)
        else:
            p_ref_eff = float(p_ref_w)

        return (
            delta_rel,
            vfd,
            vfq,
            err_vd,
            err_vq,
            i1d_u,
            i1q_u,
            i1d_r,
            i1q_r,
            iu_mag,
            i1_mag,
            err_id,
            err_iq,
            vid_u,
            viq_u,
            vid,
            viq,
            di2d,
            di2q,
            p_pcc,
            q_pcc,
            vpcc_pu,
            z_tvi,
            p_ref_eff,
            cur_lim_f,
            v_lim_f,
            tvi_f,
        )

    def rhs_continuous(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
        theta_grid_offset_rad: float,
        w_grid_rad_s: float,
    ) -> np.ndarray:
        """Continuous-time 14-state ODE right-hand side ``dx/dt``."""
        omega = float(state[1])
        p_filt = float(state[2])
        q_filt = float(state[3])
        i1d = float(state[8])
        i1q = float(state[9])
        vcd = float(state[10])
        vcq = float(state[11])
        i2d = float(state[12])
        i2q = float(state[13])

        t = self._algebraic_tuple(state, p_ref_w, q_ref_var, v_grid_peak_v, theta_grid_offset_rad)
        vfd, vfq = t[1], t[2]
        err_vd, err_vq = t[3], t[4]
        i1d_u, i1q_u, i1d_r, i1q_r = t[5], t[6], t[7], t[8]
        err_id, err_iq = t[11], t[12]
        vid_u, viq_u, vid, viq = t[13], t[14], t[15], t[16]
        di2d, di2q = t[17], t[18]
        p_pcc, q_pcc = t[19], t[20]
        p_ref_eff = t[23]
        cur_lim_f, v_lim_f = t[24], t[25]

        w0 = self.grid.w0_rad_s
        wc = 2.0 * math.pi * self.grid.lpf_cutoff_hz
        d_delta = omega - float(w_grid_rad_s)
        d_omega = (p_ref_eff - p_filt - self.grid.dp_nms_rad * w0 * (omega - w0)) / (self.grid.j_kgm2 * w0)
        d_pf = wc * (p_pcc - p_filt)
        d_qf = wc * (q_pcc - q_filt)

        if cur_lim_f > 0.5:
            d_xivd = 0.0 if ((i1d_u - i1d_r) * err_vd > 0.0) else err_vd
            d_xivq = 0.0 if ((i1q_u - i1q_r) * err_vq > 0.0) else err_vq
        else:
            d_xivd = err_vd
            d_xivq = err_vq

        if v_lim_f > 0.5:
            d_xiid = 0.0 if ((vid_u - vid) * err_id > 0.0) else err_id
            d_xiiq = 0.0 if ((viq_u - viq) * err_iq > 0.0) else err_iq
        else:
            d_xiid = err_id
            d_xiiq = err_iq

        l1, r1, cf = self.lcl.l1_h, self.lcl.r1_ohm, self.lcl.cf_f
        di1d = (vid - vfd - r1 * i1d + omega * l1 * i1q) / l1
        di1q = (viq - vfq - r1 * i1q - omega * l1 * i1d) / l1
        dvcd = ((i1d - i2d) + omega * cf * vcq) / cf
        dvcq = ((i1q - i2q) - omega * cf * vcd) / cf

        return np.array(
            [
                d_delta,
                d_omega,
                d_pf,
                d_qf,
                d_xivd,
                d_xivq,
                d_xiid,
                d_xiiq,
                di1d,
                di1q,
                dvcd,
                dvcq,
                di2d,
                di2q,
            ],
            dtype=float,
        )


def _integrate_frt_trajectory(
    model: ElectromagneticFRTVSGModel,
    *,
    scenario: Literal["voltage_sag", "phase_jump"],
    p_dispatch_w: float,
    duration_s: float,
    dt_s: float,
    event_start_s: float,
    event_end_s: float,
    sag_residual_pu: float = 0.15,
    phase_jump_deg: float = -45.0,
) -> ElectromagneticFRTTrajectory:
    if duration_s <= 0.0 or dt_s <= 0.0 or dt_s > 2.0e-4:
        raise ValueError("duration_s > 0 and 0 < dt_s <= 2e-4 required")
    if not (0.0 <= event_start_s < duration_s):
        raise ValueError("event_start_s must lie in [0, duration_s)")
    if scenario == "voltage_sag":
        if not (event_start_s < event_end_s <= duration_s):
            raise ValueError("event_end_s must satisfy event_start_s < event_end_s <= duration_s")
        if not (0.0 <= sag_residual_pu <= 1.0):
            raise ValueError("sag_residual_pu must lie in [0.0, 1.0]")

    v0 = model.grid.v_peak_nom_v
    w0 = model.grid.w0_rad_s
    inv_2pi = 1.0 / (2.0 * math.pi)
    phi_jump_rad = math.radians(float(phase_jump_deg))

    n_steps = int(round(duration_s / dt_s)) + 1
    time_s = np.linspace(0.0, float(duration_s), n_steps)
    state = model.solve_pre_event_equilibrium(p_dispatch_w)

    f_vsg_hz = np.empty(n_steps, dtype=float)
    delta_rel_deg = np.empty(n_steps, dtype=float)
    i1_peak_a = np.empty(n_steps, dtype=float)
    i1_cmd_unsat_a = np.empty(n_steps, dtype=float)
    p_pcc_kw = np.empty(n_steps, dtype=float)
    q_pcc_kvar = np.empty(n_steps, dtype=float)
    v_pcc_pu = np.empty(n_steps, dtype=float)
    z_tvi_ohm = np.empty(n_steps, dtype=float)
    current_limited = np.zeros(n_steps, dtype=bool)
    tvi_active = np.zeros(n_steps, dtype=bool)

    def grid_inputs(t_val: float) -> tuple[float, float]:
        if scenario == "voltage_sag":
            vg = (sag_residual_pu * v0) if (event_start_s - 1.0e-12 <= t_val < event_end_s - 1.0e-12) else v0
            return vg, 0.0
        phi = phi_jump_rad if t_val >= event_start_s - 1.0e-12 else 0.0
        return v0, phi

    for k, t_now in enumerate(time_s):
        tv = float(t_now)
        vg_k, phi_k = grid_inputs(tv)
        alg = model._algebraic_tuple(state, p_dispatch_w, 0.0, vg_k, phi_k)
        f_vsg_hz[k] = float(state[1]) * inv_2pi
        delta_rel_deg[k] = math.degrees(alg[0])
        i1_cmd_unsat_a[k] = alg[9]
        i1_peak_a[k] = alg[10]
        p_pcc_kw[k] = alg[19] / 1000.0
        q_pcc_kvar[k] = alg[20] / 1000.0
        v_pcc_pu[k] = alg[21]
        z_tvi_ohm[k] = alg[22]
        current_limited[k] = alg[24] > 0.5
        tvi_active[k] = alg[26] > 0.5

        if k < n_steps - 1:
            vg_m, phi_m = grid_inputs(tv + 0.5 * dt_s)
            vg_e, phi_e = grid_inputs(tv + dt_s)
            k1 = model.rhs_continuous(state, p_dispatch_w, 0.0, vg_k, phi_k, w0)
            k2 = model.rhs_continuous(state + 0.5 * dt_s * k1, p_dispatch_w, 0.0, vg_m, phi_m, w0)
            k3 = model.rhs_continuous(state + 0.5 * dt_s * k2, p_dispatch_w, 0.0, vg_m, phi_m, w0)
            k4 = model.rhs_continuous(state + dt_s * k3, p_dispatch_w, 0.0, vg_e, phi_e, w0)
            state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    peak_i1 = float(np.max(i1_peak_a))
    max_delta = float(np.max(np.abs(delta_rel_deg)))
    sat_pct = float(100.0 * np.mean(current_limited))
    if scenario == "voltage_sag":
        mask_ev = (time_s >= event_start_s + 0.03) & (time_s < event_end_s - 0.02)
    else:
        mask_ev = (time_s >= event_start_s) & (time_s < min(duration_s, event_start_s + 0.15))
    mean_q = float(np.mean(q_pcc_kvar[mask_ev])) if np.any(mask_ev) else 0.0
    p_end = float(p_pcc_kw[-1])
    p_target_kw = float(p_dispatch_w) / 1000.0

    rode_through = bool(
        peak_i1 <= model.grid.i_max_peak_a + 0.5
        and max_delta < 120.0
        and abs(p_end - p_target_kw) < 0.8
    )
    return ElectromagneticFRTTrajectory(
        mode=model.mode,
        label=PROTECTION_LABELS[model.mode],
        scenario=scenario,
        p_dispatch_w=float(p_dispatch_w),
        event_start_s=float(event_start_s),
        event_end_s=float(event_end_s),
        time_s=time_s,
        f_vsg_hz=f_vsg_hz,
        delta_rel_deg=delta_rel_deg,
        i1_peak_a=i1_peak_a,
        i1_cmd_unsat_a=i1_cmd_unsat_a,
        p_pcc_kw=p_pcc_kw,
        q_pcc_kvar=q_pcc_kvar,
        v_pcc_pu=v_pcc_pu,
        z_tvi_ohm=z_tvi_ohm,
        current_limited=current_limited,
        tvi_active=tvi_active,
        peak_bridge_current_a=peak_i1,
        max_abs_delta_deg=max_delta,
        saturation_duty_pct=sat_pct,
        fault_mean_q_kvar=mean_q,
        post_event_power_kw=p_end,
        rode_through=rode_through,
    )


def simulate_em_fault_sag(
    mode: ProtectionMode,
    *,
    grid: TheveninGridSpec | None = None,
    lcl: LCLFilterSpec | None = None,
    tvi_cfg: TVIConfig | None = None,
    p_dispatch_w: float = 17_000.0,
    sag_residual_pu: float = 0.15,
    t_fault_s: float = 0.20,
    t_clear_s: float = 0.65,
    duration_s: float = 1.40,
    dt_s: float = 80.0e-6,
) -> ElectromagneticFRTTrajectory:
    """Simulate a deep symmetrical three-phase voltage sag on the 14-state EM VSG plant."""
    m = ElectromagneticFRTVSGModel(grid=grid, lcl=lcl, tvi_cfg=tvi_cfg, mode=mode)
    return _integrate_frt_trajectory(
        m,
        scenario="voltage_sag",
        p_dispatch_w=p_dispatch_w,
        duration_s=duration_s,
        dt_s=dt_s,
        event_start_s=t_fault_s,
        event_end_s=t_clear_s,
        sag_residual_pu=sag_residual_pu,
    )


def simulate_em_phase_jump(
    mode: ProtectionMode,
    *,
    grid: TheveninGridSpec | None = None,
    lcl: LCLFilterSpec | None = None,
    tvi_cfg: TVIConfig | None = None,
    p_dispatch_w: float = 12_000.0,
    phase_jump_deg: float = -45.0,
    t_jump_s: float = 0.20,
    duration_s: float = 1.10,
    dt_s: float = 80.0e-6,
) -> ElectromagneticFRTTrajectory:
    """Simulate an instantaneous grid phase-angle jump on the 14-state EM VSG plant."""
    m = ElectromagneticFRTVSGModel(grid=grid, lcl=lcl, tvi_cfg=tvi_cfg, mode=mode)
    return _integrate_frt_trajectory(
        m,
        scenario="phase_jump",
        p_dispatch_w=p_dispatch_w,
        duration_s=duration_s,
        dt_s=dt_s,
        event_start_s=t_jump_s,
        event_end_s=duration_s,
        phase_jump_deg=phase_jump_deg,
    )


def run_electromagnetic_frt_benchmark(
    *,
    sag_residual_pu: float = 0.15,
    sag_dispatch_w: float = 17_000.0,
    phase_jump_deg: float = -45.0,
    phase_jump_dispatch_w: float = 12_000.0,
    dt_s: float = 80.0e-6,
) -> ElectromagneticFRTBenchmarkResult:
    """Run the complete 14-state electromagnetic FRT & phase-jump benchmark suite."""
    grid = TheveninGridSpec(scr=4.0, xr_ratio=7.0, j_kgm2=0.8, dp_nms_rad=12.0, i_max_peak_a=48.0)
    lcl = LCLFilterSpec()
    tvi_cfg = TVIConfig.from_overcurrent_envelope(grid, i_thresh_peak_a=39.0)

    sag_kw = {
        "grid": grid,
        "lcl": lcl,
        "tvi_cfg": tvi_cfg,
        "p_dispatch_w": sag_dispatch_w,
        "sag_residual_pu": sag_residual_pu,
        "dt_s": dt_s,
    }
    jump_kw = {
        "grid": grid,
        "lcl": lcl,
        "tvi_cfg": tvi_cfg,
        "p_dispatch_w": phase_jump_dispatch_w,
        "phase_jump_deg": phase_jump_deg,
        "dt_s": dt_s,
    }

    return ElectromagneticFRTBenchmarkResult(
        grid=grid,
        lcl=lcl,
        tvi_cfg=tvi_cfg,
        sag_residual_pu=float(sag_residual_pu),
        sag_dispatch_w=float(sag_dispatch_w),
        phase_jump_deg=float(phase_jump_deg),
        phase_jump_dispatch_w=float(phase_jump_dispatch_w),
        sag_unprotected=simulate_em_fault_sag("unprotected", **sag_kw),
        sag_hard_clamp=simulate_em_fault_sag("hard_clamp_only", **sag_kw),
        sag_tvi_only=simulate_em_fault_sag("tvi_only", **sag_kw),
        sag_tvi_paw=simulate_em_fault_sag("tvi_and_paw", **sag_kw),
        jump_unprotected=simulate_em_phase_jump("unprotected", **jump_kw),
        jump_hard_clamp=simulate_em_phase_jump("hard_clamp_only", **jump_kw),
        jump_tvi_paw=simulate_em_phase_jump("tvi_and_paw", **jump_kw),
    )
