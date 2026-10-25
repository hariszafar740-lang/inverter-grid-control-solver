"""Comparative Grid-Forming Control Triad on the 6-State Electromagnetic dq LCL Plant.

Unifies three canonical Grid-Forming (GFM) primary synchronizing laws on the identical
6-state electromagnetic ``dq`` LCL filter + cascaded dual-loop PI voltage/current inner
architecture + Thévenin AC grid network:

  1. ``MultiLoopDroopGFMModel`` (13 continuous states):
     First-order P/Q measurement LPFs (``f_c = 15 Hz``) driving algebraic P-omega and Q-V
     droop without virtual rotor momentum.
  2. ``ElectromagneticGFMModel`` / VSG (14 continuous states, ``src/cascaded_control.py``):
     Second-order synchronous swing equation (``J``, ``D_p``, ``H = 4.93 s``) providing
     synthetic inertia and reduced initial RoCoF at the expense of underdamped rotor swing.
  3. ``MultiLoopDVOCGFMModel`` (12 continuous states):
     Nonlinear Andronov-Hopf Dispatchable Virtual Oscillator Control (``dVOC``) in polar
     synchronous coordinates ``(delta, v_hat)`` with zero power LPFs, fast overdamped
     Kuramoto phase synchronization, and built-in cubic voltage stabilization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .cascaded_control import CascadedPIConfig, ElectromagneticGFMModel
from .dq_frames import rotate_dq_frame
from .dvoc_control import BlackStartExcitationResult, DispatchableVirtualOscillator, DVOCConfig
from .lcl_plant import LCLFilterSpec, LCLGridPlant
from .thevenin_network import TheveninGridSpec

GFMMode = Literal["droop", "vsg", "dvoc"]


@dataclass(frozen=True)
class GFMTriadMemberResult:
    """Standardized electromagnetic trajectory and stability metrics for a single GFM mode."""

    mode: GFMMode
    state_dim: int
    grid: TheveninGridSpec
    lcl: LCLFilterSpec
    pi_cfg: CascadedPIConfig
    time_s: np.ndarray
    f_gfm_hz: np.ndarray
    f_grid_hz: np.ndarray
    delta_deg: np.ndarray
    p_ref_w: np.ndarray
    p_pcc_w: np.ndarray
    q_pcc_var: np.ndarray
    e_int_pu: np.ndarray
    v_pcc_pu: np.ndarray
    i1_peak_a: np.ndarray
    i2_peak_a: np.ndarray
    current_limited: np.ndarray
    voltage_clamped: np.ndarray
    eigenvalues: np.ndarray
    max_real_eig_s_inv: float
    step_rise_time_90_ms: float
    step_peak_power_kw: float
    freq_drop_peak_power_kw: float
    nadir_freq_hz: float
    initial_drop_rocof_20ms_hz_s: float
    peak_drop_rocof_hz_s: float
    final_power_kw: float


@dataclass(frozen=True)
class GFMTriadBenchmarkResult:
    """Complete comparative benchmark bundle across Droop (13), VSG (14), and dVOC (12)."""

    grid: TheveninGridSpec
    lcl: LCLFilterSpec
    pi_cfg: CascadedPIConfig
    droop: GFMTriadMemberResult
    vsg: GFMTriadMemberResult
    dvoc: GFMTriadMemberResult
    black_start: BlackStartExcitationResult


class MultiLoopDroopGFMModel:
    """13-state multi-loop P-f / Q-V Droop GFM model on the electromagnetic dq LCL plant.

    State vector ordering (length 13):
        0: delta_rad   (virtual phase angle relative to Thévenin grid bus)
        1: p_filt_w    (LPF active power measurement)
        2: q_filt_var  (LPF reactive power measurement)
        3..4: xi_vd, xi_vq (outer capacitor-voltage PI integrators)
        5..6: xi_id, xi_iq (inner bridge-current PI integrators)
        7..12: x_lcl   ([i_1d, i_1q, v_cd, v_cq, i_2d, i_2q] in inverter dq frame)
    """

    STATE_DIM = 13

    def __init__(
        self,
        grid: TheveninGridSpec | None = None,
        lcl: LCLFilterSpec | None = None,
        pi_cfg: CascadedPIConfig | None = None,
    ) -> None:
        self.grid = grid if grid is not None else TheveninGridSpec()
        self.lcl = lcl if lcl is not None else LCLFilterSpec()
        self.pi_cfg = pi_cfg if pi_cfg is not None else CascadedPIConfig.from_bandwidths(self.lcl)
        self.plant = LCLGridPlant(lcl=self.lcl, grid=self.grid)
        self.mp_rad_per_ws = 1.0 / (self.grid.dp_nms_rad * self.grid.w0_rad_s)

    def steady_initial_state(self) -> np.ndarray:
        """Exact zero-power nominal equilibrium ``dx/dt = 0`` across all 13 states."""
        vg_d = self.grid.v_peak_nom_v
        w0 = self.grid.w0_rad_s
        x_lcl0, v_inv_d0, v_inv_q0 = self.plant.steady_state_equilibrium(vg_d, 0.0, vg_d, 0.0, w0)
        i1d0, i1q0, _, _, i2d0, i2q0 = (float(v) for v in x_lcl0)
        vfd0, vfq0 = self.plant.filter_bus_voltage(x_lcl0)

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
        y0[3:7] = [xi_vd0, xi_vq0, xi_id0, xi_iq0]
        y0[7:13] = x_lcl0
        return y0

    def _algebraic_tuple(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
    ) -> tuple[float, ...]:
        delta = float(state[0])
        p_filt = float(state[1])
        q_filt = float(state[2])
        xi_vd = float(state[3])
        xi_vq = float(state[4])
        xi_id = float(state[5])
        xi_iq = float(state[6])
        i1d = float(state[7])
        i1q = float(state[8])
        vcd = float(state[9])
        vcq = float(state[10])
        i2d = float(state[11])
        i2q = float(state[12])

        omega = self.grid.w0_rad_s + self.mp_rad_per_ws * (float(p_ref_w) - p_filt)
        vg_d = float(v_grid_peak_v) * math.cos(delta)
        vg_q = -float(v_grid_peak_v) * math.sin(delta)

        v_nom = self.grid.v_peak_nom_v
        e_raw = v_nom - self.grid.nq_v_per_var * (q_filt - float(q_ref_var))
        e_star = min(max(e_raw, 0.75 * v_nom), 1.20 * v_nom)

        rv = self.grid.r_v_ohm
        xv = omega * self.grid.l_v_h
        vf_d_ref = e_star - (rv * i2d - xv * i2q)
        vf_q_ref = -(rv * i2q + xv * i2d)

        rd = self.lcl.rd_ohm
        vfd = vcd + rd * (i1d - i2d)
        vfq = vcq + rd * (i1q - i2q)

        err_vd = vf_d_ref - vfd
        err_vq = vf_q_ref - vfq
        ff = self.pi_cfg.feedforward_current_gain
        cf = self.lcl.cf_f
        i1d_unsat = self.pi_cfg.kp_v * err_vd + self.pi_cfg.ki_v * xi_vd + ff * i2d - omega * cf * vfq
        i1q_unsat = self.pi_cfg.kp_v * err_vq + self.pi_cfg.ki_v * xi_vq + ff * i2q + omega * cf * vfd

        i_mag = math.hypot(i1d_unsat, i1q_unsat)
        i_max = self.grid.i_max_peak_a
        if i_mag > i_max:
            scale_i = i_max / i_mag
            i1d_ref = i1d_unsat * scale_i
            i1q_ref = i1q_unsat * scale_i
            cur_lim_f = 1.0
        else:
            i1d_ref = i1d_unsat
            i1q_ref = i1q_unsat
            cur_lim_f = 0.0

        err_id = i1d_ref - i1d
        err_iq = i1q_ref - i1q
        l1 = self.lcl.l1_h
        vinv_d_unsat = self.pi_cfg.kp_i * err_id + self.pi_cfg.ki_i * xi_id + vfd - omega * l1 * i1q
        vinv_q_unsat = self.pi_cfg.kp_i * err_iq + self.pi_cfg.ki_i * xi_iq + vfq + omega * l1 * i1d

        v_mag_cmd = math.hypot(vinv_d_unsat, vinv_q_unsat)
        v_max = self.lcl.v_inv_max_peak_v
        if v_mag_cmd > v_max:
            scale_v = v_max / v_mag_cmd
            vinv_d = vinv_d_unsat * scale_v
            vinv_q = vinv_q_unsat * scale_v
            v_lim_f = 1.0
        else:
            vinv_d = vinv_d_unsat
            vinv_q = vinv_q_unsat
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
        v_pcc_mag = math.hypot(vpcc_d, vpcc_q)

        return (
            omega,
            e_star,
            vg_d,
            vg_q,
            vfd,
            vfq,
            vf_d_ref,
            vf_q_ref,
            err_vd,
            err_vq,
            i1d_unsat,
            i1q_unsat,
            i1d_ref,
            i1q_ref,
            err_id,
            err_iq,
            vinv_d_unsat,
            vinv_q_unsat,
            vinv_d,
            vinv_q,
            di2d,
            di2q,
            p_pcc,
            q_pcc,
            v_pcc_mag,
            cur_lim_f,
            v_lim_f,
        )

    def evaluate_algebraic(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
    ) -> dict[str, float | bool]:
        """Public inspection dictionary of control signals and PCC observables at ``state``."""
        _ = rotate_dq_frame(float(v_grid_peak_v), 0.0, float(state[0]))
        t = self._algebraic_tuple(state, p_ref_w, q_ref_var, v_grid_peak_v)
        return {
            "omega": t[0],
            "e_star": t[1],
            "vg_d": t[2],
            "vg_q": t[3],
            "vfd": t[4],
            "vfq": t[5],
            "vf_d_ref": t[6],
            "vf_q_ref": t[7],
            "i1d_ref": t[12],
            "i1q_ref": t[13],
            "vinv_d": t[18],
            "vinv_q": t[19],
            "p_pcc": t[22],
            "q_pcc": t[23],
            "v_pcc_mag": t[24],
            "cur_lim": t[25] > 0.5,
            "v_lim": t[26] > 0.5,
        }

    def rhs_continuous(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
        w_grid_rad_s: float,
    ) -> np.ndarray:
        """Continuous-time 13-state ODE right-hand side ``dx/dt``."""
        p_filt = float(state[1])
        q_filt = float(state[2])
        i1d = float(state[7])
        i1q = float(state[8])
        vcd = float(state[9])
        vcq = float(state[10])
        i2d = float(state[11])
        i2q = float(state[12])

        t = self._algebraic_tuple(state, p_ref_w, q_ref_var, v_grid_peak_v)
        omega = t[0]
        vfd, vfq = t[4], t[5]
        err_vd, err_vq = t[8], t[9]
        i1d_unsat, i1q_unsat, i1d_ref, i1q_ref = t[10], t[11], t[12], t[13]
        err_id, err_iq = t[14], t[15]
        vinv_d_unsat, vinv_q_unsat, vinv_d, vinv_q = t[16], t[17], t[18], t[19]
        di2d, di2q = t[20], t[21]
        p_pcc, q_pcc = t[22], t[23]
        cur_lim_f, v_lim_f = t[25], t[26]

        wc = 2.0 * math.pi * self.grid.lpf_cutoff_hz
        d_delta = omega - float(w_grid_rad_s)
        d_pfilt = wc * (p_pcc - p_filt)
        d_qfilt = wc * (q_pcc - q_filt)

        if cur_lim_f > 0.5:
            d_xivd = 0.0 if ((i1d_unsat - i1d_ref) * err_vd > 0.0) else err_vd
            d_xivq = 0.0 if ((i1q_unsat - i1q_ref) * err_vq > 0.0) else err_vq
        else:
            d_xivd = err_vd
            d_xivq = err_vq

        if v_lim_f > 0.5:
            d_xiid = 0.0 if ((vinv_d_unsat - vinv_d) * err_id > 0.0) else err_id
            d_xiiq = 0.0 if ((vinv_q_unsat - vinv_q) * err_iq > 0.0) else err_iq
        else:
            d_xiid = err_id
            d_xiiq = err_iq

        l1, r1, cf = self.lcl.l1_h, self.lcl.r1_ohm, self.lcl.cf_f
        di1d = (vinv_d - vfd - r1 * i1d + omega * l1 * i1q) / l1
        di1q = (vinv_q - vfq - r1 * i1q - omega * l1 * i1d) / l1
        dvcd = ((i1d - i2d) + omega * cf * vcq) / cf
        dvcq = ((i1q - i2q) - omega * cf * vcd) / cf

        return np.array(
            [
                d_delta,
                d_pfilt,
                d_qfilt,
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

    def linearized_eigenvalues(
        self,
        equilibrium_state: np.ndarray | None = None,
        *,
        p_ref_w: float = 0.0,
        q_ref_var: float = 0.0,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute the 13 small-signal eigenvalues of ``df/dx`` via central differences."""
        x0 = self.steady_initial_state() if equilibrium_state is None else np.array(equilibrium_state, dtype=float)
        jac = np.empty((self.STATE_DIM, self.STATE_DIM), dtype=float)
        vg = self.grid.v_peak_nom_v
        w0 = self.grid.w0_rad_s
        for col in range(self.STATE_DIM):
            xp = x0.copy()
            xm = x0.copy()
            xp[col] += eps
            xm[col] -= eps
            fp = self.rhs_continuous(xp, p_ref_w, q_ref_var, vg, w0)
            fm = self.rhs_continuous(xm, p_ref_w, q_ref_var, vg, w0)
            jac[:, col] = (fp - fm) / (2.0 * eps)
        return np.linalg.eigvals(jac)


class MultiLoopDVOCGFMModel:
    """12-state nonlinear Andronov-Hopf dVOC GFM model on the electromagnetic dq LCL plant.

    State vector ordering (length 12):
        0: delta_rad   (dVOC oscillator phase angle relative to Thévenin grid bus)
        1: v_hat_v     (dVOC oscillator internal peak EMF amplitude)
        2..3: xi_vd, xi_vq (outer capacitor-voltage PI integrators)
        4..5: xi_id, xi_iq (inner bridge-current PI integrators)
        6..11: x_lcl   ([i_1d, i_1q, v_cd, v_cq, i_2d, i_2q] in inverter dq frame)
    """

    STATE_DIM = 12

    def __init__(
        self,
        grid: TheveninGridSpec | None = None,
        lcl: LCLFilterSpec | None = None,
        pi_cfg: CascadedPIConfig | None = None,
        dvoc_cfg: DVOCConfig | None = None,
    ) -> None:
        self.grid = grid if grid is not None else TheveninGridSpec()
        self.lcl = lcl if lcl is not None else LCLFilterSpec()
        self.pi_cfg = pi_cfg if pi_cfg is not None else CascadedPIConfig.from_bandwidths(self.lcl)
        self.dvoc_cfg = dvoc_cfg if dvoc_cfg is not None else DVOCConfig.from_grid_spec(self.grid)
        self.oscillator = DispatchableVirtualOscillator(self.dvoc_cfg)
        self.plant = LCLGridPlant(lcl=self.lcl, grid=self.grid)

    def steady_initial_state(self) -> np.ndarray:
        """Exact zero-power nominal equilibrium ``dx/dt = 0`` across all 12 states."""
        vg_d = self.grid.v_peak_nom_v
        w0 = self.grid.w0_rad_s
        x_lcl0, v_inv_d0, v_inv_q0 = self.plant.steady_state_equilibrium(vg_d, 0.0, vg_d, 0.0, w0)
        i1d0, i1q0, _, _, i2d0, i2q0 = (float(v) for v in x_lcl0)
        vfd0, vfq0 = self.plant.filter_bus_voltage(x_lcl0)

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
        y0[0] = 0.0
        y0[1] = self.dvoc_cfg.v0_peak_v
        y0[2:6] = [xi_vd0, xi_vq0, xi_id0, xi_iq0]
        y0[6:12] = x_lcl0
        return y0

    def _algebraic_tuple(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
    ) -> tuple[float, ...]:
        delta = float(state[0])
        v_hat = max(float(state[1]), 0.0)
        xi_vd = float(state[2])
        xi_vq = float(state[3])
        xi_id = float(state[4])
        xi_iq = float(state[5])
        i1d = float(state[6])
        i1q = float(state[7])
        vcd = float(state[8])
        vcq = float(state[9])
        i2d = float(state[10])
        i2q = float(state[11])

        p_osc = 1.5 * v_hat * i2d
        q_osc = -1.5 * v_hat * i2q
        omega, dv_hat = self.oscillator.polar_kinetics(
            v_hat_v=v_hat,
            p_elec_w=p_osc,
            q_elec_var=q_osc,
            p_ref_w=float(p_ref_w),
            q_ref_var=float(q_ref_var),
        )

        vg_d = float(v_grid_peak_v) * math.cos(delta)
        vg_q = -float(v_grid_peak_v) * math.sin(delta)

        v_nom = self.grid.v_peak_nom_v
        e_star = min(max(v_hat, 0.75 * v_nom), 1.20 * v_nom)

        rv = self.grid.r_v_ohm
        xv = omega * self.grid.l_v_h
        vf_d_ref = e_star - (rv * i2d - xv * i2q)
        vf_q_ref = -(rv * i2q + xv * i2d)

        rd = self.lcl.rd_ohm
        vfd = vcd + rd * (i1d - i2d)
        vfq = vcq + rd * (i1q - i2q)

        err_vd = vf_d_ref - vfd
        err_vq = vf_q_ref - vfq
        ff = self.pi_cfg.feedforward_current_gain
        cf = self.lcl.cf_f
        i1d_unsat = self.pi_cfg.kp_v * err_vd + self.pi_cfg.ki_v * xi_vd + ff * i2d - omega * cf * vfq
        i1q_unsat = self.pi_cfg.kp_v * err_vq + self.pi_cfg.ki_v * xi_vq + ff * i2q + omega * cf * vfd

        i_mag = math.hypot(i1d_unsat, i1q_unsat)
        i_max = self.grid.i_max_peak_a
        if i_mag > i_max:
            scale_i = i_max / i_mag
            i1d_ref = i1d_unsat * scale_i
            i1q_ref = i1q_unsat * scale_i
            cur_lim_f = 1.0
        else:
            i1d_ref = i1d_unsat
            i1q_ref = i1q_unsat
            cur_lim_f = 0.0

        err_id = i1d_ref - i1d
        err_iq = i1q_ref - i1q
        l1 = self.lcl.l1_h
        vinv_d_unsat = self.pi_cfg.kp_i * err_id + self.pi_cfg.ki_i * xi_id + vfd - omega * l1 * i1q
        vinv_q_unsat = self.pi_cfg.kp_i * err_iq + self.pi_cfg.ki_i * xi_iq + vfq + omega * l1 * i1d

        v_mag_cmd = math.hypot(vinv_d_unsat, vinv_q_unsat)
        v_max = self.lcl.v_inv_max_peak_v
        if v_mag_cmd > v_max:
            scale_v = v_max / v_mag_cmd
            vinv_d = vinv_d_unsat * scale_v
            vinv_q = vinv_q_unsat * scale_v
            v_lim_f = 1.0
        else:
            vinv_d = vinv_d_unsat
            vinv_q = vinv_q_unsat
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
        v_pcc_mag = math.hypot(vpcc_d, vpcc_q)

        return (
            omega,
            dv_hat,
            e_star,
            vg_d,
            vg_q,
            vfd,
            vfq,
            vf_d_ref,
            vf_q_ref,
            err_vd,
            err_vq,
            i1d_unsat,
            i1q_unsat,
            i1d_ref,
            i1q_ref,
            err_id,
            err_iq,
            vinv_d_unsat,
            vinv_q_unsat,
            vinv_d,
            vinv_q,
            di2d,
            di2q,
            p_pcc,
            q_pcc,
            v_pcc_mag,
            cur_lim_f,
            v_lim_f,
        )

    def evaluate_algebraic(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
    ) -> dict[str, float | bool]:
        """Public inspection dictionary of control signals and PCC observables at ``state``."""
        _ = rotate_dq_frame(float(v_grid_peak_v), 0.0, float(state[0]))
        t = self._algebraic_tuple(state, p_ref_w, q_ref_var, v_grid_peak_v)
        return {
            "omega": t[0],
            "dv_hat": t[1],
            "e_star": t[2],
            "vg_d": t[3],
            "vg_q": t[4],
            "vfd": t[5],
            "vfq": t[6],
            "vf_d_ref": t[7],
            "vf_q_ref": t[8],
            "i1d_ref": t[13],
            "i1q_ref": t[14],
            "vinv_d": t[19],
            "vinv_q": t[20],
            "p_pcc": t[23],
            "q_pcc": t[24],
            "v_pcc_mag": t[25],
            "cur_lim": t[26] > 0.5,
            "v_lim": t[27] > 0.5,
        }

    def rhs_continuous(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
        w_grid_rad_s: float,
    ) -> np.ndarray:
        """Continuous-time 12-state ODE right-hand side ``dx/dt``."""
        i1d = float(state[6])
        i1q = float(state[7])
        vcd = float(state[8])
        vcq = float(state[9])
        i2d = float(state[10])
        i2q = float(state[11])

        t = self._algebraic_tuple(state, p_ref_w, q_ref_var, v_grid_peak_v)
        omega, dv_hat = t[0], t[1]
        vfd, vfq = t[5], t[6]
        err_vd, err_vq = t[9], t[10]
        i1d_unsat, i1q_unsat, i1d_ref, i1q_ref = t[11], t[12], t[13], t[14]
        err_id, err_iq = t[15], t[16]
        vinv_d_unsat, vinv_q_unsat, vinv_d, vinv_q = t[17], t[18], t[19], t[20]
        di2d, di2q = t[21], t[22]
        cur_lim_f, v_lim_f = t[26], t[27]

        d_delta = omega - float(w_grid_rad_s)
        if cur_lim_f > 0.5:
            d_xivd = 0.0 if ((i1d_unsat - i1d_ref) * err_vd > 0.0) else err_vd
            d_xivq = 0.0 if ((i1q_unsat - i1q_ref) * err_vq > 0.0) else err_vq
        else:
            d_xivd = err_vd
            d_xivq = err_vq

        if v_lim_f > 0.5:
            d_xiid = 0.0 if ((vinv_d_unsat - vinv_d) * err_id > 0.0) else err_id
            d_xiiq = 0.0 if ((vinv_q_unsat - vinv_q) * err_iq > 0.0) else err_iq
        else:
            d_xiid = err_id
            d_xiiq = err_iq

        l1, r1, cf = self.lcl.l1_h, self.lcl.r1_ohm, self.lcl.cf_f
        di1d = (vinv_d - vfd - r1 * i1d + omega * l1 * i1q) / l1
        di1q = (vinv_q - vfq - r1 * i1q - omega * l1 * i1d) / l1
        dvcd = ((i1d - i2d) + omega * cf * vcq) / cf
        dvcq = ((i1q - i2q) - omega * cf * vcd) / cf

        return np.array(
            [
                d_delta,
                dv_hat,
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

    def linearized_eigenvalues(
        self,
        equilibrium_state: np.ndarray | None = None,
        *,
        p_ref_w: float = 0.0,
        q_ref_var: float = 0.0,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute the 12 small-signal eigenvalues of ``df/dx`` via central differences."""
        x0 = self.steady_initial_state() if equilibrium_state is None else np.array(equilibrium_state, dtype=float)
        jac = np.empty((self.STATE_DIM, self.STATE_DIM), dtype=float)
        vg = self.grid.v_peak_nom_v
        w0 = self.grid.w0_rad_s
        for col in range(self.STATE_DIM):
            xp = x0.copy()
            xm = x0.copy()
            xp[col] += eps
            xm[col] -= eps
            fp = self.rhs_continuous(xp, p_ref_w, q_ref_var, vg, w0)
            fm = self.rhs_continuous(xm, p_ref_w, q_ref_var, vg, w0)
            jac[:, col] = (fp - fm) / (2.0 * eps)
        return np.linalg.eigvals(jac)


def _compute_triad_metrics(
    *,
    mode: GFMMode,
    state_dim: int,
    grid: TheveninGridSpec,
    lcl: LCLFilterSpec,
    pi_cfg: CascadedPIConfig,
    time_s: np.ndarray,
    f_gfm_hz: np.ndarray,
    f_grid_hz: np.ndarray,
    delta_deg: np.ndarray,
    p_ref_w: np.ndarray,
    p_pcc_w: np.ndarray,
    q_pcc_var: np.ndarray,
    e_int_pu: np.ndarray,
    v_pcc_pu: np.ndarray,
    i1_peak_a: np.ndarray,
    i2_peak_a: np.ndarray,
    current_limited: np.ndarray,
    voltage_clamped: np.ndarray,
    eigenvalues: np.ndarray,
    p_step_time_s: float,
    p_step_w: float,
    grid_freq_drop_time_s: float,
) -> GFMTriadMemberResult:
    dt_s = float(time_s[1] - time_s[0])
    step_end_s = grid_freq_drop_time_s if grid_freq_drop_time_s > p_step_time_s else float(time_s[-1])
    mask_step = (time_s >= p_step_time_s - 1e-12) & (time_s < step_end_s - 1e-12)
    if np.any(mask_step) and p_step_w > 0.0:
        t_step = time_s[mask_step]
        p_step = p_pcc_w[mask_step]
        crossed = np.nonzero(p_step >= 0.90 * p_step_w)[0]
        rise_ms = float((t_step[int(crossed[0])] - p_step_time_s) * 1e3) if crossed.size > 0 else float("nan")
        step_pk_kw = float(np.max(p_step) / 1000.0)
    else:
        rise_ms = 0.0
        step_pk_kw = float(np.max(p_pcc_w) / 1000.0)

    mask_drop = time_s >= grid_freq_drop_time_s - 1e-12
    if np.count_nonzero(mask_drop) >= 3:
        f_drop = f_gfm_hz[mask_drop]
        p_drop = p_pcc_w[mask_drop]
        drop_pk_kw = float(np.max(p_drop) / 1000.0)
        peak_rocof = float(np.max(np.abs(np.diff(f_drop) / dt_s)))
        win_20ms = max(1, min(int(round(0.020 / dt_s)), f_drop.size - 1))
        init_rocof_20ms = float(abs(f_drop[win_20ms] - f_drop[0]) / (win_20ms * dt_s))
    else:
        drop_pk_kw = float(np.max(p_pcc_w) / 1000.0)
        peak_rocof = float(np.max(np.abs(np.diff(f_gfm_hz) / dt_s)))
        init_rocof_20ms = peak_rocof

    return GFMTriadMemberResult(
        mode=mode,
        state_dim=state_dim,
        grid=grid,
        lcl=lcl,
        pi_cfg=pi_cfg,
        time_s=time_s,
        f_gfm_hz=f_gfm_hz,
        f_grid_hz=f_grid_hz,
        delta_deg=delta_deg,
        p_ref_w=p_ref_w,
        p_pcc_w=p_pcc_w,
        q_pcc_var=q_pcc_var,
        e_int_pu=e_int_pu,
        v_pcc_pu=v_pcc_pu,
        i1_peak_a=i1_peak_a,
        i2_peak_a=i2_peak_a,
        current_limited=current_limited,
        voltage_clamped=voltage_clamped,
        eigenvalues=eigenvalues,
        max_real_eig_s_inv=float(np.max(eigenvalues.real)),
        step_rise_time_90_ms=rise_ms,
        step_peak_power_kw=step_pk_kw,
        freq_drop_peak_power_kw=drop_pk_kw,
        nadir_freq_hz=float(np.min(f_gfm_hz)),
        initial_drop_rocof_20ms_hz_s=init_rocof_20ms,
        peak_drop_rocof_hz_s=peak_rocof,
        final_power_kw=float(p_pcc_w[-1] / 1000.0),
    )


def simulate_gfm_mode_transient(
    mode: GFMMode,
    grid: TheveninGridSpec | None = None,
    lcl: LCLFilterSpec | None = None,
    pi_cfg: CascadedPIConfig | None = None,
    *,
    duration_s: float = 2.5,
    dt_s: float = 80.0e-6,
    p_step_time_s: float = 0.25,
    p_step_w: float = 10_000.0,
    grid_freq_drop_time_s: float = 1.15,
    grid_freq_drop_hz: float = -0.20,
) -> GFMTriadMemberResult:
    """Integrate ``mode`` in ``{'droop', 'vsg', 'dvoc'}`` on the electromagnetic dq LCL plant."""
    if mode not in ("droop", "vsg", "dvoc"):
        raise ValueError(f"Unsupported GFM mode {mode!r}; expected 'droop', 'vsg', or 'dvoc'")
    if duration_s <= 0.0 or dt_s <= 0.0 or dt_s > 2.5e-4:
        raise ValueError("duration_s must be > 0 and dt_s must lie in (0, 2.5e-4] s for LCL stability")

    cfg_grid = grid if grid is not None else TheveninGridSpec()
    cfg_lcl = lcl if lcl is not None else LCLFilterSpec()
    cfg_pi = pi_cfg if pi_cfg is not None else CascadedPIConfig.from_bandwidths(cfg_lcl)

    if mode == "vsg":
        model_vsg = ElectromagneticGFMModel(grid=cfg_grid, lcl=cfg_lcl, pi_cfg=cfg_pi)
        eigs = model_vsg.linearized_eigenvalues()
        n_steps = int(round(duration_s / dt_s)) + 1
        time_s = np.linspace(0.0, float(duration_s), n_steps)

        f_gfm_hz = np.empty(n_steps, dtype=float)
        f_grid_hz = np.empty(n_steps, dtype=float)
        delta_deg = np.empty(n_steps, dtype=float)
        p_ref_arr = np.empty(n_steps, dtype=float)
        p_pcc_w = np.empty(n_steps, dtype=float)
        q_pcc_var = np.empty(n_steps, dtype=float)
        e_int_pu = np.empty(n_steps, dtype=float)
        v_pcc_pu = np.empty(n_steps, dtype=float)
        i1_peak_a = np.empty(n_steps, dtype=float)
        i2_peak_a = np.empty(n_steps, dtype=float)
        cur_lim_arr = np.zeros(n_steps, dtype=bool)
        v_lim_arr = np.zeros(n_steps, dtype=bool)

        state = model_vsg.steady_initial_state()
        vg_nom = cfg_grid.v_peak_nom_v
        inv_2pi = 1.0 / (2.0 * math.pi)
        rhs = model_vsg.rhs_continuous
        alg_fn = model_vsg._algebraic_tuple

        for k, t_now in enumerate(time_s):
            t_val = float(t_now)
            p_cmd = p_step_w if t_val >= p_step_time_s - 1e-12 else 0.0
            f_g = cfg_grid.f_nom_hz + (grid_freq_drop_hz if t_val >= grid_freq_drop_time_s - 1e-12 else 0.0)
            w_g = 2.0 * math.pi * f_g
            alg = alg_fn(state, p_cmd, 0.0, vg_nom)

            e_raw = vg_nom - cfg_grid.nq_v_per_var * float(state[3])
            e_star = min(max(e_raw, 0.75 * vg_nom), 1.20 * vg_nom)

            delta_deg[k] = math.degrees(float(state[0]))
            f_gfm_hz[k] = float(state[1]) * inv_2pi
            f_grid_hz[k] = f_g
            p_ref_arr[k] = p_cmd
            p_pcc_w[k] = alg[20]
            q_pcc_var[k] = alg[21]
            e_int_pu[k] = e_star / vg_nom
            v_pcc_pu[k] = alg[22] / vg_nom
            i1_peak_a[k] = math.hypot(float(state[8]), float(state[9]))
            i2_peak_a[k] = math.hypot(float(state[12]), float(state[13]))
            cur_lim_arr[k] = alg[23] > 0.5
            v_lim_arr[k] = alg[24] > 0.5

            if k < n_steps - 1:
                k1 = rhs(state, p_cmd, 0.0, vg_nom, w_g)
                t_mid = t_val + 0.5 * dt_s
                p_mid = p_step_w if t_mid >= p_step_time_s - 1e-12 else 0.0
                fg_mid = cfg_grid.f_nom_hz + (grid_freq_drop_hz if t_mid >= grid_freq_drop_time_s - 1e-12 else 0.0)
                wg_mid = 2.0 * math.pi * fg_mid
                k2 = rhs(state + 0.5 * dt_s * k1, p_mid, 0.0, vg_nom, wg_mid)
                k3 = rhs(state + 0.5 * dt_s * k2, p_mid, 0.0, vg_nom, wg_mid)
                t_end = t_val + dt_s
                p_end = p_step_w if t_end >= p_step_time_s - 1e-12 else 0.0
                fg_end = cfg_grid.f_nom_hz + (grid_freq_drop_hz if t_end >= grid_freq_drop_time_s - 1e-12 else 0.0)
                wg_end = 2.0 * math.pi * fg_end
                k4 = rhs(state + dt_s * k3, p_end, 0.0, vg_nom, wg_end)
                state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return _compute_triad_metrics(
            mode="vsg",
            state_dim=model_vsg.STATE_DIM,
            grid=cfg_grid,
            lcl=cfg_lcl,
            pi_cfg=cfg_pi,
            time_s=time_s,
            f_gfm_hz=f_gfm_hz,
            f_grid_hz=f_grid_hz,
            delta_deg=delta_deg,
            p_ref_w=p_ref_arr,
            p_pcc_w=p_pcc_w,
            q_pcc_var=q_pcc_var,
            e_int_pu=e_int_pu,
            v_pcc_pu=v_pcc_pu,
            i1_peak_a=i1_peak_a,
            i2_peak_a=i2_peak_a,
            current_limited=cur_lim_arr,
            voltage_clamped=v_lim_arr,
            eigenvalues=eigs,
            p_step_time_s=p_step_time_s,
            p_step_w=p_step_w,
            grid_freq_drop_time_s=grid_freq_drop_time_s,
        )

    model: MultiLoopDroopGFMModel | MultiLoopDVOCGFMModel
    if mode == "droop":
        model = MultiLoopDroopGFMModel(grid=cfg_grid, lcl=cfg_lcl, pi_cfg=cfg_pi)
        i1_idx, i2_idx = 7, 11
        e_idx, p_idx, q_idx, vpcc_idx, clim_idx, vlim_idx = 1, 22, 23, 24, 25, 26
    else:
        model = MultiLoopDVOCGFMModel(grid=cfg_grid, lcl=cfg_lcl, pi_cfg=cfg_pi)
        i1_idx, i2_idx = 6, 10
        e_idx, p_idx, q_idx, vpcc_idx, clim_idx, vlim_idx = 2, 23, 24, 25, 26, 27

    eigs = model.linearized_eigenvalues()
    n_steps = int(round(duration_s / dt_s)) + 1
    time_s = np.linspace(0.0, float(duration_s), n_steps)

    f_gfm_hz = np.empty(n_steps, dtype=float)
    f_grid_hz = np.empty(n_steps, dtype=float)
    delta_deg = np.empty(n_steps, dtype=float)
    p_ref_arr = np.empty(n_steps, dtype=float)
    p_pcc_w = np.empty(n_steps, dtype=float)
    q_pcc_var = np.empty(n_steps, dtype=float)
    e_int_pu = np.empty(n_steps, dtype=float)
    v_pcc_pu = np.empty(n_steps, dtype=float)
    i1_peak_a = np.empty(n_steps, dtype=float)
    i2_peak_a = np.empty(n_steps, dtype=float)
    cur_lim_arr = np.zeros(n_steps, dtype=bool)
    v_lim_arr = np.zeros(n_steps, dtype=bool)

    state = model.steady_initial_state()
    vg_nom = cfg_grid.v_peak_nom_v
    inv_2pi = 1.0 / (2.0 * math.pi)
    rhs = model.rhs_continuous
    alg_fn = model._algebraic_tuple

    for k, t_now in enumerate(time_s):
        t_val = float(t_now)
        p_cmd = p_step_w if t_val >= p_step_time_s - 1e-12 else 0.0
        f_g = cfg_grid.f_nom_hz + (grid_freq_drop_hz if t_val >= grid_freq_drop_time_s - 1e-12 else 0.0)
        w_g = 2.0 * math.pi * f_g
        alg = alg_fn(state, p_cmd, 0.0, vg_nom)

        delta_deg[k] = math.degrees(float(state[0]))
        f_gfm_hz[k] = alg[0] * inv_2pi
        f_grid_hz[k] = f_g
        p_ref_arr[k] = p_cmd
        p_pcc_w[k] = alg[p_idx]
        q_pcc_var[k] = alg[q_idx]
        e_int_pu[k] = alg[e_idx] / vg_nom
        v_pcc_pu[k] = alg[vpcc_idx] / vg_nom
        i1_peak_a[k] = math.hypot(float(state[i1_idx]), float(state[i1_idx + 1]))
        i2_peak_a[k] = math.hypot(float(state[i2_idx]), float(state[i2_idx + 1]))
        cur_lim_arr[k] = alg[clim_idx] > 0.5
        v_lim_arr[k] = alg[vlim_idx] > 0.5

        if k < n_steps - 1:
            k1 = rhs(state, p_cmd, 0.0, vg_nom, w_g)
            t_mid = t_val + 0.5 * dt_s
            p_mid = p_step_w if t_mid >= p_step_time_s - 1e-12 else 0.0
            fg_mid = cfg_grid.f_nom_hz + (grid_freq_drop_hz if t_mid >= grid_freq_drop_time_s - 1e-12 else 0.0)
            wg_mid = 2.0 * math.pi * fg_mid
            k2 = rhs(state + 0.5 * dt_s * k1, p_mid, 0.0, vg_nom, wg_mid)
            k3 = rhs(state + 0.5 * dt_s * k2, p_mid, 0.0, vg_nom, wg_mid)
            t_end = t_val + dt_s
            p_end = p_step_w if t_end >= p_step_time_s - 1e-12 else 0.0
            fg_end = cfg_grid.f_nom_hz + (grid_freq_drop_hz if t_end >= grid_freq_drop_time_s - 1e-12 else 0.0)
            wg_end = 2.0 * math.pi * fg_end
            k4 = rhs(state + dt_s * k3, p_end, 0.0, vg_nom, wg_end)
            state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return _compute_triad_metrics(
        mode=mode,
        state_dim=model.STATE_DIM,
        grid=cfg_grid,
        lcl=cfg_lcl,
        pi_cfg=cfg_pi,
        time_s=time_s,
        f_gfm_hz=f_gfm_hz,
        f_grid_hz=f_grid_hz,
        delta_deg=delta_deg,
        p_ref_w=p_ref_arr,
        p_pcc_w=p_pcc_w,
        q_pcc_var=q_pcc_var,
        e_int_pu=e_int_pu,
        v_pcc_pu=v_pcc_pu,
        i1_peak_a=i1_peak_a,
        i2_peak_a=i2_peak_a,
        current_limited=cur_lim_arr,
        voltage_clamped=v_lim_arr,
        eigenvalues=eigs,
        p_step_time_s=p_step_time_s,
        p_step_w=p_step_w,
        grid_freq_drop_time_s=grid_freq_drop_time_s,
    )


def run_gfm_triad_benchmark(
    grid: TheveninGridSpec | None = None,
    lcl: LCLFilterSpec | None = None,
    pi_cfg: CascadedPIConfig | None = None,
    *,
    duration_s: float = 2.5,
    dt_s: float = 80.0e-6,
    p_step_time_s: float = 0.25,
    p_step_w: float = 10_000.0,
    grid_freq_drop_time_s: float = 1.15,
    grid_freq_drop_hz: float = -0.20,
) -> GFMTriadBenchmarkResult:
    """Execute the complete Droop (13) vs VSG (14) vs dVOC (12) comparative benchmark."""
    cfg_grid = grid if grid is not None else TheveninGridSpec()
    cfg_lcl = lcl if lcl is not None else LCLFilterSpec()
    cfg_pi = pi_cfg if pi_cfg is not None else CascadedPIConfig.from_bandwidths(cfg_lcl)

    common_kw = {
        "grid": cfg_grid,
        "lcl": cfg_lcl,
        "pi_cfg": cfg_pi,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "p_step_time_s": p_step_time_s,
        "p_step_w": p_step_w,
        "grid_freq_drop_time_s": grid_freq_drop_time_s,
        "grid_freq_drop_hz": grid_freq_drop_hz,
    }
    droop_res = simulate_gfm_mode_transient("droop", **common_kw)
    vsg_res = simulate_gfm_mode_transient("vsg", **common_kw)
    dvoc_res = simulate_gfm_mode_transient("dvoc", **common_kw)
    osc = DispatchableVirtualOscillator(DVOCConfig.from_grid_spec(cfg_grid))
    black_start = osc.simulate_black_start_excitation(v_init_pu=0.10, duration_s=0.35, dt_s=40.0e-6)

    return GFMTriadBenchmarkResult(
        grid=cfg_grid,
        lcl=cfg_lcl,
        pi_cfg=cfg_pi,
        droop=droop_res,
        vsg=vsg_res,
        dvoc=dvoc_res,
        black_start=black_start,
    )
