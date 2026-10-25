"""Small-Signal State-Space Modal Participation, dq Admittance Passivity & Generalized Nyquist.

Unifies time-domain state-space eigenvalue analysis and frequency-domain terminal impedance /
admittance stability theory across Grid-Forming (VSG, Droop, dVOC) and Grid-Following
(SRF-PLL) converters on the 6-state electromagnetic ``dq`` LCL + Thévenin network plant
(Pérez-Arriaga et al., *IEEE TPAS*, 1982; Harnefors et al., *IEEE TIE*, 2007 & *IEEE TPEL*,
2015; Wen et al., *IEEE TPEL*, 2016; Wang & Blaabjerg, *IEEE TPEL*, 2019; Qoria et al.,
*IEEE TPEL*, 2020):

  1. **Normalized Participation Factors (``compute_modal_participation``):**
     Decomposes every right/left eigenpair ``(lambda_i, v_i, w_i)`` of the coupled state
     matrix ``A`` into exact dimensionless state participation ``P_{ki} = |w_{ki} v_{ki}| /
     sum_m |w_{mi} v_{mi}|`` and physical subsystem shares (``sync_outer``, ``voltage_pi``,
     and ``current_pi`` / ``lcl_network``).
  2. **Exact Analytical State-Space Synchronous-dq Input Admittance ``Y_inv,dq(s)``:**
     Cuts the converter-network loop at the PCC bus in the synchronous grid ``dq`` frame
     (``Delta i_in,dq^g = -Delta i_2,dq^g = Y_inv,dq(s) Delta v_pcc,dq^g``) around the exact
     loaded PCC operating point, satisfying the Schur-complement pole identity:
     det(I_2 + Z_g,dq(lambda_i) Y_inv,dq(lambda_i)) = 0 at every coupled eigenvalue.
  3. **Passivity Index ``nu(f)`` & Generalized Nyquist Cauchy Winding (``Z = P + N``):**
     Evaluates the Hermitian passivity index ``nu(f) = lambda_min(0.5 * [Y_dq + Y_dq^H])``,
     incremental ``q``-axis conductance ``Re{Y_qq(j 2 pi f)}``, and unwrapped Cauchy argument
     winding of ``det(I_2 + Z_g,dq(j 2 pi f) Y_inv,dq(j 2 pi f))`` to recover the exact
     closed-loop RHP pole count ``Z = P + N``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import eig, eigvals, eigvalsh
from scipy.optimize import root

from .cascaded_control import CascadedPIConfig, ElectromagneticGFMModel
from .gfl_pll import ElectromagneticGFLModel, SRFPLLConfig
from .gfm_triad import MultiLoopDroopGFMModel, MultiLoopDVOCGFMModel
from .lcl_plant import LCLFilterSpec
from .thevenin_network import TheveninGridSpec

ImpedanceArchitecture = Literal[
    "vsg",
    "vsg_unity_ff",
    "droop",
    "dvoc",
    "gfl_pll_10hz",
    "gfl_pll_35hz",
    "gfl_pll_55hz",
]

VSG_STATE_NAMES: tuple[str, ...] = (
    "delta",
    "omega",
    "P_filt",
    "Q_filt",
    "xi_vd",
    "xi_vq",
    "xi_id",
    "xi_iq",
    "i_1d",
    "i_1q",
    "v_cd",
    "v_cq",
    "i_2d",
    "i_2q",
)

SUBSYSTEM_NAMES: tuple[str, ...] = (
    "sync_outer",
    "voltage_pi",
    "current_pi",
    "lcl_network",
)


@dataclass(frozen=True)
class OscillatoryModeSummary:
    """Eigenvalue, damping ratio, and subsystem participation breakdown for one mode/pair."""

    label: str
    eigenvalue: complex
    conjugate_pair: bool
    freq_hz: float
    damping_ratio: float
    dominant_state: str
    dominant_state_share_pct: float
    sync_outer_pct: float
    voltage_pi_pct: float
    current_pi_pct: float
    lcl_network_pct: float


@dataclass(frozen=True)
class ModalParticipationResult:
    """Complete coupled state-space modal decomposition and participation matrix."""

    architecture: str
    scr: float
    p_ref_w: float
    state_names: tuple[str, ...]
    a_matrix: np.ndarray
    eigenvalues: np.ndarray
    participation_matrix: np.ndarray
    modes: tuple[OscillatoryModeSummary, ...]


@dataclass(frozen=True)
class DQAdmittanceSweepResult:
    """Synchronous-dq input admittance, passivity index & Generalized Nyquist sweep."""

    architecture: ImpedanceArchitecture
    label: str
    scr: float
    p_ref_w: float
    freqs_hz: np.ndarray
    y_dq_s: np.ndarray
    passivity_index_s: np.ndarray
    re_ydd_s: np.ndarray
    re_yqq_s: np.ndarray
    det_return_diff: np.ndarray
    cumulative_rhp_index: np.ndarray
    min_passivity_high_freq_s: float
    nonpassive_band_hz: tuple[float, float] | None
    neg_re_yqq_band_hz: tuple[float, float] | None
    open_loop_rhp_poles_p: int
    nyquist_encirclements_n: int
    nyquist_rhp_poles_z: int
    state_space_rhp_poles: int
    coupled_eigenvalues: np.ndarray
    max_schur_det_residual: float


@dataclass(frozen=True)
class SmallSignalImpedanceBenchmarkResult:
    """Complete Module 6 modal participation, dq passivity & Nyquist benchmark bundle."""

    scr_operating: float
    p_ref_w: float
    vsg_participation: ModalParticipationResult
    locus_scr_values: np.ndarray
    locus_vsg_eigs: np.ndarray
    locus_dvoc_eigs: np.ndarray
    locus_gfl35_eigs: np.ndarray
    locus_gfl55_eigs: np.ndarray
    admittance_vsg: DQAdmittanceSweepResult
    admittance_vsg_unity_ff: DQAdmittanceSweepResult
    admittance_dvoc: DQAdmittanceSweepResult
    admittance_gfl10: DQAdmittanceSweepResult
    admittance_gfl35: DQAdmittanceSweepResult
    admittance_gfl55: DQAdmittanceSweepResult


def _central_jacobian(fn, x0: np.ndarray, eps: float = 1.0e-7) -> np.ndarray:
    x_base = np.asarray(x0, dtype=float)
    f0 = np.asarray(fn(x_base), dtype=float)
    n = x_base.size
    jac = np.empty((f0.size, n), dtype=float)
    for j in range(n):
        step = eps * max(1.0, abs(float(x_base[j])))
        dx = np.zeros(n, dtype=float)
        dx[j] = step
        fp = np.asarray(fn(x_base + dx), dtype=float)
        fm = np.asarray(fn(x_base - dx), dtype=float)
        jac[:, j] = (fp - fm) / (2.0 * step)
    return jac


def grid_thevenin_impedance_dq(grid: TheveninGridSpec, s_val: complex) -> np.ndarray:
    """2x2 synchronous-dq Thévenin grid impedance ``Z_g,dq(s)`` at Laplace variable ``s``."""
    rg = grid.r_g_ohm
    lg = grid.l_g_h
    w0 = grid.w0_rad_s
    diag = rg + complex(s_val) * lg
    off = w0 * lg
    return np.array([[diag, -off], [off, diag]], dtype=complex)


def _solve_architecture_setup(
    architecture: ImpedanceArchitecture,
    *,
    scr: float,
    p_ref_w: float,
    xr_ratio: float = 7.0,
):
    if scr <= 0.0 or not math.isfinite(scr):
        raise ValueError("scr must be finite and > 0")
    if not math.isfinite(p_ref_w):
        raise ValueError("p_ref_w must be finite")

    grid = TheveninGridSpec(scr=float(scr), xr_ratio=float(xr_ratio))
    lcl = LCLFilterSpec()
    v0 = grid.v_peak_nom_v
    w0 = grid.w0_rad_s
    p_cmd = float(p_ref_w)

    if architecture in ("vsg", "vsg_unity_ff"):
        fi = 1.00 if architecture == "vsg_unity_ff" else 0.75
        label = (
            "GFM VSG (Naive F_i = 1.00)"
            if architecture == "vsg_unity_ff"
            else "GFM VSG (Qoria F_i = 0.75)"
        )
        pi_cfg = CascadedPIConfig.from_bandwidths(lcl, feedforward_current_gain=fi)
        model = ElectromagneticGFMModel(grid=grid, lcl=lcl, pi_cfg=pi_cfg)
        y_seed = model.steady_initial_state()
        y_seed[0] = min(0.65, max(0.0, p_cmd / (grid.s_rated_va * grid.scr)))
        sol = root(lambda y: model.rhs_continuous(y, p_cmd, 0.0, v0, w0), y_seed, tol=1.0e-12)
        if not sol.success or float(np.linalg.norm(model.rhs_continuous(sol.x, p_cmd, 0.0, v0, w0))) > 1.0e-7:
            raise ValueError(f"Equilibrium failed for {architecture} at SCR={scr}")
        x_eq = np.asarray(sol.x, dtype=float)
        def coupled_fn(y: np.ndarray) -> np.ndarray:
            return model.rhs_continuous(y, p_cmd, 0.0, v0, w0)
        idx_i2 = (12, 13)

        def pcc_rhs(state: np.ndarray, vd_g: float, vq_g: float) -> np.ndarray:
            delta, omega, p_filt, q_filt = (float(state[k]) for k in range(4))
            xi_vd, xi_vq, xi_id, xi_iq = (float(state[k]) for k in range(4, 8))
            i1d, i1q, vcd, vcq, i2d, i2q = (float(state[k]) for k in range(8, 14))
            c_d, s_d = math.cos(delta), math.sin(delta)
            vpcc_d = vd_g * c_d + vq_g * s_d
            vpcc_q = -vd_g * s_d + vq_g * c_d
            e_star = min(max(v0 - grid.nq_v_per_var * q_filt, 0.75 * v0), 1.20 * v0)
            vf_d_ref = e_star - (grid.r_v_ohm * i2d - omega * grid.l_v_h * i2q)
            vf_q_ref = -(grid.r_v_ohm * i2q + omega * grid.l_v_h * i2d)
            vfd = vcd + lcl.rd_ohm * (i1d - i2d)
            vfq = vcq + lcl.rd_ohm * (i1q - i2q)
            err_vd, err_vq = vf_d_ref - vfd, vf_q_ref - vfq
            i1d_r = pi_cfg.kp_v * err_vd + pi_cfg.ki_v * xi_vd + fi * i2d - omega * lcl.cf_f * vfq
            i1q_r = pi_cfg.kp_v * err_vq + pi_cfg.ki_v * xi_vq + fi * i2q + omega * lcl.cf_f * vfd
            err_id, err_iq = i1d_r - i1d, i1q_r - i1q
            vid = pi_cfg.kp_i * err_id + pi_cfg.ki_i * xi_id + vfd - omega * lcl.l1_h * i1q
            viq = pi_cfg.kp_i * err_iq + pi_cfg.ki_i * xi_iq + vfq + omega * lcl.l1_h * i1d
            di1d = (vid - vfd - lcl.r1_ohm * i1d + omega * lcl.l1_h * i1q) / lcl.l1_h
            di1q = (viq - vfq - lcl.r1_ohm * i1q - omega * lcl.l1_h * i1d) / lcl.l1_h
            dvcd = ((i1d - i2d) + omega * lcl.cf_f * vcq) / lcl.cf_f
            dvcq = ((i1q - i2q) - omega * lcl.cf_f * vcd) / lcl.cf_f
            di2d = (vfd - vpcc_d - lcl.r2_ohm * i2d + omega * lcl.l2_h * i2q) / lcl.l2_h
            di2q = (vfq - vpcc_q - lcl.r2_ohm * i2q - omega * lcl.l2_h * i2d) / lcl.l2_h
            p_pcc = 1.5 * (vpcc_d * i2d + vpcc_q * i2q)
            q_pcc = 1.5 * (vpcc_q * i2d - vpcc_d * i2q)
            wc = 2.0 * math.pi * grid.lpf_cutoff_hz
            return np.array(
                [
                    omega - w0,
                    (p_cmd - p_filt - grid.dp_nms_rad * w0 * (omega - w0)) / (grid.j_kgm2 * w0),
                    wc * (p_pcc - p_filt),
                    wc * (q_pcc - q_filt),
                    err_vd,
                    err_vq,
                    err_id,
                    err_iq,
                    di1d,
                    di1q,
                    dvcd,
                    dvcq,
                    di2d,
                    di2q,
                ],
                dtype=float,
            )

        return grid, x_eq, coupled_fn, pcc_rhs, idx_i2, label

    if architecture == "droop":
        pi_cfg = CascadedPIConfig.from_bandwidths(lcl)
        model_d = MultiLoopDroopGFMModel(grid=grid, lcl=lcl, pi_cfg=pi_cfg)
        y_seed = model_d.steady_initial_state()
        y_seed[0] = min(0.65, max(0.0, p_cmd / (grid.s_rated_va * grid.scr)))
        sol = root(lambda y: model_d.rhs_continuous(y, p_cmd, 0.0, v0, w0), y_seed, tol=1.0e-12)
        if not sol.success or float(np.linalg.norm(model_d.rhs_continuous(sol.x, p_cmd, 0.0, v0, w0))) > 1.0e-7:
            raise ValueError(f"Equilibrium failed for droop at SCR={scr}")
        x_eq = np.asarray(sol.x, dtype=float)
        def coupled_fn(y: np.ndarray) -> np.ndarray:
            return model_d.rhs_continuous(y, p_cmd, 0.0, v0, w0)
        idx_i2 = (11, 12)
        fi = pi_cfg.feedforward_current_gain

        def pcc_rhs_d(state: np.ndarray, vd_g: float, vq_g: float) -> np.ndarray:
            delta, p_filt, q_filt = float(state[0]), float(state[1]), float(state[2])
            xi_vd, xi_vq, xi_id, xi_iq = (float(state[k]) for k in range(3, 7))
            i1d, i1q, vcd, vcq, i2d, i2q = (float(state[k]) for k in range(7, 13))
            omega = w0 + model_d.mp_rad_per_ws * (p_cmd - p_filt)
            c_d, s_d = math.cos(delta), math.sin(delta)
            vpcc_d = vd_g * c_d + vq_g * s_d
            vpcc_q = -vd_g * s_d + vq_g * c_d
            e_star = min(max(v0 - grid.nq_v_per_var * q_filt, 0.75 * v0), 1.20 * v0)
            vf_d_ref = e_star - (grid.r_v_ohm * i2d - omega * grid.l_v_h * i2q)
            vf_q_ref = -(grid.r_v_ohm * i2q + omega * grid.l_v_h * i2d)
            vfd = vcd + lcl.rd_ohm * (i1d - i2d)
            vfq = vcq + lcl.rd_ohm * (i1q - i2q)
            err_vd, err_vq = vf_d_ref - vfd, vf_q_ref - vfq
            i1d_r = pi_cfg.kp_v * err_vd + pi_cfg.ki_v * xi_vd + fi * i2d - omega * lcl.cf_f * vfq
            i1q_r = pi_cfg.kp_v * err_vq + pi_cfg.ki_v * xi_vq + fi * i2q + omega * lcl.cf_f * vfd
            err_id, err_iq = i1d_r - i1d, i1q_r - i1q
            vid = pi_cfg.kp_i * err_id + pi_cfg.ki_i * xi_id + vfd - omega * lcl.l1_h * i1q
            viq = pi_cfg.kp_i * err_iq + pi_cfg.ki_i * xi_iq + vfq + omega * lcl.l1_h * i1d
            di1d = (vid - vfd - lcl.r1_ohm * i1d + omega * lcl.l1_h * i1q) / lcl.l1_h
            di1q = (viq - vfq - lcl.r1_ohm * i1q - omega * lcl.l1_h * i1d) / lcl.l1_h
            dvcd = ((i1d - i2d) + omega * lcl.cf_f * vcq) / lcl.cf_f
            dvcq = ((i1q - i2q) - omega * lcl.cf_f * vcd) / lcl.cf_f
            di2d = (vfd - vpcc_d - lcl.r2_ohm * i2d + omega * lcl.l2_h * i2q) / lcl.l2_h
            di2q = (vfq - vpcc_q - lcl.r2_ohm * i2q - omega * lcl.l2_h * i2d) / lcl.l2_h
            p_pcc = 1.5 * (vpcc_d * i2d + vpcc_q * i2q)
            q_pcc = 1.5 * (vpcc_q * i2d - vpcc_d * i2q)
            wc = 2.0 * math.pi * grid.lpf_cutoff_hz
            return np.array(
                [
                    omega - w0,
                    wc * (p_pcc - p_filt),
                    wc * (q_pcc - q_filt),
                    err_vd,
                    err_vq,
                    err_id,
                    err_iq,
                    di1d,
                    di1q,
                    dvcd,
                    dvcq,
                    di2d,
                    di2q,
                ],
                dtype=float,
            )

        return grid, x_eq, coupled_fn, pcc_rhs_d, idx_i2, "GFM Droop (13-State)"

    if architecture == "dvoc":
        pi_cfg = CascadedPIConfig.from_bandwidths(lcl)
        model_o = MultiLoopDVOCGFMModel(grid=grid, lcl=lcl, pi_cfg=pi_cfg)
        y_seed = model_o.steady_initial_state()
        y_seed[0] = min(0.65, max(0.0, p_cmd / (grid.s_rated_va * grid.scr)))
        sol = root(lambda y: model_o.rhs_continuous(y, p_cmd, 0.0, v0, w0), y_seed, tol=1.0e-12)
        if not sol.success or float(np.linalg.norm(model_o.rhs_continuous(sol.x, p_cmd, 0.0, v0, w0))) > 1.0e-7:
            raise ValueError(f"Equilibrium failed for dvoc at SCR={scr}")
        x_eq = np.asarray(sol.x, dtype=float)
        def coupled_fn(y: np.ndarray) -> np.ndarray:
            return model_o.rhs_continuous(y, p_cmd, 0.0, v0, w0)
        idx_i2 = (10, 11)
        fi = pi_cfg.feedforward_current_gain

        def pcc_rhs_o(state: np.ndarray, vd_g: float, vq_g: float) -> np.ndarray:
            delta, v_hat = float(state[0]), float(state[1])
            xi_vd, xi_vq, xi_id, xi_iq = (float(state[k]) for k in range(2, 6))
            i1d, i1q, vcd, vcq, i2d, i2q = (float(state[k]) for k in range(6, 12))
            c_d, s_d = math.cos(delta), math.sin(delta)
            vpcc_d = vd_g * c_d + vq_g * s_d
            vpcc_q = -vd_g * s_d + vq_g * c_d
            p_osc = 1.5 * v_hat * i2d
            q_osc = -1.5 * v_hat * i2q
            omega, dv_hat = model_o.oscillator.polar_kinetics(
                v_hat_v=v_hat,
                p_elec_w=p_osc,
                q_elec_var=q_osc,
                p_ref_w=p_cmd,
                q_ref_var=0.0,
            )
            e_star = min(max(v_hat, 0.75 * v0), 1.20 * v0)
            vf_d_ref = e_star - (grid.r_v_ohm * i2d - omega * grid.l_v_h * i2q)
            vf_q_ref = -(grid.r_v_ohm * i2q + omega * grid.l_v_h * i2d)
            vfd = vcd + lcl.rd_ohm * (i1d - i2d)
            vfq = vcq + lcl.rd_ohm * (i1q - i2q)
            err_vd, err_vq = vf_d_ref - vfd, vf_q_ref - vfq
            i1d_r = pi_cfg.kp_v * err_vd + pi_cfg.ki_v * xi_vd + fi * i2d - omega * lcl.cf_f * vfq
            i1q_r = pi_cfg.kp_v * err_vq + pi_cfg.ki_v * xi_vq + fi * i2q + omega * lcl.cf_f * vfd
            err_id, err_iq = i1d_r - i1d, i1q_r - i1q
            vid = pi_cfg.kp_i * err_id + pi_cfg.ki_i * xi_id + vfd - omega * lcl.l1_h * i1q
            viq = pi_cfg.kp_i * err_iq + pi_cfg.ki_i * xi_iq + vfq + omega * lcl.l1_h * i1d
            di1d = (vid - vfd - lcl.r1_ohm * i1d + omega * lcl.l1_h * i1q) / lcl.l1_h
            di1q = (viq - vfq - lcl.r1_ohm * i1q - omega * lcl.l1_h * i1d) / lcl.l1_h
            dvcd = ((i1d - i2d) + omega * lcl.cf_f * vcq) / lcl.cf_f
            dvcq = ((i1q - i2q) - omega * lcl.cf_f * vcd) / lcl.cf_f
            di2d = (vfd - vpcc_d - lcl.r2_ohm * i2d + omega * lcl.l2_h * i2q) / lcl.l2_h
            di2q = (vfq - vpcc_q - lcl.r2_ohm * i2q - omega * lcl.l2_h * i2d) / lcl.l2_h
            return np.array(
                [
                    omega - w0,
                    dv_hat,
                    err_vd,
                    err_vq,
                    err_id,
                    err_iq,
                    di1d,
                    di1q,
                    dvcd,
                    dvcq,
                    di2d,
                    di2q,
                ],
                dtype=float,
            )

        return grid, x_eq, coupled_fn, pcc_rhs_o, idx_i2, "GFM Andronov-Hopf dVOC (12-State)"

    bw_map = {"gfl_pll_10hz": 10.0, "gfl_pll_35hz": 35.0, "gfl_pll_55hz": 55.0}
    if architecture in bw_map:
        bw_hz = bw_map[architecture]
        pi_cfg = CascadedPIConfig.from_bandwidths(lcl)
        pll_cfg = SRFPLLConfig.from_grid_spec(grid, f_bw_hz=bw_hz)
        model_g = ElectromagneticGFLModel(grid=grid, lcl=lcl, pi_cfg=pi_cfg, pll_cfg=pll_cfg)
        x_eq_opt, ok = model_g.solve_loaded_equilibrium(p_cmd)
        if not ok or x_eq_opt is None:
            raise ValueError(f"GFL equilibrium failed for {architecture} at SCR={scr}")
        x_eq = x_eq_opt
        def coupled_fn(y: np.ndarray) -> np.ndarray:
            return model_g.rhs_continuous(y, p_cmd, 0.0, v0, w0)
        idx_i2 = (9, 10)

        def pcc_rhs_g(state: np.ndarray, vd_g: float, vq_g: float) -> np.ndarray:
            delta, xi_pll, vd_f, xi_id, xi_iq = (float(state[k]) for k in range(5))
            i1d, i1q, vcd, vcq, i2d, i2q = (float(state[k]) for k in range(5, 11))
            c_d, s_d = math.cos(delta), math.sin(delta)
            vpcc_d = vd_g * c_d + vq_g * s_d
            vpcc_q = -vd_g * s_d + vq_g * c_d
            vfd = vcd + lcl.rd_ohm * (i1d - i2d)
            vfq = vcq + lcl.rd_ohm * (i1q - i2q)
            vq_pu = vfq / pll_cfg.v0_peak_v
            w_pll = pll_cfg.w0_rad_s + pll_cfg.kp_rad_s * vq_pu + pll_cfg.ki_rad_s2 * xi_pll
            v_denom = max(vd_f, 0.35 * pll_cfg.v0_peak_v)
            i1d_r = p_cmd / (1.5 * v_denom)
            i1q_r = -model_g.k_qv_a_per_v * (vfd - pll_cfg.v0_peak_v)
            err_id, err_iq = i1d_r - i1d, i1q_r - i1q
            vid = pi_cfg.kp_i * err_id + pi_cfg.ki_i * xi_id + vfd - w_pll * lcl.l1_h * i1q
            viq = pi_cfg.kp_i * err_iq + pi_cfg.ki_i * xi_iq + vfq + w_pll * lcl.l1_h * i1d
            di1d = (vid - vfd - lcl.r1_ohm * i1d + w_pll * lcl.l1_h * i1q) / lcl.l1_h
            di1q = (viq - vfq - lcl.r1_ohm * i1q - w_pll * lcl.l1_h * i1d) / lcl.l1_h
            dvcd = ((i1d - i2d) + w_pll * lcl.cf_f * vcq) / lcl.cf_f
            dvcq = ((i1q - i2q) - w_pll * lcl.cf_f * vcd) / lcl.cf_f
            di2d = (vfd - vpcc_d - lcl.r2_ohm * i2d + w_pll * lcl.l2_h * i2q) / lcl.l2_h
            di2q = (vfq - vpcc_q - lcl.r2_ohm * i2q - w_pll * lcl.l2_h * i2d) / lcl.l2_h
            wc_v = 2.0 * math.pi * pll_cfg.vd_filter_hz
            return np.array(
                [
                    w_pll - w0,
                    vq_pu,
                    wc_v * (vfd - vd_f),
                    err_id,
                    err_iq,
                    di1d,
                    di1q,
                    dvcd,
                    dvcq,
                    di2d,
                    di2q,
                ],
                dtype=float,
            )

        return grid, x_eq, coupled_fn, pcc_rhs_g, idx_i2, f"GFL SRF-PLL ({bw_hz:.0f} Hz)"

    raise ValueError(f"Unsupported ImpedanceArchitecture: {architecture!r}")


def compute_modal_participation(
    *,
    scr: float = 2.0,
    p_ref_w: float = 15_000.0,
    xr_ratio: float = 7.0,
    feedforward_current_gain: float = 0.75,
) -> ModalParticipationResult:
    """Compute exact 14-state VSG eigenvalues, participation matrix ``P``, and subsystem shares."""
    if scr <= 0.0 or not math.isfinite(scr):
        raise ValueError("scr must be finite and > 0")
    grid = TheveninGridSpec(scr=float(scr), xr_ratio=float(xr_ratio))
    lcl = LCLFilterSpec()
    pi_cfg = CascadedPIConfig.from_bandwidths(lcl, feedforward_current_gain=float(feedforward_current_gain))
    model = ElectromagneticGFMModel(grid=grid, lcl=lcl, pi_cfg=pi_cfg)
    v0, w0 = grid.v_peak_nom_v, grid.w0_rad_s

    y_seed = model.steady_initial_state()
    y_seed[0] = min(0.65, max(0.0, float(p_ref_w) / (grid.s_rated_va * grid.scr)))
    sol = root(lambda y: model.rhs_continuous(y, float(p_ref_w), 0.0, v0, w0), y_seed, tol=1.0e-12)
    if not sol.success:
        raise ValueError(f"VSG equilibrium failed at SCR={scr}, P_ref={p_ref_w}")

    a_mat = _central_jacobian(lambda y: model.rhs_continuous(y, float(p_ref_w), 0.0, v0, w0), sol.x)
    vals, vl, vr = eig(a_mat, left=True, right=True)
    n = len(vals)
    p_mat = np.zeros((n, n), dtype=float)
    for col in range(n):
        raw = np.abs(np.conj(vl[:, col]) * vr[:, col])
        denom = float(np.sum(raw))
        p_mat[:, col] = raw / denom if denom > 0.0 else 0.0

    # Group distinct real modes and upper-half-plane representatives of conjugate pairs
    used = np.zeros(n, dtype=bool)
    order = np.argsort(np.abs(vals))
    summaries: list[OscillatoryModeSummary] = []

    for idx in order:
        if used[idx]:
            continue
        lam = complex(vals[idx])
        used[idx] = True
        is_pair = False
        if abs(lam.imag) > 1.0e-5:
            # Find conjugate partner
            partner = None
            best_d = 1.0e9
            for k in range(n):
                if not used[k]:
                    d = abs(vals[k] - np.conj(lam))
                    if d < best_d:
                        best_d = d
                        partner = k
            if partner is not None and best_d < 1.0e-2 * max(1.0, abs(lam)):
                used[partner] = True
                is_pair = True
                if lam.imag < 0.0:
                    lam = complex(vals[partner])
                    idx = partner

        p_col = p_mat[:, idx]
        dom_idx = int(np.argmax(p_col))
        dom_name = VSG_STATE_NAMES[dom_idx]
        f_hz = abs(lam.imag) / (2.0 * math.pi)
        zeta = -lam.real / abs(lam) if abs(lam) > 0.0 else 1.0
        sync_pct = float(100.0 * np.sum(p_col[0:4]))
        vpi_pct = float(100.0 * np.sum(p_col[4:6]))
        ipi_pct = float(100.0 * np.sum(p_col[6:8]))
        lcl_pct = float(100.0 * np.sum(p_col[8:14]))

        if is_pair:
            lbl = f"{f_hz:.1f} Hz ({dom_name})"
        else:
            lbl = f"Real {lam.real:.0f} s^-1 ({dom_name})"

        summaries.append(
            OscillatoryModeSummary(
                label=lbl,
                eigenvalue=lam,
                conjugate_pair=is_pair,
                freq_hz=float(f_hz),
                damping_ratio=float(zeta),
                dominant_state=dom_name,
                dominant_state_share_pct=float(100.0 * p_col[dom_idx]),
                sync_outer_pct=sync_pct,
                voltage_pi_pct=vpi_pct,
                current_pi_pct=ipi_pct,
                lcl_network_pct=lcl_pct,
            )
        )

    return ModalParticipationResult(
        architecture=f"vsg_fi_{feedforward_current_gain:.2f}",
        scr=float(scr),
        p_ref_w=float(p_ref_w),
        state_names=VSG_STATE_NAMES,
        a_matrix=a_mat,
        eigenvalues=vals,
        participation_matrix=p_mat,
        modes=tuple(summaries),
    )


def evaluate_dq_admittance_and_nyquist(
    architecture: ImpedanceArchitecture,
    *,
    scr: float = 1.65,
    p_ref_w: float = 15_000.0,
    xr_ratio: float = 7.0,
    freqs_hz: np.ndarray | None = None,
    high_freq_passivity_min_hz: float = 25.0,
) -> DQAdmittanceSweepResult:
    """Compute exact state-space ``Y_inv,dq(j 2 pi f)``, passivity index & Nyquist winding."""
    grid, x_eq, coupled_fn, pcc_rhs, (id_idx, iq_idx), label = _solve_architecture_setup(
        architecture,
        scr=scr,
        p_ref_w=p_ref_w,
        xr_ratio=xr_ratio,
    )
    freqs = (
        np.geomspace(0.2, 3500.0, 480)
        if freqs_hz is None
        else np.asarray(freqs_hz, dtype=float)
    )
    if freqs.ndim != 1 or freqs.size < 8 or np.any(freqs <= 0.0):
        raise ValueError("freqs_hz must be a 1-D strictly positive array with >= 8 points")

    v0, w0 = grid.v_peak_nom_v, grid.w0_rad_s
    d0 = float(x_eq[0])
    i2d0, i2q0 = float(x_eq[id_idx]), float(x_eq[iq_idx])
    i2d_g0 = i2d0 * math.cos(d0) - i2q0 * math.sin(d0)
    i2q_g0 = i2d0 * math.sin(d0) + i2q0 * math.cos(d0)
    vpcc_d_g0 = v0 + grid.r_g_ohm * i2d_g0 - w0 * grid.l_g_h * i2q_g0
    vpcc_q_g0 = w0 * grid.l_g_h * i2d_g0 + grid.r_g_ohm * i2q_g0

    a_coupled = _central_jacobian(coupled_fn, x_eq)
    coupled_eigs = eigvals(a_coupled)
    ss_rhp = int(np.sum(coupled_eigs.real > 1.0e-5))

    a0 = _central_jacobian(lambda y: pcc_rhs(y, vpcc_d_g0, vpcc_q_g0), x_eq)
    b0 = _central_jacobian(
        lambda u: pcc_rhs(x_eq, float(u[0]), float(u[1])),
        np.array([vpcc_d_g0, vpcc_q_g0], dtype=float),
    )

    def port_current_in(state: np.ndarray) -> np.ndarray:
        ang = float(state[0])
        id_v, iq_v = float(state[id_idx]), float(state[iq_idx])
        return np.array(
            [
                -(id_v * math.cos(ang) - iq_v * math.sin(ang)),
                -(id_v * math.sin(ang) + iq_v * math.cos(ang)),
            ],
            dtype=float,
        )

    c0 = _central_jacobian(port_current_in, x_eq)
    n_dim = x_eq.size
    eye_n = np.eye(n_dim, dtype=complex)
    eye_2 = np.eye(2, dtype=complex)

    ol_eigs = eigvals(a0)
    ol_rhp_p = int(np.sum(ol_eigs.real > 1.0e-5))

    # Verify Schur-complement zero determinant at coupled eigenvalues
    max_schur_err = 0.0
    for lam in coupled_eigs:
        if np.min(np.abs(ol_eigs - lam)) < 1.0e-3:
            continue
        y_lam = c0 @ np.linalg.solve(complex(lam) * eye_n - a0, b0)
        m_lam = eye_2 + grid_thevenin_impedance_dq(grid, complex(lam)) @ y_lam
        svals = np.linalg.svd(m_lam, compute_uv=False)
        max_schur_err = max(max_schur_err, float(svals[-1]))

    n_f = freqs.size
    y_dq = np.empty((n_f, 2, 2), dtype=complex)
    nu_arr = np.empty(n_f, dtype=float)
    re_ydd = np.empty(n_f, dtype=float)
    re_yqq = np.empty(n_f, dtype=float)
    det_rd = np.empty(n_f, dtype=complex)

    for k, f_val in enumerate(freqs):
        s_jw = 1j * 2.0 * math.pi * float(f_val)
        yk = c0 @ np.linalg.solve(s_jw * eye_n - a0, b0)
        y_dq[k] = yk
        herm = 0.5 * (yk + yk.conj().T)
        nu_arr[k] = float(eigvalsh(herm)[0])
        re_ydd[k] = float(yk[0, 0].real)
        re_yqq[k] = float(yk[1, 1].real)
        zg = grid_thevenin_impedance_dq(grid, s_jw)
        det_rd[k] = complex(np.linalg.det(eye_2 + zg @ yk))

    # Unwrapped Cauchy phase winding over (-inf, +inf) via conjugate symmetry:
    # Delta arg_{(0+, +inf)} * 2; clockwise encirclements N = -Delta arg_{total} / (2 pi) = -Delta arg_+ / pi
    unwrapped_arg = np.unwrap(np.angle(det_rd))
    cum_winding_n = -(unwrapped_arg - unwrapped_arg[0]) / math.pi
    cum_rhp_z = float(ol_rhp_p) + cum_winding_n

    # Extend to asymptotic high frequency (1e5 Hz) for exact integer encirclement count N
    f_ext = np.geomspace(float(freqs[-1]), 2.0e5, 80)[1:]
    det_ext = np.empty(f_ext.size, dtype=complex)
    for k, fv in enumerate(f_ext):
        s_jw = 1j * 2.0 * math.pi * float(fv)
        yk = c0 @ np.linalg.solve(s_jw * eye_n - a0, b0)
        det_ext[k] = complex(np.linalg.det(eye_2 + grid_thevenin_impedance_dq(grid, s_jw) @ yk))
    s_dc = 1j * 2.0 * math.pi * 1.0e-3
    y_dc = c0 @ np.linalg.solve(s_dc * eye_n - a0, b0)
    det_dc = complex(np.linalg.det(eye_2 + grid_thevenin_impedance_dq(grid, s_dc) @ y_dc))
    arg_full = np.unwrap(np.angle(np.concatenate(([det_dc], det_rd, det_ext))))
    n_encirc = int(round(-(float(arg_full[-1]) - float(arg_full[0])) / math.pi))
    z_nyquist = ol_rhp_p + n_encirc

    mask_hf = freqs >= float(high_freq_passivity_min_hz)
    min_nu_hf = float(np.min(nu_arr[mask_hf])) if np.any(mask_hf) else float(np.min(nu_arr))
    neg_hf = freqs[mask_hf & (nu_arr < 0.0)]
    nonpass_band = (float(neg_hf[0]), float(neg_hf[-1])) if neg_hf.size > 0 else None

    mask_q = freqs >= 1.0
    neg_q = freqs[mask_q & (re_yqq < 0.0)]
    neg_q_band = (float(neg_q[0]), float(neg_q[-1])) if neg_q.size > 0 else None

    return DQAdmittanceSweepResult(
        architecture=architecture,
        label=label,
        scr=float(scr),
        p_ref_w=float(p_ref_w),
        freqs_hz=freqs,
        y_dq_s=y_dq,
        passivity_index_s=nu_arr,
        re_ydd_s=re_ydd,
        re_yqq_s=re_yqq,
        det_return_diff=det_rd,
        cumulative_rhp_index=cum_rhp_z,
        min_passivity_high_freq_s=min_nu_hf,
        nonpassive_band_hz=nonpass_band,
        neg_re_yqq_band_hz=neg_q_band,
        open_loop_rhp_poles_p=ol_rhp_p,
        nyquist_encirclements_n=n_encirc,
        nyquist_rhp_poles_z=z_nyquist,
        state_space_rhp_poles=ss_rhp,
        coupled_eigenvalues=coupled_eigs,
        max_schur_det_residual=max_schur_err,
    )


def run_small_signal_impedance_benchmark(
    *,
    scr_operating: float = 1.65,
    p_ref_w: float = 15_000.0,
    xr_ratio: float = 7.0,
) -> SmallSignalImpedanceBenchmarkResult:
    """Run the complete Module 6 modal participation, root-locus, dq passivity & Nyquist suite."""
    part = compute_modal_participation(scr=2.0, p_ref_w=p_ref_w, xr_ratio=xr_ratio)

    scr_grid = np.linspace(5.0, 1.40, 22)
    l_vsg: list[np.ndarray] = []
    l_dvoc: list[np.ndarray] = []
    l_gfl35: list[np.ndarray] = []
    l_gfl55: list[np.ndarray] = []

    for scr_val in scr_grid:
        sv = float(scr_val)
        _, x_v, fn_v, _, _, _ = _solve_architecture_setup("vsg", scr=sv, p_ref_w=p_ref_w, xr_ratio=xr_ratio)
        l_vsg.append(eigvals(_central_jacobian(fn_v, x_v)))

        _, x_o, fn_o, _, _, _ = _solve_architecture_setup("dvoc", scr=sv, p_ref_w=p_ref_w, xr_ratio=xr_ratio)
        l_dvoc.append(eigvals(_central_jacobian(fn_o, x_o)))

        _, x_35, fn_35, _, _, _ = _solve_architecture_setup("gfl_pll_35hz", scr=sv, p_ref_w=p_ref_w, xr_ratio=xr_ratio)
        l_gfl35.append(eigvals(_central_jacobian(fn_35, x_35)))

        _, x_55, fn_55, _, _, _ = _solve_architecture_setup("gfl_pll_55hz", scr=sv, p_ref_w=p_ref_w, xr_ratio=xr_ratio)
        l_gfl55.append(eigvals(_central_jacobian(fn_55, x_55)))

    freqs = np.geomspace(0.25, 6000.0, 500)
    kw = {"scr": scr_operating, "p_ref_w": p_ref_w, "xr_ratio": xr_ratio, "freqs_hz": freqs}

    return SmallSignalImpedanceBenchmarkResult(
        scr_operating=float(scr_operating),
        p_ref_w=float(p_ref_w),
        vsg_participation=part,
        locus_scr_values=scr_grid,
        locus_vsg_eigs=np.asarray(l_vsg, dtype=complex),
        locus_dvoc_eigs=np.asarray(l_dvoc, dtype=complex),
        locus_gfl35_eigs=np.asarray(l_gfl35, dtype=complex),
        locus_gfl55_eigs=np.asarray(l_gfl55, dtype=complex),
        admittance_vsg=evaluate_dq_admittance_and_nyquist("vsg", **kw),
        admittance_vsg_unity_ff=evaluate_dq_admittance_and_nyquist("vsg_unity_ff", **kw),
        admittance_dvoc=evaluate_dq_admittance_and_nyquist("dvoc", **kw),
        admittance_gfl10=evaluate_dq_admittance_and_nyquist("gfl_pll_10hz", **kw),
        admittance_gfl35=evaluate_dq_admittance_and_nyquist("gfl_pll_35hz", **kw),
        admittance_gfl55=evaluate_dq_admittance_and_nyquist("gfl_pll_55hz", **kw),
    )
