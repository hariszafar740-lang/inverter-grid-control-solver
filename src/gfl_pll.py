"""Synchronous Reference Frame PLL (SRF-PLL) & 11-state Electromagnetic GFL Inverter Model.

Models a three-phase Grid-Following (GFL) inverter synchronized via a classic Synchronous
Reference Frame Phase-Locked Loop (Kaura & Blasko, *IEEE Trans. Ind. Appl.*, 1997) and
coupled to the identical 6-state electromagnetic ``dq`` LCL filter + Thévenin AC grid
plant used by the Grid-Forming models (Wen et al., *IEEE TPEL*, 2016; Dong et al.,
*IEEE TPWRS*, 2015):

    State vector (length 11):
        0: delta_pll_rad  (PLL estimated phase angle relative to Thévenin grid bus)
        1: xi_pll_s       (SRF-PLL normalized q-axis voltage integrator)
        2: vd_filt_v      (LPF d-axis filter-bus voltage for P/Q current dispatch)
        3: xi_id          (inner d-axis bridge-current PI integrator)
        4: xi_iq          (inner q-axis bridge-current PI integrator)
        5..10: x_lcl      ([i_1d, i_1q, v_cd, v_cq, i_2d, i_2q] in PLL dq frame)

Because ``v_fq = -V_g * sin(delta_pll) + R_net * i_2q + omega_pll * L_net * i_2d + ...``,
high PLL proportional gain ``K_p,pll`` forms a positive-feedback algebraic loop with weak
inductive grid reactance ``omega_pll * L_net * i_2d`` (Hopf bifurcation), while below the
static nose-curve ceiling ``SCR_static`` no steady-state PLL equilibrium exists at all.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from .cascaded_control import CascadedPIConfig
from .dq_frames import alphabeta_to_dq, rotate_dq_frame
from .lcl_plant import LCLFilterSpec, LCLGridPlant
from .thevenin_network import TheveninGridSpec


@dataclass(frozen=True)
class SRFPLLConfig:
    """Closed-loop bandwidth, damping, and frequency-limiter parameters for an SRF-PLL."""

    f_bw_hz: float = 35.0
    zeta: float = 0.7071067811865476
    w0_rad_s: float = 100.0 * math.pi
    v0_peak_v: float = 230.0 * math.sqrt(2.0)
    vd_filter_hz: float = 60.0
    f_min_hz: float = 35.0
    f_max_hz: float = 65.0

    def __post_init__(self) -> None:
        vals = (
            self.f_bw_hz,
            self.zeta,
            self.w0_rad_s,
            self.v0_peak_v,
            self.vd_filter_hz,
            self.f_min_hz,
            self.f_max_hz,
        )
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("All SRFPLLConfig parameters must be finite")
        if (
            self.f_bw_hz <= 0.0
            or self.zeta <= 0.0
            or self.w0_rad_s <= 0.0
            or self.v0_peak_v <= 0.0
            or self.vd_filter_hz <= 0.0
        ):
            raise ValueError("f_bw_hz, zeta, w0_rad_s, v0_peak_v, and vd_filter_hz must be > 0")
        f_nom = self.w0_rad_s / (2.0 * math.pi)
        if not (0.0 < self.f_min_hz < f_nom < self.f_max_hz):
            raise ValueError("Frequency limits must satisfy 0 < f_min_hz < f_nom_hz < f_max_hz")

    @property
    def wn_rad_s(self) -> float:
        """Closed-loop natural frequency ``2 * pi * f_bw_hz`` (rad/s)."""
        return 2.0 * math.pi * self.f_bw_hz

    @property
    def kp_rad_s(self) -> float:
        """Normalized proportional gain ``2 * zeta * w_n`` acting on ``v_q / V_0`` (rad/s)."""
        return 2.0 * self.zeta * self.wn_rad_s

    @property
    def ki_rad_s2(self) -> float:
        """Normalized integral gain ``w_n^2`` acting on ``int(v_q / V_0) dt`` (rad/s^2)."""
        wn = self.wn_rad_s
        return wn * wn

    @property
    def w_min_rad_s(self) -> float:
        return 2.0 * math.pi * self.f_min_hz

    @property
    def w_max_rad_s(self) -> float:
        return 2.0 * math.pi * self.f_max_hz

    @classmethod
    def from_grid_spec(
        cls,
        grid: TheveninGridSpec | None = None,
        *,
        f_bw_hz: float = 35.0,
        zeta: float = 0.7071067811865476,
        vd_filter_hz: float = 60.0,
    ) -> SRFPLLConfig:
        """Construct an ``SRFPLLConfig`` matched to nominal voltage and frequency of ``grid``."""
        g = grid if grid is not None else TheveninGridSpec()
        return cls(
            f_bw_hz=f_bw_hz,
            zeta=zeta,
            w0_rad_s=g.w0_rad_s,
            v0_peak_v=g.v_peak_nom_v,
            vd_filter_hz=vd_filter_hz,
        )


class SynchronousReferenceFramePLL:
    """Standalone Synchronous Reference Frame Phase-Locked Loop (SRF-PLL) block."""

    def __init__(self, config: SRFPLLConfig | None = None) -> None:
        self.config = config if config is not None else SRFPLLConfig()

    def phase_detector_dq(
        self,
        v_alpha_v: float,
        v_beta_v: float,
        theta_pll_rad: float,
    ) -> tuple[float, float, float]:
        """Return ``(v_d_v, v_q_v, v_q_pu)`` in the PLL synchronous frame ``theta_pll_rad``."""
        vd, vq = alphabeta_to_dq(v_alpha_v, v_beta_v, theta_pll_rad)
        return vd, vq, vq / self.config.v0_peak_v

    def step_kinetics(
        self,
        vq_v: float,
        xi_pll_s: float,
    ) -> tuple[float, float, bool]:
        """Return ``(omega_pll_rad_s, d_xi_pll_dt, freq_clamped)`` with conditional anti-windup."""
        vq = float(vq_v)
        xi = float(xi_pll_s)
        if not (math.isfinite(vq) and math.isfinite(xi)):
            raise ValueError("vq_v and xi_pll_s must be finite")

        cfg = self.config
        vq_pu = vq / cfg.v0_peak_v
        w_raw = cfg.w0_rad_s + cfg.kp_rad_s * vq_pu + cfg.ki_rad_s2 * xi
        w_clamped = min(max(w_raw, cfg.w_min_rad_s), cfg.w_max_rad_s)
        clamped = w_clamped != w_raw
        if clamped and (w_raw - w_clamped) * vq_pu > 0.0:
            d_xi = 0.0
        else:
            d_xi = vq_pu
        return w_clamped, d_xi, clamped

    def small_signal_stiff_poles(self) -> tuple[complex, complex]:
        """Exact closed-loop roots of ``s^2 + K_p * s + K_i = 0`` on an ideal stiff grid."""
        kp = self.config.kp_rad_s
        ki = self.config.ki_rad_s2
        disc = cmath.sqrt(kp * kp - 4.0 * ki)
        return (-0.5 * kp + 0.5 * disc, -0.5 * kp - 0.5 * disc)


class ElectromagneticGFLModel:
    """11-state Grid-Following (GFL) SRF-PLL + inner dq current PI + LCL + Thévenin grid model."""

    STATE_DIM = 11

    def __init__(
        self,
        grid: TheveninGridSpec | None = None,
        lcl: LCLFilterSpec | None = None,
        pi_cfg: CascadedPIConfig | None = None,
        pll_cfg: SRFPLLConfig | None = None,
        *,
        k_qv_a_per_v: float = 0.0,
    ) -> None:
        if not math.isfinite(float(k_qv_a_per_v)) or float(k_qv_a_per_v) < 0.0:
            raise ValueError("k_qv_a_per_v must be finite and >= 0")
        self.grid = grid if grid is not None else TheveninGridSpec()
        self.lcl = lcl if lcl is not None else LCLFilterSpec()
        self.pi_cfg = pi_cfg if pi_cfg is not None else CascadedPIConfig.from_bandwidths(self.lcl)
        self.pll_cfg = pll_cfg if pll_cfg is not None else SRFPLLConfig.from_grid_spec(self.grid)
        self.pll = SynchronousReferenceFramePLL(self.pll_cfg)
        self.plant = LCLGridPlant(lcl=self.lcl, grid=self.grid)
        self.k_qv_a_per_v = float(k_qv_a_per_v)

    def steady_initial_state(self) -> np.ndarray:
        """Exact zero-dispatch nominal equilibrium ``dx/dt = 0`` across all 11 states."""
        w0 = self.grid.w0_rad_s
        v0 = self.grid.v_peak_nom_v
        z_net = complex(self.plant.r_net_ohm, w0 * self.plant.l_net_h)
        denom_c = complex(1.0, w0 * self.lcl.cf_f * self.lcl.rd_ohm)
        y_branch = complex(0.0, -w0 * self.lcl.cf_f) / denom_c
        h_oc = 1.0 - z_net * y_branch
        vfd0 = v0 / abs(h_oc)
        delta0 = -cmath.phase(h_oc)
        vc0 = vfd0 / denom_c
        i20 = complex(0.0, -w0 * self.lcl.cf_f) * vc0
        y0 = np.array(
            [delta0, 0.0, vfd0, 0.0, 0.0, 0.0, 0.0, vc0.real, vc0.imag, i20.real, i20.imag],
            dtype=float,
        )
        if self.k_qv_a_per_v > 0.0:
            sol = root(lambda y: self.rhs_continuous(y, 0.0, 0.0, v0, w0), y0, tol=1.0e-12)
            return np.asarray(sol.x, dtype=float)
        return y0

    def solve_loaded_equilibrium(
        self,
        p_ref_w: float,
        q_ref_var: float = 0.0,
        *,
        initial_guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, bool]:
        """Solve ``rhs_continuous(y, P_ref, Q_ref) = 0`` on the normal low-angle branch."""
        v0 = self.grid.v_peak_nom_v
        w0 = self.grid.w0_rad_s
        p_cmd = float(p_ref_w)
        q_cmd = float(q_ref_var)

        seeds: list[np.ndarray] = []
        if initial_guess is not None:
            seeds.append(np.array(initial_guess, dtype=float))

        base = self.steady_initial_state()
        id_est = p_cmd / (1.5 * v0)
        sin_est = min(0.85, max(-0.85, (w0 * self.plant.l_net_h * id_est) / v0))
        seed_phys = base.copy()
        seed_phys[0] = math.asin(sin_est)
        seed_phys[5] = id_est
        seed_phys[9] = id_est
        seeds.append(seed_phys)

        for delta_try in (0.15, 0.45, 0.72):
            s_alt = seed_phys.copy()
            s_alt[0] = delta_try
            seeds.append(s_alt)

        for guess in seeds:
            sol = root(
                lambda y: self.rhs_continuous(y, p_cmd, q_cmd, v0, w0),
                guess,
                tol=1.0e-11,
            )
            res_norm = float(np.linalg.norm(self.rhs_continuous(sol.x, p_cmd, q_cmd, v0, w0)))
            delta_wrapped = math.atan2(math.sin(float(sol.x[0])), math.cos(float(sol.x[0])))
            vd_eq = float(sol.x[2])
            if sol.success and res_norm < 1.0e-7 and abs(delta_wrapped) < math.radians(85.0) and vd_eq > 0.45 * v0:
                y_eq = np.asarray(sol.x, dtype=float)
                y_eq[0] = delta_wrapped
                return y_eq, True

        return None, False

    def _algebraic_tuple(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
    ) -> tuple[float, ...]:
        delta = float(state[0])
        xi_pll = float(state[1])
        vd_f = float(state[2])
        xi_id = float(state[3])
        xi_iq = float(state[4])
        i1d = float(state[5])
        i1q = float(state[6])
        vcd = float(state[7])
        vcq = float(state[8])
        i2d = float(state[9])
        i2q = float(state[10])

        vg_d = float(v_grid_peak_v) * math.cos(delta)
        vg_q = -float(v_grid_peak_v) * math.sin(delta)

        rd = self.lcl.rd_ohm
        vfd = vcd + rd * (i1d - i2d)
        vfq = vcq + rd * (i1q - i2q)

        cfg_p = self.pll_cfg
        v0 = cfg_p.v0_peak_v
        vq_pu = vfq / v0
        w_raw = cfg_p.w0_rad_s + cfg_p.kp_rad_s * vq_pu + cfg_p.ki_rad_s2 * xi_pll
        w_pll = min(max(w_raw, cfg_p.w_min_rad_s), cfg_p.w_max_rad_s)
        pll_lim_f = 1.0 if w_pll != w_raw else 0.0

        v_denom = max(vd_f, 0.35 * v0)
        i1d_unsat = float(p_ref_w) / (1.5 * v_denom)
        i1q_unsat = -float(q_ref_var) / (1.5 * v_denom) - self.k_qv_a_per_v * (vfd - v0)

        i_mag = math.hypot(i1d_unsat, i1q_unsat)
        i_max = self.grid.i_max_peak_a
        if i_mag > i_max:
            sc_i = i_max / i_mag
            i1d_ref = i1d_unsat * sc_i
            i1q_ref = i1q_unsat * sc_i
            cur_lim_f = 1.0
        else:
            i1d_ref = i1d_unsat
            i1q_ref = i1q_unsat
            cur_lim_f = 0.0

        err_id = i1d_ref - i1d
        err_iq = i1q_ref - i1q
        l1 = self.lcl.l1_h
        vinv_d_unsat = self.pi_cfg.kp_i * err_id + self.pi_cfg.ki_i * xi_id + vfd - w_pll * l1 * i1q
        vinv_q_unsat = self.pi_cfg.kp_i * err_iq + self.pi_cfg.ki_i * xi_iq + vfq + w_pll * l1 * i1d

        v_mag = math.hypot(vinv_d_unsat, vinv_q_unsat)
        v_max = self.lcl.v_inv_max_peak_v
        if v_mag > v_max:
            sc_v = v_max / v_mag
            vinv_d = vinv_d_unsat * sc_v
            vinv_q = vinv_q_unsat * sc_v
            v_lim_f = 1.0
        else:
            vinv_d = vinv_d_unsat
            vinv_q = vinv_q_unsat
            v_lim_f = 0.0

        r_net = self.plant.r_net_ohm
        l_net = self.plant.l_net_h
        di2d = (vfd - vg_d - r_net * i2d + w_pll * l_net * i2q) / l_net
        di2q = (vfq - vg_q - r_net * i2q - w_pll * l_net * i2d) / l_net

        rg = self.grid.r_g_ohm
        lg = self.grid.l_g_h
        vpcc_d = vg_d + rg * i2d + lg * di2d - w_pll * lg * i2q
        vpcc_q = vg_q + rg * i2q + lg * di2q + w_pll * lg * i2d
        p_pcc = 1.5 * (vpcc_d * i2d + vpcc_q * i2q)
        q_pcc = 1.5 * (vpcc_q * i2d - vpcc_d * i2q)
        v_pcc_mag = math.hypot(vpcc_d, vpcc_q)

        return (
            w_pll,
            w_raw,
            vq_pu,
            vg_d,
            vg_q,
            vfd,
            vfq,
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
            pll_lim_f,
        )

    def evaluate_algebraic(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
    ) -> dict[str, float | bool]:
        """Public inspection dictionary of GFL PLL, current-loop, and PCC observables."""
        _ = rotate_dq_frame(float(v_grid_peak_v), 0.0, float(state[0]))
        t = self._algebraic_tuple(state, p_ref_w, q_ref_var, v_grid_peak_v)
        return {
            "omega_pll": t[0],
            "vq_pu": t[2],
            "vg_d": t[3],
            "vg_q": t[4],
            "vfd": t[5],
            "vfq": t[6],
            "i1d_ref": t[7],
            "i1q_ref": t[8],
            "vinv_d": t[13],
            "vinv_q": t[14],
            "p_pcc": t[17],
            "q_pcc": t[18],
            "v_pcc_mag": t[19],
            "cur_lim": t[20] > 0.5,
            "v_lim": t[21] > 0.5,
            "pll_lim": t[22] > 0.5,
        }

    def rhs_continuous(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
        w_grid_rad_s: float,
    ) -> np.ndarray:
        """Continuous-time 11-state ODE right-hand side ``dx/dt``."""
        vd_f = float(state[2])
        i1d = float(state[5])
        i1q = float(state[6])
        vcd = float(state[7])
        vcq = float(state[8])
        i2d = float(state[9])
        i2q = float(state[10])

        t = self._algebraic_tuple(state, p_ref_w, q_ref_var, v_grid_peak_v)
        w_pll, w_raw, vq_pu = t[0], t[1], t[2]
        vfd, vfq = t[5], t[6]
        err_id, err_iq = t[9], t[10]
        vinv_d_unsat, vinv_q_unsat, vinv_d, vinv_q = t[11], t[12], t[13], t[14]
        di2d, di2q = t[15], t[16]
        v_lim_f, pll_lim_f = t[21], t[22]

        d_delta = w_pll - float(w_grid_rad_s)
        if pll_lim_f > 0.5 and (w_raw - w_pll) * vq_pu > 0.0:
            d_xi_pll = 0.0
        else:
            d_xi_pll = vq_pu

        wc_vd = 2.0 * math.pi * self.pll_cfg.vd_filter_hz
        d_vdf = wc_vd * (vfd - vd_f)

        if v_lim_f > 0.5:
            d_xiid = 0.0 if ((vinv_d_unsat - vinv_d) * err_id > 0.0) else err_id
            d_xiiq = 0.0 if ((vinv_q_unsat - vinv_q) * err_iq > 0.0) else err_iq
        else:
            d_xiid = err_id
            d_xiiq = err_iq

        l1, r1, cf = self.lcl.l1_h, self.lcl.r1_ohm, self.lcl.cf_f
        di1d = (vinv_d - vfd - r1 * i1d + w_pll * l1 * i1q) / l1
        di1q = (vinv_q - vfq - r1 * i1q - w_pll * l1 * i1d) / l1
        dvcd = ((i1d - i2d) + w_pll * cf * vcq) / cf
        dvcq = ((i1q - i2q) - w_pll * cf * vcd) / cf

        return np.array(
            [
                d_delta,
                d_xi_pll,
                d_vdf,
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
        eps: float = 1.0e-6,
    ) -> np.ndarray:
        """Compute the 11 small-signal eigenvalues of ``df/dx`` via central differences."""
        if equilibrium_state is None:
            eq, ok = self.solve_loaded_equilibrium(p_ref_w=p_ref_w, q_ref_var=q_ref_var)
            if not ok or eq is None:
                raise ValueError(f"No steady-state GFL equilibrium exists at SCR={self.grid.scr}, P_ref={p_ref_w} W")
            x0 = eq
        else:
            x0 = np.array(equilibrium_state, dtype=float)

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
