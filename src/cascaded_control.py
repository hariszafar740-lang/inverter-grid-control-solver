"""Cascaded synchronous-dq PI voltage/current controller & 14-state electromagnetic GFM solver.

Couples:
  - Outer primary VSG swing equation + Q-V droop + virtual impedance (4 states:
    ``[delta, omega, P_filt, Q_filt]``)
  - Outer synchronous-dq filter-voltage PI controller with ``omega * C_f`` decoupling,
    partial load-current feedforward ``F_i * i_2,dq`` (Qoria passivity tuning ``F_i = 0.75``),
    radial bridge-current saturation ``|i_1,dq*| <= I_max``, and conditional anti-windup
    (2 states: ``[xi_vd, xi_vq]``)
  - Inner synchronous-dq inverter-current PI controller with ``omega * L_1`` decoupling,
    capacitor-voltage feedforward ``v_f,dq``, SVPWM DC-bus voltage clamping, and
    conditional anti-windup (2 states: ``[xi_id, xi_iq]``)
  - Full 6-state electromagnetic LCL + Thévenin AC grid plant (6 states:
    ``[i_1d, i_1q, v_cd, v_cq, i_2d, i_2q]``)

Total coupled differential order: 14 continuous states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dq_frames import rotate_dq_frame
from .lcl_plant import LCLFilterSpec, LCLGridPlant
from .thevenin_network import TheveninGridSpec


@dataclass(frozen=True)
class CascadedPIConfig:
    """Synchronous-dq dual-loop PI gains and partial current feedforward scaling."""

    kp_v: float
    ki_v: float
    kp_i: float
    ki_i: float
    feedforward_current_gain: float = 0.75

    def __post_init__(self) -> None:
        vals = (self.kp_v, self.ki_v, self.kp_i, self.ki_i, self.feedforward_current_gain)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("All CascadedPIConfig gains must be finite")
        if self.kp_v <= 0.0 or self.ki_v < 0.0 or self.kp_i <= 0.0 or self.ki_i < 0.0:
            raise ValueError("Proportional gains must be > 0 and integral gains >= 0")
        if not (0.0 <= self.feedforward_current_gain <= 1.2):
            raise ValueError("feedforward_current_gain must lie in [0.0, 1.2]")

    @classmethod
    def from_bandwidths(
        cls,
        lcl: LCLFilterSpec,
        *,
        f_bw_current_hz: float = 1000.0,
        f_bw_voltage_hz: float = 180.0,
        feedforward_current_gain: float = 0.75,
    ) -> CascadedPIConfig:
        """Timescale-separated IMC dual-loop tuning with Qoria partial feedforward."""
        if f_bw_current_hz <= 0.0 or f_bw_voltage_hz <= 0.0:
            raise ValueError("Loop bandwidths must be positive")
        if f_bw_voltage_hz >= 0.6 * f_bw_current_hz:
            raise ValueError("Outer voltage bandwidth must be < 0.6 * inner current bandwidth")

        w_ci = 2.0 * math.pi * float(f_bw_current_hz)
        w_cv = 2.0 * math.pi * float(f_bw_voltage_hz)
        kp_i = w_ci * lcl.l1_h
        ki_i = w_ci * max(lcl.r1_ohm, 0.05)
        kp_v = w_cv * lcl.cf_f
        ki_v = 0.25 * (w_cv * w_cv) * lcl.cf_f
        return cls(
            kp_v=kp_v,
            ki_v=ki_v,
            kp_i=kp_i,
            ki_i=ki_i,
            feedforward_current_gain=feedforward_current_gain,
        )


@dataclass(frozen=True)
class ElectromagneticTransientResult:
    """Trajectory bundle from 14-state electromagnetic LCL + cascaded PI GFM simulation."""

    grid: TheveninGridSpec
    lcl: LCLFilterSpec
    pi_cfg: CascadedPIConfig
    time_s: np.ndarray
    f_vsg_hz: np.ndarray
    f_grid_hz: np.ndarray
    delta_deg: np.ndarray
    p_ref_w: np.ndarray
    p_pcc_w: np.ndarray
    q_pcc_var: np.ndarray
    vf_d_ref_v: np.ndarray
    vf_q_ref_v: np.ndarray
    vf_d_v: np.ndarray
    vf_q_v: np.ndarray
    i1_d_ref_a: np.ndarray
    i1_q_ref_a: np.ndarray
    i1_d_a: np.ndarray
    i1_q_a: np.ndarray
    i2_d_a: np.ndarray
    i2_q_a: np.ndarray
    v_pcc_pu: np.ndarray
    current_limited: np.ndarray
    voltage_clamped: np.ndarray
    nadir_freq_hz: float
    peak_power_kw: float
    final_power_kw: float


class ElectromagneticGFMModel:
    """Full 14-state electromagnetic GFM inverter model in synchronous inverter ``dq`` frame.

    State vector ordering (length 14):
        0: delta_rad   (rotor angle relative to Thévenin grid bus)
        1: omega_rad_s (VSG virtual rotor angular speed)
        2: p_filt_w    (LPF active power measurement)
        3: q_filt_var  (LPF reactive power measurement)
        4: xi_vd       (outer d-axis voltage PI integrator)
        5: xi_vq       (outer q-axis voltage PI integrator)
        6: xi_id       (inner d-axis current PI integrator)
        7: xi_iq       (inner q-axis current PI integrator)
        8..13: x_lcl   ([i_1d, i_1q, v_cd, v_cq, i_2d, i_2q] in inverter dq frame)
    """

    STATE_DIM = 14

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

    def steady_initial_state(self) -> np.ndarray:
        """Compute exact zero-power nominal equilibrium ``dx/dt = 0`` across all 14 states."""
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
        y0[0] = 0.0
        y0[1] = w0
        y0[4:8] = [xi_vd0, xi_vq0, xi_id0, xi_iq0]
        y0[8:14] = x_lcl0
        return y0

    def _algebraic_tuple(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
    ) -> tuple[float, ...]:
        """Fast allocation-free algebraic evaluation returning a 17-tuple of floats."""
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

        c_d = math.cos(delta)
        s_d = math.sin(delta)
        vg_d = float(v_grid_peak_v) * c_d
        vg_q = -float(v_grid_peak_v) * s_d

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
            "vg_d": t[0],
            "vg_q": t[1],
            "vfd": t[2],
            "vfq": t[3],
            "vf_d_ref": t[4],
            "vf_q_ref": t[5],
            "err_vd": t[6],
            "err_vq": t[7],
            "i1d_unsat": t[8],
            "i1q_unsat": t[9],
            "i1d_ref": t[10],
            "i1q_ref": t[11],
            "err_id": t[12],
            "err_iq": t[13],
            "vinv_d_unsat": t[14],
            "vinv_q_unsat": t[15],
            "vinv_d": t[16],
            "vinv_q": t[17],
            "p_pcc": t[20],
            "q_pcc": t[21],
            "v_pcc_mag": t[22],
            "cur_lim": t[23] > 0.5,
            "v_lim": t[24] > 0.5,
        }

    def rhs_continuous(
        self,
        state: np.ndarray,
        p_ref_w: float,
        q_ref_var: float,
        v_grid_peak_v: float,
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

        t = self._algebraic_tuple(state, p_ref_w, q_ref_var, v_grid_peak_v)
        vfd, vfq = t[2], t[3]
        err_vd, err_vq = t[6], t[7]
        i1d_unsat, i1q_unsat, i1d_ref, i1q_ref = t[8], t[9], t[10], t[11]
        err_id, err_iq = t[12], t[13]
        vinv_d_unsat, vinv_q_unsat, vinv_d, vinv_q = t[14], t[15], t[16], t[17]
        di2d, di2q = t[18], t[19]
        p_pcc, q_pcc = t[20], t[21]
        cur_lim_f, v_lim_f = t[23], t[24]

        wc = 2.0 * math.pi * self.grid.lpf_cutoff_hz
        d_delta = omega - float(w_grid_rad_s)
        d_omega = (
            (float(p_ref_w) - p_filt) / self.grid.w0_rad_s
            - self.grid.dp_nms_rad * (omega - self.grid.w0_rad_s)
        ) / self.grid.j_kgm2
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
                d_omega,
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
        """Compute the 14 small-signal eigenvalues of the Jacobian ``df/dx`` via central differences."""
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


def simulate_electromagnetic_gfm_transient(
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
) -> ElectromagneticTransientResult:
    """Integrate the 14-state electromagnetic GFM + LCL + Thévenin system using RK4."""
    if duration_s <= 0.0 or dt_s <= 0.0 or dt_s > 2.5e-4:
        raise ValueError("duration_s must be > 0 and dt_s must lie in (0, 2.5e-4] s for LCL stability")

    model = ElectromagneticGFMModel(grid=grid, lcl=lcl, pi_cfg=pi_cfg)
    cfg = model.grid
    n_steps = int(round(duration_s / dt_s)) + 1
    time_s = np.linspace(0.0, float(duration_s), n_steps)

    f_vsg_hz = np.empty(n_steps, dtype=float)
    f_grid_hz = np.empty(n_steps, dtype=float)
    delta_deg = np.empty(n_steps, dtype=float)
    p_ref_arr = np.empty(n_steps, dtype=float)
    p_pcc_w = np.empty(n_steps, dtype=float)
    q_pcc_var = np.empty(n_steps, dtype=float)
    vf_d_ref_v = np.empty(n_steps, dtype=float)
    vf_q_ref_v = np.empty(n_steps, dtype=float)
    vf_d_v = np.empty(n_steps, dtype=float)
    vf_q_v = np.empty(n_steps, dtype=float)
    i1_d_ref_a = np.empty(n_steps, dtype=float)
    i1_q_ref_a = np.empty(n_steps, dtype=float)
    i1_d_a = np.empty(n_steps, dtype=float)
    i1_q_a = np.empty(n_steps, dtype=float)
    i2_d_a = np.empty(n_steps, dtype=float)
    i2_q_a = np.empty(n_steps, dtype=float)
    v_pcc_pu = np.empty(n_steps, dtype=float)
    cur_lim_arr = np.zeros(n_steps, dtype=bool)
    v_lim_arr = np.zeros(n_steps, dtype=bool)

    state = model.steady_initial_state()
    vg_nom = cfg.v_peak_nom_v
    inv_2pi = 1.0 / (2.0 * math.pi)
    rhs = model.rhs_continuous
    alg_fn = model._algebraic_tuple

    for k, t_now in enumerate(time_s):
        t_val = float(t_now)
        p_cmd = p_step_w if t_val >= p_step_time_s - 1e-12 else 0.0
        f_g = cfg.f_nom_hz + (grid_freq_drop_hz if t_val >= grid_freq_drop_time_s - 1e-12 else 0.0)
        w_g = 2.0 * math.pi * f_g
        alg = alg_fn(state, p_cmd, 0.0, vg_nom)

        delta_deg[k] = math.degrees(float(state[0]))
        f_vsg_hz[k] = float(state[1]) * inv_2pi
        f_grid_hz[k] = f_g
        p_ref_arr[k] = p_cmd
        vf_d_v[k] = alg[2]
        vf_q_v[k] = alg[3]
        vf_d_ref_v[k] = alg[4]
        vf_q_ref_v[k] = alg[5]
        i1_d_ref_a[k] = alg[10]
        i1_q_ref_a[k] = alg[11]
        i1_d_a[k] = float(state[8])
        i1_q_a[k] = float(state[9])
        i2_d_a[k] = float(state[12])
        i2_q_a[k] = float(state[13])
        p_pcc_w[k] = alg[20]
        q_pcc_var[k] = alg[21]
        v_pcc_pu[k] = alg[22] / vg_nom
        cur_lim_arr[k] = alg[23] > 0.5
        v_lim_arr[k] = alg[24] > 0.5

        if k < n_steps - 1:
            k1 = rhs(state, p_cmd, 0.0, vg_nom, w_g)
            t_mid = t_val + 0.5 * dt_s
            p_mid = p_step_w if t_mid >= p_step_time_s - 1e-12 else 0.0
            fg_mid = cfg.f_nom_hz + (grid_freq_drop_hz if t_mid >= grid_freq_drop_time_s - 1e-12 else 0.0)
            wg_mid = 2.0 * math.pi * fg_mid
            k2 = rhs(state + 0.5 * dt_s * k1, p_mid, 0.0, vg_nom, wg_mid)
            k3 = rhs(state + 0.5 * dt_s * k2, p_mid, 0.0, vg_nom, wg_mid)
            t_end = t_val + dt_s
            p_end = p_step_w if t_end >= p_step_time_s - 1e-12 else 0.0
            fg_end = cfg.f_nom_hz + (grid_freq_drop_hz if t_end >= grid_freq_drop_time_s - 1e-12 else 0.0)
            wg_end = 2.0 * math.pi * fg_end
            k4 = rhs(state + dt_s * k3, p_end, 0.0, vg_nom, wg_end)
            state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ElectromagneticTransientResult(
        grid=cfg,
        lcl=model.lcl,
        pi_cfg=model.pi_cfg,
        time_s=time_s,
        f_vsg_hz=f_vsg_hz,
        f_grid_hz=f_grid_hz,
        delta_deg=delta_deg,
        p_ref_w=p_ref_arr,
        p_pcc_w=p_pcc_w,
        q_pcc_var=q_pcc_var,
        vf_d_ref_v=vf_d_ref_v,
        vf_q_ref_v=vf_q_ref_v,
        vf_d_v=vf_d_v,
        vf_q_v=vf_q_v,
        i1_d_ref_a=i1_d_ref_a,
        i1_q_ref_a=i1_q_ref_a,
        i1_d_a=i1_d_a,
        i1_q_a=i1_q_a,
        i2_d_a=i2_d_a,
        i2_q_a=i2_q_a,
        v_pcc_pu=v_pcc_pu,
        current_limited=cur_lim_arr,
        voltage_clamped=v_lim_arr,
        nadir_freq_hz=float(np.min(f_vsg_hz)),
        peak_power_kw=float(np.max(p_pcc_w) / 1000.0),
        final_power_kw=float(p_pcc_w[-1] / 1000.0),
    )
