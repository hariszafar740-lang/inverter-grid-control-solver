"""Weak-Grid SCR Collapse & Stability Boundary Benchmark: GFL (SRF-PLL) vs. GFM Triad.

Evaluates two fundamental weak-grid stability mechanisms at high active-power dispatch
(``P_ref = 15 kW = 0.75 pu`` on the ``20 kVA`` 6-state electromagnetic ``dq`` LCL plant):

  1. **Dynamic PLL-Grid Impedance Hopf Bifurcation (``SCR_crit(f_bw,pll)``):**
     Increasing ``SRF-PLL`` closed-loop bandwidth amplifies positive feedback between
     PLL speed ``omega_pll`` and weak inductive voltage drop ``omega_pll * L_net * i_2d``,
     driving complex-conjugate eigenvalues into the right-half plane even when a static
     operating point still exists.
  2. **Static Synchronizing / Nose-Curve Saddle-Node Collapse (``SCR_static``):**
     Below ``SCR_static ~ 1.32`` at ``0.75 pu`` dispatch, ``V_g * sin(delta) = X_net * i_2d``
     loses real solutions and *any* pure P/Q GFL current source undergoes unbounded PLL
     pole-slipping (Loss of Synchronism, ``LOS``), whereas Grid-Forming ``Droop``, ``VSG``,
     and ``dVOC`` controllers autonomously support PCC voltage and remain strictly Hurwitz
     all the way down to ``SCR = 1.15``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import root

from .cascaded_control import CascadedPIConfig, ElectromagneticGFMModel
from .gfl_pll import ElectromagneticGFLModel, SRFPLLConfig
from .gfm_triad import MultiLoopDroopGFMModel, MultiLoopDVOCGFMModel
from .lcl_plant import LCLFilterSpec
from .thevenin_network import TheveninGridSpec

WeakGridArchitecture = Literal[
    "gfl_fast_pll",
    "gfl_med_pll",
    "gfl_slow_pll",
    "gfm_droop",
    "gfm_vsg",
    "gfm_dvoc",
]

DEFAULT_SCR_SWEEP: tuple[float, ...] = (
    6.0,
    4.5,
    3.5,
    3.1,
    2.9,
    2.6,
    2.3,
    2.0,
    1.85,
    1.75,
    1.65,
    1.55,
    1.45,
    1.36,
    1.30,
    1.24,
    1.18,
    1.15,
)


@dataclass(frozen=True)
class SCRStabilityPoint:
    """Loaded equilibrium and small-signal Hurwitz metrics at a single grid ``SCR``."""

    scr: float
    equilibrium_exists: bool
    is_small_signal_stable: bool
    max_real_eig_s_inv: float
    dominant_freq_hz: float
    delta_eq_deg: float
    v_pcc_eq_pu: float


@dataclass(frozen=True)
class WeakGridLineTripResult:
    """Time-domain trajectory across a sudden weak-grid parallel-line opening contingency."""

    label: str
    architecture: WeakGridArchitecture
    scr_pre: float
    scr_post: float
    p_dispatch_w: float
    trip_time_s: float
    time_s: np.ndarray
    f_est_hz: np.ndarray
    delta_deg: np.ndarray
    p_pcc_kw: np.ndarray
    q_pcc_kvar: np.ndarray
    v_pcc_pu: np.ndarray
    i2_peak_a: np.ndarray
    lost_synchronism: bool
    min_v_pcc_pu: float
    max_abs_delta_deg: float
    final_power_kw: float


@dataclass(frozen=True)
class WeakGridBenchmarkResult:
    """Complete comparative weak-grid SCR sweep & line-opening collapse benchmark bundle."""

    p_dispatch_w: float
    scr_grid: np.ndarray
    gfl_fast_sweep: tuple[SCRStabilityPoint, ...]
    gfl_med_sweep: tuple[SCRStabilityPoint, ...]
    gfl_slow_sweep: tuple[SCRStabilityPoint, ...]
    vsg_sweep: tuple[SCRStabilityPoint, ...]
    dvoc_sweep: tuple[SCRStabilityPoint, ...]
    scr_crit_gfl_fast: float
    scr_crit_gfl_med: float
    scr_crit_gfl_slow: float
    scr_static_limit_gfl: float
    trip_gfl_med: WeakGridLineTripResult
    trip_gfl_slow: WeakGridLineTripResult
    trip_gfm_vsg: WeakGridLineTripResult
    trip_gfm_dvoc: WeakGridLineTripResult


def _pll_bandwidth_for_arch(arch: WeakGridArchitecture) -> float:
    if arch == "gfl_fast_pll":
        return 55.0
    if arch == "gfl_med_pll":
        return 35.0
    if arch == "gfl_slow_pll":
        return 10.0
    raise ValueError(f"Architecture {arch!r} is not a GFL PLL configuration")


def _solve_gfm_loaded_equilibrium(
    model: MultiLoopDroopGFMModel | ElectromagneticGFMModel | MultiLoopDVOCGFMModel,
    p_dispatch_w: float,
    prev_state: np.ndarray | None = None,
) -> tuple[np.ndarray | None, bool]:
    g = model.grid
    v0 = g.v_peak_nom_v
    w0 = g.w0_rad_s
    seeds: list[np.ndarray] = []
    if prev_state is not None:
        seeds.append(np.array(prev_state, dtype=float))
    base = model.steady_initial_state()
    base[0] = min(0.75, max(0.05, float(p_dispatch_w) / (g.s_rated_va * g.scr)))
    seeds.append(base)

    for guess in seeds:
        sol = root(
            lambda y: model.rhs_continuous(y, float(p_dispatch_w), 0.0, v0, w0),
            guess,
            tol=1.0e-11,
        )
        res_norm = float(np.linalg.norm(model.rhs_continuous(sol.x, float(p_dispatch_w), 0.0, v0, w0)))
        delta_w = math.atan2(math.sin(float(sol.x[0])), math.cos(float(sol.x[0])))
        if sol.success and res_norm < 1.0e-7 and abs(delta_w) < math.radians(85.0):
            y_eq = np.asarray(sol.x, dtype=float)
            y_eq[0] = delta_w
            return y_eq, True
    return None, False


def sweep_scr_stability(
    architecture: WeakGridArchitecture,
    *,
    scr_values: Sequence[float] = DEFAULT_SCR_SWEEP,
    p_dispatch_w: float = 15_000.0,
    xr_ratio: float = 7.0,
    lcl: LCLFilterSpec | None = None,
    pi_cfg: CascadedPIConfig | None = None,
) -> tuple[SCRStabilityPoint, ...]:
    """Sweep descending ``SCR`` values using branch continuation and Jacobian eigenvalues."""
    if p_dispatch_w < 0.0 or not math.isfinite(p_dispatch_w):
        raise ValueError("p_dispatch_w must be finite and >= 0")
    scr_sorted = sorted((float(s) for s in scr_values), reverse=True)
    if not scr_sorted or any((not math.isfinite(s) or s <= 0.0) for s in scr_sorted):
        raise ValueError("scr_values must be non-empty finite positive numbers")

    cfg_lcl = lcl if lcl is not None else LCLFilterSpec()
    cfg_pi = pi_cfg if pi_cfg is not None else CascadedPIConfig.from_bandwidths(cfg_lcl)
    prev_eq: np.ndarray | None = None
    out: list[SCRStabilityPoint] = []

    is_gfl = architecture in ("gfl_fast_pll", "gfl_med_pll", "gfl_slow_pll")
    for scr in scr_sorted:
        grid = TheveninGridSpec(scr=scr, xr_ratio=xr_ratio)
        v0 = grid.v_peak_nom_v

        if is_gfl:
            f_bw = _pll_bandwidth_for_arch(architecture)
            pll_cfg = SRFPLLConfig.from_grid_spec(grid, f_bw_hz=f_bw)
            m_gfl = ElectromagneticGFLModel(grid=grid, lcl=cfg_lcl, pi_cfg=cfg_pi, pll_cfg=pll_cfg)
            y_eq, ok = m_gfl.solve_loaded_equilibrium(p_dispatch_w, 0.0, initial_guess=prev_eq)
            if not ok or y_eq is None:
                prev_eq = None
                out.append(
                    SCRStabilityPoint(
                        scr=scr,
                        equilibrium_exists=False,
                        is_small_signal_stable=False,
                        max_real_eig_s_inv=float("nan"),
                        dominant_freq_hz=float("nan"),
                        delta_eq_deg=float("nan"),
                        v_pcc_eq_pu=float("nan"),
                    )
                )
                continue
            prev_eq = y_eq
            ev = m_gfl.linearized_eigenvalues(y_eq, p_ref_w=p_dispatch_w)
            info = m_gfl.evaluate_algebraic(y_eq, p_dispatch_w, 0.0, v0)
        else:
            m_gfm: MultiLoopDroopGFMModel | ElectromagneticGFMModel | MultiLoopDVOCGFMModel
            if architecture == "gfm_droop":
                m_gfm = MultiLoopDroopGFMModel(grid=grid, lcl=cfg_lcl, pi_cfg=cfg_pi)
            elif architecture == "gfm_vsg":
                m_gfm = ElectromagneticGFMModel(grid=grid, lcl=cfg_lcl, pi_cfg=cfg_pi)
            elif architecture == "gfm_dvoc":
                m_gfm = MultiLoopDVOCGFMModel(grid=grid, lcl=cfg_lcl, pi_cfg=cfg_pi)
            else:
                raise ValueError(f"Unsupported architecture {architecture!r}")

            y_eq, ok = _solve_gfm_loaded_equilibrium(m_gfm, p_dispatch_w, prev_state=prev_eq)
            if not ok or y_eq is None:
                prev_eq = None
                out.append(
                    SCRStabilityPoint(
                        scr=scr,
                        equilibrium_exists=False,
                        is_small_signal_stable=False,
                        max_real_eig_s_inv=float("nan"),
                        dominant_freq_hz=float("nan"),
                        delta_eq_deg=float("nan"),
                        v_pcc_eq_pu=float("nan"),
                    )
                )
                continue
            prev_eq = y_eq
            ev = m_gfm.linearized_eigenvalues(y_eq, p_ref_w=p_dispatch_w)
            info = m_gfm.evaluate_algebraic(y_eq, p_dispatch_w, 0.0, v0)

        idx_dom = int(np.argmax(ev.real))
        z_dom = ev[idx_dom]
        max_re = float(z_dom.real)
        f_dom = float(abs(z_dom.imag) / (2.0 * math.pi))
        out.append(
            SCRStabilityPoint(
                scr=scr,
                equilibrium_exists=True,
                is_small_signal_stable=max_re < 0.0,
                max_real_eig_s_inv=max_re,
                dominant_freq_hz=f_dom,
                delta_eq_deg=math.degrees(float(y_eq[0])),
                v_pcc_eq_pu=float(info["v_pcc_mag"]) / v0,
            )
        )

    return tuple(out)


def estimate_critical_scr(points: Sequence[SCRStabilityPoint]) -> tuple[float, float]:
    """Estimate ``(scr_crit_dynamic, scr_static_limit)`` from a descending-SCR sweep."""
    pts = sorted(points, key=lambda p: p.scr, reverse=True)
    if not pts:
        raise ValueError("points must be non-empty")

    scr_static = pts[-1].scr
    for k in range(len(pts) - 1):
        if pts[k].equilibrium_exists and not pts[k + 1].equilibrium_exists:
            scr_static = 0.5 * (pts[k].scr + pts[k + 1].scr)
            break

    scr_crit = scr_static
    for k in range(len(pts) - 1):
        p_hi, p_lo = pts[k], pts[k + 1]
        if p_hi.equilibrium_exists and p_hi.max_real_eig_s_inv < 0.0:
            if not p_lo.equilibrium_exists:
                scr_crit = scr_static
                break
            if p_lo.max_real_eig_s_inv >= 0.0:
                re_hi = p_hi.max_real_eig_s_inv
                re_lo = p_lo.max_real_eig_s_inv
                frac = (0.0 - re_hi) / (re_lo - re_hi)
                scr_crit = p_hi.scr + frac * (p_lo.scr - p_hi.scr)
                break

    return float(scr_crit), float(scr_static)


def simulate_weak_grid_line_trip(
    architecture: WeakGridArchitecture,
    *,
    scr_pre: float = 4.0,
    scr_post: float = 1.22,
    p_dispatch_w: float = 15_000.0,
    xr_ratio: float = 7.0,
    duration_s: float = 1.20,
    dt_s: float = 80.0e-6,
    trip_time_s: float = 0.30,
    lcl: LCLFilterSpec | None = None,
    pi_cfg: CascadedPIConfig | None = None,
) -> WeakGridLineTripResult:
    """Simulate sudden grid weakening ``scr_pre -> scr_post`` at ``trip_time_s`` via RK4."""
    if scr_pre <= 0.0 or scr_post <= 0.0 or duration_s <= 0.0 or dt_s <= 0.0 or dt_s > 2.5e-4:
        raise ValueError("Invalid SCR or time-step parameters for simulate_weak_grid_line_trip")
    if not (0.0 <= trip_time_s < duration_s):
        raise ValueError("trip_time_s must lie in [0, duration_s)")

    cfg_lcl = lcl if lcl is not None else LCLFilterSpec()
    cfg_pi = pi_cfg if pi_cfg is not None else CascadedPIConfig.from_bandwidths(cfg_lcl)
    g_pre = TheveninGridSpec(scr=scr_pre, xr_ratio=xr_ratio)
    g_post = TheveninGridSpec(scr=scr_post, xr_ratio=xr_ratio)
    v0 = g_pre.v_peak_nom_v
    w0 = g_pre.w0_rad_s
    inv_2pi = 1.0 / (2.0 * math.pi)

    n_steps = int(round(duration_s / dt_s)) + 1
    time_s = np.linspace(0.0, float(duration_s), n_steps)
    f_est_hz = np.empty(n_steps, dtype=float)
    delta_deg = np.empty(n_steps, dtype=float)
    p_pcc_kw = np.empty(n_steps, dtype=float)
    q_pcc_kvar = np.empty(n_steps, dtype=float)
    v_pcc_pu = np.empty(n_steps, dtype=float)
    i2_peak_a = np.empty(n_steps, dtype=float)

    is_gfl = architecture in ("gfl_fast_pll", "gfl_med_pll", "gfl_slow_pll")
    if is_gfl:
        f_bw = _pll_bandwidth_for_arch(architecture)
        label = f"GFL SRF-PLL ({f_bw:.0f} Hz)"
        m_pre_gfl = ElectromagneticGFLModel(
            grid=g_pre,
            lcl=cfg_lcl,
            pi_cfg=cfg_pi,
            pll_cfg=SRFPLLConfig.from_grid_spec(g_pre, f_bw_hz=f_bw),
        )
        m_post_gfl = ElectromagneticGFLModel(
            grid=g_post,
            lcl=cfg_lcl,
            pi_cfg=cfg_pi,
            pll_cfg=SRFPLLConfig.from_grid_spec(g_post, f_bw_hz=f_bw),
        )
        state_opt, ok = m_pre_gfl.solve_loaded_equilibrium(p_dispatch_w, 0.0)
        if not ok or state_opt is None:
            raise ValueError(f"Pre-trip GFL equilibrium failed at SCR={scr_pre}")
        state = state_opt

        for k, t_now in enumerate(time_s):
            t_val = float(t_now)
            m_cur = m_post_gfl if t_val >= trip_time_s - 1.0e-12 else m_pre_gfl
            alg = m_cur._algebraic_tuple(state, p_dispatch_w, 0.0, v0)
            f_est_hz[k] = alg[0] * inv_2pi
            delta_deg[k] = math.degrees(float(state[0]))
            p_pcc_kw[k] = alg[17] / 1000.0
            q_pcc_kvar[k] = alg[18] / 1000.0
            v_pcc_pu[k] = alg[19] / v0
            i2_peak_a[k] = math.hypot(float(state[9]), float(state[10]))

            if k < n_steps - 1:
                m_mid = m_post_gfl if (t_val + 0.5 * dt_s) >= trip_time_s - 1.0e-12 else m_pre_gfl
                m_end = m_post_gfl if (t_val + dt_s) >= trip_time_s - 1.0e-12 else m_pre_gfl
                k1 = m_cur.rhs_continuous(state, p_dispatch_w, 0.0, v0, w0)
                k2 = m_mid.rhs_continuous(state + 0.5 * dt_s * k1, p_dispatch_w, 0.0, v0, w0)
                k3 = m_mid.rhs_continuous(state + 0.5 * dt_s * k2, p_dispatch_w, 0.0, v0, w0)
                k4 = m_end.rhs_continuous(state + dt_s * k3, p_dispatch_w, 0.0, v0, w0)
                state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        if architecture == "gfm_droop":
            label = "GFM Droop (13-State)"
            m_pre_gfm: MultiLoopDroopGFMModel | ElectromagneticGFMModel | MultiLoopDVOCGFMModel = (
                MultiLoopDroopGFMModel(grid=g_pre, lcl=cfg_lcl, pi_cfg=cfg_pi)
            )
            m_post_gfm: MultiLoopDroopGFMModel | ElectromagneticGFMModel | MultiLoopDVOCGFMModel = (
                MultiLoopDroopGFMModel(grid=g_post, lcl=cfg_lcl, pi_cfg=cfg_pi)
            )
            i2_idx, p_idx, q_idx, vpcc_idx = 11, 22, 23, 24
        elif architecture == "gfm_vsg":
            label = "GFM VSG (14-State)"
            m_pre_gfm = ElectromagneticGFMModel(grid=g_pre, lcl=cfg_lcl, pi_cfg=cfg_pi)
            m_post_gfm = ElectromagneticGFMModel(grid=g_post, lcl=cfg_lcl, pi_cfg=cfg_pi)
            i2_idx, p_idx, q_idx, vpcc_idx = 12, 20, 21, 22
        elif architecture == "gfm_dvoc":
            label = "GFM dVOC (12-State)"
            m_pre_gfm = MultiLoopDVOCGFMModel(grid=g_pre, lcl=cfg_lcl, pi_cfg=cfg_pi)
            m_post_gfm = MultiLoopDVOCGFMModel(grid=g_post, lcl=cfg_lcl, pi_cfg=cfg_pi)
            i2_idx, p_idx, q_idx, vpcc_idx = 10, 23, 24, 25
        else:
            raise ValueError(f"Unsupported architecture {architecture!r}")

        state_opt, ok = _solve_gfm_loaded_equilibrium(m_pre_gfm, p_dispatch_w)
        if not ok or state_opt is None:
            raise ValueError(f"Pre-trip GFM equilibrium failed at SCR={scr_pre}")
        state = state_opt

        for k, t_now in enumerate(time_s):
            t_val = float(t_now)
            m_cur_g = m_post_gfm if t_val >= trip_time_s - 1.0e-12 else m_pre_gfm
            alg = m_cur_g._algebraic_tuple(state, p_dispatch_w, 0.0, v0)
            w_inst = float(state[1]) if architecture == "gfm_vsg" else alg[0]
            f_est_hz[k] = w_inst * inv_2pi
            delta_deg[k] = math.degrees(float(state[0]))
            p_pcc_kw[k] = alg[p_idx] / 1000.0
            q_pcc_kvar[k] = alg[q_idx] / 1000.0
            v_pcc_pu[k] = alg[vpcc_idx] / v0
            i2_peak_a[k] = math.hypot(float(state[i2_idx]), float(state[i2_idx + 1]))

            if k < n_steps - 1:
                m_mid_g = m_post_gfm if (t_val + 0.5 * dt_s) >= trip_time_s - 1.0e-12 else m_pre_gfm
                m_end_g = m_post_gfm if (t_val + dt_s) >= trip_time_s - 1.0e-12 else m_pre_gfm
                k1 = m_cur_g.rhs_continuous(state, p_dispatch_w, 0.0, v0, w0)
                k2 = m_mid_g.rhs_continuous(state + 0.5 * dt_s * k1, p_dispatch_w, 0.0, v0, w0)
                k3 = m_mid_g.rhs_continuous(state + 0.5 * dt_s * k2, p_dispatch_w, 0.0, v0, w0)
                k4 = m_end_g.rhs_continuous(state + dt_s * k3, p_dispatch_w, 0.0, v0, w0)
                state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    max_abs_delta = float(np.max(np.abs(delta_deg)))
    min_v = float(np.min(v_pcc_pu))
    lost_sync = bool(max_abs_delta > 180.0 or min_v < 0.50)
    return WeakGridLineTripResult(
        label=label,
        architecture=architecture,
        scr_pre=float(scr_pre),
        scr_post=float(scr_post),
        p_dispatch_w=float(p_dispatch_w),
        trip_time_s=float(trip_time_s),
        time_s=time_s,
        f_est_hz=f_est_hz,
        delta_deg=delta_deg,
        p_pcc_kw=p_pcc_kw,
        q_pcc_kvar=q_pcc_kvar,
        v_pcc_pu=v_pcc_pu,
        i2_peak_a=i2_peak_a,
        lost_synchronism=lost_sync,
        min_v_pcc_pu=min_v,
        max_abs_delta_deg=max_abs_delta,
        final_power_kw=float(p_pcc_kw[-1]),
    )


def run_weak_grid_scr_benchmark(
    *,
    p_dispatch_w: float = 15_000.0,
    scr_values: Sequence[float] = DEFAULT_SCR_SWEEP,
    scr_pre: float = 4.0,
    scr_post: float = 1.22,
    duration_s: float = 1.20,
    dt_s: float = 80.0e-6,
    trip_time_s: float = 0.30,
) -> WeakGridBenchmarkResult:
    """Execute the full GFL SRF-PLL vs. GFM weak-grid SCR stability & collapse benchmark."""
    cfg_lcl = LCLFilterSpec()
    cfg_pi = CascadedPIConfig.from_bandwidths(cfg_lcl)
    sweep_kw = {"scr_values": scr_values, "p_dispatch_w": p_dispatch_w, "lcl": cfg_lcl, "pi_cfg": cfg_pi}

    gfl_fast = sweep_scr_stability("gfl_fast_pll", **sweep_kw)
    gfl_med = sweep_scr_stability("gfl_med_pll", **sweep_kw)
    gfl_slow = sweep_scr_stability("gfl_slow_pll", **sweep_kw)
    vsg_sw = sweep_scr_stability("gfm_vsg", **sweep_kw)
    dvoc_sw = sweep_scr_stability("gfm_dvoc", **sweep_kw)

    crit_fast, _ = estimate_critical_scr(gfl_fast)
    crit_med, _ = estimate_critical_scr(gfl_med)
    crit_slow, static_lim = estimate_critical_scr(gfl_slow)

    trip_kw = {
        "scr_pre": scr_pre,
        "scr_post": scr_post,
        "p_dispatch_w": p_dispatch_w,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "trip_time_s": trip_time_s,
        "lcl": cfg_lcl,
        "pi_cfg": cfg_pi,
    }
    trip_med = simulate_weak_grid_line_trip("gfl_med_pll", **trip_kw)
    trip_slow = simulate_weak_grid_line_trip("gfl_slow_pll", **trip_kw)
    trip_vsg = simulate_weak_grid_line_trip("gfm_vsg", **trip_kw)
    trip_dvoc = simulate_weak_grid_line_trip("gfm_dvoc", **trip_kw)

    scr_arr = np.array([p.scr for p in gfl_med], dtype=float)
    return WeakGridBenchmarkResult(
        p_dispatch_w=float(p_dispatch_w),
        scr_grid=scr_arr,
        gfl_fast_sweep=gfl_fast,
        gfl_med_sweep=gfl_med,
        gfl_slow_sweep=gfl_slow,
        vsg_sweep=vsg_sw,
        dvoc_sweep=dvoc_sw,
        scr_crit_gfl_fast=crit_fast,
        scr_crit_gfl_med=crit_med,
        scr_crit_gfl_slow=crit_slow,
        scr_static_limit_gfl=static_lim,
        trip_gfl_med=trip_med,
        trip_gfl_slow=trip_slow,
        trip_gfm_vsg=trip_vsg,
        trip_gfm_dvoc=trip_dvoc,
    )
