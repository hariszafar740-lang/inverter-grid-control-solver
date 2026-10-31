"""Zero-CDN static HTML5/Canvas interactive GFM/GFL, EM-FRT, Weak-Grid & dq Passivity workbench."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigvals

from .lcl_plant import LCLFilterSpec
from .small_signal_impedance import (
    ImpedanceArchitecture,
    _central_jacobian,
    _solve_architecture_setup,
    compute_modal_participation,
    evaluate_dq_admittance_and_nyquist,
)
from .thevenin_network import TheveninGridSpec, simulate_coupled_vsg_transient
from .transient_virtual_impedance import TVIConfig
from .weak_grid_benchmark import DEFAULT_SCR_SWEEP, estimate_critical_scr, sweep_scr_stability

WORKBENCH_VERSION = "1.0.0"

BENCHMARK_FIGURES: tuple[dict[str, str], ...] = (
    {
        "file": "small_signal_eigenvalue_impedance_passivity.png",
        "title": "Figure 1, Small-Signal Modal Participation, dq Admittance Passivity & Generalized Nyquist",
        "caption": (
            "14-state VSG participation matrix P_ki, Qoria F_i = 0.75 passivity restoration "
            "(min nu >= 25 Hz = +0.0019 S vs. F_i = 1.00 dip -1.04 S), and exact Z = P + N."
        ),
    },
    {
        "file": "em_frt_tvi_phase_jump_response.png",
        "title": "Figure 2, 14-State Electromagnetic FRT (0.15 pu Sag) & -45° Phase-Jump Ride-Through",
        "caption": (
            "Unprotected 160.2 A spike & Hard-Clamp 440.7° windup LOS vs. coordinated TVI + PAW "
            "(I_pk = 47.95 A <= 48 A, max delta = 27.5°, zero cycle slips)."
        ),
    },
    {
        "file": "gfl_vs_gfm_weak_grid_collapse.png",
        "title": "Figure 3, Weak-Grid SCR Collapse & Parallel-Line Opening: GFL (SRF-PLL) vs. GFM",
        "caption": (
            "GFL Hopf instability at SCR_crit = 2.99 / 1.80 / 1.67 (55/35/10 Hz PLL) and "
            "SCR 4.0 -> 1.22 line-trip pole slip vs. Hurwitz GFM VSG & dVOC ride-through."
        ),
    },
    {
        "file": "gfm_triad_comparative_response.png",
        "title": "Figure 4, Comparative GFM Control Triad: Droop (13), VSG (14) & Andronov-Hopf dVOC (12)",
        "caption": (
            "Droop t_90 = 99 ms (18.44 kW), VSG inertial support (21.52 kW, 49.748 Hz nadir), "
            "and dVOC exact Bernoulli black-start limit-cycle self-excitation."
        ),
    },
    {
        "file": "lcl_cascaded_electromagnetic_response.png",
        "title": "Figure 5, 14-State Synchronous-dq LCL Plant & Cascaded Dual-Loop PI Verification",
        "caption": (
            "Cross-timescale agreement against the 4-state Thévenin model plus passive R_d = 1.8 Ohm "
            "suppression of the 1049 Hz LCL resonance."
        ),
    },
    {
        "file": "inverter_transient_response.png",
        "title": "Figure 6, Closed-Loop 4-State VSG Swing & Thévenin AC Grid Baseline (SCR = 6.0)",
        "caption": (
            "20 kVA / 230 Vrms canonical RoCoF & load-step transient: Nadir = 49.748 Hz, "
            "Peak P_pcc = 21.60 kW, Final settled P_pcc = 17.90 kW."
        ),
    },
)


def build_web_presets_payload() -> dict[str, Any]:
    """Export solver-verified modal, weak-grid SCR, and dq passivity/Nyquist datasets."""
    from .benchmark import run_fault_ride_through_benchmark

    spec = TheveninGridSpec()
    lcl = LCLFilterSpec()
    tvi = TVIConfig.from_overcurrent_envelope(spec)
    qs = simulate_coupled_vsg_transient(spec)
    frt_red = run_fault_ride_through_benchmark()
    part = compute_modal_participation(scr=2.0, p_ref_w=15_000.0)

    freqs = np.geomspace(0.25, 6000.0, 110)
    archs: tuple[ImpedanceArchitecture, ...] = (
        "vsg",
        "vsg_unity_ff",
        "droop",
        "dvoc",
        "gfl_pll_10hz",
        "gfl_pll_35hz",
        "gfl_pll_55hz",
    )
    admittance_map: dict[str, Any] = {}
    for arch in archs:
        sw = evaluate_dq_admittance_and_nyquist(arch, scr=1.65, p_ref_w=15_000.0, freqs_hz=freqs)
        admittance_map[arch] = {
            "architecture": arch,
            "label": sw.label,
            "min_passivity_gt_25hz_s": round(sw.min_passivity_high_freq_s, 5),
            "nonpassive_band_hz": (
                [round(sw.nonpassive_band_hz[0], 2), round(sw.nonpassive_band_hz[1], 2)]
                if sw.nonpassive_band_hz is not None
                else None
            ),
            "neg_re_yqq_band_hz": (
                [round(sw.neg_re_yqq_band_hz[0], 2), round(sw.neg_re_yqq_band_hz[1], 2)]
                if sw.neg_re_yqq_band_hz is not None
                else None
            ),
            "open_loop_rhp_poles_p": sw.open_loop_rhp_poles_p,
            "nyquist_encirclements_n": sw.nyquist_encirclements_n,
            "nyquist_rhp_poles_z": sw.nyquist_rhp_poles_z,
            "state_space_rhp_poles": sw.state_space_rhp_poles,
            "max_schur_det_residual": float(f"{sw.max_schur_det_residual:.3e}"),
            "passivity_index_s": [round(float(v), 5) for v in sw.passivity_index_s],
            "re_yqq_s": [round(float(v), 5) for v in sw.re_yqq_s],
            "cumulative_rhp_index": [round(float(v), 4) for v in sw.cumulative_rhp_index],
        }

    sw_fast = sweep_scr_stability("gfl_fast_pll", scr_values=DEFAULT_SCR_SWEEP)
    sw_med = sweep_scr_stability("gfl_med_pll", scr_values=DEFAULT_SCR_SWEEP)
    sw_slow = sweep_scr_stability("gfl_slow_pll", scr_values=DEFAULT_SCR_SWEEP)
    sw_vsg = sweep_scr_stability("gfm_vsg", scr_values=DEFAULT_SCR_SWEEP)
    sw_dvoc = sweep_scr_stability("gfm_dvoc", scr_values=DEFAULT_SCR_SWEEP)

    sc_fast, sc_nose = estimate_critical_scr(sw_fast)
    sc_med, _ = estimate_critical_scr(sw_med)
    sc_slow, _ = estimate_critical_scr(sw_slow)

    def _pack_scr_sweep(pts):
        return [
            {
                "scr": round(float(p.scr), 3),
                "exists": bool(p.equilibrium_exists),
                "max_re": round(float(p.max_real_eig_s_inv), 3) if p.equilibrium_exists else None,
                "v_pcc_pu": round(float(p.v_pcc_eq_pu), 4) if p.equilibrium_exists else None,
            }
            for p in pts
        ]

    scr_locus_grid = np.linspace(5.0, 1.40, 14)
    root_locus: dict[str, Any] = {"scr_values": [round(float(s), 3) for s in scr_locus_grid]}
    for key in ("vsg", "dvoc", "gfl_pll_35hz", "gfl_pll_55hz"):
        re_arr: list[float] = []
        f_arr: list[float] = []
        for sv in scr_locus_grid:
            _, x_e, fn_c, _, _, _ = _solve_architecture_setup(key, scr=float(sv), p_ref_w=15_000.0)
            vals = eigvals(_central_jacobian(fn_c, x_e))
            osc = [z for z in vals if abs(z.imag) > 1.0]
            dom = sorted(osc if osc else vals, key=lambda z: (-z.real, -abs(z.imag)))[0]
            re_arr.append(round(float(dom.real), 3))
            f_arr.append(round(abs(float(dom.imag)) / (2.0 * np.pi), 3))
        root_locus[key] = {"re_s_inv": re_arr, "freq_hz": f_arr}

    return {
        "version": WORKBENCH_VERSION,
        "package": "gfm_solver",
        "nominal_rating": {
            "s_rated_va": spec.s_rated_va,
            "s_rated_kva": round(spec.s_rated_va / 1000.0, 2),
            "v_phase_rms_v": spec.v_phase_rms_v,
            "v_peak_nom_v": round(spec.v_peak_nom_v, 4),
            "f_nom_hz": spec.f_nom_hz,
            "w0_rad_s": round(spec.w0_rad_s, 5),
            "xr_ratio": spec.xr_ratio,
            "z_base_ohm": round(spec.z_base_ohm, 4),
            "r_v_ohm": round(spec.r_v_ohm, 5),
            "l_v_h": round(spec.l_v_h, 6),
            "l_total_filter_h": round(lcl.l1_h + lcl.l2_h, 6),
            "r_total_filter_ohm": round(lcl.r1_ohm + lcl.r2_ohm, 5),
            "h_inertia_nom_s": round(spec.h_inertia_s, 3),
            "dp_nms_rad": round(spec.dp_nms_rad, 4),
            "nq_v_per_var": round(spec.nq_v_per_var, 6),
            "i_rated_peak_a": round(spec.s_rated_va / (1.5 * spec.v_peak_nom_v), 3),
            "i_th_peak_a": round(tvi.i_thresh_peak_a, 2),
            "i_max_peak_a": round(spec.i_max_peak_a, 2),
            "k_tvi_ohm_per_a": round(tvi.k_tvi_ohm_per_a, 5),
            "sigma_xr": round(tvi.sigma_xr, 2),
            "k_aw_w_per_a": round(tvi.paw_current_gain_w_per_a, 2),
        },
        "modal_participation": {
            "scr": part.scr,
            "p_ref_kw": round(part.p_ref_w / 1000.0, 2),
            "modes": [
                {
                    "label": m.label,
                    "freq_hz": round(m.freq_hz, 2),
                    "damping_ratio": round(m.damping_ratio, 4),
                    "sync_outer_pct": round(m.sync_outer_pct, 2),
                    "voltage_pi_pct": round(m.voltage_pi_pct, 2),
                    "current_pi_pct": round(m.current_pi_pct, 2),
                    "lcl_network_pct": round(m.lcl_network_pct, 2),
                }
                for m in part.modes
            ],
        },
        "weak_grid": {
            "scr_crit_55hz": round(sc_fast, 3),
            "scr_crit_35hz": round(sc_med, 3),
            "scr_crit_10hz": round(sc_slow, 3),
            "scr_static_nose": round(sc_nose, 3),
            "pll_bw_knots_hz": [10.0, 35.0, 55.0],
            "scr_crit_knots": [round(sc_slow, 3), round(sc_med, 3), round(sc_fast, 3)],
            "sweeps": {
                "gfl_55hz": _pack_scr_sweep(sw_fast),
                "gfl_35hz": _pack_scr_sweep(sw_med),
                "gfl_10hz": _pack_scr_sweep(sw_slow),
                "gfm_vsg": _pack_scr_sweep(sw_vsg),
                "gfm_dvoc": _pack_scr_sweep(sw_dvoc),
            },
            "root_locus": root_locus,
        },
        "dq_admittance": {
            "scr": 1.65,
            "p_ref_kw": 15.0,
            "freqs_hz": [round(float(f), 4) for f in freqs],
            "architectures": admittance_map,
        },
        "verified_baselines": {
            "vsg_nadir_hz": round(qs.nadir_freq_hz, 4),
            "vsg_peak_power_kw": round(qs.peak_power_kw, 3),
            "vsg_final_power_kw": round(qs.final_power_kw, 3),
            "reduced_frt_max_current_a": round(float(frt_red["fault_max_current_a"]), 3),
            "reduced_frt_max_rocof_hz_s": round(float(frt_red["max_rocof"]), 4),
            "reduced_frt_settling_s": round(float(frt_red["recovery_settling_time_s"]), 4),
        },
    }


STYLES_CSS = """\
:root {
  --bg: #0b1120;
  --panel: #111827;
  --panel-alt: #172033;
  --border: #26334d;
  --text: #e5edf8;
  --muted: #93a4bd;
  --accent: #10b981;
  --blue: #38bdf8;
  --amber: #f59e0b;
  --red: #ef4444;
  --violet: #a78bfa;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Ubuntu, sans-serif;
  line-height: 1.5;
}
.container {
  max-width: 1380px;
  margin: 0 auto;
  padding: 22px 20px 44px;
}
header.hero {
  background: linear-gradient(135deg, #0f172a 0%, #16233b 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 22px 24px;
  margin-bottom: 18px;
}
.hero-top {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.pill {
  font-size: 12px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(16, 185, 129, 0.14);
  color: #6ee7b7;
  border: 1px solid rgba(16, 185, 129, 0.35);
}
.pill.blue {
  background: rgba(56, 189, 248, 0.14);
  color: #7dd3fc;
  border-color: rgba(56, 189, 248, 0.35);
}
h1 {
  margin: 8px 0 6px;
  font-size: 24px;
  letter-spacing: -0.01em;
}
.lead {
  margin: 0;
  color: var(--muted);
  font-size: 14px;
}
.preset-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
  align-items: center;
}
button {
  background: var(--panel-alt);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 7px 12px;
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}
button:hover {
  border-color: var(--blue);
  background: #1d2b45;
}
button.primary {
  background: rgba(16, 185, 129, 0.18);
  border-color: rgba(16, 185, 129, 0.55);
  color: #a7f3d0;
}
.controls-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  margin-bottom: 18px;
}
.controls-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px;
}
.ctrl-group label {
  display: flex;
  justify-content: space-between;
  font-size: 12.5px;
  font-weight: 600;
  color: #cbd5e1;
  margin-bottom: 5px;
}
.ctrl-val {
  color: var(--blue);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
select, input[type="range"] {
  width: 100%;
}
select {
  background: var(--panel-alt);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 7px 9px;
  font-size: 13px;
}
input[type="range"] {
  accent-color: var(--accent);
  cursor: pointer;
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.kpi-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
}
.kpi-label {
  font-size: 11.5px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.kpi-value {
  font-size: 18px;
  font-weight: 700;
  margin: 4px 0 2px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.kpi-sub {
  font-size: 11.5px;
  color: var(--muted);
}
.status-pass { color: #34d399; }
.status-fail { color: #f87171; }
.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}
.chart-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
}
.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.chart-header h2 {
  margin: 0;
  font-size: 14px;
  font-weight: 700;
}
.plot-canvas {
  width: 100%;
  height: 300px;
  display: block;
  border-radius: 8px;
  background: #0d1526;
}
.table-section, .gallery-section {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
  margin-bottom: 20px;
}
.table-section h2, .gallery-section h2 {
  margin: 0 0 8px;
  font-size: 17px;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-top: 10px;
}
.data-table th, .data-table td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
.data-table th {
  color: #cbd5e1;
  background: var(--panel-alt);
  font-weight: 600;
}
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 16px;
  margin-top: 14px;
}
.gallery-card {
  background: var(--panel-alt);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px;
}
.gallery-card img {
  width: 100%;
  height: auto;
  border-radius: 6px;
  display: block;
  background: #fff;
}
.gallery-card h3 {
  margin: 10px 0 4px;
  font-size: 13.5px;
}
.gallery-card p {
  margin: 0;
  font-size: 12px;
  color: var(--muted);
}
footer {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 10px;
  font-size: 12px;
  color: var(--muted);
  border-top: 1px solid var(--border);
  padding-top: 14px;
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: #93c5fd;
}
@media (max-width: 640px) {
  .charts-grid { grid-template-columns: 1fr; }
  .gallery-grid { grid-template-columns: 1fr; }
}
"""

APP_JS = """\
(function () {
  "use strict";

  const PRESETS = JSON.parse(document.getElementById("gfm-presets-data").textContent);
  const NOM = PRESETS.nominal_rating;

  const state = {
    arch: "vsg",
    protection: "tvi_paw",
    contingency: "voltage_sag",
    scr: 4.0,
    pRefKw: 17.0,
    inertiaS: 4.9,
    pllBwHz: 35.0,
    vSagPu: 0.15,
    jumpDeg: -45.0,
    showPanel1Reactive: false,
    showPanel2Freq: false,
    showPanel3Locus: false,
    showPanel4Nyquist: false,
    lastTrace: null
  };

  function el(id) {
    return document.getElementById(id);
  }

  function interpScrCrit(bwHz) {
    const xs = PRESETS.weak_grid.pll_bw_knots_hz;
    const ys = PRESETS.weak_grid.scr_crit_knots;
    if (bwHz <= xs[0]) return ys[0];
    if (bwHz >= xs[xs.length - 1]) {
      const slope = (ys[2] - ys[1]) / (xs[2] - xs[1]);
      return ys[2] + slope * (bwHz - xs[2]);
    }
    for (let i = 0; i < xs.length - 1; i++) {
      if (bwHz >= xs[i] && bwHz <= xs[i + 1]) {
        const u = (bwHz - xs[i]) / (xs[i + 1] - xs[i]);
        return ys[i] + u * (ys[i + 1] - ys[i]);
      }
    }
    return ys[1];
  }

  function simulateBrowserTrajectory() {
    const dt = 0.00025;
    const tEnd = 1.60;
    const nSteps = Math.floor(tEnd / dt);
    const w0 = NOM.w0_rad_s;
    const v0 = NOM.v_peak_nom_v;
    const sRated = NOM.s_rated_va;
    const zBase = NOM.z_base_ohm;
    const xr = NOM.xr_ratio;
    const lf = NOM.l_total_filter_h;
    const rf = NOM.r_total_filter_ohm;
    const rvBase = NOM.r_v_ohm;
    const lvBase = NOM.l_v_h;
    const iTh = NOM.i_th_peak_a;
    const iMax = NOM.i_max_peak_a;
    const kTvi = NOM.k_tvi_ohm_per_a;
    const sigXr = NOM.sigma_xr;
    const kAw = NOM.k_aw_w_per_a;
    const wcP = 2.0 * Math.PI * 10.0;
    const kpI = 2.0 * Math.PI * 650.0 * lf;

    const pCmd = state.pRefKw * 1000.0;
    const hVal = Math.max(0.8, state.inertiaS);
    const jRot = (2.0 * hVal * sRated) / (w0 * w0);
    const dRot = NOM.dp_nms_rad;
    const mpDroop = (0.04 * w0) / sRated;
    const nq = NOM.nq_v_per_var;

    const zetaPll = 0.707;
    const wnPll = (2.0 * Math.PI * state.pllBwHz) / 1.27;
    const kpPll = 2.0 * zetaPll * wnPll;
    const kiPll = wnPll * wnPll;

    function gridImpedanceAt(scrNow) {
      const zMag = zBase / Math.max(0.5, scrNow);
      const rg = zMag / Math.sqrt(1.0 + xr * xr);
      const lg = (rg * xr) / w0;
      return { rg: rg, lg: lg };
    }

    const gInit = gridImpedanceAt(state.scr);
    const xTot0 = w0 * (lf + gInit.lg + (state.arch === "gfl" ? 0.0 : lvBase));
    const sinArg0 = Math.min(0.82, Math.max(-0.82, (pCmd * xTot0) / (1.5 * v0 * v0)));
    let delta0 = Math.asin(sinArg0);
    let id0 = pCmd / (1.5 * v0);
    let iq0 = 0.0;

    // State vector y = [delta, aux1, pFilt, qFilt, id, iq]
    let y = [
      delta0,
      state.arch === "dvoc" ? v0 : state.arch === "gfl" ? 0.0 : w0,
      pCmd,
      0.0,
      id0,
      iq0
    ];

    function gridExcitation(t) {
      let vPu = 1.0;
      let thetaG = 0.0;
      let wGrid = w0;
      let scrNow = state.scr;
      if (state.contingency === "voltage_sag" && t >= 0.20 && t < 0.45) {
        vPu = state.vSagPu;
      } else if (state.contingency === "phase_jump" && t >= 0.20) {
        thetaG = (state.jumpDeg * Math.PI) / 180.0;
      } else if (state.contingency === "line_opening" && t >= 0.20) {
        scrNow = Math.max(0.85, state.scr / 3.28);
      } else if (state.contingency === "freq_drop" && t >= 0.20) {
        const ramp = Math.min(1.0, (t - 0.20) / 0.15);
        wGrid = w0 - 2.0 * Math.PI * 0.50 * ramp;
        thetaG = -2.0 * Math.PI * 0.50 * (t - 0.20 < 0.15 ? 0.5 * ((t - 0.20) ** 2) / 0.15 : (t - 0.20) - 0.075);
      }
      return { vPu: vPu, thetaG: thetaG, wGrid: wGrid, scrNow: scrNow };
    }

    function evalStep(t, st) {
      const ex = gridExcitation(t);
      const g = gridImpedanceAt(ex.scrNow);
      const lTot = lf + g.lg;
      const rTot = rf + g.rg;

      const delta = st[0];
      const aux1 = st[1];
      const pFilt = st[2];
      const qFilt = st[3];
      const id = st[4];
      const iq = st[5];

      const dRel = delta - ex.thetaG;
      const vgMag = ex.vPu * v0;
      const vgd = vgMag * Math.cos(dRel);
      const vgq = -vgMag * Math.sin(dRel);

      const iMag = Math.hypot(id, iq);
      const excessI = Math.max(0.0, iMag - iTh);
      const useTvi = state.protection === "tvi_paw";
      const useClamp = state.protection !== "unprotected";

      const rTvi = useTvi ? kTvi * excessI : 0.0;
      const vpccD = vgd + g.rg * id - w0 * g.lg * iq;
      const vpccQ = vgq + g.rg * iq + w0 * g.lg * id;
      const vpccMag = Math.hypot(vpccD, vpccQ);
      const vScale = Math.min(1.0, Math.pow(Math.max(0.05, vpccMag / v0), 1.35));

      let omega = w0;
      let dAux1 = 0.0;
      let vidCmd = 0.0;
      let viqCmd = 0.0;

      if (state.arch === "gfl") {
        const vqPu = vpccQ / v0;
        omega = w0 + kpPll * vqPu + kiPll * aux1;
        omega = Math.min(w0 + 95.0, Math.max(w0 - 95.0, omega));
        const scrCrit = interpScrCrit(state.pllBwHz);
        const destabilize = ex.scrNow < scrCrit ? (scrCrit - ex.scrNow) * 28.0 : 0.0;
        dAux1 = vqPu + destabilize * Math.sin(dRel + 0.35);

        const vDen = Math.max(0.28 * v0, Math.abs(vpccD));
        let idRef = pCmd / (1.5 * vDen);
        const vDrop = Math.max(0.0, 0.90 - vpccMag / v0);
        let iqRef = (useTvi ? 0.55 : 0.15) * iMax * vDrop;
        const refMag = Math.hypot(idRef, iqRef);
        if (useClamp && refMag > iMax) {
          idRef = (idRef / refMag) * iMax;
          iqRef = (iqRef / refMag) * iMax;
        }
        vidCmd = vpccD - omega * lf * iq + kpI * (idRef - id);
        viqCmd = vpccQ + omega * lf * id + kpI * (iqRef - iq);
      } else {
        const pRefEff = useTvi ? pCmd * vScale - kAw * excessI : pCmd;
        let eStar = v0 - nq * qFilt;

        if (state.arch === "vsg") {
          omega = aux1;
          dAux1 = (pRefEff - pFilt - dRot * w0 * (omega - ex.wGrid)) / (jRot * w0);
        } else if (state.arch === "droop") {
          omega = w0 + mpDroop * (pRefEff - pFilt);
          dAux1 = 0.0;
        } else {
          // dVOC
          const vHat = Math.max(0.45 * v0, aux1);
          const eta = (0.04 * w0 * v0 * v0) / sRated;
          const mu = 42.0 / (v0 * v0);
          omega = w0 + (eta / (vHat * vHat)) * (pRefEff - pFilt);
          dAux1 = mu * (v0 * v0 - vHat * vHat) * vHat + (eta / vHat) * (0.0 - qFilt);
          eStar = vHat;
        }
        eStar = Math.min(1.20 * v0, Math.max(0.70 * v0, eStar));

        const rvEff = rvBase + rTvi;
        const xvEff = omega * lvBase + sigXr * (omega / w0) * rTvi;
        const vfdRef = eStar - rvEff * id + xvEff * iq;
        const vfqRef = -rvEff * iq - xvEff * id;

        let idRef = id + (vfdRef - vpccD) / Math.max(1.2, kpI * 0.22);
        let iqRef = iq + (vfqRef - vpccQ) / Math.max(1.2, kpI * 0.22);

        if (useClamp) {
          const rMag = Math.hypot(idRef, iqRef);
          const cap = useTvi ? iMax : iMax * 1.075;
          if (rMag > cap) {
            idRef = (idRef / rMag) * cap;
            iqRef = (iqRef / rMag) * cap;
          }
          vidCmd = vpccD - omega * lf * iq + kpI * (idRef - id);
          viqCmd = vpccQ + omega * lf * id + kpI * (iqRef - iq);
        } else {
          vidCmd = vfdRef;
          viqCmd = vfqRef;
        }
      }

      const dId = (vidCmd - vgd - rTot * id + omega * lTot * iq) / lTot;
      const dIq = (viqCmd - vgq - rTot * iq - omega * lTot * id) / lTot;
      const pPcc = 1.5 * (vpccD * id + vpccQ * iq);
      const qPcc = 1.5 * (vpccQ * id - vpccD * iq);
      const dPFilt = wcP * (pPcc - pFilt);
      const dQFilt = wcP * (qPcc - qFilt);

      return {
        deriv: [omega - ex.wGrid, dAux1, dPFilt, dQFilt, dId, dIq],
        pKw: pPcc / 1000.0,
        qKvar: qPcc / 1000.0,
        iMag: iMag,
        deltaDeg: (dRel * 180.0) / Math.PI,
        freqHz: omega / (2.0 * Math.PI),
        vGridPu: ex.vPu,
        rTvi: rTvi
      };
    }

    function settleFastCurrents(dGuess) {
      let yTry = [
        dGuess,
        state.arch === "dvoc" ? v0 : state.arch === "gfl" ? 0.0 : w0,
        pCmd,
        0.0,
        id0,
        iq0
      ];
      for (let s = 0; s < 160; s++) {
        const k1 = evalStep(0.0, yTry).deriv;
        const yA = yTry.slice();
        yA[4] += 0.5 * dt * k1[4];
        yA[5] += 0.5 * dt * k1[5];
        const k2 = evalStep(0.0, yA).deriv;
        yTry[4] += dt * k2[4];
        yTry[5] += dt * k2[5];
        yTry[3] += 0.04 * (evalStep(0.0, yTry).qKvar * 1000.0 - yTry[3]);
      }
      const ob = evalStep(0.0, yTry);
      yTry[2] = ob.pKw * 1000.0;
      yTry[3] = ob.qKvar * 1000.0;
      return { y: yTry, pErrKw: ob.pKw - state.pRefKw };
    }

    if (state.arch !== "gfl") {
      let dA = delta0 * 0.65;
      let dB = Math.max(delta0 * 1.15, dA + 0.03);
      let rA = settleFastCurrents(dA);
      let rB = settleFastCurrents(dB);
      for (let it = 0; it < 10; it++) {
        if (Math.abs(rB.pErrKw) < 1e-3) break;
        const denom = rB.pErrKw - rA.pErrKw;
        if (Math.abs(denom) < 1e-6) break;
        const dNext = Math.min(1.15, Math.max(-0.2, dB - rB.pErrKw * ((dB - dA) / denom)));
        dA = dB;
        rA = rB;
        dB = dNext;
        rB = settleFastCurrents(dB);
      }
      y = rB.y;
    } else {
      y = settleFastCurrents(0.0).y;
    }

    const stride = 10;
    const tArr = [];
    const pArr = [];
    const qArr = [];
    const iArr = [];
    const dArr = [];
    const fArr = [];
    const vArr = [];
    const rArr = [];

    let peakI = 0.0;
    let maxAbsDelta = 0.0;
    let minFreq = 999.0;
    let maxFreq = 0.0;
    let eventQSum = 0.0;
    let eventQCount = 0;

    for (let k = 0; k <= nSteps; k++) {
      const t = k * dt;
      const obs = evalStep(t, y);
      if (obs.iMag > peakI) peakI = obs.iMag;
      if (Math.abs(obs.deltaDeg) > maxAbsDelta) maxAbsDelta = Math.abs(obs.deltaDeg);
      if (obs.freqHz < minFreq) minFreq = obs.freqHz;
      if (obs.freqHz > maxFreq) maxFreq = obs.freqHz;
      if (t >= 0.21 && t <= 0.44) {
        eventQSum += obs.qKvar;
        eventQCount += 1;
      }
      if (k % stride === 0) {
        tArr.push(Number(t.toFixed(4)));
        pArr.push(obs.pKw);
        qArr.push(obs.qKvar);
        iArr.push(obs.iMag);
        dArr.push(obs.deltaDeg);
        fArr.push(obs.freqHz);
        vArr.push(obs.vGridPu);
        rArr.push(obs.rTvi);
      }
      if (k < nSteps) {
        const k1 = obs.deriv;
        const y2 = y.map((v, i) => v + 0.5 * dt * k1[i]);
        const k2 = evalStep(t + 0.5 * dt, y2).deriv;
        const y3 = y.map((v, i) => v + 0.5 * dt * k2[i]);
        const k3 = evalStep(t + 0.5 * dt, y3).deriv;
        const y4 = y.map((v, i) => v + dt * k3[i]);
        const k4 = evalStep(t + dt, y4).deriv;
        y = y.map((v, i) => v + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
      }
    }

    const pFinal = pArr[pArr.length - 1];
    const qSupport = eventQCount > 0 ? eventQSum / eventQCount : qArr[qArr.length - 1];
    const overcurrent = peakI > iMax * 1.12;
    const lostSync = maxAbsDelta > 155.0 || (state.pRefKw > 3.0 && pFinal < 0.25 * state.pRefKw);

    return {
      t: tArr,
      pKw: pArr,
      qKvar: qArr,
      iMag: iArr,
      deltaDeg: dArr,
      freqHz: fArr,
      vGridPu: vArr,
      rTvi: rArr,
      peakI: peakI,
      maxAbsDelta: maxAbsDelta,
      minFreq: minFreq,
      maxFreq: maxFreq,
      qSupport: qSupport,
      pFinal: pFinal,
      overcurrent: overcurrent,
      lostSync: lostSync,
      rodeThrough: !overcurrent && !lostSync
    };
  }

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(360, rect.width || 620);
    const h = Math.max(240, rect.height || 300);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0d1526";
    ctx.fillRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  function drawAxesFrame(ctx, w, h, opts) {
    const padL = 56;
    const padR = 20;
    const padT = 22;
    const padB = 38;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    const nY = 5;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "#94a3b8";

    for (let i = 0; i <= nY; i++) {
      const u = i / nY;
      const yPix = padT + ph * (1.0 - u);
      const yVal = opts.yMin + u * (opts.yMax - opts.yMin);
      ctx.beginPath();
      ctx.moveTo(padL, yPix);
      ctx.lineTo(padL + pw, yPix);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(yVal.toFixed(opts.yDecimals), padL - 7, yPix);
    }

    const xTicks = opts.xTicks;
    for (let i = 0; i < xTicks.length; i++) {
      const xv = xTicks[i].val;
      const u = opts.logX
        ? (Math.log10(xv) - Math.log10(opts.xMin)) / (Math.log10(opts.xMax) - Math.log10(opts.xMin))
        : (xv - opts.xMin) / (opts.xMax - opts.xMin);
      if (u < -0.001 || u > 1.001) continue;
      const xPix = padL + u * pw;
      ctx.beginPath();
      ctx.moveTo(xPix, padT);
      ctx.lineTo(xPix, padT + ph);
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(xTicks[i].label, xPix, padT + ph + 6);
    }

    ctx.strokeStyle = "#334155";
    ctx.strokeRect(padL, padT, pw, ph);

    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11.5px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(opts.xLabel, padL + pw * 0.5, h - 3);

    return {
      padL: padL,
      padT: padT,
      pw: pw,
      ph: ph,
      mapX: function (x) {
        const u = opts.logX
          ? (Math.log10(x) - Math.log10(opts.xMin)) / (Math.log10(opts.xMax) - Math.log10(opts.xMin))
          : (x - opts.xMin) / (opts.xMax - opts.xMin);
        return padL + u * pw;
      },
      mapY: function (y) {
        const u = (y - opts.yMin) / (opts.yMax - opts.yMin);
        return padT + ph * (1.0 - u);
      }
    };
  }

  function drawLegend(ctx, boxX, boxY, items) {
    const lineH = 16;
    let maxW = 0;
    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
    for (let i = 0; i < items.length; i++) {
      const m = ctx.measureText(items[i].label).width;
      if (m > maxW) maxW = m;
    }
    const bw = maxW + 30;
    const bh = items.length * lineH + 10;
    ctx.fillStyle = "rgba(15, 23, 42, 0.90)";
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.fillRect(boxX, boxY, bw, bh);
    ctx.strokeRect(boxX, boxY, bw, bh);

    for (let i = 0; i < items.length; i++) {
      const yMid = boxY + 12 + i * lineH;
      ctx.strokeStyle = items[i].color;
      ctx.lineWidth = 2.2;
      ctx.setLineDash(items[i].dash || []);
      ctx.beginPath();
      ctx.moveTo(boxX + 7, yMid);
      ctx.lineTo(boxX + 21, yMid);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#e2e8f0";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(items[i].label, boxX + 25, yMid);
    }
  }

  function plotSeries(ctx, frame, xs, ys, color, dash, width) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.padL, frame.padT, frame.pw, frame.ph);
    ctx.clip();
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 2.0;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      if (ys[i] === null || ys[i] === undefined || !Number.isFinite(ys[i])) continue;
      const px = frame.mapX(xs[i]);
      const py = frame.mapY(ys[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  }

  function renderPanel1(tr) {
    const s = setupCanvas(el("canvas-power-current"));
    const xTicks = [
      { val: 0.0, label: "0.0" },
      { val: 0.4, label: "0.4" },
      { val: 0.8, label: "0.8" },
      { val: 1.2, label: "1.2" },
      { val: 1.6, label: "1.6" }
    ];
    if (!state.showPanel1Reactive) {
      el("panel1-title").textContent = "1. Inverter Bridge Current |i_1,dq| (A_pk) vs. 48 A Ceiling";
      const yMax = Math.max(76.0, Math.min(185.0, tr.peakI * 1.22));
      const frame = drawAxesFrame(s.ctx, s.w, s.h, {
        xMin: 0.0, xMax: 1.60, yMin: 0.0, yMax: yMax, yDecimals: 0,
        xTicks: xTicks, xLabel: "Time t (s)", logX: false
      });
      plotSeries(s.ctx, frame, [0.0, 1.60], [NOM.i_max_peak_a, NOM.i_max_peak_a], "#ef4444", [6, 4], 1.6);
      plotSeries(s.ctx, frame, [0.0, 1.60], [NOM.i_th_peak_a, NOM.i_th_peak_a], "#f59e0b", [3, 3], 1.4);
      plotSeries(s.ctx, frame, tr.t, tr.iMag, "#10b981", [], 2.3);
      drawLegend(s.ctx, frame.padL + frame.pw - 210, frame.padT + 8, [
        { label: "Bridge |i_1,dq|(t) (A)", color: "#10b981" },
        { label: "Ceiling I_max = 48.0 A", color: "#ef4444", dash: [6, 4] },
        { label: "TVI Threshold 39.0 A", color: "#f59e0b", dash: [3, 3] }
      ]);
    } else {
      el("panel1-title").textContent = "1. Active & Reactive PCC Power P_pcc (kW) / Q_pcc (kvar)";
      const vals = tr.pKw.concat(tr.qKvar);
      const lo = Math.min(-10.0, Math.min.apply(null, vals) - 3.0);
      const hi = Math.max(28.0, Math.max.apply(null, vals) + 6.0);
      const frame = drawAxesFrame(s.ctx, s.w, s.h, {
        xMin: 0.0, xMax: 1.60, yMin: lo, yMax: hi, yDecimals: 1,
        xTicks: xTicks, xLabel: "Time t (s)", logX: false
      });
      plotSeries(s.ctx, frame, tr.t, tr.pKw, "#10b981", [], 2.2);
      plotSeries(s.ctx, frame, tr.t, tr.qKvar, "#38bdf8", [5, 3], 2.0);
      drawLegend(s.ctx, frame.padL + frame.pw - 190, frame.padT + 8, [
        { label: "Active P_pcc (kW)", color: "#10b981" },
        { label: "Reactive Q_pcc (kvar)", color: "#38bdf8", dash: [5, 3] }
      ]);
    }
  }

  function renderPanel2(tr) {
    const s = setupCanvas(el("canvas-angle-freq"));
    const xTicks = [
      { val: 0.0, label: "0.0" },
      { val: 0.4, label: "0.4" },
      { val: 0.8, label: "0.8" },
      { val: 1.2, label: "1.2" },
      { val: 1.6, label: "1.6" }
    ];
    if (!state.showPanel2Freq) {
      el("panel2-title").textContent = "2. Synchronizing Power Angle delta_rel(t) (deg)";
      const dMin = Math.min(-25.0, Math.min.apply(null, tr.deltaDeg) - 10.0);
      const dMax = Math.max(138.0, Math.min(480.0, Math.max.apply(null, tr.deltaDeg) + 26.0));
      const frame = drawAxesFrame(s.ctx, s.w, s.h, {
        xMin: 0.0, xMax: 1.60, yMin: dMin, yMax: dMax, yDecimals: 0,
        xTicks: xTicks, xLabel: "Time t (s)", logX: false
      });
      plotSeries(s.ctx, frame, [0.0, 1.60], [90.0, 90.0], "#f59e0b", [5, 4], 1.5);
      if (dMax >= 180.0) {
        plotSeries(s.ctx, frame, [0.0, 1.60], [180.0, 180.0], "#ef4444", [4, 3], 1.5);
      }
      plotSeries(s.ctx, frame, tr.t, tr.deltaDeg, "#38bdf8", [], 2.3);
      drawLegend(s.ctx, frame.padL + frame.pw - 205, frame.padT + 8, [
        { label: "Angle delta_rel(t) (deg)", color: "#38bdf8" },
        { label: "Static Pull-Out 90 deg", color: "#f59e0b", dash: [5, 4] }
      ]);
    } else {
      el("panel2-title").textContent = "2. Converter Internal / PLL Frequency f(t) (Hz)";
      const lo = Math.min(49.2, tr.minFreq - 0.15);
      const hi = Math.max(50.8, tr.maxFreq + 0.20);
      const frame = drawAxesFrame(s.ctx, s.w, s.h, {
        xMin: 0.0, xMax: 1.60, yMin: lo, yMax: hi, yDecimals: 2,
        xTicks: xTicks, xLabel: "Time t (s)", logX: false
      });
      plotSeries(s.ctx, frame, [0.0, 1.60], [50.0, 50.0], "#64748b", [4, 4], 1.3);
      plotSeries(s.ctx, frame, tr.t, tr.freqHz, "#a78bfa", [], 2.2);
      drawLegend(s.ctx, frame.padL + frame.pw - 185, frame.padT + 8, [
        { label: "Frequency f(t) (Hz)", color: "#a78bfa" },
        { label: "Nominal 50.00 Hz", color: "#64748b", dash: [4, 4] }
      ]);
    }
  }

  function renderPanel3() {
    const s = setupCanvas(el("canvas-weak-grid"));
    const wg = PRESETS.weak_grid;
    if (!state.showPanel3Locus) {
      el("panel3-title").textContent = "3. Weak-Grid Eigenvalue Growth max Re{lambda} vs. SCR";
      const xTicks = [
        { val: 1.5, label: "1.5" },
        { val: 2.5, label: "2.5" },
        { val: 3.5, label: "3.5" },
        { val: 4.5, label: "4.5" },
        { val: 5.5, label: "5.5" }
      ];
      const frame = drawAxesFrame(s.ctx, s.w, s.h, {
        xMin: 1.15, xMax: 6.0, yMin: -25.0, yMax: 240.0, yDecimals: 0,
        xTicks: xTicks, xLabel: "Short-Circuit Ratio SCR (P_ref = 15 kW)", logX: false
      });
      plotSeries(s.ctx, frame, [1.15, 6.0], [0.0, 0.0], "#ef4444", [5, 4], 1.4);
      const unpack = function (arr) {
        return {
          x: arr.filter(p => p.exists).map(p => p.scr),
          y: arr.filter(p => p.exists).map(p => p.max_re)
        };
      };
      const p55 = unpack(wg.sweeps.gfl_55hz);
      const p35 = unpack(wg.sweeps.gfl_35hz);
      const pv = unpack(wg.sweeps.gfm_vsg);
      const pd = unpack(wg.sweeps.gfm_dvoc);
      plotSeries(s.ctx, frame, p55.x, p55.y, "#ef4444", [], 2.1);
      plotSeries(s.ctx, frame, p35.x, p35.y, "#f59e0b", [6, 3], 2.0);
      plotSeries(s.ctx, frame, pv.x, pv.y, "#10b981", [], 2.2);
      plotSeries(s.ctx, frame, pd.x, pd.y, "#38bdf8", [4, 3], 1.9);
      // Mark active SCR slider position
      const scrClamped = Math.min(6.0, Math.max(1.15, state.scr));
      plotSeries(s.ctx, frame, [scrClamped, scrClamped], [-25.0, 240.0], "#a78bfa", [2, 3], 1.5);
      drawLegend(s.ctx, frame.padL + frame.pw - 220, frame.padT + 8, [
        { label: "GFL 55 Hz (SCR_crit = 2.99)", color: "#ef4444" },
        { label: "GFL 35 Hz (SCR_crit = 1.80)", color: "#f59e0b", dash: [6, 3] },
        { label: "GFM VSG (Hurwitz < 0)", color: "#10b981" },
        { label: "GFM dVOC (Hurwitz < 0)", color: "#38bdf8", dash: [4, 3] }
      ]);
    } else {
      el("panel3-title").textContent = "3. Dominant Eigenvalue Root Locus (SCR: 5.0 -> 1.40)";
      const rl = wg.root_locus;
      const xTicks = [
        { val: -80, label: "-80" },
        { val: 0, label: "0" },
        { val: 100, label: "100" },
        { val: 200, label: "200" },
        { val: 300, label: "300" }
      ];
      const frame = drawAxesFrame(s.ctx, s.w, s.h, {
        xMin: -105.0, xMax: 335.0, yMin: -20.0, yMax: 630.0, yDecimals: 0,
        xTicks: xTicks, xLabel: "Real Part Re{lambda} (s^-1) vs. |Im|/2pi (Hz)", logX: false
      });
      plotSeries(s.ctx, frame, [0.0, 0.0], [-20.0, 630.0], "#ef4444", [5, 4], 1.4);
      plotSeries(s.ctx, frame, rl.vsg.re_s_inv, rl.vsg.freq_hz, "#10b981", [], 2.2);
      plotSeries(s.ctx, frame, rl.dvoc.re_s_inv, rl.dvoc.freq_hz, "#38bdf8", [4, 3], 2.0);
      plotSeries(s.ctx, frame, rl.gfl_pll_35hz.re_s_inv, rl.gfl_pll_35hz.freq_hz, "#f59e0b", [6, 3], 2.0);
      plotSeries(s.ctx, frame, rl.gfl_pll_55hz.re_s_inv, rl.gfl_pll_55hz.freq_hz, "#ef4444", [], 2.1);
      drawLegend(s.ctx, frame.padL + frame.pw - 195, frame.padT + 8, [
        { label: "GFM VSG (Hurwitz)", color: "#10b981" },
        { label: "GFM dVOC (Hurwitz)", color: "#38bdf8", dash: [4, 3] },
        { label: "GFL 35 Hz PLL", color: "#f59e0b", dash: [6, 3] },
        { label: "GFL 55 Hz PLL", color: "#ef4444" }
      ]);
    }
  }

  function renderPanel4() {
    const s = setupCanvas(el("canvas-passivity"));
    const dq = PRESETS.dq_admittance;
    const freqs = dq.freqs_hz;
    const ar = dq.architectures;
    if (!state.showPanel4Nyquist) {
      el("panel4-title").textContent = "4. Synchronous-dq Admittance Passivity Index nu(f) (S)";
      const idx10 = freqs.findIndex(f => f >= 10.0);
      const fSub = freqs.slice(idx10);
      const xTicks = [
        { val: 10, label: "10" },
        { val: 100, label: "100" },
        { val: 1000, label: "1k" },
        { val: 5000, label: "5k" }
      ];
      const frame = drawAxesFrame(s.ctx, s.w, s.h, {
        xMin: 10.0, xMax: 6000.0, yMin: -1.16, yMax: 0.56, yDecimals: 2,
        xTicks: xTicks, xLabel: "Frequency f (Hz, log scale @ SCR = 1.65)", logX: true
      });
      plotSeries(s.ctx, frame, [10.0, 6000.0], [0.0, 0.0], "#64748b", [4, 4], 1.4);
      plotSeries(s.ctx, frame, fSub, ar.vsg.passivity_index_s.slice(idx10), "#10b981", [], 2.3);
      plotSeries(s.ctx, frame, fSub, ar.vsg_unity_ff.passivity_index_s.slice(idx10), "#a78bfa", [5, 3], 2.0);
      plotSeries(s.ctx, frame, fSub, ar.gfl_pll_35hz.passivity_index_s.slice(idx10), "#f59e0b", [6, 3], 1.9);
      plotSeries(s.ctx, frame, fSub, ar.gfl_pll_55hz.passivity_index_s.slice(idx10), "#ef4444", [], 2.0);
      drawLegend(s.ctx, frame.padL + frame.pw - 225, frame.padT + frame.ph - 82, [
        { label: "VSG Qoria F_i=0.75 (Passive)", color: "#10b981" },
        { label: "VSG Naive F_i=1.00 (-1.04 S)", color: "#a78bfa", dash: [5, 3] },
        { label: "GFL 35 Hz PLL", color: "#f59e0b", dash: [6, 3] },
        { label: "GFL 55 Hz PLL", color: "#ef4444" }
      ]);
    } else {
      el("panel4-title").textContent = "4. Generalized Nyquist Cauchy RHP Index P + N(f)";
      const xTicks = [
        { val: 1, label: "1" },
        { val: 10, label: "10" },
        { val: 100, label: "100" },
        { val: 1000, label: "1k" },
        { val: 5000, label: "5k" }
      ];
      const frame = drawAxesFrame(s.ctx, s.w, s.h, {
        xMin: 0.25, xMax: 6000.0, yMin: -1.25, yMax: 5.55, yDecimals: 1,
        xTicks: xTicks, xLabel: "Frequency f (Hz, log scale @ SCR = 1.65)", logX: true
      });
      plotSeries(s.ctx, frame, [0.25, 6000.0], [0.0, 0.0], "#10b981", [3, 3], 1.2);
      plotSeries(s.ctx, frame, freqs, ar.vsg.cumulative_rhp_index, "#10b981", [], 2.2);
      plotSeries(s.ctx, frame, freqs, ar.dvoc.cumulative_rhp_index, "#38bdf8", [5, 3], 1.9);
      plotSeries(s.ctx, frame, freqs, ar.vsg_unity_ff.cumulative_rhp_index, "#a78bfa", [3, 3], 1.9);
      plotSeries(s.ctx, frame, freqs, ar.gfl_pll_35hz.cumulative_rhp_index, "#f59e0b", [6, 3], 2.0);
      plotSeries(s.ctx, frame, freqs, ar.gfl_pll_55hz.cumulative_rhp_index, "#ef4444", [], 2.1);
      drawLegend(s.ctx, frame.padL + 10, frame.padT + 8, [
        { label: "VSG F_i=0.75 (Z = 0)", color: "#10b981" },
        { label: "dVOC (Z = 0)", color: "#38bdf8", dash: [5, 3] },
        { label: "VSG F_i=1.00 (Z = 2)", color: "#a78bfa", dash: [3, 3] },
        { label: "GFL 35 Hz (Z = 2)", color: "#f59e0b", dash: [6, 3] },
        { label: "GFL 55 Hz (Z = 4)", color: "#ef4444" }
      ]);
    }
  }

  function updateKpis(tr) {
    const verdictEl = el("kpi-verdict");
    if (tr.rodeThrough) {
      verdictEl.textContent = "PASS, BOUNDED";
      verdictEl.className = "kpi-value status-pass";
      el("kpi-verdict-sub").textContent = "Zero pole slips & current bounded";
    } else if (tr.overcurrent) {
      verdictEl.textContent = "FAIL, OVERCURRENT";
      verdictEl.className = "kpi-value status-fail";
      el("kpi-verdict-sub").textContent = "Exceeds 48.0 A bridge ceiling";
    } else {
      verdictEl.textContent = "FAIL, POLE SLIP";
      verdictEl.className = "kpi-value status-fail";
      el("kpi-verdict-sub").textContent = "Synchronism lost (|delta| > 165 deg)";
    }

    el("kpi-ipeak").textContent = tr.peakI.toFixed(2) + " A";
    el("kpi-ipeak-sub").textContent = (tr.peakI / NOM.i_rated_peak_a).toFixed(2) + " pu (Max 48.0 A)";

    el("kpi-delta").textContent = tr.maxAbsDelta.toFixed(1) + " deg";
    el("kpi-delta-sub").textContent = tr.maxAbsDelta < 90.0 ? "Inside static pull-out" : "Beyond 90 deg static limit";

    const fDev = Math.abs(50.0 - tr.minFreq) >= Math.abs(tr.maxFreq - 50.0) ? tr.minFreq : tr.maxFreq;
    el("kpi-freq").textContent = fDev.toFixed(3) + " Hz";
    el("kpi-freq-sub").textContent = "Range [" + tr.minFreq.toFixed(2) + ", " + tr.maxFreq.toFixed(2) + "] Hz";

    const scrCrit = interpScrCrit(state.pllBwHz);
    el("kpi-scrcrit").textContent = scrCrit.toFixed(2);
    el("kpi-scrcrit-sub").textContent = "GFL Hopf @ " + state.pllBwHz.toFixed(0) + " Hz PLL";

    const arMap = PRESETS.dq_admittance.architectures;
    let admEntry = arMap.vsg;
    if (state.arch === "droop") admEntry = arMap.droop;
    else if (state.arch === "dvoc") admEntry = arMap.dvoc;
    else if (state.arch === "gfl") {
      if (state.pllBwHz >= 45) admEntry = arMap.gfl_pll_55hz;
      else if (state.pllBwHz >= 22) admEntry = arMap.gfl_pll_35hz;
      else admEntry = arMap.gfl_pll_10hz;
    }
    const nuVal = admEntry.min_passivity_gt_25hz_s;
    el("kpi-passivity").textContent = (nuVal >= 0 ? "+" : "") + nuVal.toFixed(4) + " S";
    el("kpi-passivity-sub").textContent = "Z_nyq = " + admEntry.nyquist_rhp_poles_z + " RHP @ SCR 1.65";

    el("kpi-qsupport").textContent = (tr.qSupport >= 0 ? "+" : "") + tr.qSupport.toFixed(2) + " kvar";
    el("kpi-qsupport-sub").textContent = "Mean fault-window reactive injection";

    el("kpi-pfinal").textContent = tr.pFinal.toFixed(2) + " kW";
    el("kpi-pfinal-sub").textContent = "Dispatch target " + state.pRefKw.toFixed(1) + " kW";
  }

  function populateImpedanceTable() {
    const tbody = el("impedance-table-body");
    tbody.innerHTML = "";
    const order = [
      "vsg", "vsg_unity_ff", "dvoc", "droop",
      "gfl_pll_10hz", "gfl_pll_35hz", "gfl_pll_55hz"
    ];
    for (let i = 0; i < order.length; i++) {
      const item = PRESETS.dq_admittance.architectures[order[i]];
      const tr = document.createElement("tr");
      const npBand = item.nonpassive_band_hz
        ? "[" + item.nonpassive_band_hz[0] + ", " + item.nonpassive_band_hz[1] + "] Hz"
        : "None (Passive >= 25 Hz)";
      const qBand = item.neg_re_yqq_band_hz
        ? "[" + item.neg_re_yqq_band_hz[0] + ", " + item.neg_re_yqq_band_hz[1] + "] Hz"
        : "None";
      const nuStr = (item.min_passivity_gt_25hz_s >= 0 ? "+" : "") + item.min_passivity_gt_25hz_s.toFixed(4) + " S";
      const zOk = item.nyquist_rhp_poles_z === 0 ? "0 (Hurwitz Stable)" : item.nyquist_rhp_poles_z + " (Unstable RHP)";
      tr.innerHTML =
        "<td><strong>" + item.label + "</strong></td>" +
        "<td><code>" + nuStr + "</code></td>" +
        "<td>" + npBand + "</td>" +
        "<td>" + qBand + "</td>" +
        "<td><code>" + item.open_loop_rhp_poles_p + "</code></td>" +
        "<td><code>" + (item.nyquist_encirclements_n >= 0 ? "+" : "") + item.nyquist_encirclements_n + "</code></td>" +
        "<td><strong>" + zOk + "</strong></td>";
      tbody.appendChild(tr);
    }
  }

  function syncControlsToDom() {
    el("sel-arch").value = state.arch;
    el("sel-protection").value = state.protection;
    el("sel-contingency").value = state.contingency;
    el("rng-scr").value = String(state.scr);
    el("rng-pref").value = String(state.pRefKw);
    el("rng-inertia").value = String(state.inertiaS);
    el("rng-pllbw").value = String(state.pllBwHz);
    el("rng-vsag").value = String(state.vSagPu);
    el("rng-jump").value = String(state.jumpDeg);

    el("val-scr").textContent = state.scr.toFixed(2);
    el("val-pref").textContent = state.pRefKw.toFixed(1) + " kW";
    el("val-inertia").textContent = state.inertiaS.toFixed(1) + " s";
    el("val-pllbw").textContent = state.pllBwHz.toFixed(0) + " Hz";
    el("val-vsag").textContent = state.vSagPu.toFixed(2) + " pu";
    el("val-jump").textContent = state.jumpDeg.toFixed(0) + " deg";
  }

  function refreshAll() {
    syncControlsToDom();
    const tr = simulateBrowserTrajectory();
    state.lastTrace = tr;
    updateKpis(tr);
    renderPanel1(tr);
    renderPanel2(tr);
    renderPanel3();
    renderPanel4();
  }

  function applyPreset(p) {
    Object.assign(state, p);
    refreshAll();
  }

  function exportTraceCsv() {
    const tr = state.lastTrace;
    if (!tr) return;
    const rows = ["t_s,v_grid_pu,p_pcc_kw,q_pcc_kvar,i_bridge_a,delta_deg,freq_hz,r_tvi_ohm"];
    for (let i = 0; i < tr.t.length; i++) {
      rows.push([
        tr.t[i].toFixed(4),
        tr.vGridPu[i].toFixed(4),
        tr.pKw[i].toFixed(4),
        tr.qKvar[i].toFixed(4),
        tr.iMag[i].toFixed(4),
        tr.deltaDeg[i].toFixed(4),
        tr.freqHz[i].toFixed(5),
        tr.rTvi[i].toFixed(5)
      ].join(","));
    }
    const blob = new Blob([rows.join("\\n") + "\\n"], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "gfm_simulation_trace.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function bindEvents() {
    el("sel-arch").addEventListener("change", function (e) { state.arch = e.target.value; refreshAll(); });
    el("sel-protection").addEventListener("change", function (e) { state.protection = e.target.value; refreshAll(); });
    el("sel-contingency").addEventListener("change", function (e) {
      state.contingency = e.target.value;
      refreshAll();
    });
    el("rng-scr").addEventListener("input", function (e) { state.scr = parseFloat(e.target.value); refreshAll(); });
    el("rng-pref").addEventListener("input", function (e) { state.pRefKw = parseFloat(e.target.value); refreshAll(); });
    el("rng-inertia").addEventListener("input", function (e) {
      state.inertiaS = parseFloat(e.target.value);
      refreshAll();
    });
    el("rng-pllbw").addEventListener("input", function (e) {
      state.pllBwHz = parseFloat(e.target.value);
      refreshAll();
    });
    el("rng-vsag").addEventListener("input", function (e) { state.vSagPu = parseFloat(e.target.value); refreshAll(); });
    el("rng-jump").addEventListener("input", function (e) {
      state.jumpDeg = parseFloat(e.target.value);
      refreshAll();
    });

    el("btn-preset-sag-tvi").addEventListener("click", function () {
      applyPreset({
        arch: "vsg", protection: "tvi_paw", contingency: "voltage_sag",
        scr: 4.0, pRefKw: 17.0, inertiaS: 4.9, vSagPu: 0.15
      });
    });
    el("btn-preset-sag-clamp").addEventListener("click", function () {
      applyPreset({
        arch: "vsg", protection: "hard_clamp", contingency: "voltage_sag",
        scr: 4.0, pRefKw: 17.0, inertiaS: 4.9, vSagPu: 0.15
      });
    });
    el("btn-preset-jump").addEventListener("click", function () {
      applyPreset({
        arch: "vsg", protection: "tvi_paw", contingency: "phase_jump",
        scr: 4.0, pRefKw: 12.0, inertiaS: 4.9, jumpDeg: -45.0
      });
    });
    el("btn-preset-weak-gfl").addEventListener("click", function () {
      applyPreset({
        arch: "gfl", protection: "hard_clamp", contingency: "line_opening",
        scr: 4.0, pRefKw: 15.0, pllBwHz: 55.0
      });
    });
    el("btn-preset-dvoc-trip").addEventListener("click", function () {
      applyPreset({ arch: "dvoc", protection: "tvi_paw", contingency: "line_opening", scr: 4.0, pRefKw: 15.0 });
    });

    el("btn-toggle-panel1").addEventListener("click", function () {
      state.showPanel1Reactive = !state.showPanel1Reactive;
      el("btn-toggle-panel1").textContent = state.showPanel1Reactive ? "Show Bridge Current" : "Show P & Q Power";
      renderPanel1(state.lastTrace);
    });
    el("btn-toggle-panel2").addEventListener("click", function () {
      state.showPanel2Freq = !state.showPanel2Freq;
      el("btn-toggle-panel2").textContent = state.showPanel2Freq ? "Show Power Angle" : "Show Frequency f(t)";
      renderPanel2(state.lastTrace);
    });
    el("btn-toggle-panel3").addEventListener("click", function () {
      state.showPanel3Locus = !state.showPanel3Locus;
      el("btn-toggle-panel3").textContent = state.showPanel3Locus ? "Show SCR Sweep" : "Show s-Plane Root Locus";
      renderPanel3();
    });
    el("btn-toggle-panel4").addEventListener("click", function () {
      state.showPanel4Nyquist = !state.showPanel4Nyquist;
      el("btn-toggle-panel4").textContent = state.showPanel4Nyquist ? "Show Passivity nu(f)" : "Show Nyquist Z(f)";
      renderPanel4();
    });

    el("btn-export-csv").addEventListener("click", exportTraceCsv);
    window.addEventListener("resize", function () {
      if (state.lastTrace) {
        renderPanel1(state.lastTrace);
        renderPanel2(state.lastTrace);
        renderPanel3();
        renderPanel4();
      }
    });
  }

  populateImpedanceTable();
  bindEvents();
  refreshAll();
})();
"""


def _render_index_html(presets: dict[str, Any], *, has_figures: bool) -> str:
    presets_json = json.dumps(presets, separators=(",", ":"), allow_nan=False)
    cards = []
    for item in BENCHMARK_FIGURES:
        img_html = (
            f'<img src="results/{item["file"]}" alt="{item["title"]}" loading="lazy">'
            if has_figures
            else ""
        )
        cards.append(
            f'<article class="gallery-card">{img_html}'
            f'<h3>{item["title"]}</h3><p>{item["caption"]}</p></article>'
        )
    gallery_html = "\n        ".join(cards)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GFM &amp; Weak-Grid Inverter Control Workbench, gfm_solver v{WORKBENCH_VERSION}</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="container">
    <header class="hero">
      <div class="hero-top">
        <div class="badge-row">
          <span class="pill">gfm_solver v{WORKBENCH_VERSION}</span>
          <span class="pill blue">Zero-CDN Browser RK4 + Python State-Space</span>
          <span class="pill">20 kVA | 230 Vrms | 50 Hz | LCL + TVI + PAW</span>
        </div>
        <button id="btn-export-csv" class="primary" type="button">Export Trace CSV</button>
      </div>
      <h1>Grid-Forming (VSG, Droop, dVOC) &amp; Weak-Grid GFL Electromagnetic Workbench</h1>
      <p class="lead">
        Interactive browser-side 4th-order Runge-Kutta (<code>dt = 250 us</code>) electromagnetic fault &amp; weak-grid
        simulator coupled with exact Python state-space modal participation, synchronous-<code>dq</code> admittance
        passivity <code>nu(f)</code>, and Generalized Nyquist <code>Z = P + N</code> datasets.
        <em>(Model-consistency research workbench only, not hardware or grid-code certification.)</em>
      </p>
      <div class="preset-bar">
        <button id="btn-preset-sag-tvi" type="button">1. 0.15 pu Sag, TVI + PAW Ride-Through</button>
        <button id="btn-preset-sag-clamp" type="button">2. 0.15 pu Sag, Hard Clamp Windup LOS</button>
        <button id="btn-preset-jump" type="button">3. -45° Phase Jump, TVI + PAW</button>
        <button id="btn-preset-weak-gfl" type="button">4. Weak-Grid Line Trip, GFL 55 Hz LOS</button>
        <button id="btn-preset-dvoc-trip" type="button">5. dVOC Line-Opening Ride-Through</button>
      </div>
    </header>

    <section class="controls-panel" aria-label="Simulation Parameters">
      <div class="controls-grid">
        <div class="ctrl-group">
          <label for="sel-arch"><span>Converter Architecture</span></label>
          <select id="sel-arch">
            <option value="vsg">GFM, Virtual Synchronous Generator (VSG)</option>
            <option value="droop">GFM, Multi-Loop P-f / Q-V Droop</option>
            <option value="dvoc">GFM, Andronov-Hopf Oscillator (dVOC)</option>
            <option value="gfl">GFL, Synchronous-Ref-Frame PLL (SRF-PLL)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label for="sel-protection"><span>Overcurrent &amp; Anti-Windup Law</span></label>
          <select id="sel-protection">
            <option value="tvi_paw">Threshold TVI + Power Anti-Windup</option>
            <option value="hard_clamp">Hard Circular Clamp Only (Windup)</option>
            <option value="unprotected">Unprotected Bridge (No Clamp)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label for="sel-contingency"><span>Grid Contingency Event (t = 0.20 s)</span></label>
          <select id="sel-contingency">
            <option value="voltage_sag">Deep Symmetrical Voltage Sag (250 ms)</option>
            <option value="phase_jump">Instantaneous Grid Phase Jump</option>
            <option value="line_opening">Parallel-Line Opening (SCR -&gt; SCR/3.28)</option>
            <option value="freq_drop">Grid Frequency Drop (-0.50 Hz RoCoF)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <label for="rng-scr">
            <span>Grid Short-Circuit Ratio SCR</span><span id="val-scr" class="ctrl-val">4.00</span>
          </label>
          <input id="rng-scr" type="range" min="1.15" max="8.00" step="0.05" value="4.00">
        </div>
        <div class="ctrl-group">
          <label for="rng-pref">
            <span>Active Dispatch P_ref</span><span id="val-pref" class="ctrl-val">17.0 kW</span>
          </label>
          <input id="rng-pref" type="range" min="2.0" max="20.0" step="0.5" value="17.0">
        </div>
        <div class="ctrl-group">
          <label for="rng-inertia">
            <span>VSG Virtual Inertia H</span><span id="val-inertia" class="ctrl-val">4.9 s</span>
          </label>
          <input id="rng-inertia" type="range" min="1.0" max="8.0" step="0.2" value="4.9">
        </div>
        <div class="ctrl-group">
          <label for="rng-pllbw">
            <span>GFL SRF-PLL Bandwidth</span><span id="val-pllbw" class="ctrl-val">35 Hz</span>
          </label>
          <input id="rng-pllbw" type="range" min="10" max="60" step="1" value="35">
        </div>
        <div class="ctrl-group">
          <label for="rng-vsag">
            <span>Sag Residual Voltage V_sag</span><span id="val-vsag" class="ctrl-val">0.15 pu</span>
          </label>
          <input id="rng-vsag" type="range" min="0.10" max="0.80" step="0.05" value="0.15">
        </div>
        <div class="ctrl-group">
          <label for="rng-jump">
            <span>Grid Phase-Jump Angle</span><span id="val-jump" class="ctrl-val">-45 deg</span>
          </label>
          <input id="rng-jump" type="range" min="-60" max="-10" step="5" value="-45">
        </div>
      </div>
    </section>

    <section class="kpi-grid" aria-label="Key Telemetry Indicators">
      <div class="kpi-card">
        <div class="kpi-label">Ride-Through Verdict</div>
        <div id="kpi-verdict" class="kpi-value status-pass">PASS</div>
        <div id="kpi-verdict-sub" class="kpi-sub">Bounded</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Peak Bridge |i_1,dq|</div>
        <div id="kpi-ipeak" class="kpi-value">--</div>
        <div id="kpi-ipeak-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Max Power Angle |delta|</div>
        <div id="kpi-delta" class="kpi-value">--</div>
        <div id="kpi-delta-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Frequency Extremum</div>
        <div id="kpi-freq" class="kpi-value">--</div>
        <div id="kpi-freq-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">GFL Critical Hopf SCR</div>
        <div id="kpi-scrcrit" class="kpi-value">--</div>
        <div id="kpi-scrcrit-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">dq Passivity min nu(&gt;=25Hz)</div>
        <div id="kpi-passivity" class="kpi-value">--</div>
        <div id="kpi-passivity-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Fault Reactive Q_pcc</div>
        <div id="kpi-qsupport" class="kpi-value">--</div>
        <div id="kpi-qsupport-sub" class="kpi-sub">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Post-Event Settled P_pcc</div>
        <div id="kpi-pfinal" class="kpi-value">--</div>
        <div id="kpi-pfinal-sub" class="kpi-sub">--</div>
      </div>
    </section>

    <section class="charts-grid" aria-label="Interactive Simulation &amp; Frequency-Domain Plots">
      <div class="chart-card">
        <div class="chart-header">
          <h2 id="panel1-title">1. Inverter Bridge Current |i_1,dq| (A_pk) vs. 48 A Ceiling</h2>
          <button id="btn-toggle-panel1" type="button">Show P &amp; Q Power</button>
        </div>
        <canvas id="canvas-power-current" class="plot-canvas"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-header">
          <h2 id="panel2-title">2. Synchronizing Power Angle delta_rel(t) (deg)</h2>
          <button id="btn-toggle-panel2" type="button">Show Frequency f(t)</button>
        </div>
        <canvas id="canvas-angle-freq" class="plot-canvas"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-header">
          <h2 id="panel3-title">3. Weak-Grid Eigenvalue Growth max Re{{lambda}} vs. SCR</h2>
          <button id="btn-toggle-panel3" type="button">Show s-Plane Root Locus</button>
        </div>
        <canvas id="canvas-weak-grid" class="plot-canvas"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-header">
          <h2 id="panel4-title">4. Synchronous-dq Admittance Passivity Index nu(f) (S)</h2>
          <button id="btn-toggle-panel4" type="button">Show Nyquist Z(f)</button>
        </div>
        <canvas id="canvas-passivity" class="plot-canvas"></canvas>
      </div>
    </section>

    <section class="table-section" aria-label="Synchronous-dq Admittance Passivity Table">
      <h2>
        Solver-Verified Synchronous-<code>dq</code> Admittance Passivity &amp; Nyquist (SCR = 1.65, P = 15 kW)
      </h2>
      <table class="data-table">
        <thead>
          <tr>
            <th>Converter Architecture</th>
            <th>min nu (f &gt;= 25 Hz)</th>
            <th>Non-Passive Band (&gt;= 25 Hz)</th>
            <th>Negative Re{{Y_qq}} Band</th>
            <th>Stiff-Bus RHP P</th>
            <th>Nyquist Winding N</th>
            <th>Closed-Loop Z = P + N</th>
          </tr>
        </thead>
        <tbody id="impedance-table-body"></tbody>
      </table>
    </section>

    <section class="gallery-section" aria-label="Validated Benchmark Gallery">
      <h2>Reproducible Python Solver Benchmark Gallery (main.py)</h2>
      <p class="lead">
        All 6 publication figures are deterministically generated by <code>python main.py</code> and verified in CI
        across Python 3.10–3.13 against state-space eigenvalues, SciPy RK45, and Generalized Nyquist invariants.
      </p>
      <div class="gallery-grid">{gallery_html}</div>
    </section>

    <footer>
      <span>MIT License, Haris Zafar (<code>hariszafar740-lang/inverter-grid-control-solver</code>)</span>
      <span>
        Grid-Forming &amp; Weak-Grid Electromagnetic LCL, FRT/TVI &amp; dq Passivity Solver (v{WORKBENCH_VERSION})
      </span>
    </footer>
  </div>

  <script id="gfm-presets-data" type="application/json">{presets_json}</script>
  <script src="app.js"></script>
</body>
</html>
"""


def build_static_web_workbench(
    output_dir: str | Path = "site",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate the standalone zero-CDN GitHub Pages workbench into ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    results_src = root / "results"
    copied_figures: list[str] = []
    if results_src.is_dir():
        results_dst = out / "results"
        results_dst.mkdir(parents=True, exist_ok=True)
        for item in BENCHMARK_FIGURES:
            src_png = results_src / item["file"]
            if src_png.is_file():
                shutil.copy2(src_png, results_dst / item["file"])
                copied_figures.append(item["file"])

    presets = build_web_presets_payload()
    (out / ".nojekyll").write_text("", encoding="utf-8")
    (out / "gfm_presets.json").write_text(
        json.dumps(presets, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (out / "app.js").write_text(APP_JS, encoding="utf-8")
    (out / "index.html").write_text(
        _render_index_html(presets, has_figures=len(copied_figures) == len(BENCHMARK_FIGURES)),
        encoding="utf-8",
    )

    return {
        "output_dir": str(out),
        "version": WORKBENCH_VERSION,
        "files_written": [".nojekyll", "index.html", "styles.css", "app.js", "gfm_presets.json"],
        "copied_figures": copied_figures,
        "architectures": sorted(presets["dq_admittance"]["architectures"].keys()),
    }
