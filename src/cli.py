"""Command-line interface (`gfm-sim` & `python -m gfm_solver`) for GFM/GFL benchmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from . import __version__
from .benchmark import run_fault_ride_through_benchmark, validate_frt_benchmark_results
from .em_frt_benchmark import run_electromagnetic_frt_benchmark
from .gfm_triad import run_gfm_triad_benchmark
from .small_signal_impedance import run_small_signal_impedance_benchmark
from .thevenin_network import TheveninGridSpec
from .visualizer import (
    plot_electromagnetic_frt_benchmark,
    plot_gfm_triad_benchmark,
    plot_small_signal_impedance_benchmark,
    plot_weak_grid_scr_benchmark,
)
from .weak_grid_benchmark import run_weak_grid_scr_benchmark
from .web_dashboard import build_static_web_workbench

CLI_VERSION = __version__


def _emit(payload: dict[str, Any], *, as_json: bool, lines: Sequence[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        for line in lines:
            print(line)


def _cmd_info(args: argparse.Namespace) -> int:
    spec = TheveninGridSpec()
    payload: dict[str, Any] = {
        "package": "gfm_solver",
        "distribution": "inverter-grid-control-solver",
        "version": CLI_VERSION,
        "nominal_rating": {
            "s_rated_kva": round(spec.s_rated_va / 1000.0, 2),
            "v_phase_rms_v": spec.v_phase_rms_v,
            "f_nom_hz": spec.f_nom_hz,
            "i_max_peak_a": spec.i_max_peak_a,
        },
        "models": [
            "4-state CoupledVSGTheveninModel",
            "14-state ElectromagneticGFMModel (VSG)",
            "13-state MultiLoopDroopGFMModel",
            "12-state MultiLoopDVOCGFMModel (Andronov-Hopf)",
            "11-state ElectromagneticGFLModel (SRF-PLL)",
            "14-state ElectromagneticFRTVSGModel (TVI + Power Anti-Windup)",
            "2x2 Synchronous-dq Admittance Passivity & Generalized Nyquist",
        ],
    }
    lines = [
        f"gfm_solver v{CLI_VERSION} (inverter-grid-control-solver)",
        (
            f"Nominal Rating: {spec.s_rated_va / 1e3:.1f} kVA | {spec.v_phase_rms_v:.0f} Vrms | "
            f"{spec.f_nom_hz:.1f} Hz | I_max = {spec.i_max_peak_a:.1f} A_pk"
        ),
        "Available Subcommands: info | triad | weak-grid | frt | impedance | web",
    ]
    _emit(payload, as_json=bool(args.json), lines=lines)
    return 0


def _cmd_triad(args: argparse.Namespace) -> int:
    grid = TheveninGridSpec(scr=float(args.scr), xr_ratio=float(args.xr))
    res = run_gfm_triad_benchmark(grid=grid, dt_s=float(args.dt_us) * 1.0e-6)
    saved = str(plot_gfm_triad_benchmark(res, save_path=args.save_plot)) if args.save_plot else None
    dr, vs, dv = res.droop, res.vsg, res.dvoc
    payload: dict[str, Any] = {
        "scr": grid.scr,
        "xr_ratio": grid.xr_ratio,
        "droop": {
            "rise_t90_ms": round(dr.step_rise_time_90_ms, 2),
            "peak_power_kw": round(dr.freq_drop_peak_power_kw, 3),
            "max_real_eig_s_inv": round(dr.max_real_eig_s_inv, 4),
        },
        "vsg": {
            "rise_t90_ms": round(vs.step_rise_time_90_ms, 2),
            "peak_power_kw": round(vs.freq_drop_peak_power_kw, 3),
            "max_real_eig_s_inv": round(vs.max_real_eig_s_inv, 4),
        },
        "dvoc": {
            "rise_t90_ms": round(dv.step_rise_time_90_ms, 2),
            "peak_power_kw": round(dv.freq_drop_peak_power_kw, 3),
            "max_real_eig_s_inv": round(dv.max_real_eig_s_inv, 4),
        },
        "saved_figure": saved,
    }
    lines = [
        f"GFM Control Triad @ SCR = {grid.scr:.2f}, X/R = {grid.xr_ratio:.1f}",
        (
            f"  Droop (13): t_90 = {dr.step_rise_time_90_ms:.1f} ms | "
            f"Peak P = {dr.freq_drop_peak_power_kw:.2f} kW | max Re = {dr.max_real_eig_s_inv:.2f} s^-1"
        ),
        (
            f"  VSG   (14): t_90 = {vs.step_rise_time_90_ms:.1f} ms | "
            f"Peak P = {vs.freq_drop_peak_power_kw:.2f} kW | max Re = {vs.max_real_eig_s_inv:.2f} s^-1"
        ),
        (
            f"  dVOC  (12): t_90 = {dv.step_rise_time_90_ms:.1f} ms | "
            f"Peak P = {dv.freq_drop_peak_power_kw:.2f} kW | max Re = {dv.max_real_eig_s_inv:.2f} s^-1"
        ),
    ]
    if saved:
        lines.append(f"  Saved figure: {saved}")
    _emit(payload, as_json=bool(args.json), lines=lines)
    return 0


def _cmd_weak_grid(args: argparse.Namespace) -> int:
    res = run_weak_grid_scr_benchmark(p_dispatch_w=float(args.p_kw) * 1000.0, xr_ratio=float(args.xr))
    saved = str(plot_weak_grid_scr_benchmark(res, save_path=args.save_plot)) if args.save_plot else None
    payload: dict[str, Any] = {
        "p_dispatch_kw": round(res.p_dispatch_w / 1000.0, 2),
        "scr_crit_gfl_55hz": round(res.scr_crit_gfl_fast, 3),
        "scr_crit_gfl_35hz": round(res.scr_crit_gfl_med, 3),
        "scr_crit_gfl_10hz": round(res.scr_crit_gfl_slow, 3),
        "line_trip_gfl_los": res.trip_gfl_med.lost_synchronism,
        "line_trip_vsg_los": res.trip_gfm_vsg.lost_synchronism,
        "line_trip_dvoc_los": res.trip_gfm_dvoc.lost_synchronism,
        "saved_figure": saved,
    }
    lines = [
        f"Weak-Grid SCR Benchmark @ P_ref = {res.p_dispatch_w / 1e3:.1f} kW",
        (
            f"  GFL Hopf SCR_crit (55/35/10 Hz): {res.scr_crit_gfl_fast:.2f} / "
            f"{res.scr_crit_gfl_med:.2f} / {res.scr_crit_gfl_slow:.2f}"
        ),
        (
            f"  Line-Opening LOS: GFL = {res.trip_gfl_med.lost_synchronism} | "
            f"VSG = {res.trip_gfm_vsg.lost_synchronism} | dVOC = {res.trip_gfm_dvoc.lost_synchronism}"
        ),
    ]
    if saved:
        lines.append(f"  Saved figure: {saved}")
    _emit(payload, as_json=bool(args.json), lines=lines)
    return 0


def _cmd_frt(args: argparse.Namespace) -> int:
    if args.reduced_order:
        frt = run_fault_ride_through_benchmark()
        validate_frt_benchmark_results(frt, require_current_limit=True)
        payload = {
            "mode": "reduced_order_algebraic_frt",
            "fault_max_current_a": round(float(frt["fault_max_current_a"]), 3),
            "max_rocof_hz_s": round(float(frt["max_rocof"]), 4),
            "recovery_settling_time_s": round(float(frt["recovery_settling_time_s"]), 4),
            "passed": True,
        }
        lines = [
            (
                f"Reduced-Order FRT Check: I_max = {frt['fault_max_current_a']:.2f} A | "
                f"RoCoF = {frt['max_rocof']:.4f} Hz/s | Settling = {frt['recovery_settling_time_s']:.3f} s | PASS"
            )
        ]
        _emit(payload, as_json=bool(args.json), lines=lines)
        return 0

    res = run_electromagnetic_frt_benchmark(
        sag_residual_pu=float(args.sag_pu),
        sag_dispatch_w=float(args.p_sag_kw) * 1000.0,
        phase_jump_deg=float(args.jump_deg),
        dt_s=float(args.dt_us) * 1.0e-6,
    )
    saved = str(plot_electromagnetic_frt_benchmark(res, save_path=args.save_plot)) if args.save_plot else None
    sp, sh, jp, jh = res.sag_tvi_paw, res.sag_hard_clamp, res.jump_tvi_paw, res.jump_hard_clamp
    payload = {
        "sag_residual_pu": res.sag_residual_pu,
        "phase_jump_deg": res.phase_jump_deg,
        "sag_tvi_paw": {
            "peak_current_a": round(sp.peak_bridge_current_a, 3),
            "max_delta_deg": round(sp.max_abs_delta_deg, 2),
            "rode_through": sp.rode_through,
        },
        "sag_hard_clamp": {
            "peak_current_a": round(sh.peak_bridge_current_a, 3),
            "max_delta_deg": round(sh.max_abs_delta_deg, 2),
            "rode_through": sh.rode_through,
        },
        "jump_tvi_paw": {
            "peak_current_a": round(jp.peak_bridge_current_a, 3),
            "max_delta_deg": round(jp.max_abs_delta_deg, 2),
            "rode_through": jp.rode_through,
        },
        "jump_hard_clamp": {
            "max_delta_deg": round(jh.max_abs_delta_deg, 2),
            "rode_through": jh.rode_through,
        },
        "saved_figure": saved,
    }
    lines = [
        f"14-State EM FRT ({res.sag_residual_pu:.2f} pu Sag) & {res.phase_jump_deg:.0f} deg Phase Jump",
        (
            f"  Sag TVI+PAW:  I_pk = {sp.peak_bridge_current_a:.2f} A | "
            f"max delta = {sp.max_abs_delta_deg:.1f} deg | rode = {sp.rode_through}"
        ),
        (
            f"  Sag Clamp:    I_pk = {sh.peak_bridge_current_a:.2f} A | "
            f"max delta = {sh.max_abs_delta_deg:.1f} deg | rode = {sh.rode_through}"
        ),
        (
            f"  Jump TVI+PAW: I_pk = {jp.peak_bridge_current_a:.2f} A | "
            f"max delta = {jp.max_abs_delta_deg:.1f} deg | rode = {jp.rode_through}"
        ),
    ]
    if saved:
        lines.append(f"  Saved figure: {saved}")
    _emit(payload, as_json=bool(args.json), lines=lines)
    return 0


def _cmd_impedance(args: argparse.Namespace) -> int:
    res = run_small_signal_impedance_benchmark(
        scr_operating=float(args.scr),
        p_ref_w=float(args.p_kw) * 1000.0,
    )
    saved = str(plot_small_signal_impedance_benchmark(res, save_path=args.save_plot)) if args.save_plot else None
    av, au, g35, g55 = (
        res.admittance_vsg,
        res.admittance_vsg_unity_ff,
        res.admittance_gfl35,
        res.admittance_gfl55,
    )
    f_q35 = g35.neg_re_yqq_band_hz[1] if g35.neg_re_yqq_band_hz else 0.0
    f_q55 = g55.neg_re_yqq_band_hz[1] if g55.neg_re_yqq_band_hz else 0.0
    payload: dict[str, Any] = {
        "scr_operating": res.scr_operating,
        "p_ref_kw": round(res.p_ref_w / 1000.0, 2),
        "vsg_qoria_fi_075": {
            "min_passivity_gt_25hz_s": round(av.min_passivity_high_freq_s, 5),
            "nyquist_rhp_poles_z": av.nyquist_rhp_poles_z,
            "state_space_rhp_poles": av.state_space_rhp_poles,
        },
        "vsg_naive_fi_100": {
            "min_passivity_gt_25hz_s": round(au.min_passivity_high_freq_s, 5),
            "nyquist_rhp_poles_z": au.nyquist_rhp_poles_z,
            "state_space_rhp_poles": au.state_space_rhp_poles,
        },
        "gfl_pll_35hz": {
            "neg_re_yqq_upper_hz": round(f_q35, 2),
            "nyquist_rhp_poles_z": g35.nyquist_rhp_poles_z,
            "state_space_rhp_poles": g35.state_space_rhp_poles,
        },
        "gfl_pll_55hz": {
            "neg_re_yqq_upper_hz": round(f_q55, 2),
            "nyquist_rhp_poles_z": g55.nyquist_rhp_poles_z,
            "state_space_rhp_poles": g55.state_space_rhp_poles,
        },
        "saved_figure": saved,
    }
    lines = [
        f"Modal Participation & dq Admittance @ SCR = {res.scr_operating:.2f}, P = {res.p_ref_w / 1e3:.1f} kW",
        (
            f"  VSG (F_i=0.75): min nu(>=25Hz) = {av.min_passivity_high_freq_s:+.4f} S | "
            f"Z_nyq = {av.nyquist_rhp_poles_z} (Z_ss = {av.state_space_rhp_poles})"
        ),
        (
            f"  VSG (F_i=1.00): min nu(>=25Hz) = {au.min_passivity_high_freq_s:+.4f} S | "
            f"Z_nyq = {au.nyquist_rhp_poles_z} (Z_ss = {au.state_space_rhp_poles})"
        ),
        f"  GFL 35 Hz PLL:  Re(Yqq)<0 up to {f_q35:.1f} Hz | Z_nyq = {g35.nyquist_rhp_poles_z}",
        f"  GFL 55 Hz PLL:  Re(Yqq)<0 up to {f_q55:.1f} Hz | Z_nyq = {g55.nyquist_rhp_poles_z}",
    ]
    if saved:
        lines.append(f"  Saved figure: {saved}")
    _emit(payload, as_json=bool(args.json), lines=lines)
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    summary = build_static_web_workbench(output_dir=args.output_dir)
    lines = [
        f"Static GFM & Weak-Grid Workbench v{summary['version']} built at: {summary['output_dir']}",
        f"  Files written:  {', '.join(summary['files_written'])}",
        f"  Copied figures: {len(summary['copied_figures'])} PNGs",
    ]
    _emit(summary, as_json=bool(args.json), lines=lines)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gfm-sim",
        description="Grid-Forming & Grid-Following Electromagnetic LCL, FRT & dq Impedance Solver CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {CLI_VERSION}")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_info = sub.add_parser("info", help="Display solver metadata, nominal ratings, and available models.")
    p_info.add_argument("--json", action="store_true", help="Emit structured JSON output.")
    p_info.set_defaults(func=_cmd_info)

    p_triad = sub.add_parser("triad", help="Run Comparative GFM Control Triad (Droop, VSG, dVOC).")
    p_triad.add_argument("--scr", type=float, default=6.0)
    p_triad.add_argument("--xr", type=float, default=7.0)
    p_triad.add_argument("--dt-us", type=float, default=80.0)
    p_triad.add_argument("--save-plot", type=str, default=None)
    p_triad.add_argument("--json", action="store_true")
    p_triad.set_defaults(func=_cmd_triad)

    p_wg = sub.add_parser("weak-grid", help="Run Weak-Grid GFL (SRF-PLL) vs. GFM SCR collapse benchmark.")
    p_wg.add_argument("--p-kw", type=float, default=15.0)
    p_wg.add_argument("--xr", type=float, default=7.0)
    p_wg.add_argument("--save-plot", type=str, default=None)
    p_wg.add_argument("--json", action="store_true")
    p_wg.set_defaults(func=_cmd_weak_grid)

    p_frt = sub.add_parser("frt", help="Run 14-state Electromagnetic FRT / TVI / PAW or reduced-order FRT check.")
    p_frt.add_argument("--reduced-order", action="store_true", help="Run fast reduced-order algebraic FRT check.")
    p_frt.add_argument("--sag-pu", type=float, default=0.15)
    p_frt.add_argument("--p-sag-kw", type=float, default=17.0)
    p_frt.add_argument("--jump-deg", type=float, default=-45.0)
    p_frt.add_argument("--dt-us", type=float, default=80.0)
    p_frt.add_argument("--save-plot", type=str, default=None)
    p_frt.add_argument("--json", action="store_true")
    p_frt.set_defaults(func=_cmd_frt)

    p_imp = sub.add_parser("impedance", help="Run Small-Signal Modal Participation, dq Passivity & Nyquist sweep.")
    p_imp.add_argument("--scr", type=float, default=1.65)
    p_imp.add_argument("--p-kw", type=float, default=15.0)
    p_imp.add_argument("--save-plot", type=str, default=None)
    p_imp.add_argument("--json", action="store_true")
    p_imp.set_defaults(func=_cmd_impedance)


    p_web = sub.add_parser("web", help="Build standalone zero-CDN HTML5/Canvas workbench for GitHub Pages.")
    p_web.add_argument("--output-dir", type=str, default="site")
    p_web.add_argument("--json", action="store_true")
    p_web.set_defaults(func=_cmd_web)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
