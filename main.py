"""Reproducible Thévenin, 14-state LCL, GFM Triad, Weak-Grid, EM FRT & dq Impedance entry point."""

from __future__ import annotations

from pathlib import Path

from src.benchmark import run_fault_ride_through_benchmark, validate_frt_benchmark_results
from src.cascaded_control import simulate_electromagnetic_gfm_transient
from src.em_frt_benchmark import run_electromagnetic_frt_benchmark
from src.gfm_triad import run_gfm_triad_benchmark
from src.small_signal_impedance import run_small_signal_impedance_benchmark
from src.thevenin_network import TheveninGridSpec, simulate_coupled_vsg_transient
from src.visualizer import (
    plot_closed_loop_thevenin_benchmark,
    plot_electromagnetic_frt_benchmark,
    plot_gfm_triad_benchmark,
    plot_lcl_cascaded_electromagnetic_benchmark,
    plot_small_signal_impedance_benchmark,
    plot_weak_grid_scr_benchmark,
)
from src.weak_grid_benchmark import run_weak_grid_scr_benchmark


def generate_reference_transient_figure(
    save_path: str | Path = "results/inverter_transient_response.png",
):
    """Run the canonical 4-state closed-loop Thévenin GFM benchmark and write the figure."""
    spec = TheveninGridSpec()
    res = simulate_coupled_vsg_transient(spec)
    out = plot_closed_loop_thevenin_benchmark(res, save_path=save_path)
    return res, out


def generate_lcl_electromagnetic_figure(
    qs_res=None,
    save_path: str | Path = "results/lcl_cascaded_electromagnetic_response.png",
):
    """Run the 14-state electromagnetic dq LCL + cascaded PI benchmark and write the figure."""
    spec = TheveninGridSpec() if qs_res is None else qs_res.spec
    qs = simulate_coupled_vsg_transient(spec) if qs_res is None else qs_res
    em_res = simulate_electromagnetic_gfm_transient(grid=spec)
    out = plot_lcl_cascaded_electromagnetic_benchmark(em_res, qs, save_path=save_path)
    return em_res, out


def generate_gfm_triad_figure(
    spec: TheveninGridSpec | None = None,
    save_path: str | Path = "results/gfm_triad_comparative_response.png",
):
    """Run the Comparative GFM Control Triad (Droop, VSG, dVOC) benchmark and write the figure."""
    grid_spec = TheveninGridSpec() if spec is None else spec
    triad_res = run_gfm_triad_benchmark(grid=grid_spec)
    out = plot_gfm_triad_benchmark(triad_res, save_path=save_path)
    return triad_res, out


def generate_weak_grid_scr_figure(
    save_path: str | Path = "results/gfl_vs_gfm_weak_grid_collapse.png",
):
    """Run the GFL (SRF-PLL) vs. GFM weak-grid SCR collapse benchmark and write the figure."""
    wg_res = run_weak_grid_scr_benchmark()
    out = plot_weak_grid_scr_benchmark(wg_res, save_path=save_path)
    return wg_res, out


def generate_electromagnetic_frt_figure(
    save_path: str | Path = "results/em_frt_tvi_phase_jump_response.png",
):
    """Run the 14-state EM FRT, TVI, PAW & -45 deg phase-jump benchmark and write the figure."""
    em_frt_res = run_electromagnetic_frt_benchmark()
    out = plot_electromagnetic_frt_benchmark(em_frt_res, save_path=save_path)
    return em_frt_res, out


def generate_small_signal_impedance_figure(
    save_path: str | Path = "results/small_signal_eigenvalue_impedance_passivity.png",
):
    """Run the Small-Signal Modal Participation, dq Passivity & Nyquist benchmark and plot."""
    ssi_res = run_small_signal_impedance_benchmark()
    out = plot_small_signal_impedance_benchmark(ssi_res, save_path=save_path)
    return ssi_res, out


def main(
    save_path: str | Path = "results/inverter_transient_response.png",
    lcl_save_path: str | Path | None = None,
    triad_save_path: str | Path | None = None,
    weak_grid_save_path: str | Path | None = None,
    em_frt_save_path: str | Path | None = None,
    ssi_save_path: str | Path | None = None,
) -> int:
    out_main = Path(save_path)
    out_lcl = (
        Path(lcl_save_path)
        if lcl_save_path is not None
        else out_main.parent / "lcl_cascaded_electromagnetic_response.png"
    )
    out_triad = (
        Path(triad_save_path)
        if triad_save_path is not None
        else out_main.parent / "gfm_triad_comparative_response.png"
    )
    out_wg = (
        Path(weak_grid_save_path)
        if weak_grid_save_path is not None
        else out_main.parent / "gfl_vs_gfm_weak_grid_collapse.png"
    )
    out_em_frt = (
        Path(em_frt_save_path)
        if em_frt_save_path is not None
        else out_main.parent / "em_frt_tvi_phase_jump_response.png"
    )
    out_ssi = (
        Path(ssi_save_path)
        if ssi_save_path is not None
        else out_main.parent / "small_signal_eigenvalue_impedance_passivity.png"
    )

    qs_res, fig_qs = generate_reference_transient_figure(save_path=out_main)
    em_res, fig_em = generate_lcl_electromagnetic_figure(qs_res=qs_res, save_path=out_lcl)
    triad_res, fig_triad = generate_gfm_triad_figure(spec=qs_res.spec, save_path=out_triad)
    wg_res, fig_wg = generate_weak_grid_scr_figure(save_path=out_wg)
    em_frt_res, fig_em_frt = generate_electromagnetic_frt_figure(save_path=out_em_frt)
    ssi_res, fig_ssi = generate_small_signal_impedance_figure(save_path=out_ssi)
    frt = run_fault_ride_through_benchmark()
    validate_frt_benchmark_results(frt, require_current_limit=True)

    f_res = em_res.lcl.coupled_resonance_hz(em_res.grid.l_g_h)
    print("=" * 92)
    print("GRID-FORMING & WEAK-GRID GFL/GFM LCL, TRIAD, EM-FRT & DQ IMPEDANCE PASSIVITY BENCHMARK")
    print("=" * 92)
    print(
        f"Grid Rating & Strength: S_rated = {qs_res.spec.s_rated_va / 1e3:.1f} kVA | "
        f"SCR = {qs_res.spec.scr:.1f} | X/R = {qs_res.spec.xr_ratio:.1f} | H = {qs_res.spec.h_inertia_s:.2f} s"
    )
    print(
        f"4-State Quasi-Static:   Nadir = {qs_res.nadir_freq_hz:.3f} Hz | "
        f"Peak P_pcc = {qs_res.peak_power_kw:.2f} kW | P_final = {qs_res.final_power_kw:.2f} kW"
    )
    print(
        f"14-State EM VSG LCL:    Nadir = {em_res.nadir_freq_hz:.3f} Hz | "
        f"Peak P_pcc = {em_res.peak_power_kw:.2f} kW | "
        f"P_final = {em_res.final_power_kw:.2f} kW | f_res = {f_res:.0f} Hz"
    )
    dr, vs, dv = triad_res.droop, triad_res.vsg, triad_res.dvoc
    print(
        f"GFM Triad Rise & Peak:  "
        f"Droop t_90 = {dr.step_rise_time_90_ms:.1f} ms ({dr.freq_drop_peak_power_kw:.2f} kW) | "
        f"VSG t_90 = {vs.step_rise_time_90_ms:.1f} ms ({vs.freq_drop_peak_power_kw:.2f} kW) | "
        f"dVOC t_90 = {dv.step_rise_time_90_ms:.1f} ms ({dv.freq_drop_peak_power_kw:.2f} kW)"
    )
    print(
        f"Weak-Grid SCR Collapse: "
        f"GFL SCR_crit (55/35/10 Hz) = {wg_res.scr_crit_gfl_fast:.2f} / "
        f"{wg_res.scr_crit_gfl_med:.2f} / {wg_res.scr_crit_gfl_slow:.2f} | "
        f"GFL LOS = {wg_res.trip_gfl_med.lost_synchronism} vs GFM LOS = {wg_res.trip_gfm_vsg.lost_synchronism}"
    )
    sp, sh, jp, jh = (
        em_frt_res.sag_tvi_paw,
        em_frt_res.sag_hard_clamp,
        em_frt_res.jump_tvi_paw,
        em_frt_res.jump_hard_clamp,
    )
    print(
        f"14-State EM FRT & Jump: "
        f"0.15pu Sag TVI+PAW I_pk = {sp.peak_bridge_current_a:.2f} A (delta {sp.max_abs_delta_deg:.1f} deg vs "
        f"Clamp {sh.max_abs_delta_deg:.0f} deg) | "
        f"-45deg Jump I_pk = {jp.peak_bridge_current_a:.2f} A (Clamp LOS {jh.max_abs_delta_deg:.0f} deg)"
    )
    av, au, g35, g55 = (
        ssi_res.admittance_vsg,
        ssi_res.admittance_vsg_unity_ff,
        ssi_res.admittance_gfl35,
        ssi_res.admittance_gfl55,
    )
    print(
        f"dq Admittance & Nyquist:"
        f" VSG(F_i=0.75) nu_min = {av.min_passivity_high_freq_s:+.3f} S (Z={av.nyquist_rhp_poles_z}) |"
        f" VSG(F_i=1.0) nu_min = {au.min_passivity_high_freq_s:+.2f} S (Z={au.nyquist_rhp_poles_z}) |"
        f" GFL(35/55Hz) Z = {g35.nyquist_rhp_poles_z}/{g55.nyquist_rhp_poles_z}"
    )
    print(
        f"Reduced-Order 0.5pu Sag: I_max = {frt['fault_max_current_a']:.2f} A | "
        f"Fault RoCoF = {frt['max_rocof']:.4f} Hz/s | Settling = {frt['recovery_settling_time_s']:.3f} s"
    )
    print(
        f"Saved Figures:          {fig_qs} | {fig_em} | {fig_triad} | {fig_wg} | {fig_em_frt} | {fig_ssi}"
    )
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
