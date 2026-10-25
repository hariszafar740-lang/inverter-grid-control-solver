"""Reproducible closed-loop Thévenin, 14-state electromagnetic LCL & FRT benchmark entry point."""

from __future__ import annotations

from pathlib import Path

from src.benchmark import run_fault_ride_through_benchmark, validate_frt_benchmark_results
from src.cascaded_control import simulate_electromagnetic_gfm_transient
from src.thevenin_network import TheveninGridSpec, simulate_coupled_vsg_transient
from src.visualizer import (
    plot_closed_loop_thevenin_benchmark,
    plot_lcl_cascaded_electromagnetic_benchmark,
)


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


def main(
    save_path: str | Path = "results/inverter_transient_response.png",
    lcl_save_path: str | Path | None = None,
) -> int:
    out_main = Path(save_path)
    out_lcl = (
        Path(lcl_save_path)
        if lcl_save_path is not None
        else out_main.parent / "lcl_cascaded_electromagnetic_response.png"
    )

    qs_res, fig_qs = generate_reference_transient_figure(save_path=out_main)
    em_res, fig_em = generate_lcl_electromagnetic_figure(qs_res=qs_res, save_path=out_lcl)
    frt = run_fault_ride_through_benchmark()
    validate_frt_benchmark_results(frt, require_current_limit=True)

    f_res = em_res.lcl.coupled_resonance_hz(em_res.grid.l_g_h)
    print("=" * 78)
    print("GRID-FORMING INVERTER CLOSED-LOOP THÉVENIN, 14-STATE LCL & FRT BENCHMARK")
    print("=" * 78)
    print(
        f"Grid Rating & Strength: S_rated = {qs_res.spec.s_rated_va / 1e3:.1f} kVA | "
        f"SCR = {qs_res.spec.scr:.1f} | X/R = {qs_res.spec.xr_ratio:.1f} | H = {qs_res.spec.h_inertia_s:.2f} s"
    )
    print(
        f"4-State Quasi-Static:   Nadir = {qs_res.nadir_freq_hz:.3f} Hz | "
        f"Peak P_pcc = {qs_res.peak_power_kw:.2f} kW | P_final = {qs_res.final_power_kw:.2f} kW"
    )
    print(
        f"14-State EM dq LCL:     Nadir = {em_res.nadir_freq_hz:.3f} Hz | "
        f"Peak P_pcc = {em_res.peak_power_kw:.2f} kW | "
        f"P_final = {em_res.final_power_kw:.2f} kW | f_res = {f_res:.0f} Hz"
    )
    print(
        f"Reduced-Order 0.5pu Sag: I_max = {frt['fault_max_current_a']:.2f} A | "
        f"Fault RoCoF = {frt['max_rocof']:.4f} Hz/s | Settling = {frt['recovery_settling_time_s']:.3f} s"
    )
    print(f"Saved Figures:          {fig_qs} | {fig_em}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
