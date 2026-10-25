"""Reproducible closed-loop Thévenin GFM & reduced-order FRT benchmark entry point."""

from __future__ import annotations

from pathlib import Path

from src.benchmark import run_fault_ride_through_benchmark, validate_frt_benchmark_results
from src.thevenin_network import TheveninGridSpec, simulate_coupled_vsg_transient
from src.visualizer import plot_closed_loop_thevenin_benchmark


def generate_reference_transient_figure(
    save_path: str | Path = "results/inverter_transient_response.png",
):
    """Run the canonical closed-loop Thévenin GFM benchmark and write the 4-panel figure."""
    spec = TheveninGridSpec()
    res = simulate_coupled_vsg_transient(spec)
    out = plot_closed_loop_thevenin_benchmark(res, save_path=save_path)
    return res, out


def main(
    save_path: str | Path = "results/inverter_transient_response.png",
) -> int:
    res, fig_path = generate_reference_transient_figure(save_path=save_path)
    frt = run_fault_ride_through_benchmark()
    validate_frt_benchmark_results(frt, require_current_limit=True)

    print("=" * 78)
    print("GRID-FORMING INVERTER CLOSED-LOOP THÉVENIN & FRT VERIFICATION BENCHMARK")
    print("=" * 78)
    print(
        f"Grid Rating & Strength: S_rated = {res.spec.s_rated_va / 1e3:.1f} kVA | "
        f"SCR = {res.spec.scr:.1f} | X/R = {res.spec.xr_ratio:.1f} | H = {res.spec.h_inertia_s:.2f} s"
    )
    print(
        f"Closed-Loop VSG Step:   Nadir = {res.nadir_freq_hz:.3f} Hz | "
        f"Peak RoCoF = {res.peak_rocof_hz_s:.3f} Hz/s | Peak P_pcc = {res.peak_power_kw:.2f} kW"
    )
    print(
        f"Post-Droop Equilibrium: P_final = {res.final_power_kw:.2f} kW | "
        f"delta_final = {res.final_delta_deg:.2f} deg | |V_pcc| = {res.final_v_pcc_pu:.4f} pu"
    )
    print(
        f"Reduced-Order 0.5pu Sag: I_max = {frt['fault_max_current_a']:.2f} A | "
        f"Fault RoCoF = {frt['max_rocof']:.4f} Hz/s | Settling = {frt['recovery_settling_time_s']:.3f} s"
    )
    print(f"Saved Figure Artifact:  {fig_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
