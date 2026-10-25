"""Deterministic Matplotlib visualization routines for Grid-Forming Inverter benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cascaded_control import ElectromagneticTransientResult
from .em_frt_benchmark import ElectromagneticFRTBenchmarkResult
from .gfm_triad import GFMTriadBenchmarkResult
from .lcl_plant import simulate_lcl_damping_step
from .small_signal_impedance import SmallSignalImpedanceBenchmarkResult
from .thevenin_network import ClosedLoopTransientResult
from .weak_grid_benchmark import SCRStabilityPoint, WeakGridBenchmarkResult


def plot_inverter_response(
    time_array: Sequence[float] | np.ndarray,
    f_vsm: Sequence[float] | np.ndarray,
    p_filt: Sequence[float] | np.ndarray,
    v_mag: Sequence[float] | np.ndarray,
    save_path: str | Path = "results/inverter_transient_response.png",
) -> Path:
    """Backward-compatible 3-panel transient response plotter."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

    axs[0].plot(time_array, f_vsm, color="#1f77b4", linewidth=1.8, label="VSM Frequency f(t)")
    axs[0].axhline(y=50.0, color="#334155", linestyle="--", alpha=0.8, label="Nominal Frequency (50 Hz)")
    axs[0].set_ylabel("Frequency (Hz)", fontsize=9.5, fontweight="bold")
    axs[0].set_title("Grid-Forming Inverter Dynamic Transient Response", fontsize=11.5, fontweight="bold")
    axs[0].grid(True, ls="--", alpha=0.45)
    axs[0].legend(loc="upper right")

    axs[1].plot(
        time_array,
        np.asarray(p_filt, dtype=float) / 1000.0,
        color="#2ca02c",
        linewidth=1.8,
        label="Active Power P_filt(t)",
    )
    axs[1].set_ylabel("Power (kW)", fontsize=9.5, fontweight="bold")
    axs[1].grid(True, ls="--", alpha=0.45)
    axs[1].legend(loc="upper right")

    axs[2].plot(time_array, v_mag, color="#d62728", linewidth=1.8, label="Peak Voltage Magnitude |V_ref*|(t)")
    axs[2].set_xlabel("Time (Seconds)", fontsize=10, fontweight="bold")
    axs[2].set_ylabel("Voltage (V)", fontsize=9.5, fontweight="bold")
    axs[2].grid(True, ls="--", alpha=0.45)
    axs[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=220, metadata={"Software": "inverter-grid-control-solver"})
    plt.close(fig)
    return out


def plot_closed_loop_thevenin_benchmark(
    res: ClosedLoopTransientResult,
    save_path: str | Path = "results/inverter_transient_response.png",
) -> Path:
    """Render the 4-panel closed-loop VSG + Q-V Droop + Virtual Impedance Thévenin benchmark figure."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), dpi=220)
    fig.suptitle(
        f"Closed-Loop Grid-Forming VSG + Virtual Impedance on Thévenin AC Grid "
        f"(SCR = {res.spec.scr:.1f}, X/R = {res.spec.xr_ratio:.0f}, H = {res.spec.h_inertia_s:.2f} s)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    t = res.time_s

    # (a) VSG Rotor vs Grid Frequency
    ax = axes[0, 0]
    ax.plot(t, res.f_grid_hz, color="#64748b", ls="--", lw=1.6, label="Grid Frequency f_grid(t)")
    ax.plot(
        t,
        res.f_vsg_hz,
        color="#1d4ed8",
        lw=2.1,
        label=f"VSG Rotor f_vsg(t) (Nadir {res.nadir_freq_hz:.3f} Hz)",
    )
    ax.axvline(0.25, color="#94a3b8", ls=":", lw=1.1)
    ax.axvline(1.15, color="#94a3b8", ls=":", lw=1.1)
    ax.set_title("(a) Inertial Frequency Response & Synchronizing Lock", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Frequency (Hz)", fontsize=9.5)
    ax.set_ylim(49.72, 50.17)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.94)

    # (b) Active & Reactive Power Injection at PCC
    ax = axes[0, 1]
    ax.plot(t, res.p_ref_w / 1000.0, color="#475569", ls="--", lw=1.5, label="Setpoint P_ref(t)")
    ax.plot(
        t,
        res.p_pcc_w / 1000.0,
        color="#059669",
        lw=2.1,
        label=f"PCC Active Power P_pcc (Peak {res.peak_power_kw:.2f} kW)",
    )
    ax.plot(t, res.q_pcc_var / 1000.0, color="#d97706", lw=1.8, label="PCC Reactive Power Q_pcc (kvar)")
    ax.set_title("(b) Dispatch Tracking & Inertial Droop Support (+Delta P)", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Three-Phase Power (kW / kvar)", fontsize=9.5)
    ax.set_ylim(-9.5, 24.5)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.94)

    # (c) Rotor Power Angle delta(t) & PCC / Internal Voltage
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(t, res.delta_deg, color="#7c3aed", lw=2.1, label="Power Angle delta(t) (deg)")
    l2 = ax2.plot(
        t,
        res.v_pcc_peak_v / res.spec.v_peak_nom_v,
        color="#dc2626",
        lw=1.8,
        ls="-.",
        label="PCC Voltage |V_pcc| (pu)",
    )
    l3 = ax2.plot(
        t,
        res.e_int_peak_v / res.spec.v_peak_nom_v,
        color="#0284c7",
        lw=1.5,
        ls=":",
        label="Q-V Droop Internal |E*| (pu)",
    )
    ax.set_title("(c) Rotor Angle delta(t) & Q-V Droop Voltage Regulation", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Rotor Angle delta (deg)", fontsize=9.5, color="#6d28d9")
    ax2.set_ylabel("Voltage Magnitude (pu)", fontsize=9.5, color="#b91c1c")
    ax.set_ylim(-1.0, 18.2)
    ax2.set_ylim(0.94, 1.04)
    ax.grid(True, ls="--", alpha=0.4)
    lines = l1 + l2 + l3
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8.2, framealpha=0.94)

    # (d) Power-Angle P(delta) Synchronizing Curve & Operating Trajectory
    ax = axes[1, 1]
    ax.plot(
        res.p_delta_curve_deg,
        res.p_delta_curve_kw,
        color="#0f172a",
        lw=2.0,
        label="Steady-State P_pcc(delta) Characteristic",
    )
    ax.plot(
        res.delta_deg,
        res.p_pcc_w / 1000.0,
        color="#10b981",
        lw=1.7,
        alpha=0.88,
        label="Dynamic Trajectory (delta(t), P_pcc(t))",
    )
    ax.scatter(
        [res.delta_deg[0], res.final_delta_deg],
        [res.p_pcc_w[0] / 1000.0, res.final_power_kw],
        color=["#64748b", "#dc2626"],
        zorder=5,
        s=42,
    )
    ax.annotate(
        f"Post-Droop Equilibrium\n({res.final_delta_deg:.2f} deg, {res.final_power_kw:.2f} kW)",
        xy=(res.final_delta_deg, res.final_power_kw),
        xytext=(res.final_delta_deg + 8.5, res.final_power_kw - 9.5),
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.94},
        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.2},
    )
    ax.set_title("(d) Synchronizing Torque P(delta) Locus & Equilibrium", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Power Angle delta (deg)", fontsize=9.5)
    ax.set_ylabel("Active Power P_pcc (kW)", fontsize=9.5)
    ax.set_ylim(-25.5, 26.5)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.4, framealpha=0.94)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=220, metadata={"Software": "inverter-grid-control-solver"})
    plt.close(fig)
    return out


def plot_lcl_cascaded_electromagnetic_benchmark(
    em_res: ElectromagneticTransientResult,
    qs_res: ClosedLoopTransientResult,
    save_path: str | Path = "results/lcl_cascaded_electromagnetic_response.png",
) -> Path:
    """Render the 4-panel 14-state electromagnetic LCL + cascaded dq PI benchmark figure."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), dpi=220)
    f_res_coupled = em_res.lcl.coupled_resonance_hz(em_res.grid.l_g_h)
    fig.suptitle(
        f"14-State Electromagnetic dq LCL Plant + Cascaded Dual-Loop PI GFM Control "
        f"(f_res = {f_res_coupled:.0f} Hz, F_i = {em_res.pi_cfg.feedforward_current_gain:.2f})",
        fontsize=12.3,
        fontweight="bold",
        y=0.98,
    )

    # (a) Macro-timescale comparison: 14-state Electromagnetic LCL vs 4-state Quasi-Static
    ax = axes[0, 0]
    ax.plot(
        qs_res.time_s,
        qs_res.p_pcc_w / 1000.0,
        color="#64748b",
        ls="--",
        lw=1.8,
        label=f"4-State Quasi-Static P_pcc ({qs_res.final_power_kw:.2f} kW)",
    )
    ax.plot(
        em_res.time_s,
        em_res.p_pcc_w / 1000.0,
        color="#059669",
        lw=2.0,
        alpha=0.92,
        label=f"14-State Electromagnetic LCL P_pcc ({em_res.final_power_kw:.2f} kW)",
    )
    ax.plot(
        em_res.time_s,
        em_res.p_ref_w / 1000.0,
        color="#0f172a",
        ls=":",
        lw=1.4,
        label="Dispatch Setpoint P_ref(t)",
    )
    ax.set_title("(a) Macro-Timescale Fidelity: 14-State LCL vs. Quasi-Static", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("PCC Active Power (kW)", fontsize=9.5)
    ax.set_ylim(-3.5, 25.5)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.3, framealpha=0.94)

    # (b) Inner dq Current Loop Tracking across the t = 1.15 s Grid Frequency Drop
    ax = axes[0, 1]
    mask_zoom = (em_res.time_s >= 1.13) & (em_res.time_s <= 1.45)
    tz = em_res.time_s[mask_zoom]
    ax.plot(tz, em_res.i1_d_ref_a[mask_zoom], color="#1d4ed8", ls="--", lw=1.7, label="Reference i_1d* (A)")
    ax.plot(tz, em_res.i1_d_a[mask_zoom], color="#3b82f6", lw=1.9, alpha=0.9, label="Bridge Current i_1d (A)")
    ax.plot(tz, em_res.i1_q_ref_a[mask_zoom], color="#b45309", ls="--", lw=1.7, label="Reference i_1q* (A)")
    ax.plot(tz, em_res.i1_q_a[mask_zoom], color="#f59e0b", lw=1.9, alpha=0.9, label="Bridge Current i_1q (A)")
    ax.axvline(1.15, color="#94a3b8", ls=":", lw=1.1)
    ax.set_title("(b) Inner dq Current PI Tracking (f_bw,i = 1.0 kHz)", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Synchronous dq Current (A)", fontsize=9.5)
    ax.set_ylim(-14.0, 52.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="center right", fontsize=8.2, framealpha=0.94)

    # (c) Outer dq Filter-Capacitor Voltage Regulation & Cross-Coupling Decoupling
    ax = axes[1, 0]
    ax.plot(
        em_res.time_s,
        em_res.vf_d_ref_v,
        color="#7c3aed",
        ls="--",
        lw=1.6,
        label="Reference v_fd* (V)",
    )
    ax.plot(
        em_res.time_s,
        em_res.vf_d_v,
        color="#a855f7",
        lw=1.9,
        alpha=0.9,
        label="Filter Bus v_fd (V)",
    )
    ax.plot(
        em_res.time_s,
        em_res.vf_q_ref_v,
        color="#dc2626",
        ls="--",
        lw=1.6,
        label="Reference v_fq* (V)",
    )
    ax.plot(
        em_res.time_s,
        em_res.vf_q_v,
        color="#fb7185",
        lw=1.8,
        alpha=0.9,
        label="Filter Bus v_fq (V)",
    )
    ax.set_title("(c) Outer dq Capacitor-Voltage PI & Z_v Conditioning", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Filter Bus Voltage v_f,dq (V)", fontsize=9.5)
    ax.set_ylim(-65.0, 385.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="center right", fontsize=8.2, framealpha=0.94)

    # (d) Sub-cycle LCL Filter Resonance Damping (Rd = 1.80 Ohm vs Underdamped Rd = 0.02 Ohm)
    ax = axes[1, 1]
    t_ud, vcd_ud = simulate_lcl_damping_step(rd_ohm=0.02, grid=em_res.grid)
    t_dp, vcd_dp = simulate_lcl_damping_step(rd_ohm=em_res.lcl.rd_ohm, grid=em_res.grid)
    ax.plot(
        t_ud * 1e3,
        vcd_ud,
        color="#ef4444",
        lw=1.5,
        alpha=0.88,
        label="Underdamped R_d = 0.02 Ohm (1.05 kHz Ringing)",
    )
    ax.plot(
        t_dp * 1e3,
        vcd_dp,
        color="#0f172a",
        lw=2.1,
        label=f"Passively Damped R_d = {em_res.lcl.rd_ohm:.2f} Ohm",
    )
    ax.axvline(1.0, color="#94a3b8", ls=":", lw=1.1)
    ax.set_title("(d) Sub-Cycle LCL Capacitor Step & Passive Resonance Damping", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=9.5)
    ax.set_ylabel("Capacitor Voltage v_cd (V)", fontsize=9.5)
    ax.set_ylim(322.0, 352.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.3, framealpha=0.94)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=220, metadata={"Software": "inverter-grid-control-solver"})
    plt.close(fig)
    return out


def plot_gfm_triad_benchmark(
    triad: GFMTriadBenchmarkResult,
    save_path: str | Path = "results/gfm_triad_comparative_response.png",
) -> Path:
    """Render the 4-panel Comparative GFM Control Triad (Droop vs VSG vs dVOC) figure."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.2), dpi=220)
    fig.suptitle(
        f"Comparative Grid-Forming Control Triad on 6-State Electromagnetic dq LCL Plant "
        f"(SCR = {triad.grid.scr:.1f}, X/R = {triad.grid.xr_ratio:.0f})",
        fontsize=12.4,
        fontweight="bold",
        y=0.98,
    )

    t = triad.vsg.time_s

    # (a) Internal GFM Synchronizing Frequency & Nadir Response
    ax = axes[0, 0]
    ax.plot(t, triad.vsg.f_grid_hz, color="#64748b", ls="--", lw=1.5, label="Grid Frequency f_grid(t)")
    ax.plot(
        t,
        triad.droop.f_gfm_hz,
        color="#d97706",
        lw=1.8,
        alpha=0.92,
        label=f"Droop (13-State, Nadir {triad.droop.nadir_freq_hz:.3f} Hz, 15 Hz LPF)",
    )
    ax.plot(
        t,
        triad.vsg.f_gfm_hz,
        color="#1d4ed8",
        lw=2.1,
        label=f"VSG (14-State, Nadir {triad.vsg.nadir_freq_hz:.3f} Hz, H = {triad.grid.h_inertia_s:.2f} s)",
    )
    ax.plot(
        t,
        triad.dvoc.f_gfm_hz,
        color="#059669",
        lw=1.9,
        alpha=0.92,
        label=f"dVOC (12-State, Nadir {triad.dvoc.nadir_freq_hz:.3f} Hz, Zero LPF)",
    )
    ax.axvline(0.25, color="#94a3b8", ls=":", lw=1.1)
    ax.axvline(1.15, color="#94a3b8", ls=":", lw=1.1)
    ax.set_title("(a) Internal Frequency Trajectory & Inertial vs. Oscillator Lock", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Frequency (Hz)", fontsize=9.5)
    ax.set_ylim(49.70, 50.30)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.95)

    # (b) Active Power Dispatch Rise Time (t_90) & Primary Frequency Support
    ax = axes[0, 1]
    ax.plot(t, triad.vsg.p_ref_w / 1000.0, color="#0f172a", ls=":", lw=1.4, label="Dispatch Setpoint P_ref(t)")
    ax.plot(
        t,
        triad.droop.p_pcc_w / 1000.0,
        color="#d97706",
        lw=1.85,
        label=(
            f"Droop (t_90 = {triad.droop.step_rise_time_90_ms:.1f} ms, "
            f"Peak {triad.droop.freq_drop_peak_power_kw:.2f} kW)"
        ),
    )
    ax.plot(
        t,
        triad.vsg.p_pcc_w / 1000.0,
        color="#1d4ed8",
        lw=2.05,
        label=(
            f"VSG (t_90 = {triad.vsg.step_rise_time_90_ms:.1f} ms, "
            f"Peak {triad.vsg.freq_drop_peak_power_kw:.2f} kW)"
        ),
    )
    ax.plot(
        t,
        triad.dvoc.p_pcc_w / 1000.0,
        color="#059669",
        lw=1.9,
        label=(
            f"dVOC (t_90 = {triad.dvoc.step_rise_time_90_ms:.1f} ms, "
            f"Peak {triad.dvoc.freq_drop_peak_power_kw:.2f} kW)"
        ),
    )
    ax.set_title("(b) PCC Active Power Rise Time & Inertial Over-Power Trade-Off", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("PCC Active Power P_pcc (kW)", fontsize=9.5)
    ax.set_ylim(-2.5, 24.8)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.94)

    # (c) Dominant Outer-Loop & Synchronizing Small-Signal Eigenvalue Locus
    ax = axes[1, 0]
    ax.axvline(0.0, color="#dc2626", ls="--", lw=1.4, label="Hurwitz Boundary Re(lambda) = 0")
    ax.axvspan(0.0, 8.0, color="#fee2e2", alpha=0.55)
    two_pi = 2.0 * np.pi
    for member, color, marker, label_prefix in (
        (triad.droop, "#d97706", "s", "Droop 13-State"),
        (triad.vsg, "#1d4ed8", "o", "VSG 14-State"),
        (triad.dvoc, "#059669", "^", "dVOC 12-State"),
    ):
        dom = member.eigenvalues[member.eigenvalues.real >= -110.0]
        ax.scatter(
            dom.real,
            dom.imag / two_pi,
            color=color,
            marker=marker,
            s=52,
            alpha=0.92,
            edgecolors="#0f172a",
            linewidths=0.5,
            zorder=4,
            label=f"{label_prefix} (max Re = {member.max_real_eig_s_inv:.2f} s^-1)",
        )
    ax.annotate(
        "dVOC Kuramoto Sync\n(-10.65 s^-1, 0 Hz)",
        xy=(-10.65, 0.15),
        xytext=(-52.0, 2.8),
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#cbd5e1", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "#059669", "lw": 1.15},
    )
    ax.annotate(
        "VSG Swing Mode\n(+-1.53 Hz)",
        xy=(-4.24, -1.53),
        xytext=(-48.0, -5.8),
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#cbd5e1", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "#1d4ed8", "lw": 1.15},
    )
    ax.set_title("(c) Dominant Synchronizing & Outer-Loop Eigenvalue Spectrum", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Real Part Re(lambda) (s^-1)", fontsize=9.5)
    ax.set_ylabel("Oscillation Frequency Im(lambda)/(2*pi) (Hz)", fontsize=9.5)
    ax.set_xlim(-112.0, 8.0)
    ax.set_ylim(-11.5, 11.5)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7.9, framealpha=0.95)

    # (d) Nonlinear Andronov-Hopf dVOC Black-Start Self-Excitation Spiral (220 ms)
    ax = axes[1, 1]
    bs = triad.black_start
    v0 = bs.config.v0_peak_v
    mask_sp = bs.time_s <= 0.22 + 1e-12
    va_pu = bs.v_alpha_v[mask_sp] / v0
    vb_pu = bs.v_beta_v[mask_sp] / v0
    ang = np.linspace(0.0, 2.0 * np.pi, 361)

    # Invisible bounding anchor keeps aspect="equal" warning-free with an unobstructed left legend bay
    ax.plot([-2.85, 1.35], [-1.25, 1.30], alpha=0.0)
    ax.plot(np.cos(ang), np.sin(ang), color="#0f172a", ls="--", lw=1.6, label="Limit Cycle ||v_ab|| = 1.00 pu")
    ax.plot(
        va_pu,
        vb_pu,
        color="#059669",
        lw=1.55,
        alpha=0.92,
        label="220 ms Hopf Spiral (0.10 -> 1.00 pu)",
    )
    ax.scatter([va_pu[0]], [vb_pu[0]], color="#dc2626", s=46, zorder=5, label="Black-Start Seed (0.10 pu)")
    ax.scatter([va_pu[-1]], [vb_pu[-1]], color="#1d4ed8", s=46, zorder=5, label="220 ms Lock (0.996 pu)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("(d) Nonlinear dVOC Black-Start Limit-Cycle Self-Excitation", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Stationary EMF v_alpha / V_0 (pu)", fontsize=9.5)
    ax.set_ylabel("Stationary EMF v_beta / V_0 (pu)", fontsize=9.5)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.95)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=220, metadata={"Software": "inverter-grid-control-solver"})
    plt.close(fig)
    return out


def _valid_scr_xy(
    pts: Sequence[SCRStabilityPoint],
    attr: str,
) -> tuple[np.ndarray, np.ndarray]:
    valid = [p for p in pts if p.equilibrium_exists]
    return (
        np.array([p.scr for p in valid], dtype=float),
        np.array([getattr(p, attr) for p in valid], dtype=float),
    )


def plot_weak_grid_scr_benchmark(
    bench: WeakGridBenchmarkResult,
    save_path: str | Path = "results/gfl_vs_gfm_weak_grid_collapse.png",
) -> Path:
    """Render 4-panel GFL SRF-PLL vs. GFM weak-grid SCR stability & collapse benchmark."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8))
    p_kw = bench.p_dispatch_w / 1000.0
    fig.suptitle(
        f"Weak-Grid SCR Stability & Line-Opening Collapse: GFL (SRF-PLL) vs. GFM ({p_kw:.1f} kW = 0.75 pu Dispatch)",
        fontsize=12.4,
        fontweight="bold",
        y=0.985,
    )

    # (a) Small-Signal Dominant Real Eigenvalue vs. Grid SCR
    ax = axes[0, 0]
    ax.axhline(0.0, color="#dc2626", ls="--", lw=1.35, label="Hopf Boundary Re(lambda) = 0")
    ax.axhspan(0.0, 62.0, color="#fee2e2", alpha=0.50)
    ax.axvline(
        bench.scr_static_limit_gfl,
        color="#7f1d1d",
        ls=":",
        lw=1.45,
        label=f"GFL Nose-Curve Limit (SCR ~ {bench.scr_static_limit_gfl:.2f})",
    )

    for pts, color, marker, label in (
        (
            bench.gfl_fast_sweep,
            "#dc2626",
            "v",
            f"GFL PLL 55 Hz (SCR_crit = {bench.scr_crit_gfl_fast:.2f})",
        ),
        (
            bench.gfl_med_sweep,
            "#ea580c",
            "D",
            f"GFL PLL 35 Hz (SCR_crit = {bench.scr_crit_gfl_med:.2f})",
        ),
        (
            bench.gfl_slow_sweep,
            "#ca8a04",
            "s",
            f"GFL PLL 10 Hz (SCR_crit = {bench.scr_crit_gfl_slow:.2f})",
        ),
        (bench.vsg_sweep, "#1d4ed8", "o", "GFM VSG 14-State (Hurwitz to 1.15)"),
        (bench.dvoc_sweep, "#059669", "^", "GFM dVOC 12-State (Hurwitz to 1.15)"),
    ):
        sx, sy = _valid_scr_xy(pts, "max_real_eig_s_inv")
        ax.plot(sx, sy, color=color, marker=marker, ms=4.6, lw=1.75, label=label)

    ax.set_title("(a) Small-Signal Stability Margin vs. Grid Short-Circuit Ratio", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Grid Short-Circuit Ratio SCR (lower = weaker grid)", fontsize=9.5)
    ax.set_ylabel("Max Real Eigenvalue Re(lambda) (s^-1)", fontsize=9.5)
    ax.set_xlim(1.05, 6.15)
    ax.set_ylim(-26.0, 58.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7.65, framealpha=0.95)

    # (b) Static Weak-Grid PCC Voltage Nose-Curve Knee vs. GFM Autonomous Support
    ax = axes[0, 1]
    ax.axvline(
        bench.scr_static_limit_gfl,
        color="#7f1d1d",
        ls=":",
        lw=1.45,
        label=f"GFL Saddle-Node Collapse (SCR ~ {bench.scr_static_limit_gfl:.2f})",
    )
    ax.axvspan(1.05, bench.scr_static_limit_gfl, color="#fef2f2", alpha=0.72)

    sx_gfl, vy_gfl = _valid_scr_xy(bench.gfl_slow_sweep, "v_pcc_eq_pu")
    sx_vsg, vy_vsg = _valid_scr_xy(bench.vsg_sweep, "v_pcc_eq_pu")
    sx_dvoc, vy_dvoc = _valid_scr_xy(bench.dvoc_sweep, "v_pcc_eq_pu")

    ax.plot(
        sx_gfl,
        vy_gfl,
        color="#ea580c",
        marker="D",
        ms=4.8,
        lw=1.9,
        label="GFL Equilibrium Branch (Q_ref = 0)",
    )
    ax.scatter(
        [sx_gfl[-1]],
        [vy_gfl[-1]],
        color="#dc2626",
        marker="X",
        s=78,
        zorder=5,
        label=f"Last GFL Equilibrium ({sx_gfl[-1]:.2f}, {vy_gfl[-1]:.2f} pu)",
    )
    ax.plot(
        sx_vsg,
        vy_vsg,
        color="#1d4ed8",
        marker="o",
        ms=4.6,
        lw=1.85,
        label=f"GFM VSG (V_pcc = {vy_vsg[-1]:.3f} pu @ SCR 1.15)",
    )
    ax.plot(
        sx_dvoc,
        vy_dvoc,
        color="#059669",
        marker="^",
        ms=4.8,
        lw=1.85,
        ls="--",
        label=f"GFM dVOC (V_pcc = {vy_dvoc[-1]:.3f} pu @ SCR 1.15)",
    )
    ax.set_title("(b) Steady-State PCC Voltage Nose-Curve Collapse vs. GFM Support", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Grid Short-Circuit Ratio SCR", fontsize=9.5)
    ax.set_ylabel("Equilibrium PCC Voltage |v_pcc| (pu)", fontsize=9.5)
    ax.set_xlim(1.05, 6.15)
    ax.set_ylim(0.76, 1.03)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.95)

    # (c) Line-Opening Contingency (SCR 4.0 -> 1.22): PCC Voltage Collapse vs. Ride-Through
    ax = axes[1, 0]
    tr_m, tr_s = bench.trip_gfl_med, bench.trip_gfl_slow
    tr_v, tr_d = bench.trip_gfm_vsg, bench.trip_gfm_dvoc
    ax.axvline(
        tr_m.trip_time_s,
        color="#334155",
        ls=":",
        lw=1.35,
        label=f"Line Trip SCR {tr_m.scr_pre:.1f} -> {tr_m.scr_post:.2f}",
    )
    ax.plot(
        tr_m.time_s,
        tr_m.v_pcc_pu,
        color="#ea580c",
        lw=1.55,
        alpha=0.92,
        label=f"GFL PLL 35 Hz (Collapse V_min = {tr_m.min_v_pcc_pu:.2f} pu)",
    )
    ax.plot(
        tr_s.time_s,
        tr_s.v_pcc_pu,
        color="#ca8a04",
        lw=1.45,
        ls="--",
        alpha=0.90,
        label=f"GFL PLL 10 Hz (Collapse V_min = {tr_s.min_v_pcc_pu:.2f} pu)",
    )
    ax.plot(
        tr_v.time_s,
        tr_v.v_pcc_pu,
        color="#1d4ed8",
        lw=2.05,
        label=f"GFM VSG Ride-Through (V_min = {tr_v.min_v_pcc_pu:.3f} pu)",
    )
    ax.plot(
        tr_d.time_s,
        tr_d.v_pcc_pu,
        color="#059669",
        lw=1.95,
        label=f"GFM dVOC Ride-Through (V_min = {tr_d.min_v_pcc_pu:.3f} pu)",
    )
    ax.set_title(
        "(c) Line-Opening Contingency (SCR 4.0 -> 1.22): PCC Voltage |v_pcc|",
        fontsize=10.3,
        fontweight="bold",
    )
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("PCC Voltage Magnitude |v_pcc| (pu)", fontsize=9.5)
    ax.set_ylim(-0.05, 1.76)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7.65, framealpha=0.95)

    # (d) Synchronization Phase Angle Trajectory & Unbounded PLL Pole-Slipping (LOS)
    ax = axes[1, 1]
    ax.axvline(tr_m.trip_time_s, color="#334155", ls=":", lw=1.35)
    ax.axhline(90.0, color="#64748b", ls="--", lw=1.15, label="90 deg Static Power-Transfer Ceiling")
    ax.axhline(180.0, color="#dc2626", ls="-.", lw=1.2, label="180 deg Pole-Slip / LOS Threshold")
    ax.plot(
        tr_m.time_s,
        tr_m.delta_deg,
        color="#ea580c",
        lw=1.75,
        label="GFL PLL 35 Hz (Unbounded Pole-Slipping LOS)",
    )
    ax.plot(
        tr_s.time_s,
        tr_s.delta_deg,
        color="#ca8a04",
        lw=1.65,
        ls="--",
        label="GFL PLL 10 Hz (Unbounded Pole-Slipping LOS)",
    )
    ax.plot(
        tr_v.time_s,
        tr_v.delta_deg,
        color="#1d4ed8",
        lw=2.1,
        label=f"GFM VSG Locked (delta_max = {tr_v.max_abs_delta_deg:.1f} deg)",
    )
    ax.plot(
        tr_d.time_s,
        tr_d.delta_deg,
        color="#059669",
        lw=2.0,
        label=f"GFM dVOC Locked (delta_max = {tr_d.max_abs_delta_deg:.1f} deg)",
    )
    ax.set_title("(d) Phase Angle delta(t): PLL Loss of Synchronism vs. GFM Lock", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Grid-Relative Phase Angle delta (deg)", fontsize=9.5)
    ax.set_ylim(-35.0, 1680.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7.65, framealpha=0.95)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=220, metadata={"Software": "inverter-grid-control-solver"})
    plt.close(fig)
    return out


def plot_electromagnetic_frt_benchmark(
    frt: ElectromagneticFRTBenchmarkResult,
    save_path: str | Path = "results/em_frt_tvi_phase_jump_response.png",
) -> Path:
    """Render 4-panel 14-state electromagnetic FRT, TVI, PAW & phase-jump benchmark figure."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.9))
    fig.suptitle(
        "14-State Electromagnetic GFM Fault Ride-Through & -45 deg Phase Jump: TVI + Power Anti-Windup",
        fontsize=12.3,
        fontweight="bold",
        y=0.985,
    )

    su = frt.sag_unprotected
    sh = frt.sag_hard_clamp
    st = frt.sag_tvi_only
    sp = frt.sag_tvi_paw
    t0_f, t1_f = sp.event_start_s, sp.event_end_s
    i_max = frt.grid.i_max_peak_a
    i_th = frt.tvi_cfg.i_thresh_peak_a

    # (a) Deep 0.15 pu Fault: Bridge Peak Current |i_1,dq|(t)
    ax = axes[0, 0]
    ax.axvspan(t0_f, t1_f, color="#fee2e2", alpha=0.45, label="0.15 pu Symmetrical Fault")
    ax.axhline(i_max, color="#dc2626", ls="--", lw=1.4, label=f"Hardware Ceiling I_max = {i_max:.0f} A")
    ax.axhline(i_th, color="#475569", ls=":", lw=1.2, label=f"TVI Threshold I_th = {i_th:.0f} A")
    ax.plot(
        su.time_s,
        su.i1_peak_a,
        color="#991b1b",
        lw=1.35,
        ls="-.",
        alpha=0.85,
        label=f"Unprotected (I_pk = {su.peak_bridge_current_a:.1f} A [3.91 pu])",
    )
    ax.plot(
        sh.time_s,
        sh.i1_peak_a,
        color="#ea580c",
        lw=1.65,
        label=f"Hard Clamp Only (I_pk = {sh.peak_bridge_current_a:.1f} A, sat {sh.saturation_duty_pct:.0f}%)",
    )
    ax.plot(
        st.time_s,
        st.i1_peak_a,
        color="#ca8a04",
        lw=1.65,
        ls="--",
        label=f"TVI Only (I_pk = {st.peak_bridge_current_a:.2f} A)",
    )
    ax.plot(
        sp.time_s,
        sp.i1_peak_a,
        color="#059669",
        lw=2.1,
        label=f"TVI + Power AW (I_pk = {sp.peak_bridge_current_a:.2f} A <= {i_max:.0f} A)",
    )
    ax.set_title("(a) Deep 0.15 pu Fault: Inverter Bridge Peak Current |i_1,dq|", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Bridge Peak Current |i_1,dq| (A)", fontsize=9.5)
    ax.set_ylim(0.0, 176.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7.45, framealpha=0.95)

    # (b) Deep 0.15 pu Fault: Virtual Rotor Angle Windup vs. PAW Stabilization
    ax = axes[0, 1]
    ax.axvspan(t0_f, t1_f, color="#fee2e2", alpha=0.45)
    ax.axhline(90.0, color="#64748b", ls="--", lw=1.15, label="90 deg Static Ceiling")
    ax.axhline(180.0, color="#dc2626", ls="-.", lw=1.2, label="180 deg Pole-Slip / LOS")
    ax.plot(
        su.time_s,
        su.delta_rel_deg,
        color="#991b1b",
        lw=1.4,
        ls="-.",
        alpha=0.85,
        label=f"Unprotected (delta_max = {su.max_abs_delta_deg:.1f} deg, 160 A!)",
    )
    ax.plot(
        sh.time_s,
        sh.delta_rel_deg,
        color="#ea580c",
        lw=1.8,
        label=f"Hard Clamp Only (Windup {sh.max_abs_delta_deg:.0f} deg, LOS)",
    )
    ax.plot(
        st.time_s,
        st.delta_rel_deg,
        color="#ca8a04",
        lw=1.75,
        ls="--",
        label=f"TVI Only (1 Cycle Slip {st.max_abs_delta_deg:.0f} deg)",
    )
    ax.plot(
        sp.time_s,
        sp.delta_rel_deg,
        color="#059669",
        lw=2.15,
        label=f"TVI + Power AW (Locked, max {sp.max_abs_delta_deg:.1f} deg)",
    )
    ax.set_title("(b) Deep 0.15 pu Fault: Virtual Rotor Angle delta_rel(t)", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Grid-Relative Rotor Angle delta (deg)", fontsize=9.5)
    ax.set_ylim(-15.0, 540.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7.55, framealpha=0.95)

    # (c) Deep 0.15 pu Fault: Active Power Recovery & Reactive Fault Support
    ax = axes[1, 0]
    ax.axvspan(t0_f, t1_f, color="#fee2e2", alpha=0.45)
    ax.axhline(0.0, color="#334155", lw=0.9, alpha=0.7)
    ax.plot(
        sh.time_s,
        sh.p_pcc_kw,
        color="#ea580c",
        lw=1.6,
        alpha=0.92,
        label=f"Hard Clamp P_pcc (Post-Fault {sh.post_event_power_kw:.1f} kW Reversal)",
    )
    ax.plot(
        st.time_s,
        st.p_pcc_kw,
        color="#ca8a04",
        lw=1.55,
        ls="--",
        label=f"TVI Only P_pcc (Slip Swing, P_fin = {st.post_event_power_kw:.1f} kW)",
    )
    ax.plot(
        sp.time_s,
        sp.p_pcc_kw,
        color="#059669",
        lw=2.1,
        label=f"TVI + PAW P_pcc (Smooth Recovery {sp.post_event_power_kw:.2f} kW)",
    )
    ax.plot(
        sp.time_s,
        sp.q_pcc_kvar,
        color="#1d4ed8",
        lw=1.85,
        ls="-.",
        label=f"TVI + PAW Q_pcc (+{sp.fault_mean_q_kvar:.2f} kvar Inductive Support)",
    )
    ax.set_title("(c) PCC Active Power Recovery & Reactive Fault Support", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("PCC Power P_pcc (kW) / Q_pcc (kvar)", fontsize=9.5)
    ax.set_ylim(-24.0, 36.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7.45, framealpha=0.95)

    # (d) Severe -45 deg Grid Phase Jump: Circular-Clamp Latch-Up vs. TVI Ride-Through
    ax = axes[1, 1]
    ju, jh, jp = frt.jump_unprotected, frt.jump_hard_clamp, frt.jump_tvi_paw
    ax.axvline(jp.event_start_s, color="#334155", ls=":", lw=1.35, label="-45 deg Grid Phase Jump (t = 0.20 s)")
    ax.axhline(180.0, color="#dc2626", ls="-.", lw=1.2, label="180 deg Latch-Up / LOS Threshold")
    ax.plot(
        ju.time_s,
        ju.delta_rel_deg,
        color="#991b1b",
        lw=1.45,
        ls="-.",
        alpha=0.85,
        label=f"Unprotected (Destructive I_pk = {ju.peak_bridge_current_a:.1f} A!)",
    )
    ax.plot(
        jh.time_s,
        jh.delta_rel_deg,
        color="#ea580c",
        lw=1.85,
        label=f"Hard Clamp Latch-Up (delta = {jh.max_abs_delta_deg:.0f} deg, sat {jh.saturation_duty_pct:.0f}%)",
    )
    ax.plot(
        jp.time_s,
        jp.delta_rel_deg,
        color="#059669",
        lw=2.15,
        label=f"TVI + PAW Ride-Through (I_pk = {jp.peak_bridge_current_a:.1f} A <= {i_max:.0f} A)",
    )
    ax.set_title("(d) -45 deg Grid Phase Jump: Circular-Clamp Latch-Up vs. TVI", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel("Grid-Relative Rotor Angle delta (deg)", fontsize=9.5)
    ax.set_ylim(-15.0, 415.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7.45, framealpha=0.95)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=220, metadata={"Software": "inverter-grid-control-solver"})
    plt.close(fig)
    return out


def plot_small_signal_impedance_benchmark(
    ssi: SmallSignalImpedanceBenchmarkResult,
    save_path: str | Path = "results/small_signal_eigenvalue_impedance_passivity.png",
) -> Path:
    """Render 4-panel modal participation, SCR root-locus, dq passivity & Nyquist figure."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    fig.suptitle(
        "Small-Signal Modal Participation, Synchronous-dq Admittance Passivity & Generalized Nyquist",
        fontsize=12.3,
        fontweight="bold",
        y=0.985,
    )

    # (a) Dominant Eigenvalue Root Locus in the s-Plane as SCR Sweeps 5.0 -> 1.40
    ax = axes[0, 0]
    ax.axvspan(0.0, 335.0, color="#fee2e2", alpha=0.45, label="Unstable RHP (Re{lambda} > 0)")
    ax.axvline(0.0, color="#dc2626", ls="--", lw=1.4, label="Hopf Boundary Re{lambda} = 0")

    def _dom_mode_track(eig_grid):
        re_arr, f_arr = [], []
        for row in eig_grid:
            osc = [z for z in row if abs(z.imag) > 1.0]
            dom = sorted(osc if osc else row, key=lambda z: (-z.real, -abs(z.imag)))[0]
            re_arr.append(float(dom.real))
            f_arr.append(abs(float(dom.imag)) / (2.0 * np.pi))
        return np.asarray(re_arr), np.asarray(f_arr)

    rv, fv = _dom_mode_track(ssi.locus_vsg_eigs)
    ro, fo = _dom_mode_track(ssi.locus_dvoc_eigs)
    r35, f35 = _dom_mode_track(ssi.locus_gfl35_eigs)
    r55, f55 = _dom_mode_track(ssi.locus_gfl55_eigs)

    ax.plot(rv, fv, "o-", color="#059669", ms=3.8, lw=1.9, label="GFM VSG (SCR 5.0 -> 1.40, Hurwitz)")
    ax.plot(ro, fo, "s--", color="#0284c7", ms=3.5, lw=1.7, label="GFM dVOC (SCR 5.0 -> 1.40, Hurwitz)")
    ax.plot(r35, f35, "^-", color="#d97706", ms=4.0, lw=1.8, label="GFL 35 Hz PLL (Crosses @ SCR = 1.80)")
    ax.plot(r55, f55, "d-.", color="#dc2626", ms=4.0, lw=1.8, label="GFL 55 Hz PLL (Crosses @ SCR = 2.99)")
    ax.set_title("(a) Dominant Eigenvalue Root Locus (SCR: 5.0 -> 1.40)", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Real Part Re{lambda} (s^-1)", fontsize=9.5)
    ax.set_ylabel("Oscillation Frequency |Im{lambda}| / 2pi (Hz)", fontsize=9.5)
    ax.set_xlim(-105.0, 335.0)
    ax.set_ylim(-28.0, 640.0)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7.45, framealpha=0.95)

    # (b) 14-State VSG Subsystem Modal Participation Breakdown
    ax = axes[0, 1]
    modes = ssi.vsg_participation.modes
    y_pos = np.arange(len(modes))
    sync_s = np.array([m.sync_outer_pct for m in modes])
    vpi_s = np.array([m.voltage_pi_pct for m in modes])
    ipi_s = np.array([m.current_pi_pct for m in modes])
    lcl_s = np.array([m.lcl_network_pct for m in modes])
    m_labels = [m.label for m in modes]

    ax.barh(y_pos, sync_s, color="#059669", height=0.66, label="Sync & Power Outer (delta, omega, P_f, Q_f)")
    ax.barh(y_pos, vpi_s, left=sync_s, color="#0284c7", height=0.66, label="Outer Voltage PI (xi_vd, xi_vq)")
    ax.barh(
        y_pos,
        ipi_s,
        left=sync_s + vpi_s,
        color="#d97706",
        height=0.66,
        label="Inner Current PI (xi_id, xi_iq)",
    )
    ax.barh(
        y_pos,
        lcl_s,
        left=sync_s + vpi_s + ipi_s,
        color="#7c3aed",
        height=0.66,
        label="Electromagnetic LCL & Network (i_1, v_c, i_2)",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(m_labels, fontsize=8.0)
    ax.set_xlim(0.0, 104.0)
    ax.set_ylim(-0.6, len(modes) + 1.45)
    ax.set_title("(b) 14-State VSG Subsystem Participation Factors P_ki", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Normalized Modal Participation Share (%)", fontsize=9.5)
    ax.grid(True, axis="x", ls="--", alpha=0.4)
    ax.legend(loc="upper center", ncol=2, fontsize=7.0, framealpha=0.95)

    # (c) Synchronous-dq Input Admittance Passivity Index nu(f) (>= 10 Hz)
    ax = axes[1, 0]
    av = ssi.admittance_vsg
    au = ssi.admittance_vsg_unity_ff
    g35 = ssi.admittance_gfl35
    g55 = ssi.admittance_gfl55
    mask_p = av.freqs_hz >= 10.0

    ax.axhspan(-1.25, 0.0, color="#fee2e2", alpha=0.40, label="Non-Passive Region (nu(f) < 0 S)")
    ax.axhline(0.0, color="#334155", ls="--", lw=1.25)
    ax.semilogx(
        av.freqs_hz[mask_p],
        av.passivity_index_s[mask_p],
        color="#059669",
        lw=2.15,
        label=f"GFM VSG F_i = 0.75 (Passive >=25 Hz, min {av.min_passivity_high_freq_s:+.3f} S)",
    )
    ax.semilogx(
        au.freqs_hz[mask_p],
        au.passivity_index_s[mask_p],
        color="#7c3aed",
        lw=1.8,
        ls="--",
        label=f"GFM VSG Naive F_i = 1.00 (Dip {au.min_passivity_high_freq_s:+.2f} S -> Unstable)",
    )
    ax.semilogx(
        g35.freqs_hz[mask_p],
        g35.passivity_index_s[mask_p],
        color="#d97706",
        lw=1.8,
        ls="-.",
        label=f"GFL 35 Hz PLL (Re{{Y_qq}}<0 to {g35.neg_re_yqq_band_hz[1]:.0f} Hz)",
    )
    ax.semilogx(
        g55.freqs_hz[mask_p],
        g55.passivity_index_s[mask_p],
        color="#dc2626",
        lw=1.85,
        label=f"GFL 55 Hz PLL (Re{{Y_qq}}<0 to {g55.neg_re_yqq_band_hz[1]:.0f} Hz)",
    )
    ax.set_title("(c) Synchronous-dq Admittance Passivity Index nu(f)", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Frequency f (Hz, log scale)", fontsize=9.5)
    ax.set_ylabel("Passivity Index nu(f) = lambda_min(0.5[Y+Y^H]) (S)", fontsize=9.2)
    ax.set_xlim(10.0, 6000.0)
    ax.set_ylim(-1.16, 0.58)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.25, framealpha=0.95)

    # (d) Generalized Nyquist Cauchy Winding Index Z(f) = P - Delta arg det(I + Z_g Y_inv) / pi
    ax = axes[1, 1]
    ad = ssi.admittance_dvoc
    ax.axhline(0.0, color="#059669", ls=":", lw=1.25)
    ax.axhline(2.0, color="#d97706", ls=":", lw=1.1)
    ax.axhline(4.0, color="#dc2626", ls=":", lw=1.1)
    ax.semilogx(
        av.freqs_hz,
        av.cumulative_rhp_index,
        color="#059669",
        lw=2.15,
        label=(
            f"GFM VSG F_i=0.75 (P={av.open_loop_rhp_poles_p}, "
            f"N={av.nyquist_encirclements_n:+d} => Z={av.nyquist_rhp_poles_z})"
        ),
    )
    ax.semilogx(
        ad.freqs_hz,
        ad.cumulative_rhp_index,
        color="#0284c7",
        lw=1.8,
        ls="--",
        label=(
            f"GFM dVOC (P={ad.open_loop_rhp_poles_p}, "
            f"N={ad.nyquist_encirclements_n:+d} => Z={ad.nyquist_rhp_poles_z})"
        ),
    )
    ax.semilogx(
        au.freqs_hz,
        au.cumulative_rhp_index,
        color="#7c3aed",
        lw=1.7,
        ls=":",
        label=(
            f"GFM VSG F_i=1.00 (P={au.open_loop_rhp_poles_p}, "
            f"N={au.nyquist_encirclements_n:+d} => Z={au.nyquist_rhp_poles_z} RHP!)"
        ),
    )
    ax.semilogx(
        g35.freqs_hz,
        g35.cumulative_rhp_index,
        color="#d97706",
        lw=1.85,
        ls="-.",
        label=(
            f"GFL 35 Hz PLL (P={g35.open_loop_rhp_poles_p}, "
            f"N={g35.nyquist_encirclements_n:+d} => Z={g35.nyquist_rhp_poles_z} RHP!)"
        ),
    )
    ax.semilogx(
        g55.freqs_hz,
        g55.cumulative_rhp_index,
        color="#dc2626",
        lw=2.0,
        label=(
            f"GFL 55 Hz PLL (P={g55.open_loop_rhp_poles_p}, "
            f"N={g55.nyquist_encirclements_n:+d} => Z={g55.nyquist_rhp_poles_z} RHP!)"
        ),
    )
    ax.set_title(
        f"(d) Generalized Nyquist Cauchy RHP Pole Index @ SCR = {ssi.scr_operating:.2f}",
        fontsize=10.3,
        fontweight="bold",
    )
    ax.set_xlabel("Frequency f (Hz, log scale)", fontsize=9.5)
    ax.set_ylabel("Cumulative RHP Pole Index P + N(f)", fontsize=9.5)
    ax.set_xlim(0.25, 6000.0)
    ax.set_ylim(-1.25, 5.55)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7.25, framealpha=0.95)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=220, metadata={"Software": "inverter-grid-control-solver"})
    plt.close(fig)
    return out
