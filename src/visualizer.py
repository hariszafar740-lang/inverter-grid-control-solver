"""Deterministic Matplotlib visualization routines for Grid-Forming Inverter benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cascaded_control import ElectromagneticTransientResult
from .gfm_triad import GFMTriadBenchmarkResult
from .lcl_plant import simulate_lcl_damping_step
from .thevenin_network import ClosedLoopTransientResult


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
