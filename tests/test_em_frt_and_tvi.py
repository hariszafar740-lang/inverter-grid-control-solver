"""Tests for Module 5 Threshold TVI, Power Anti-Windup, 14-State EM FRT & Phase-Jump Benchmark."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.em_frt_benchmark import (
    ElectromagneticFRTVSGModel,
    run_electromagnetic_frt_benchmark,
    simulate_em_fault_sag,
    simulate_em_phase_jump,
)
from src.thevenin_network import TheveninGridSpec
from src.transient_virtual_impedance import TransientVirtualImpedance, TVIConfig
from src.visualizer import plot_electromagnetic_frt_benchmark


class TestTransientVirtualImpedanceAndPAW:
    def test_tvi_config_validation_and_analytical_bolted_fault_sizing(self):
        with pytest.raises(ValueError):
            TVIConfig(i_thresh_peak_a=-5.0)
        with pytest.raises(ValueError):
            TVIConfig(i_thresh_peak_a=50.0, i_max_peak_a=48.0)
        with pytest.raises(ValueError):
            TVIConfig(paw_voltage_knee_pu=1.2)
        with pytest.raises(ValueError):
            TVIConfig.from_overcurrent_envelope(bolted_fault_margin=0.9)

        grid = TheveninGridSpec(i_max_peak_a=48.0)
        cfg = TVIConfig.from_overcurrent_envelope(
            grid,
            i_thresh_peak_a=39.0,
            sigma_xr=5.0,
            bolted_fault_margin=1.08,
        )
        tvi = TransientVirtualImpedance(cfg)
        dr, dx, active = tvi.compute_incremental_zv(48.0, grid.w0_rad_s)
        assert active is True
        assert dx / dr == pytest.approx(5.0, rel=1e-12)
        z_tot = math.hypot(grid.r_v_ohm + dr, grid.w0_rad_s * grid.l_v_h + dx)
        assert z_tot >= (grid.v_peak_nom_v / 48.0) * 1.07

    def test_tvi_inactive_below_threshold_and_smooth_above_threshold(self):
        tvi = TransientVirtualImpedance(TVIConfig(i_thresh_peak_a=39.0, i_max_peak_a=48.0))
        dr0, dx0, act0 = tvi.compute_incremental_zv(38.9)
        assert (dr0, dx0, act0) == (0.0, 0.0, False)

        vfd, vfq, z_mag, act1 = tvi.apply_dq_voltage_drop(
            e_star_v=325.27,
            i_d_a=42.0,
            i_q_a=-5.0,
            r_v_base_ohm=0.12,
            l_v_base_h=2.4e-3,
            w_inst_rad_s=100.0 * math.pi,
        )
        assert act1 is True and z_mag > 0.0
        assert vfd < 325.27 and vfq < 0.0

    def test_synchronizing_power_anti_windup_derating(self):
        tvi = TransientVirtualImpedance(
            TVIConfig(
                i_thresh_peak_a=39.0,
                i_max_peak_a=48.0,
                paw_voltage_knee_pu=0.88,
                paw_voltage_exponent=1.5,
                paw_current_gain_w_per_a=350.0,
            )
        )
        assert tvi.compute_anti_windup_power_ref(16_000.0, 0.95, 34.0) == pytest.approx(16_000.0)
        p_sag = tvi.compute_anti_windup_power_ref(16_000.0, 0.20, 45.0)
        expected = 16_000.0 * (0.20**1.5) - 350.0 * (45.0 - 39.0)
        assert p_sag == pytest.approx(expected, rel=1e-12)


class TestElectromagneticFRTAndPhaseJump:
    def test_pre_event_loaded_equilibrium_exactness(self):
        model = ElectromagneticFRTVSGModel(mode="tvi_and_paw")
        y_eq = model.solve_pre_event_equilibrium(17_000.0)
        res = model.rhs_continuous(y_eq, 17_000.0, 0.0, model.grid.v_peak_nom_v, 0.0, model.grid.w0_rad_s)
        assert float(np.linalg.norm(res)) < 1.0e-7

    def test_rk4_matches_scipy_solve_ivp_during_nonlinear_tvi_fault_segment(self):
        model = ElectromagneticFRTVSGModel(mode="tvi_and_paw")
        tr = simulate_em_fault_sag(
            "tvi_and_paw",
            p_dispatch_w=17_000.0,
            sag_residual_pu=0.15,
            t_fault_s=0.0,
            t_clear_s=0.25,
            duration_s=0.25,
            dt_s=50.0e-6,
        )
        y0 = model.solve_pre_event_equilibrium(17_000.0)
        vg_sag = 0.15 * model.grid.v_peak_nom_v
        w0 = model.grid.w0_rad_s
        sol = solve_ivp(
            fun=lambda _t, y: model.rhs_continuous(y, 17_000.0, 0.0, vg_sag, 0.0, w0),
            t_span=(0.0, 0.25),
            y0=y0,
            t_eval=tr.time_s,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        assert sol.success
        np.testing.assert_allclose(np.degrees(sol.y[0]), tr.delta_rel_deg, atol=5.0e-3)

    def test_deep_sag_unprotected_overcurrent_and_hard_clamp_windup_los(self):
        unprot = simulate_em_fault_sag("unprotected", duration_s=1.10)
        clamp = simulate_em_fault_sag("hard_clamp_only", duration_s=1.10)
        assert unprot.peak_bridge_current_a > 130.0
        assert unprot.rode_through is False
        assert clamp.max_abs_delta_deg > 240.0
        assert clamp.saturation_duty_pct > 65.0
        assert clamp.rode_through is False

    def test_tvi_only_caps_current_but_paw_eliminates_cycle_slip(self):
        tvi_only = simulate_em_fault_sag("tvi_only")
        tvi_paw = simulate_em_fault_sag("tvi_and_paw")

        assert tvi_only.peak_bridge_current_a <= 48.1
        assert tvi_only.max_abs_delta_deg > 300.0
        assert tvi_only.rode_through is False

        assert tvi_paw.peak_bridge_current_a <= 48.0
        assert tvi_paw.max_abs_delta_deg < 35.0
        assert tvi_paw.saturation_duty_pct < 2.0
        assert tvi_paw.fault_mean_q_kvar > 6.0
        assert tvi_paw.rode_through is True

    def test_severe_phase_jump_qoria_circular_clamp_latchup_vs_tvi_ridethrough(self):
        unprot = simulate_em_phase_jump("unprotected")
        clamp = simulate_em_phase_jump("hard_clamp_only")
        tvi_paw = simulate_em_phase_jump("tvi_and_paw")

        assert unprot.peak_bridge_current_a > 120.0
        assert unprot.rode_through is False

        assert clamp.max_abs_delta_deg > 200.0
        assert clamp.saturation_duty_pct > 65.0
        assert clamp.rode_through is False

        assert tvi_paw.peak_bridge_current_a <= 48.0
        assert tvi_paw.saturation_duty_pct == pytest.approx(0.0, abs=0.5)
        assert tvi_paw.max_abs_delta_deg < 65.0
        assert tvi_paw.rode_through is True

    def test_electromagnetic_frt_benchmark_bundle_and_figure_generation(self, tmp_path):
        bench = run_electromagnetic_frt_benchmark()
        out_png = plot_electromagnetic_frt_benchmark(bench, save_path=tmp_path / "em_frt.png")
        assert out_png.is_file() and out_png.stat().st_size > 50_000
        assert bench.sag_tvi_paw.rode_through is True
        assert bench.jump_tvi_paw.rode_through is True
        assert bench.sag_hard_clamp.rode_through is False
        assert bench.jump_hard_clamp.rode_through is False

    def test_em_frt_parameter_validation_guards(self):
        with pytest.raises(ValueError):
            ElectromagneticFRTVSGModel(mode="invalid_mode")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            simulate_em_fault_sag("tvi_and_paw", dt_s=1.0e-3)
        with pytest.raises(ValueError):
            simulate_em_fault_sag("tvi_and_paw", t_fault_s=0.5, t_clear_s=0.3)
        with pytest.raises(ValueError):
            simulate_em_fault_sag("tvi_and_paw", sag_residual_pu=1.4)
        with pytest.raises(ValueError):
            simulate_em_phase_jump("tvi_and_paw", t_jump_s=2.0, duration_s=1.0)
