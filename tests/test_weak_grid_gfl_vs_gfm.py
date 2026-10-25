"""Tests for Module 4 SRF-PLL Grid-Following model and Weak-Grid SCR Collapse Benchmark."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.gfl_pll import (
    ElectromagneticGFLModel,
    SRFPLLConfig,
    SynchronousReferenceFramePLL,
)
from src.thevenin_network import TheveninGridSpec
from src.visualizer import plot_weak_grid_scr_benchmark
from src.weak_grid_benchmark import (
    estimate_critical_scr,
    run_weak_grid_scr_benchmark,
    simulate_weak_grid_line_trip,
    sweep_scr_stability,
)


class TestSRFPLLUnitPhysics:
    def test_srf_pll_config_validation_and_stiff_grid_poles(self):
        with pytest.raises(ValueError):
            SRFPLLConfig(f_bw_hz=-10.0)
        with pytest.raises(ValueError):
            SRFPLLConfig(zeta=0.0)
        with pytest.raises(ValueError):
            SRFPLLConfig(f_min_hz=52.0, f_max_hz=65.0)
        with pytest.raises(ValueError):
            SRFPLLConfig(vd_filter_hz=float("nan"))

        cfg = SRFPLLConfig(f_bw_hz=35.0, zeta=0.7071067811865476)
        pll = SynchronousReferenceFramePLL(cfg)
        p1, p2 = pll.small_signal_stiff_poles()
        wn = 2.0 * math.pi * 35.0
        expected_re = -cfg.zeta * wn
        expected_im = wn * math.sqrt(1.0 - cfg.zeta * cfg.zeta)
        assert p1.real == pytest.approx(expected_re, rel=1e-12)
        assert p2.real == pytest.approx(expected_re, rel=1e-12)
        assert abs(p1.imag) == pytest.approx(expected_im, rel=1e-12)

    def test_srf_pll_phase_detector_and_conditional_anti_windup(self):
        cfg = SRFPLLConfig(f_bw_hz=35.0, f_min_hz=45.0, f_max_hz=55.0)
        pll = SynchronousReferenceFramePLL(cfg)
        theta = 0.62
        v_peak = cfg.v0_peak_v
        vd, vq, vq_pu = pll.phase_detector_dq(
            v_peak * math.cos(theta),
            v_peak * math.sin(theta),
            theta,
        )
        assert vd == pytest.approx(v_peak, rel=1e-12)
        assert vq == pytest.approx(0.0, abs=1e-12)
        assert vq_pu == pytest.approx(0.0, abs=1e-12)

        w_sat, d_xi_sat, clamped = pll.step_kinetics(vq_v=0.5 * v_peak, xi_pll_s=0.25)
        assert clamped is True
        assert w_sat == pytest.approx(cfg.w_max_rad_s, rel=1e-12)
        assert d_xi_sat == 0.0

        w_rel, d_xi_rel, clamped_rel = pll.step_kinetics(vq_v=-0.05 * v_peak, xi_pll_s=0.25)
        assert clamped_rel is True
        assert w_rel == pytest.approx(cfg.w_max_rad_s, rel=1e-12)
        assert d_xi_rel == pytest.approx(-0.05, rel=1e-12)


class TestElectromagneticGFLModel:
    def test_gfl_steady_initial_state_is_exact_equilibrium(self):
        for scr, k_qv in ((6.0, 0.0), (1.6, 0.0), (4.0, 0.08)):
            grid = TheveninGridSpec(scr=scr)
            model = ElectromagneticGFLModel(grid=grid, k_qv_a_per_v=k_qv)
            y0 = model.steady_initial_state()
            rhs0 = model.rhs_continuous(y0, 0.0, 0.0, grid.v_peak_nom_v, grid.w0_rad_s)
            assert float(np.linalg.norm(rhs0)) < 1.0e-10

    def test_gfl_loaded_equilibrium_power_balance_and_rk45_agreement(self):
        grid_pre = TheveninGridSpec(scr=4.5)
        model_pre = ElectromagneticGFLModel(grid=grid_pre)
        y_eq, ok = model_pre.solve_loaded_equilibrium(10_000.0, 0.0)
        assert ok and y_eq is not None
        rhs_eq = model_pre.rhs_continuous(y_eq, 10_000.0, 0.0, grid_pre.v_peak_nom_v, grid_pre.w0_rad_s)
        assert float(np.linalg.norm(rhs_eq)) < 1.0e-9
        info = model_pre.evaluate_algebraic(y_eq, 10_000.0, 0.0, grid_pre.v_peak_nom_v)
        assert 9_650.0 < float(info["p_pcc"]) < 10_050.0

        trip = simulate_weak_grid_line_trip(
            "gfl_med_pll",
            scr_pre=4.5,
            scr_post=2.8,
            p_dispatch_w=10_000.0,
            duration_s=0.25,
            dt_s=50.0e-6,
            trip_time_s=0.0,
        )
        grid_post = TheveninGridSpec(scr=2.8)
        model_post = ElectromagneticGFLModel(
            grid=grid_post,
            pll_cfg=SRFPLLConfig.from_grid_spec(grid_post, f_bw_hz=35.0),
        )
        sol = solve_ivp(
            fun=lambda _t, y: model_post.rhs_continuous(
                y, 10_000.0, 0.0, grid_post.v_peak_nom_v, grid_post.w0_rad_s
            ),
            t_span=(0.0, 0.25),
            y0=y_eq,
            t_eval=trip.time_s,
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
        )
        assert sol.success
        np.testing.assert_allclose(np.degrees(sol.y[0]), trip.delta_deg, atol=2.0e-3)


class TestWeakGridSCRCollapseVsGFM:
    def test_high_pll_bandwidth_triggers_weak_grid_hopf_instability_monotonicity(self):
        fast = sweep_scr_stability("gfl_fast_pll")
        med = sweep_scr_stability("gfl_med_pll")
        slow = sweep_scr_stability("gfl_slow_pll")
        c_fast, s_fast = estimate_critical_scr(fast)
        c_med, s_med = estimate_critical_scr(med)
        c_slow, s_slow = estimate_critical_scr(slow)

        assert c_fast > 2.6
        assert c_fast > c_med > c_slow > s_slow
        assert s_fast == pytest.approx(s_med, abs=0.08)
        assert s_med == pytest.approx(s_slow, abs=0.08)

    def test_gfl_static_nose_curve_collapse_below_critical_scr(self):
        grid_weak = TheveninGridSpec(scr=1.22)
        for f_bw in (55.0, 35.0, 10.0):
            model = ElectromagneticGFLModel(
                grid=grid_weak,
                pll_cfg=SRFPLLConfig.from_grid_spec(grid_weak, f_bw_hz=f_bw),
            )
            y_eq, ok = model.solve_loaded_equilibrium(15_000.0, 0.0)
            assert not ok and y_eq is None

    def test_all_three_gfm_laws_remain_hurwitz_down_to_ultra_weak_scr_1_15(self):
        for arch in ("gfm_droop", "gfm_vsg", "gfm_dvoc"):
            pts = sweep_scr_stability(arch, scr_values=(6.0, 2.0, 1.30, 1.15), p_dispatch_w=15_000.0)
            p_end = pts[-1]
            assert p_end.scr == pytest.approx(1.15)
            assert p_end.equilibrium_exists is True
            assert p_end.is_small_signal_stable is True
            assert p_end.max_real_eig_s_inv < -2.0
            assert p_end.v_pcc_eq_pu > 0.94
            assert 30.0 < p_end.delta_eq_deg < 52.0

    def test_line_opening_contingency_gfl_pole_slipping_vs_gfm_ride_through(self):
        for gfl_arch in ("gfl_med_pll", "gfl_slow_pll"):
            tr_gfl = simulate_weak_grid_line_trip(gfl_arch, duration_s=1.0, trip_time_s=0.20)
            assert tr_gfl.lost_synchronism is True
            assert tr_gfl.max_abs_delta_deg > 1_000.0
            assert tr_gfl.min_v_pcc_pu < 0.15

        for gfm_arch in ("gfm_droop", "gfm_vsg", "gfm_dvoc"):
            tr_gfm = simulate_weak_grid_line_trip(gfm_arch, duration_s=1.0, trip_time_s=0.20)
            assert tr_gfm.lost_synchronism is False
            assert tr_gfm.max_abs_delta_deg < 50.0
            assert tr_gfm.min_v_pcc_pu > 0.90
            assert 13.0 < tr_gfm.final_power_kw < 15.5

    def test_weak_grid_benchmark_bundle_and_figure_generation(self, tmp_path):
        bench = run_weak_grid_scr_benchmark(duration_s=0.90, trip_time_s=0.20)
        out_png = plot_weak_grid_scr_benchmark(bench, save_path=tmp_path / "wg_collapse.png")
        assert out_png.is_file() and out_png.stat().st_size > 50_000
        assert bench.scr_crit_gfl_fast > bench.scr_crit_gfl_med > bench.scr_crit_gfl_slow
        assert bench.trip_gfl_med.lost_synchronism is True
        assert bench.trip_gfm_vsg.lost_synchronism is False
        assert bench.trip_gfm_dvoc.lost_synchronism is False

    def test_weak_grid_parameter_validation_guards(self):
        with pytest.raises(ValueError):
            ElectromagneticGFLModel(k_qv_a_per_v=-0.1)
        with pytest.raises(ValueError):
            sweep_scr_stability("gfl_med_pll", p_dispatch_w=-100.0)
        with pytest.raises(ValueError):
            sweep_scr_stability("gfl_med_pll", scr_values=())
        with pytest.raises(ValueError):
            estimate_critical_scr(())
        with pytest.raises(ValueError):
            simulate_weak_grid_line_trip("gfl_med_pll", dt_s=1.0e-3)
        with pytest.raises(ValueError):
            simulate_weak_grid_line_trip("gfl_med_pll", duration_s=0.5, trip_time_s=0.6)
