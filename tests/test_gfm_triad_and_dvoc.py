"""Verification suite for Module 3: Andronov-Hopf dVOC & Comparative GFM Control Triad."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.cascaded_control import ElectromagneticGFMModel
from src.dvoc_control import DispatchableVirtualOscillator, DVOCConfig
from src.gfm_triad import (
    MultiLoopDroopGFMModel,
    MultiLoopDVOCGFMModel,
    run_gfm_triad_benchmark,
    simulate_gfm_mode_transient,
)
from src.lcl_plant import LCLFilterSpec
from src.thevenin_network import TheveninGridSpec
from src.visualizer import plot_gfm_triad_benchmark


class TestAndronovHopfDVOC:
    def test_dvoc_config_grid_matching_and_validation(self):
        grid = TheveninGridSpec()
        cfg = DVOCConfig.from_grid_spec(grid)
        expected_mp = 1.0 / (grid.dp_nms_rad * grid.w0_rad_s)
        assert cfg.mp_eq_rad_per_ws == pytest.approx(expected_mp, rel=1e-12)
        assert cfg.nq_eq_v_per_var == pytest.approx(grid.nq_v_per_var, rel=1e-12)
        assert cfg.eta_v2_per_ws == pytest.approx(16.838593, rel=1e-5)
        assert cfg.alpha_osc_w_per_v2 == pytest.approx(1.280991, rel=1e-5)

        with pytest.raises(ValueError):
            DVOCConfig(eta_v2_per_ws=-1.0)
        with pytest.raises(ValueError):
            DVOCConfig(rotation_angle_rad=4.0)

    def test_dvoc_cartesian_and_polar_exact_equivalence(self):
        for phi in (0.5 * math.pi, 0.38 * math.pi):
            cfg = DVOCConfig.from_grid_spec(rotation_angle_rad=phi)
            osc = DispatchableVirtualOscillator(cfg)

            v_hat = 312.4
            theta = 0.67
            va = v_hat * math.cos(theta)
            vb = v_hat * math.sin(theta)
            ia, ib = 24.5, -13.2
            p_ref, q_ref = 9_500.0, 2_200.0

            dva, dvb = osc.cartesian_rhs(va, vb, ia, ib, p_ref, q_ref)
            dv_cart = (va * dva + vb * dvb) / v_hat
            omega_cart = (va * dvb - vb * dva) / (v_hat * v_hat)

            p_elec = 1.5 * (va * ia + vb * ib)
            q_elec = 1.5 * (vb * ia - va * ib)
            omega_pol, dv_pol = osc.polar_kinetics(v_hat, p_elec, q_elec, p_ref, q_ref)

            assert omega_pol == pytest.approx(omega_cart, rel=1e-12, abs=1e-12)
            assert dv_pol == pytest.approx(dv_cart, rel=1e-12, abs=1e-12)

    def test_dvoc_black_start_matches_bernoulli_closed_form(self):
        osc = DispatchableVirtualOscillator()
        bs = osc.simulate_black_start_excitation(v_init_pu=0.10, duration_s=0.35, dt_s=40.0e-6)
        np.testing.assert_allclose(bs.v_hat_v, bs.v_hat_analytical_v, rtol=1e-7, atol=1e-7)
        assert bs.v_hat_v[-1] / osc.config.v0_peak_v == pytest.approx(1.0, abs=1e-4)
        assert osc.analytical_unloaded_amplitude(0.20, 0.0) == 0.0
        assert float(osc.analytical_unloaded_amplitude(0.0, 50.0)) == pytest.approx(50.0, rel=1e-12)


class TestComparativeGFMTriad:
    def test_droop_13_state_and_dvoc_12_state_zero_power_equilibria(self):
        grid = TheveninGridSpec()
        droop_model = MultiLoopDroopGFMModel(grid=grid)
        dvoc_model = MultiLoopDVOCGFMModel(grid=grid)
        assert droop_model.STATE_DIM == 13
        assert dvoc_model.STATE_DIM == 12

        y0_droop = droop_model.steady_initial_state()
        f0_droop = droop_model.rhs_continuous(y0_droop, 0.0, 0.0, grid.v_peak_nom_v, grid.w0_rad_s)
        assert float(np.linalg.norm(f0_droop)) < 1e-11

        y0_dvoc = dvoc_model.steady_initial_state()
        f0_dvoc = dvoc_model.rhs_continuous(y0_dvoc, 0.0, 0.0, grid.v_peak_nom_v, grid.w0_rad_s)
        assert float(np.linalg.norm(f0_dvoc)) < 1e-11

    def test_triad_hurwitz_eigenvalues_strong_and_weak_grid(self):
        for scr in (6.0, 2.0):
            g = TheveninGridSpec(scr=scr)
            ev_droop = MultiLoopDroopGFMModel(grid=g).linearized_eigenvalues()
            ev_vsg = ElectromagneticGFMModel(grid=g).linearized_eigenvalues()
            ev_dvoc = MultiLoopDVOCGFMModel(grid=g).linearized_eigenvalues()
            assert float(np.max(ev_droop.real)) < -2.5
            assert float(np.max(ev_vsg.real)) < -2.5
            assert float(np.max(ev_dvoc.real)) < -2.5

        ev_dvoc_nom = MultiLoopDVOCGFMModel(grid=TheveninGridSpec(scr=6.0)).linearized_eigenvalues()
        real_modes = ev_dvoc_nom[np.abs(ev_dvoc_nom.imag) < 1e-6].real
        assert np.any(np.isclose(real_modes, -10.6537, atol=5e-2))

    def test_droop_and_dvoc_rk4_match_scipy_solve_ivp_rk45(self):
        grid = TheveninGridSpec()
        lcl = LCLFilterSpec()
        w_g = 2.0 * math.pi * 49.90

        for mode, model in (
            ("droop", MultiLoopDroopGFMModel(grid=grid, lcl=lcl)),
            ("dvoc", MultiLoopDVOCGFMModel(grid=grid, lcl=lcl)),
        ):
            res = simulate_gfm_mode_transient(
                mode,
                grid=grid,
                lcl=lcl,
                duration_s=0.30,
                dt_s=50.0e-6,
                p_step_time_s=0.0,
                p_step_w=8_000.0,
                grid_freq_drop_time_s=0.0,
                grid_freq_drop_hz=-0.10,
            )
            y0 = model.steady_initial_state()
            sol = solve_ivp(
                fun=lambda _t, y, m=model: m.rhs_continuous(y, 8_000.0, 0.0, grid.v_peak_nom_v, w_g),
                t_span=(0.0, 0.30),
                y0=y0,
                t_eval=res.time_s[::20],
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
            )
            assert sol.success
            np.testing.assert_allclose(np.degrees(sol.y[0]), res.delta_deg[::20], atol=3e-4)

    def test_dvoc_steady_state_pcc_power_exact_ohmic_identity(self):
        grid = TheveninGridSpec()
        lcl = LCLFilterSpec()
        res = simulate_gfm_mode_transient("dvoc", grid=grid, lcl=lcl, duration_s=2.5, dt_s=80.0e-6)

        dw = 2.0 * math.pi * 0.20
        p_droop_nom = 10_000.0 + grid.dp_nms_rad * grid.w0_rad_s * dw
        vh_pu_end = float(res.e_int_pu[-1])
        i2_pk_end = float(res.i2_peak_a[-1])
        expected_p_pcc_w = (vh_pu_end * vh_pu_end) * p_droop_nom - 1.5 * (grid.r_v_ohm + lcl.r2_ohm) * (i2_pk_end**2)
        assert float(res.p_pcc_w[-1]) == pytest.approx(expected_p_pcc_w, rel=2e-3)

    def test_comparative_triad_physical_tradeoffs_and_ordering(self):
        triad = run_gfm_triad_benchmark()
        assert triad.droop.step_rise_time_90_ms < triad.vsg.step_rise_time_90_ms < triad.dvoc.step_rise_time_90_ms
        assert (
            triad.vsg.freq_drop_peak_power_kw
            > triad.droop.freq_drop_peak_power_kw
            > triad.dvoc.freq_drop_peak_power_kw
        )
        assert triad.vsg.nadir_freq_hz < triad.droop.nadir_freq_hz < triad.dvoc.nadir_freq_hz

    def test_evaluate_algebraic_public_inspection_dicts(self):
        grid = TheveninGridSpec()
        for model in (
            ElectromagneticGFMModel(grid=grid),
            MultiLoopDroopGFMModel(grid=grid),
            MultiLoopDVOCGFMModel(grid=grid),
        ):
            y0 = model.steady_initial_state()
            info = model.evaluate_algebraic(y0, 0.0, 0.0, grid.v_peak_nom_v)
            assert info["p_pcc"] == pytest.approx(0.0, abs=1e-9)
            assert info["v_pcc_mag"] == pytest.approx(grid.v_peak_nom_v, rel=1e-9)
            assert info["cur_lim"] is False
            assert info["v_lim"] is False

        with pytest.raises(ValueError):
            simulate_gfm_mode_transient("invalid_mode")  # type: ignore[arg-type]

    def test_plot_gfm_triad_benchmark_renders_warning_free(self, tmp_path):
        triad = run_gfm_triad_benchmark(duration_s=1.4, dt_s=1.0e-4)
        out = tmp_path / "triad_test.png"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            saved = plot_gfm_triad_benchmark(triad, save_path=out)
        assert len(caught) == 0
        assert saved.is_file() and saved.stat().st_size > 60_000
