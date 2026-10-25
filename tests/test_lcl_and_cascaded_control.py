"""Verification suite for Module 2: Clarke/Park frames, 6-state LCL plant & 14-state cascaded GFM."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.cascaded_control import (
    CascadedPIConfig,
    ElectromagneticGFMModel,
    simulate_electromagnetic_gfm_transient,
)
from src.dq_frames import (
    abc_to_alphabeta,
    alphabeta_to_abc,
    alphabeta_to_dq,
    compute_pq_dq,
    dq_to_alphabeta,
    rotate_dq_frame,
)
from src.lcl_plant import LCLFilterSpec, LCLGridPlant, simulate_lcl_damping_step
from src.thevenin_network import TheveninGridSpec, simulate_coupled_vsg_transient


class TestCoordinateTransforms:
    def test_clarke_park_roundtrip_and_zero_sequence(self):
        theta = 0.73
        va, vb, vc = 310.0, -120.0, -160.0  # includes non-zero homopolar component (+10 V)
        al, be, z0 = abc_to_alphabeta(va, vb, vc)
        assert z0 == pytest.approx(10.0, abs=1e-12)
        d_val, q_val = alphabeta_to_dq(al, be, theta)
        al2, be2 = dq_to_alphabeta(d_val, q_val, theta)
        va2, vb2, vc2 = alphabeta_to_abc(al2, be2, z0)
        assert va2 == pytest.approx(va, abs=1e-12)
        assert vb2 == pytest.approx(vb, abs=1e-12)
        assert vc2 == pytest.approx(vc, abs=1e-12)

    def test_power_invariance_across_abc_alphabeta_and_dq(self):
        v_pk, i_pk = 325.27, 38.0
        wt = 1.18
        phi = 0.42
        va = v_pk * math.cos(wt)
        vb = v_pk * math.cos(wt - 2.0 * math.pi / 3.0)
        vc = v_pk * math.cos(wt + 2.0 * math.pi / 3.0)
        ia = i_pk * math.cos(wt - phi)
        ib = i_pk * math.cos(wt - phi - 2.0 * math.pi / 3.0)
        ic = i_pk * math.cos(wt - phi + 2.0 * math.pi / 3.0)

        p_abc = va * ia + vb * ib + vc * ic
        val, vbe, _ = abc_to_alphabeta(va, vb, vc)
        ial, ibe, _ = abc_to_alphabeta(ia, ib, ic)
        p_ab = 1.5 * (val * ial + vbe * ibe)
        q_ab = 1.5 * (vbe * ial - val * ibe)

        vd, vq = alphabeta_to_dq(val, vbe, wt + 0.25)
        id_, iq_ = alphabeta_to_dq(ial, ibe, wt + 0.25)
        p_dq, q_dq = compute_pq_dq(vd, vq, id_, iq_)

        expected_p = 1.5 * v_pk * i_pk * math.cos(phi)
        expected_q = 1.5 * v_pk * i_pk * math.sin(phi)
        assert p_abc == pytest.approx(expected_p, rel=1e-12)
        assert p_ab == pytest.approx(expected_p, rel=1e-12)
        assert p_dq == pytest.approx(expected_p, rel=1e-12)
        assert q_ab == pytest.approx(expected_q, rel=1e-12)
        assert q_dq == pytest.approx(expected_q, rel=1e-12)

    def test_rotate_dq_frame_composition_and_nonfinite_rejection(self):
        d1, q1 = rotate_dq_frame(300.0, -25.0, 0.3)
        d2, q2 = rotate_dq_frame(d1, q1, -0.3)
        assert d2 == pytest.approx(300.0, abs=1e-12)
        assert q2 == pytest.approx(-25.0, abs=1e-12)
        with pytest.raises(ValueError):
            abc_to_alphabeta(float("nan"), 0.0, 0.0)


class TestLCLFilterElectromagnetics:
    def test_resonance_frequencies_and_parameter_validation(self):
        lcl = LCLFilterSpec()
        grid = TheveninGridSpec()
        # Stiff resonance > grid-coupled resonance because L_2 + L_g > L_2
        assert 1600.0 < lcl.stiff_resonance_hz < 1650.0
        assert 1000.0 < lcl.coupled_resonance_hz(grid.l_g_h) < lcl.stiff_resonance_hz
        with pytest.raises(ValueError):
            LCLFilterSpec(cf_f=-1e-6)
        with pytest.raises(ValueError):
            lcl.coupled_resonance_hz(-0.01)

    def test_exact_electromagnetic_energy_conservation_passivity_identity(self):
        """Verify dE_stored/dt + P_ohmic == P_inv - P_grid (rotational w*J terms do zero work)."""
        plant = LCLGridPlant()
        x = np.array([28.5, -11.2, 318.0, -14.5, 26.8, -12.9], dtype=float)
        vinv_d, vinv_q = 342.0, 19.5
        vg_d, vg_q = 321.0, -24.0
        w = 2.0 * math.pi * 50.3

        dx = plant.rhs(x, vinv_d, vinv_q, vg_d, vg_q, w)
        # Analytical dE/dt = 1.5 * (L1*i1.di1 + Cf*vc.dvc + Lnet*i2.di2)
        de_dt = 1.5 * (
            plant.lcl.l1_h * (x[0] * dx[0] + x[1] * dx[1])
            + plant.lcl.cf_f * (x[2] * dx[2] + x[3] * dx[3])
            + plant.l_net_h * (x[4] * dx[4] + x[5] * dx[5])
        )
        p_loss = plant.ohmic_dissipation_w(x)
        p_inv, _ = compute_pq_dq(vinv_d, vinv_q, x[0], x[1])
        p_grid, _ = compute_pq_dq(vg_d, vg_q, x[4], x[5])
        assert de_dt + p_loss == pytest.approx(p_inv - p_grid, rel=1e-12)

    def test_passive_damping_suppresses_lcl_step_ringing(self):
        _, v_undamped = simulate_lcl_damping_step(rd_ohm=0.02)
        _, v_damped = simulate_lcl_damping_step(rd_ohm=1.80)
        pk2pk_undamped = float(np.max(v_undamped[250:]) - np.min(v_undamped[250:]))
        pk2pk_damped = float(np.max(v_damped[250:]) - np.min(v_damped[250:]))
        assert pk2pk_damped < 0.70 * pk2pk_undamped


class TestCascadedDualLoopGFM:
    def test_14_state_equilibrium_and_qoria_feedforward_stability(self):
        grid = TheveninGridSpec()
        lcl = LCLFilterSpec()
        pi_stable = CascadedPIConfig.from_bandwidths(lcl, feedforward_current_gain=0.75)
        model_stable = ElectromagneticGFMModel(grid=grid, lcl=lcl, pi_cfg=pi_stable)

        y0 = model_stable.steady_initial_state()
        f0 = model_stable.rhs_continuous(y0, 0.0, 0.0, grid.v_peak_nom_v, grid.w0_rad_s)
        assert float(np.linalg.norm(f0)) < 1e-11

        eigs_stable = model_stable.linearized_eigenvalues(y0)
        assert float(np.max(eigs_stable.real)) < -3.0

        # Unity current feedforward (F_i = 1.0) cancels grid damping and destabilizes EM flux mode
        pi_unstable = CascadedPIConfig.from_bandwidths(lcl, feedforward_current_gain=1.0)
        model_unstable = ElectromagneticGFMModel(grid=grid, lcl=lcl, pi_cfg=pi_unstable)
        eigs_unstable = model_unstable.linearized_eigenvalues()
        assert float(np.max(eigs_unstable.real)) > 10.0

    def test_14_state_rk4_matches_scipy_solve_ivp_rk45(self):
        grid = TheveninGridSpec()
        lcl = LCLFilterSpec()
        model = ElectromagneticGFMModel(grid=grid, lcl=lcl)
        res = simulate_electromagnetic_gfm_transient(
            grid=grid,
            lcl=lcl,
            duration_s=0.35,
            dt_s=50.0e-6,
            p_step_time_s=0.0,
            p_step_w=8_000.0,
            grid_freq_drop_time_s=0.0,
            grid_freq_drop_hz=-0.10,
        )

        y0 = model.steady_initial_state()
        w_g = 2.0 * math.pi * 49.90
        sol = solve_ivp(
            fun=lambda _t, y: model.rhs_continuous(y, 8_000.0, 0.0, grid.v_peak_nom_v, w_g),
            t_span=(0.0, 0.35),
            y0=y0,
            t_eval=res.time_s[::20],
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        assert sol.success
        np.testing.assert_allclose(np.degrees(sol.y[0]), res.delta_deg[::20], atol=2e-4)
        np.testing.assert_allclose(sol.y[1] / (2.0 * math.pi), res.f_vsg_hz[::20], atol=2e-5)
        np.testing.assert_allclose(sol.y[8], res.i1_d_a[::20], atol=2e-3)
        np.testing.assert_allclose(sol.y[12], res.i2_d_a[::20], atol=2e-3)

    def test_macro_timescale_agreement_with_4_state_thevenin_model(self):
        grid = TheveninGridSpec()
        em = simulate_electromagnetic_gfm_transient(grid=grid, duration_s=2.5, dt_s=80.0e-6)
        qs = simulate_coupled_vsg_transient(spec=grid, duration_s=2.5, dt_s=1.0e-3)

        assert em.nadir_freq_hz == pytest.approx(qs.nadir_freq_hz, abs=5e-3)
        assert em.f_vsg_hz[-1] == pytest.approx(qs.f_vsg_hz[-1], abs=2e-3)
        assert em.final_power_kw == pytest.approx(qs.final_power_kw, rel=5e-3)

    def test_radial_current_saturation_and_anti_windup_under_overload(self):
        grid = TheveninGridSpec(i_max_peak_a=22.0)
        res = simulate_electromagnetic_gfm_transient(
            grid=grid,
            duration_s=0.45,
            dt_s=80.0e-6,
            p_step_time_s=0.02,
            p_step_w=16_000.0,
        )
        assert np.any(res.current_limited)
        i1_ref_mag = np.hypot(res.i1_d_ref_a, res.i1_q_ref_a)
        assert float(np.max(i1_ref_mag)) <= 22.0 + 1e-9
        assert np.all(np.isfinite(res.p_pcc_w))
