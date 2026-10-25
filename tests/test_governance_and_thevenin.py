"""Tests for Module 1 governance files, coupled Thévenin GFM dynamics, and SciPy ODE agreement."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import main as main_entry
import src
from src.droop_control import InverterDroopController
from src.thevenin_network import (
    CoupledVSGTheveninModel,
    TheveninGridSpec,
    simulate_coupled_vsg_transient,
)
from src.virtual_impedance import VirtualImpedanceController

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestGovernanceAndMetadata:
    def test_community_standards_and_config_files_exist(self):
        required = (
            "LICENSE",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "SECURITY.md",
            "ruff.toml",
            "pytest.ini",
            ".github/workflows/ci.yml",
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".github/ISSUE_TEMPLATE/config.yml",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/ISSUE_TEMPLATE/feature_request.yml",
            "results/inverter_transient_response.png",
            "results/lcl_cascaded_electromagnetic_response.png",
            "results/gfm_triad_comparative_response.png",
            "results/gfl_vs_gfm_weak_grid_collapse.png",
            "results/em_frt_tvi_phase_jump_response.png",
            "results/small_signal_eigenvalue_impedance_passivity.png",
            "pyproject.toml",
            "src/py.typed",
        )
        for rel in required:
            p = REPO_ROOT / rel
            assert p.is_file(), f"Missing required repository file: {rel}"
            assert p.stat().st_size > 20, f"Unexpectedly empty file: {rel}"

    def test_package_version_and_public_exports(self):
        assert src.__version__ == "1.0.0"
        for sym in (
            "CoupledVSGTheveninModel",
            "TheveninGridSpec",
            "simulate_coupled_vsg_transient",
            "DVOCConfig",
            "DispatchableVirtualOscillator",
            "MultiLoopDroopGFMModel",
            "MultiLoopDVOCGFMModel",
            "run_gfm_triad_benchmark",
            "SRFPLLConfig",
            "ElectromagneticGFLModel",
            "run_weak_grid_scr_benchmark",
            "TVIConfig",
            "TransientVirtualImpedance",
            "run_electromagnetic_frt_benchmark",
            "compute_modal_participation",
            "evaluate_dq_admittance_and_nyquist",
            "run_small_signal_impedance_benchmark",
        ):
            assert hasattr(src, sym)

    def test_readme_honest_scope_and_figure_reference(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "results/inverter_transient_response.png" in readme
        assert "results/lcl_cascaded_electromagnetic_response.png" in readme
        assert "results/gfm_triad_comparative_response.png" in readme
        assert "results/gfl_vs_gfm_weak_grid_collapse.png" in readme
        assert "results/em_frt_tvi_phase_jump_response.png" in readme
        assert "results/small_signal_eigenvalue_impedance_passivity.png" in readme
        assert "not evidence of real-hardware performance" in readme


class TestHardenedControlValidation:
    def test_droop_controller_validation_and_voltage_clamp(self):
        with pytest.raises(ValueError):
            InverterDroopController(w0=-1.0)
        with pytest.raises(ValueError):
            InverterDroopController(mp=-1e-4)

        ctrl = InverterDroopController(v0=325.27, nq=0.05)
        _, _, v_clamped = ctrl.compute_setpoints(0.0, 5_000.0, v_min=280.0, v_max=350.0)
        assert v_clamped == pytest.approx(280.0)

    def test_virtual_impedance_rejects_negative_or_nonfinite_inputs(self):
        with pytest.raises(ValueError):
            VirtualImpedanceController(r_v_ohm=-0.05)
        v_imp = VirtualImpedanceController(r_v_ohm=0.1, l_v_h=2e-3)
        with pytest.raises(ValueError):
            v_imp.apply_virtual_impedance(float("nan"), 0.0, 1.0, 0.0)


class TestCoupledVSGTheveninPhysics:
    def test_grid_spec_impedance_and_inertia_identities(self):
        spec = TheveninGridSpec(s_rated_va=20_000.0, v_phase_rms_v=230.0, scr=5.0, xr_ratio=7.0, j_kgm2=2.0)
        assert spec.z_base_ohm == pytest.approx(7.935, rel=1e-6)
        assert spec.z_grid_mag_ohm == pytest.approx(7.935 / 5.0, rel=1e-6)
        assert math.hypot(spec.r_g_ohm, spec.x_g_nom_ohm) == pytest.approx(spec.z_grid_mag_ohm, rel=1e-9)
        assert spec.x_g_nom_ohm / spec.r_g_ohm == pytest.approx(7.0, rel=1e-9)
        assert spec.h_inertia_s == pytest.approx(0.5 * 2.0 * (100.0 * math.pi) ** 2 / 20_000.0, rel=1e-9)

    def test_steady_state_droop_power_matches_analytical_formula(self):
        spec = TheveninGridSpec(dp_nms_rad=20.0)
        res = simulate_coupled_vsg_transient(
            spec,
            duration_s=3.5,
            dt_s=1e-3,
            p_step_time_s=0.25,
            p_step_w=10_000.0,
            grid_freq_drop_time_s=1.15,
            grid_freq_drop_hz=-0.20,
        )
        dw = 2.0 * math.pi * 0.20
        expected_p_w = 10_000.0 + spec.dp_nms_rad * spec.w0_rad_s * dw
        assert res.f_vsg_hz[-1] == pytest.approx(49.80, abs=2e-3)
        assert res.p_pcc_w[-1] == pytest.approx(expected_p_w, rel=1e-3)
        assert res.final_delta_deg > 2.0

    def test_rk4_trajectory_matches_scipy_solve_ivp_rk45(self):
        spec = TheveninGridSpec()
        model = CoupledVSGTheveninModel(spec)

        res = simulate_coupled_vsg_transient(
            spec,
            duration_s=1.2,
            dt_s=1e-3,
            p_step_time_s=0.0,
            p_step_w=8_000.0,
            grid_freq_drop_time_s=0.0,
            grid_freq_drop_hz=-0.10,
        )

        w_g = 2.0 * math.pi * 49.90
        sol = solve_ivp(
            fun=lambda _t, y: model.rhs_continuous(y, 8_000.0, 0.0, spec.v_peak_nom_v, w_g),
            t_span=(0.0, 1.2),
            y0=np.array([0.0, spec.w0_rad_s, 0.0, 0.0], dtype=float),
            t_eval=res.time_s,
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
        )
        assert sol.success
        np.testing.assert_allclose(np.degrees(sol.y[0]), res.delta_deg, atol=1e-5)
        np.testing.assert_allclose(sol.y[1] / (2.0 * math.pi), res.f_vsg_hz, atol=1e-6)
        np.testing.assert_allclose(sol.y[2], res.p_filt_w, rtol=1e-6, atol=1e-3)

    def test_severe_overload_activates_radial_current_limiter(self):
        spec = TheveninGridSpec(i_max_peak_a=18.0)
        res = simulate_coupled_vsg_transient(spec, duration_s=0.8, p_step_time_s=0.05, p_step_w=16_000.0)
        assert np.any(res.current_limited)
        assert float(np.max(res.i_peak_a)) <= 18.0 + 1e-9

    def test_main_entrypoint_and_reproducible_figure(self, tmp_path, capsys):
        fig_file = tmp_path / "transient.png"
        lcl_file = tmp_path / "lcl.png"
        triad_file = tmp_path / "triad.png"
        wg_file = tmp_path / "wg.png"
        em_frt_file = tmp_path / "em_frt.png"
        ssi_file = tmp_path / "ssi.png"
        rc = main_entry.main(
            save_path=fig_file,
            lcl_save_path=lcl_file,
            triad_save_path=triad_file,
            weak_grid_save_path=wg_file,
            em_frt_save_path=em_frt_file,
            ssi_save_path=ssi_file,
        )
        captured = capsys.readouterr().out
        assert rc == 0
        assert fig_file.is_file() and fig_file.stat().st_size > 40_000
        assert lcl_file.is_file() and lcl_file.stat().st_size > 40_000
        assert triad_file.is_file() and triad_file.stat().st_size > 40_000
        assert wg_file.is_file() and wg_file.stat().st_size > 40_000
        assert em_frt_file.is_file() and em_frt_file.stat().st_size > 40_000
        assert ssi_file.is_file() and ssi_file.stat().st_size > 40_000
        assert "DAY 19" not in captured
        assert "GRID-FORMING & WEAK-GRID GFL/GFM" in captured

    def test_benchmark_cli_invocations_emit_no_warnings(self):
        for args in ([sys.executable, "-m", "src.benchmark"], [sys.executable, "src/benchmark.py"]):
            proc = subprocess.run(
                args,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0
            assert proc.stderr == ""
            assert "Acceptance checks:      PASS" in proc.stdout
