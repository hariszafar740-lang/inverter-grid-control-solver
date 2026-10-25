"""Tests for Module 6 Small-Signal Participation, dq Passivity, Nyquist, Wheel & `gfm-sim` CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import gfm_solver
import src
from gfm_solver.cli import CLI_VERSION
from gfm_solver.cli import main as cli_main
from src.small_signal_impedance import (
    compute_modal_participation,
    evaluate_dq_admittance_and_nyquist,
    run_small_signal_impedance_benchmark,
)
from src.visualizer import plot_small_signal_impedance_benchmark


class TestSmallSignalParticipationAndImpedancePassivity:
    def test_vsg_modal_participation_matrix_normalization_and_timescale_separation(self):
        part = compute_modal_participation(scr=2.0, p_ref_w=15_000.0)
        assert part.participation_matrix.shape == (14, 14)
        col_sums = np.sum(part.participation_matrix, axis=0)
        np.testing.assert_allclose(col_sums, np.ones(14), atol=1.0e-12)

        for m in part.modes:
            total_share = m.sync_outer_pct + m.voltage_pi_pct + m.current_pi_pct + m.lcl_network_pct
            assert total_share == pytest.approx(100.0, abs=1.0e-6)

        swing_mode = part.modes[0]
        assert swing_mode.freq_hz == pytest.approx(0.77, abs=0.15)
        assert swing_mode.sync_outer_pct > 90.0

        fast_modes = [m for m in part.modes if m.freq_hz > 120.0]
        assert len(fast_modes) >= 2
        for fm in fast_modes:
            assert fm.lcl_network_pct > 85.0

    def test_schur_complement_pole_equivalence_and_nyquist_cauchy_theorem(self):
        for arch, expected_rhp in (
            ("vsg", 0),
            ("vsg_unity_ff", 2),
            ("droop", 0),
            ("dvoc", 0),
            ("gfl_pll_10hz", 2),
            ("gfl_pll_35hz", 2),
            ("gfl_pll_55hz", 4),
        ):
            res = evaluate_dq_admittance_and_nyquist(arch, scr=1.65, p_ref_w=15_000.0)
            assert res.state_space_rhp_poles == expected_rhp
            assert res.nyquist_rhp_poles_z == res.state_space_rhp_poles
            assert res.max_schur_det_residual < 1.0e-6

    def test_qoria_feedforward_passivity_restoration_vs_unity_feedforward(self):
        qoria = evaluate_dq_admittance_and_nyquist("vsg", scr=2.0, p_ref_w=15_000.0)
        naive = evaluate_dq_admittance_and_nyquist("vsg_unity_ff", scr=2.0, p_ref_w=15_000.0)

        assert qoria.min_passivity_high_freq_s > 0.0
        assert qoria.nonpassive_band_hz is None
        assert qoria.state_space_rhp_poles == 0

        assert naive.min_passivity_high_freq_s < -0.5
        assert naive.nonpassive_band_hz is not None
        assert naive.state_space_rhp_poles == 2

    def test_gfl_pll_bandwidth_monotonic_negative_q_conductance_expansion(self):
        g10 = evaluate_dq_admittance_and_nyquist("gfl_pll_10hz", scr=2.0, p_ref_w=15_000.0)
        g35 = evaluate_dq_admittance_and_nyquist("gfl_pll_35hz", scr=2.0, p_ref_w=15_000.0)
        g55 = evaluate_dq_admittance_and_nyquist("gfl_pll_55hz", scr=2.0, p_ref_w=15_000.0)

        assert g10.neg_re_yqq_band_hz is not None
        assert g35.neg_re_yqq_band_hz is not None
        assert g55.neg_re_yqq_band_hz is not None
        assert g10.neg_re_yqq_band_hz[1] < g35.neg_re_yqq_band_hz[1] < g55.neg_re_yqq_band_hz[1]

    def test_small_signal_benchmark_bundle_figure_and_validation_guards(self, tmp_path):
        bench = run_small_signal_impedance_benchmark()
        out_fig = plot_small_signal_impedance_benchmark(bench, save_path=tmp_path / "ssi.png")
        assert out_fig.is_file() and out_fig.stat().st_size > 50_000

        with pytest.raises(ValueError):
            compute_modal_participation(scr=-1.0)
        with pytest.raises(ValueError):
            evaluate_dq_admittance_and_nyquist("invalid_arch")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            evaluate_dq_admittance_and_nyquist("vsg", freqs_hz=np.array([1.0, 2.0]))


class TestPEP621PackageWheelAndGFMSimCLI:
    def test_package_namespace_version_and_py_typed(self):
        assert gfm_solver.__version__ == "0.7.0"
        assert src.__version__ == "0.7.0"
        assert CLI_VERSION == "0.7.0"
        assert set(gfm_solver.__all__) == set(src.__all__)
        pkg_dir = Path(gfm_solver.__file__).resolve().parent
        assert (pkg_dir / "py.typed").is_file()

    def test_no_hardcoded_src_or_main_imports_in_package_modules(self):
        src_dir = Path(__file__).resolve().parents[1] / "src"
        py_files = sorted(src_dir.glob("*.py"))
        assert len(py_files) >= 16
        forbidden_src = re.compile(r"^\s*(?:from|import)\s+src\b", re.M)
        forbidden_main = re.compile(r"^\s*(?:from|import)\s+main\b", re.M)
        for path in py_files:
            text = path.read_text(encoding="utf-8")
            assert not forbidden_src.search(text), f"Hardcoded `src` import in {path.name}"
            assert not forbidden_main.search(text), f"Top-level `main` import in {path.name}"

    def test_built_wheel_maps_cleanly_to_gfm_solver_without_src_leakage(self, tmp_path):
        from setuptools import build_meta

        root = Path(__file__).resolve().parents[1]
        whl_dir = tmp_path / "dist"
        whl_dir.mkdir()
        prev_cwd = Path.cwd()
        try:
            os.chdir(root)
            whl_name = build_meta.build_wheel(str(whl_dir))
        finally:
            os.chdir(prev_cwd)
            shutil.rmtree(root / "build", ignore_errors=True)

        whl_path = whl_dir / whl_name
        assert whl_path.is_file()
        with zipfile.ZipFile(whl_path) as zf:
            names = zf.namelist()
        assert "gfm_solver/__init__.py" in names
        assert "gfm_solver/py.typed" in names
        assert "gfm_solver/cli.py" in names
        assert "gfm_solver/small_signal_impedance.py" in names
        assert not any(n.startswith("src/") for n in names)

    def test_cli_info_and_reduced_frt_json_output(self, capsys):
        rc_info = cli_main(["info", "--json"])
        assert rc_info == 0
        info_data = json.loads(capsys.readouterr().out)
        assert info_data["package"] == "gfm_solver"
        assert info_data["version"] == "0.7.0"
        assert info_data["nominal_rating"]["s_rated_kva"] == 20.0

        rc_frt = cli_main(["frt", "--reduced-order", "--json"])
        assert rc_frt == 0
        frt_data = json.loads(capsys.readouterr().out)
        assert frt_data["passed"] is True
        assert frt_data["fault_max_current_a"] <= 30.001

    def test_cli_impedance_subcommand_json_and_module_entrypoint(self, tmp_path, capsys):
        fig_out = tmp_path / "cli_ssi.png"
        rc = cli_main(["impedance", "--scr", "1.65", "--p-kw", "15.0", "--save-plot", str(fig_out), "--json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vsg_qoria_fi_075"]["nyquist_rhp_poles_z"] == 0
        assert data["vsg_naive_fi_100"]["nyquist_rhp_poles_z"] == 2
        assert data["gfl_pll_55hz"]["nyquist_rhp_poles_z"] == 4
        assert fig_out.is_file()

        proc = subprocess.run(
            [sys.executable, "-m", "gfm_solver", "info", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        mod_info = json.loads(proc.stdout)
        assert mod_info["version"] == "0.7.0"
