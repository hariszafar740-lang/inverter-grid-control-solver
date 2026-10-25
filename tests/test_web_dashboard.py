"""Tests for the zero-CDN static HTML5/Canvas GFM workbench & `gfm-sim web` CLI (`v1.0.0`)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from gfm_solver import (
    WORKBENCH_VERSION,
    build_static_web_workbench,
    build_web_presets_payload,
)
from gfm_solver.cli import main as cli_main


class TestStaticWebWorkbench:
    def test_presets_payload_matches_python_solver_invariants_and_strict_json(self):
        payload = build_web_presets_payload()
        assert payload["version"] == "1.0.0"
        assert WORKBENCH_VERSION == "1.0.0"
        raw = json.dumps(payload, allow_nan=False)
        assert "NaN" not in raw and "Infinity" not in raw

        nom = payload["nominal_rating"]
        assert nom["s_rated_kva"] == 20.0
        assert nom["i_max_peak_a"] == 48.0
        assert nom["i_th_peak_a"] == 39.0

        vb = payload["verified_baselines"]
        assert vb["vsg_nadir_hz"] == pytest.approx(49.748, abs=0.005)
        assert vb["reduced_frt_max_current_a"] <= 30.001

    def test_presets_admittance_schur_and_nyquist_exactness(self):
        payload = build_web_presets_payload()
        adm = payload["dq_admittance"]["architectures"]
        assert len(payload["dq_admittance"]["freqs_hz"]) == 110

        for key, expected_z in (
            ("vsg", 0),
            ("vsg_unity_ff", 2),
            ("droop", 0),
            ("dvoc", 0),
            ("gfl_pll_10hz", 2),
            ("gfl_pll_35hz", 2),
            ("gfl_pll_55hz", 4),
        ):
            item = adm[key]
            assert item["nyquist_rhp_poles_z"] == expected_z
            assert item["state_space_rhp_poles"] == expected_z
            assert item["max_schur_det_residual"] < 1.0e-6

        assert adm["vsg"]["min_passivity_gt_25hz_s"] > 0.0
        assert adm["vsg_unity_ff"]["min_passivity_gt_25hz_s"] < -0.5

    def test_presets_weak_grid_hopf_critical_scr_ordering(self):
        wg = payload_wg = build_web_presets_payload()["weak_grid"]
        assert wg["scr_crit_10hz"] < wg["scr_crit_35hz"] < wg["scr_crit_55hz"]
        assert wg["scr_static_nose"] < wg["scr_crit_10hz"]
        assert len(payload_wg["root_locus"]["scr_values"]) == 14

    def test_build_static_bundle_files_and_six_copied_figures(self, tmp_path):
        summary = build_static_web_workbench(output_dir=tmp_path / "site")
        site_dir = Path(summary["output_dir"])
        for fname in (".nojekyll", "index.html", "styles.css", "app.js", "gfm_presets.json"):
            assert (site_dir / fname).is_file(), f"Missing {fname}"
        assert len(summary["copied_figures"]) == 6
        for png in summary["copied_figures"]:
            assert (site_dir / "results" / png).stat().st_size > 40_000

    def test_zero_cdn_offline_self_containment_invariants(self, tmp_path):
        summary = build_static_web_workbench(output_dir=tmp_path / "site")
        site_dir = Path(summary["output_dir"])
        html = (site_dir / "index.html").read_text(encoding="utf-8")
        css = (site_dir / "styles.css").read_text(encoding="utf-8")
        js = (site_dir / "app.js").read_text(encoding="utf-8")

        assert not re.search(r'<script[^>]+src=["\']https?://', html, re.I)
        assert not re.search(r'<link[^>]+href=["\']https?://', html, re.I)
        assert "https://" not in css and "http://" not in css and "@import" not in css
        assert "fetch(" not in js and "XMLHttpRequest" not in js

    def test_dom_id_integrity_between_index_html_and_app_js(self, tmp_path):
        summary = build_static_web_workbench(output_dir=tmp_path / "site")
        site_dir = Path(summary["output_dir"])
        html = (site_dir / "index.html").read_text(encoding="utf-8")
        js = (site_dir / "app.js").read_text(encoding="utf-8")

        html_ids = set(re.findall(r'\bid="([^"]+)"', html))
        js_ids = set(re.findall(r'(?:getElementById|el)\("([^"]+)"\)', js))
        assert len(js_ids) >= 30
        missing = js_ids - html_ids
        assert not missing, f"Missing DOM IDs in index.html: {missing}"

    def test_build_static_bundle_without_figures_graceful_fallback(self, tmp_path):
        empty_repo = tmp_path / "empty_repo"
        empty_repo.mkdir()
        out = tmp_path / "site_no_figs"
        summary = build_static_web_workbench(output_dir=out, repo_root=empty_repo)
        assert summary["copied_figures"] == []
        assert (out / "index.html").is_file()

    def test_cli_web_subcommand_json_and_human_modes(self, capsys, tmp_path):
        out_json = tmp_path / "web_json"
        assert cli_main(["web", "--output-dir", str(out_json), "--json"]) == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["version"] == "1.0.0"
        assert len(parsed["architectures"]) == 7
        assert (out_json / "index.html").is_file()

        out_txt = tmp_path / "web_txt"
        assert cli_main(["web", "--output-dir", str(out_txt)]) == 0
        msg = capsys.readouterr().out
        assert str(out_txt) in msg
        assert (out_txt / "gfm_presets.json").is_file()

    def test_cli_info_lists_web_subcommand_and_v1_release(self, capsys):
        assert cli_main(["info"]) == 0
        out = capsys.readouterr().out
        assert "1.0.0" in out
        assert "web" in out

    def test_deploy_pages_workflow_exists_and_invokes_gfm_sim_web(self):
        root = Path(__file__).resolve().parents[1]
        wf = root / ".github" / "workflows" / "deploy_pages.yml"
        assert wf.is_file()
        content = wf.read_text(encoding="utf-8")
        assert "gfm-sim web --output-dir site --json" in content
        assert "actions/deploy-pages@v4" in content
