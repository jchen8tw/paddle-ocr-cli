"""What the Judge's renderer needs from the machine: node, Noto Sans CJK, the vendored Droid font, Chromium.

All checks take their probes as arguments so they run here without any of those installed.
"""
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import render_env

FC_LIST = """\
/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc: Noto Sans CJK JP,Noto Sans CJK JP Regular
/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc: Noto Sans CJK TC,Noto Sans CJK TC Bold
/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc: Noto Serif CJK JP,Noto Serif CJK JP Regular
/usr/share/fonts/TTF/DejaVuSans.ttf: DejaVu Sans
"""


def problems(tmp_path, *, node="/usr/bin/node", fc_list=FC_LIST, droid=True):
    font = tmp_path / "Droid.ttf"
    if droid:
        font.write_bytes(b"font")
    return render_env.setup_problems(which=lambda name: node, fc_list=lambda: fc_list, droid_font=font)


def test_a_complete_setup_has_no_problems(tmp_path):
    assert problems(tmp_path) == []


def test_missing_node_is_named_with_what_it_is_for(tmp_path):
    (msg,) = problems(tmp_path, node=None)
    assert "node" in msg and "KaTeX" in msg


def test_missing_noto_sans_cjk_is_named_with_the_package_to_install(tmp_path):
    (msg,) = problems(tmp_path, fc_list="/usr/share/fonts/TTF/DejaVuSans.ttf: DejaVu Sans\n")
    assert "Noto Sans CJK" in msg and "noto" in msg.lower()


def test_noto_serif_alone_does_not_count_as_noto_sans(tmp_path):
    (msg,) = problems(tmp_path, fc_list="/x/NotoSerifCJK-Regular.ttc: Noto Serif CJK JP\n")
    assert "Noto Sans CJK" in msg


def test_a_missing_vendored_font_is_reported(tmp_path):
    (msg,) = problems(tmp_path, droid=False)
    assert "DroidSansFallbackFull" in msg


def test_fc_list_missing_is_reported_as_a_font_problem(tmp_path):
    def broken():
        raise FileNotFoundError("fc-list")

    font = tmp_path / "d.ttf"
    font.write_bytes(b"x")
    (msg,) = render_env.setup_problems(which=lambda n: "/usr/bin/node", fc_list=broken, droid_font=font)
    assert "fc-list" in msg


def test_several_problems_are_all_reported_at_once(tmp_path):
    assert len(problems(tmp_path, node=None, fc_list="", droid=False)) == 3


def test_only_the_noto_sans_cjk_files_are_picked_from_fc_list():
    assert render_env.noto_sans_cjk_files(FC_LIST) == [
        Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"), Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")]


def render_result(ok=True, reason=None, detail=None):
    return types.SimpleNamespace(case_id="smoke", ok=ok, render_image_path="/x.png" if ok else None,
                                 reason=reason, detail=detail)


def test_a_working_renderer_is_no_problem(tmp_path):
    seen = []
    assert render_env.smoke_render_problem(lambda cands, work: seen.append(cands) or [render_result()]) is None
    assert seen[0][0].category == "text" and Path(seen[0][0].image_path).is_file()  # the bundled example crop


def test_a_renderer_that_cannot_run_names_the_install_command_and_the_cause():
    def boom(cands, work):
        raise RuntimeError("BrowserType.launch: Executable doesn't exist at /x/chrome")

    msg = render_env.smoke_render_problem(boom)
    assert "playwright install chromium" in msg and "Executable doesn't exist" in msg


def test_a_render_that_fails_reports_the_reason_and_still_names_chromium():
    msg = render_env.smoke_render_problem(lambda cands, work: [render_result(False, "render_error", "render_failed")])
    assert "render_error" in msg and "render_failed" in msg and "playwright install chromium" in msg


# --- the fontconfig the renderer runs under ----------------------------------------------------------

def test_the_fontconfig_lists_only_droid_and_noto_sans_cjk_and_is_valid_xml(tmp_path):
    droid_dir = tmp_path / "fonts"
    droid_dir.mkdir()
    noto = tmp_path / "NotoSansCJK-Regular.ttc"
    noto.write_bytes(b"ttc")
    conf = render_env.write_fontconfig(tmp_path / "work", droid_dir, [noto], tmp_path / "cache")
    root = ET.parse(conf).getroot()
    dirs = [Path(d.text) for d in root.findall("dir")]
    assert droid_dir in dirs and len(dirs) == 2
    noto_dir = next(d for d in dirs if d != droid_dir)
    assert [p.name for p in noto_dir.iterdir()] == ["NotoSansCJK-Regular.ttc"]  # a symlink, not a copy
    assert (noto_dir / noto.name).is_symlink()
    assert Path(root.find("cachedir").text) == tmp_path / "cache"


def test_the_fontconfig_carries_the_rendering_rules_the_reference_machine_gets_from_its_conf_d(tmp_path):
    droid_dir = tmp_path / "f"
    droid_dir.mkdir()
    conf = render_env.write_fontconfig(tmp_path / "w", droid_dir, [], tmp_path / "c")
    text = conf.read_text()
    for needle in ("hintslight", "lcddefault", "antialias"):
        assert needle in text


def test_writing_the_fontconfig_twice_gives_the_same_file(tmp_path):
    droid_dir = tmp_path / "f"
    droid_dir.mkdir()
    a = render_env.write_fontconfig(tmp_path / "w", droid_dir, [], tmp_path / "c").read_text()
    b = render_env.write_fontconfig(tmp_path / "w", droid_dir, [], tmp_path / "c").read_text()
    assert a == b


def test_the_font_cache_lives_in_the_user_cache_not_in_the_run_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert render_env.font_cache_dir() == tmp_path / "xdg" / "pdf_ocr" / "fontconfig"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert render_env.font_cache_dir() == tmp_path / "home" / ".cache" / "pdf_ocr" / "fontconfig"
