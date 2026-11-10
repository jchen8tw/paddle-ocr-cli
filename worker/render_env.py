"""What the Judge's renderer needs from the machine, and the fontconfig it runs under (docs/adr/0002).

The renderer (worker/render/, the MiniCPM-OCR authors' own, vendored unchanged) draws a Reading with
headless Chromium. Its font stack names Noto Sans CJK and falls back to DroidSansFallbackFull; Latin letters
and digits come from Noto, Droid only fills CJK glyphs Noto lacks (ADR-0002, amendment). So the machine needs
Noto Sans CJK installed, plus `node` (the KaTeX precheck) and Chromium. Droid is vendored in worker/fonts/.

To keep the rendered pixels the same on every machine, the renderer runs with a private FONTCONFIG_FILE
that lists only those two font sets and carries the rendering rules the reference machine gets from its
fontconfig defaults (slight hinting, antialiasing, default LCD filter). Without those rules the pixels,
and the Scores, drift (measured, see the README).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
DROID_DIR = HERE / "fonts"
DROID_FONT = DROID_DIR / "DroidSansFallbackFull.ttf"
RENDER_DIR = HERE / "render"

INSTALL_CHROMIUM = "run once: uv run --project worker playwright install chromium"

# What the reference deployment's conf.d provides (10-hinting-slight, 10-yes-antialias, 11-lcdfilter-default,
# with 10-sub-pixel-rgb removed, as the authors' setup script does).
RENDERING_RULES = """\
  <match target="font">
    <edit name="antialias" mode="assign"><bool>true</bool></edit>
    <edit name="hinting" mode="assign"><bool>true</bool></edit>
    <edit name="hintstyle" mode="assign"><const>hintslight</const></edit>
    <edit name="lcdfilter" mode="assign"><const>lcddefault</const></edit>
  </match>
"""


def default_fc_list() -> str:
    return subprocess.run(["fc-list", ":", "file", "family"], capture_output=True, text=True,
                          check=True).stdout


def noto_sans_cjk_files(fc_list_output: str) -> list[Path]:
    """The Noto Sans CJK font files in `fc-list` output ("path: family,family"), sorted. Noto Serif CJK is not it."""
    files = set()
    for line in fc_list_output.splitlines():
        path, _, families = line.partition(": ")
        if re.search(r"\bNoto Sans CJK\b", families) and path.strip():
            files.add(Path(path.strip()))
    return sorted(files)


def setup_problems(*, which: Callable[[str], str | None] = shutil.which,
                   fc_list: Callable[[], str] = default_fc_list, droid_font: Path = DROID_FONT) -> list[str]:
    """Everything missing for the Judge's renderer (except Chromium, see smoke_render_problem), each with its fix."""
    problems = []
    if not which("node"):
        problems.append("`node` (Node.js) is not on PATH; the Judge's KaTeX precheck needs it "
                        "(install Node.js, e.g. your package manager's `nodejs`)")
    try:
        fonts = noto_sans_cjk_files(fc_list())
    except (OSError, subprocess.CalledProcessError) as e:
        problems.append(f"cannot list the installed fonts with `fc-list` ({e}); the Judge's renderer needs "
                        "Noto Sans CJK, found through fontconfig (install `fontconfig` and the Noto Sans CJK fonts)")
    else:
        if not fonts:
            problems.append("the font Noto Sans CJK is not installed; the Judge's renderer draws Latin letters, "
                            "digits and CJK text with it (install your package manager's `fonts-noto-cjk` / "
                            "`noto-fonts-cjk` / `google-noto-sans-cjk-fonts`)")
    if not Path(droid_font).is_file():
        problems.append(f"the bundled font DroidSansFallbackFull.ttf is missing at {droid_font}; "
                        "it ships with the repository (worker/fonts/), restore it with git")
    return problems


EXAMPLE_BLOCK = RENDER_DIR / "examples" / "01_text_putnam" / "block.png"


def _gist(text: str, limit: int = 300) -> str:
    """The telling line of a long error (a subprocess traceback): the last line that names an Error, else the start."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    errors = [ln for ln in lines if "Error" in ln]
    return (errors[-1] if errors else str(text).strip())[:limit]


def smoke_render_problem(render=None) -> str | None:
    """Render one tiny Reading with the real renderer: None if it works, else what to fix. This is what finds a
    missing Chromium (and anything else wrong with Playwright, node or the fonts) before a run starts."""
    from judging import Candidate

    render = render or render_with_official_renderer
    install = f"if Chromium is not installed, {INSTALL_CHROMIUM}"
    work = tempfile.mkdtemp(prefix="pdf_ocr_smoke_")
    try:
        result = render([Candidate("smoke", str(EXAMPLE_BLOCK), "Hello, 世界", "text")], work)[0]
    except Exception as e:
        return f"the Judge's renderer cannot run ({_gist(e)}); {install}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if result.ok:
        return None
    return f"a test render failed ({result.reason}: {_gist(result.detail)}); {install}"


def font_cache_dir() -> Path:
    """Where fontconfig caches the Noto Sans CJK glyph data: kept between runs (the TTCs are ~40 MB)."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "pdf_ocr" / "fontconfig"


def write_fontconfig(work_dir: Path, droid_dir: Path, noto_files: list[Path], cache_dir: Path) -> Path:
    """Write the private fontconfig file under `work_dir`; Noto is linked (not copied) into a folder of its own
    so that Noto Serif/Mono CJK in the same system folder are left out. Returns the file's path."""
    work_dir = Path(work_dir)
    noto_dir = work_dir / "noto-sans-cjk"
    noto_dir.mkdir(parents=True, exist_ok=True)
    for stale in noto_dir.iterdir():
        stale.unlink()
    for f in noto_files:
        (noto_dir / Path(f).name).symlink_to(f)
    conf = work_dir / "fonts.conf"
    conf.write_text(
        '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig>\n'
        f"  <dir>{droid_dir}</dir>\n  <dir>{noto_dir}</dir>\n  <cachedir>{cache_dir}</cachedir>\n"
        f"{RENDERING_RULES}</fontconfig>\n", encoding="utf-8")
    return conf


def prepare_environment() -> None:
    """Point the renderer's Chromium at the private fontconfig. Call once, before the first render.

    The file and the links to Noto live in a folder of this process alone (removed at exit), so two runs, or a
    restarted worker, can never find each other's half-written files; only fontconfig's cache is shared and kept."""
    import atexit

    noto = noto_sans_cjk_files(default_fc_list())
    if not noto:  # setup_problems() reports this first; never render with the wrong font silently
        raise RuntimeError("Noto Sans CJK is not installed")
    work = Path(tempfile.mkdtemp(prefix="pdf_ocr_fontconfig_"))
    atexit.register(shutil.rmtree, work, ignore_errors=True)
    os.environ["FONTCONFIG_FILE"] = str(write_fontconfig(work, DROID_DIR, noto, font_cache_dir()))


def render_with_official_renderer(candidates, work_dir):
    """The vendored renderer (precheck + Chromium) for a batch of judging.Candidates, one Chromium launch."""
    render_dir = str(RENDER_DIR)
    if render_dir not in sys.path:
        sys.path.insert(0, render_dir)
    import rm_render

    return rm_render.render_candidates(
        [rm_render.Candidate(case_id=c.case_id, image_path=c.image_path, text=c.text, category=c.category)
         for c in candidates],
        work_dir=work_dir, precheck=True)
