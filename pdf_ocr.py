#!/usr/bin/env python3
"""Transcribe a PDF to Markdown and JSON with PaddleOCR-VL-1.6 (or MiniCPM-OCR-1.0), using the local models in ./models.

    python pdf_ocr.py input.pdf                   # -> output/<stem>/<stem>.md, <stem>.json, imgs/, csv/
    python pdf_ocr.py input.pdf -o out --pages 1-5,9
    python pdf_ocr.py scans/                      # every PDF under scans/ -> output/scans/<sub>/<name>/ (one model load)
    python pdf_ocr.py scans/                      # again: the Documents already done are skipped (--force redoes them)
    python pdf_ocr.py input.pdf --reader minicpm  # MiniCPM-OCR-1.0 (in the torch worker) instead of PaddleOCR-VL-1.6
    python pdf_ocr.py input.pdf --judge           # score every Reading with the Judge, re-read the low ones

Pipeline: pypdfium2 renders each page -> PP-DocLayoutV3 finds and orders layout blocks -> the Reader
(PaddleOCR-VL-1.6, or MiniCPM-OCR-1.0 in the torch worker) recognises every block (text / table /
formula; charts and seals on request) -> block-level clean-up (lone LaTeX symbols, Simplified/Traditional
repair, truncated-block re-read) -> JSON per page, and Markdown with cross-page tables merged and title
levels fixed, plus one CSV per well-formed table.
"""
from __future__ import annotations

import argparse
import csv
import enum
import importlib.metadata
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import Path

import block_reader
import document_record
import reader_bridge
from minicpm_reader import unwrap_symbols
from script_repair import BIG5_LOOKALIKES, NO_REPAIR_LABELS, ScriptRepair, is_traditional

ROOT = Path(__file__).resolve().parent
LAYOUT_MODEL_DIR = "PP-DocLayoutV3"
VL_MODEL_DIR = "PaddleOCR-VL-1.6"
MINICPM_MODEL_DIR = "MiniCPM-OCR-1.0"
JUDGE_MODEL_DIR = "MiniCPM-OCR-Jev"
READER_MODEL_DIRS = {"paddle": VL_MODEL_DIR, "minicpm": MINICPM_MODEL_DIR}
WORKER_DIR = ROOT / "worker"  # second uv project: torch + transformers (docs/adr/0001)

# Page furniture that is kept in the JSON but left out of the Markdown by default.
# (PaddleOCR's own default also drops `footnote`; that is real content, so it is kept here.)
# Measured on pages 38-58 of one document (see the README, "Tuning notes"): 0.5 is the lowest value tried at which
# re-reading fixed anything; higher values made the same fix and re-read more Blocks (83 % at 0.98) for no gain.
DEFAULT_JUDGE_THRESHOLD = 0.5
DEFAULT_MD_IGNORE = ["header", "header_image", "footer", "footer_image", "number", "aside_text"]
MAX_RENDER_PIXELS = 40_000_000  # guard against huge pages exhausting memory at high DPI


# --------------------------------------------------------------------------- CLI

def unit_interval(text: str) -> float:
    """argparse type for a Score threshold: a number from 0 to 1 (NaN and out-of-range values are rejected)."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0.0 <= value <= 1.0:  # false for NaN as well
        raise argparse.ArgumentTypeError(f"{text} is outside 0..1 (the Judge's Scores are between 0 and 1)")
    return value


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf_ocr",
        description="Transcribe a PDF to Markdown + JSON with PaddleOCR-VL-1.6 or MiniCPM-OCR-1.0 (local models).",
    )
    p.add_argument("pdf", type=Path, metavar="PDF_OR_FOLDER",
                   help="input PDF, or a folder: every PDF under it (any depth; names starting with '.' are "
                        "skipped, symlinked folders are not entered) is transcribed in one Run")
    p.add_argument("-o", "--output-dir", type=Path,
                   help="output directory (default: output/<pdf stem>; for a folder, output/<folder name>, with "
                        "one subfolder per Document mirroring its place in the folder)")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models",
                   help=f"folder holding {LAYOUT_MODEL_DIR}/ and the Reader's folder: {VL_MODEL_DIR}/, or "
                        f"{MINICPM_MODEL_DIR}/ with --reader minicpm, plus {JUDGE_MODEL_DIR}/ with --judge "
                        "(default: ./models)")
    p.add_argument("--pages", help="1-based pages to process, e.g. '1-3,7,10-' (default: all; not with a folder)")
    p.add_argument("--device", help="paddle device, e.g. gpu:0 or cpu (default: gpu:0 if available)")

    g = p.add_argument_group("accuracy")
    g.add_argument("--reader", choices=tuple(READER_MODEL_DIRS), default="paddle",
                   help="model that reads each block: paddle = PaddleOCR-VL-1.6; minicpm = MiniCPM-OCR-1.0, run in "
                        "the torch worker (needs the worker environment, see the README) (default: %(default)s)")
    # Defaults below were tuned against the text layer of a 21-page Traditional Chinese PDF:
    # 216 vs 288 DPI were equivalent (216 is faster); layout threshold 0.3 -> 0.2 cut text lost to
    # missed layout blocks from ~85 to ~20 characters, while 0.1 starts producing false boxes.
    g.add_argument("--dpi", type=float, default=216.0, help="page render resolution (default: %(default)s)")
    g.add_argument("--layout-threshold", type=float, default=0.2,
                   help="layout detection score threshold; lower = fewer missed blocks, "
                        "too low = false boxes (default: %(default)s)")
    g.add_argument("--max-new-tokens", type=int, default=8192,
                   help="decoding limit per block; raise for very large tables. With --reader minicpm this is "
                        "the limit for tables only; other blocks have fixed limits by type (default: %(default)s)")
    g.add_argument("--charts", action="store_true",
                   help="convert charts to data tables. Off by default: values are estimated and the model "
                        "can invent data for figures it mistakes for charts; otherwise the chart is saved as an image")
    g.add_argument("--seals", action="store_true",
                   help="read text on seals/stamps (off by default; otherwise saved as an image)")
    g.add_argument("--ocr-images", action="store_true", help="also OCR text inside figure/image blocks")
    g.add_argument("--no-merge-blocks", action="store_true", help="do not merge adjacent text blocks before OCR")
    g.add_argument("--judge", action=argparse.BooleanOptionalAction, default=False,
                   help="score every block with the MiniCPM-OCR Jev Judge and re-read those below --judge-threshold "
                        "(needs the worker environment, Chromium, node and Noto Sans CJK, see the README). Off by "
                        "default: the JSON has no scores and the ink-based re-read of implausibly short blocks runs "
                        "instead")
    g.add_argument("--judge-threshold", type=unit_interval, default=None, metavar="FLOAT",
                   help="with --judge: blocks whose first Reading scores below this are re-read (up to 10 Readings, "
                        "stopping after a round that reads no new text; the best one is kept) and flagged in the JSON "
                        "(flagged) and counted on stderr. A re-read never clears the flag; a block whose first Reading "
                        "could not be scored is flagged too. 0 turns the re-reading off and keeps the Scores. The "
                        "Judge authors' own value, 0.98, flags and re-reads most Blocks for no gain on the measured "
                        f"document; see the README (default: {DEFAULT_JUDGE_THRESHOLD})")
    g.add_argument("--no-recover", action="store_true",
                   help="without --judge, do not re-read blocks whose text is implausibly short for the ink they "
                        "contain (with the Judge on, that re-read never runs)")
    g.add_argument("--script", choices=("auto", "off"), default="auto",
                   help="auto: if the document is Traditional Chinese, repair Simplified characters the "
                        "model drifted into (needs opencc); off: keep raw model output (default: %(default)s)")

    g = p.add_argument_group("output")
    g.add_argument("--md-ignore-labels", nargs="*", default=DEFAULT_MD_IGNORE, metavar="LABEL",
                   help="block labels left out of the Markdown (the JSON always keeps them). "
                        "Pass the flag with no values to keep everything. Default: %(default)s")
    g.add_argument("--no-merge-tables", action="store_true",
                   help="in the Markdown and CSV, keep a table that continues over a page break as separate "
                        "per-page tables (default: merge them into one; the JSON is always per page)")
    g.add_argument("--no-csv", action="store_true",
                   help="do not export tables as csv/pNNN_tableKK.csv (tables whose HTML is malformed "
                        "are never exported; the JSON lists them with the reason)")
    g.add_argument("--no-page-markers", action="store_true", help="omit <!-- page N --> markers in the Markdown")
    p.add_argument("--force", action="store_true",
                   help="with a folder: transcribe every Document again. Without it, a Document whose JSON shows it "
                        "was already done (same PDF, every page transcribed and none failed, same settings and "
                        "models) is skipped. "
                        "Not with a single PDF, which is always transcribed")
    p.add_argument("-v", "--verbose", action="store_true", help="show PaddleOCR/PaddleX logs")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, with the Judge settled: `judge` is a bool, and `judge_threshold` is a float exactly when
    `judge` is true (None otherwise)."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.judge_threshold is not None and not args.judge:
        parser.error("--judge-threshold only applies when the Judge is on; add --judge")
    if args.judge and args.judge_threshold is None:
        args.judge_threshold = DEFAULT_JUDGE_THRESHOLD
    folder_run = args.pdf.is_dir()
    if folder_run and args.pages is not None:
        parser.error("--pages cannot be used with a folder: Documents differ in length")
    if not folder_run and args.force:
        parser.error("--force only applies to a folder: a single PDF is always transcribed again")
    return args


def parse_pages(spec: str | None, total: int) -> list[int]:
    """'1-3,7,10-' -> sorted 0-based indices, validated against the page count."""
    if not spec:
        return list(range(total))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d*)-(\d*)|(\d+)", part)
        if not m:
            raise ValueError(f"bad page spec {part!r}")
        if m.group(3):
            lo = hi = int(m.group(3))
        else:
            lo = int(m.group(1) or 1)
            hi = int(m.group(2) or total)
        if lo < 1 or hi > total or lo > hi:
            raise ValueError(f"page range {part!r} outside 1..{total}")
        out.update(range(lo - 1, hi))
    return sorted(out)


# --------------------------------------------------------------------------- Engine spec

class Adapter(enum.Enum):
    """How the Engine reads a page: with PaddleOCR-VL alone, or with the worker, which holds MiniCPM-OCR, the Judge,
    or both."""
    PADDLEOCR_VL_ALONE = "PaddleOCR-VL alone"
    WORKER = "with the worker"


# The peaks quoted below are from the README's "Readers compared" table (pages 38-58 of one document).
MEMORY_WAYS_OUT = {
    ("paddle", False): "close other GPU programs, or try --reader minicpm, which needs less GPU memory "
                       "(5,408 against 9,657 MiB at the peak on the measured document; it runs in the worker "
                       "environment)",
    ("minicpm", False): "close other GPU programs (MiniCPM-OCR-1.0 without the Judge is the smallest configuration)",
    ("minicpm", True): "the Judge and the Reader must fit side by side; close other GPU programs, or run without "
                       "--judge",
    ("paddle", True): "the Judge and the Reader must fit side by side, and PaddleOCR-VL with the Judge is the "
                      "tightest combination; close other GPU programs, run without --judge, or use "
                      "--reader minicpm --judge",
}


@dataclass(frozen=True)
class EngineSpec:
    """What the Engine loads and how it reads, settled once per Run from the options. Every rule that depends on the
    Reader and the Judge is answered here."""
    reader: str
    judge: bool
    judge_threshold: float | None
    dpi: float
    max_new_tokens: int
    layout_threshold: float
    charts: bool
    seals: bool
    ocr_images: bool
    merge_blocks: bool
    recover: bool
    script: str

    @classmethod
    def from_options(cls, options) -> EngineSpec:
        return cls(reader=options.reader, judge=options.judge, judge_threshold=options.judge_threshold,
                   dpi=options.dpi, max_new_tokens=options.max_new_tokens, layout_threshold=options.layout_threshold,
                   charts=options.charts, seals=options.seals, ocr_images=options.ocr_images,
                   merge_blocks=not options.no_merge_blocks, recover=not options.no_recover, script=options.script)

    @property
    def adapter(self) -> Adapter:
        return Adapter.WORKER if self.reader == "minicpm" or self.judge else Adapter.PADDLEOCR_VL_ALONE

    @property
    def needs_worker(self) -> bool:
        return self.adapter is Adapter.WORKER

    @property
    def models(self) -> dict[str, str]:
        """The model names of the JSON `engine` section and of the Already-done check: the Layout model, the Reader,
        and the Judge when it is on. Each is also the folder of --models-dir that holds it, and the name the worker
        (worker/serve.py) gives it."""
        return {"layout_model": LAYOUT_MODEL_DIR, "reader": READER_MODEL_DIRS[self.reader],
                **({"judge": JUDGE_MODEL_DIR} if self.judge else {})}

    @property
    def model_dirs(self) -> list[str]:
        return list(self.models.values())

    @property
    def worker_flags(self) -> list[str]:
        return [*(["--no-reader"] if self.reader != "minicpm" else []), *([] if self.judge else ["--no-judge"])]

    @property
    def ink_reread(self) -> bool:
        return self.recover and not self.judge

    @property
    def memory_advice(self) -> str:
        return MEMORY_WAYS_OUT[self.reader, self.judge]

    def supports(self, step) -> bool:
        """Whether the Reader can do this re-read Round, known without a bridge."""
        return reader_bridge.reader_supports(self.reader)(step)


# --------------------------------------------------------------------------- script repair

def repair_script(results: list[tuple[int, object]], mode: str) -> tuple[dict, dict]:
    """Fix Simplified drift, and write `≥`/`≤` as the Big5 glyphs `≧`/`≦`, in the block objects of `results` in
    place, once the whole document is decided Traditional (ScriptRepair.fix).

    Returns (summary for the JSON, {(page_idx, block_idx): original_text} for changed blocks).
    """
    info: dict = {"mode": mode, "applied": False}
    raw: dict = {}
    if mode == "off":
        return info, raw
    rep = ScriptRepair()

    blocks = [(idx, i, b) for idx, r in results for i, b in enumerate(r["parsing_res_list"])
              if b.label not in NO_REPAIR_LABELS]
    simp, trad = rep.survey(b.content for _, _, b in blocks)
    share = simp / (simp + trad) if simp + trad else 0.0
    is_trad = is_traditional(simp, trad)
    info.update(simplified_chars=simp, traditional_chars=trad,
                document_script="traditional" if is_trad else "not traditional")
    if not is_trad:
        return info, raw
    for idx, i, b in blocks:
        fixed = rep.fix(b.content)
        if fixed != b.content:
            raw[(idx, i)] = b.content
            b.content = fixed
    info.update(applied=True, blocks_changed=len(raw),
                big5_symbols_changed=sum(text.count(c) for text in raw.values() for c in BIG5_LOOKALIKES))
    return info, raw


# --------------------------------------------------------------------------- OCR

def render_page(page, dpi: float):
    """Render to a BGR uint8 array (what PaddleX expects). Returns (array, px_per_pt)."""
    w, h = page.get_size()
    scale = dpi / 72.0
    if w * h * scale * scale > MAX_RENDER_PIXELS:
        scale = (MAX_RENDER_PIXELS / (w * h)) ** 0.5
    return page.render(scale=scale).to_numpy(), scale


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    import paddle

    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        return "gpu:0"
    print("warning: no CUDA device found, running on CPU (slow)", file=sys.stderr)
    return "cpu"


def missing_models_message(spec: EngineSpec, models_dir: Path) -> str | None:
    missing = [d for d in spec.model_dirs if not (models_dir / d).is_dir()]
    if not missing:
        return None
    other = "paddle" if spec.reader == "minicpm" else "minicpm"
    return (f"model folder(s) not found under {models_dir}: {', '.join(missing)}. "
            f"Put them there (nothing is downloaded at run time); --reader {other} needs "
            f"{READER_MODEL_DIRS[other]} instead of {READER_MODEL_DIRS[spec.reader]}.")


WORKER_FREE = "a run without --reader minicpm and --judge does not need it"


def worker_env_path() -> Path:
    return Path(os.environ.get("PDF_OCR_WORKER_ENV") or WORKER_DIR / ".venv")


def worker_env_problem() -> str | None:
    """None if the worker's virtual environment exists. Checked first because `uv run --no-sync` would
    otherwise silently create an empty one and fail later with a confusing import error."""
    path = worker_env_path()
    if path.is_dir():
        return None
    return (f"the worker environment (MiniCPM-OCR-1.0 and the Judge) is not set up ({path} not found). "
            f"Create it once with:  uv sync --project worker   ({WORKER_FREE})")


def worker_environment() -> dict:
    env = dict(os.environ)
    for name in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        env.pop(name, None)  # these name the main project's environment (`uv run` sets VIRTUAL_ENV), not the worker's
    if os.environ.get("PDF_OCR_WORKER_ENV"):
        env["UV_PROJECT_ENVIRONMENT"] = os.environ["PDF_OCR_WORKER_ENV"]
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return env


def worker_command(spec: EngineSpec, models_dir: Path, uv: str) -> list[str]:
    """The worker's flags (EngineSpec.worker_flags) are its private protocol, not the CLI's."""
    return [uv, "run", "--project", str(WORKER_DIR), "--no-sync",
            "python", str(WORKER_DIR / "serve.py"), "--models-dir", str(models_dir.resolve()), *spec.worker_flags]


def start_worker(spec: EngineSpec, models_dir: Path):
    """Start the torch worker and wait until its models are loaded. Returns (client, ready event)."""
    from worker_client import WorkerClient, WorkerError

    uv = shutil.which("uv")
    if not uv:
        sys.exit("error: `uv` is not on PATH; it starts the torch worker for --reader minicpm and --judge "
                 f"(install uv; {WORKER_FREE})")
    if problem := worker_env_problem():
        sys.exit(f"error: {problem}")
    client = WorkerClient(worker_command(spec, models_dir, uv), env=worker_environment())
    try:
        return client, client.start()
    except WorkerError as e:
        client.close()
        sys.exit(f"error: {e}")


def engine_models(args) -> dict:
    """The model names of the JSON `engine` section (the Layout model, the Reader, the Judge when it is on), known
    from the options alone. The "already done" check of a folder Run compares these. The worker's model folders
    (worker/serve.py) are these same names; tests/test_options_only.py keeps the two apart from drifting."""
    return {"layout_model": LAYOUT_MODEL_DIR, "reader": READER_MODEL_DIRS[args.reader],
            **({"judge": JUDGE_MODEL_DIR} if args.judge else {})}


def run_fingerprint(args) -> document_record.Fingerprint:
    return document_record.fingerprint(args, engine_models(args), reader_bridge.reader_supports(args.reader))


def engine_info(args, device: str, ready: dict | None) -> dict:
    """The JSON `engine` section: the Layout model, the Reader and the Judge, with the library versions."""
    models = engine_models(args)
    engine = {"layout_model": models["layout_model"], "reader": models["reader"]}
    if args.reader == "paddle":
        engine.update(pipeline="PaddleOCR-VL-1.6", vl_model=VL_MODEL_DIR)
    engine.update(paddleocr=importlib.metadata.version("paddleocr"), paddlex=importlib.metadata.version("paddlex"))
    if "judge" in models:
        engine["judge"] = models["judge"]
    if ready:
        engine.update(ready["versions"])
    engine["device"] = device
    return engine


def build_pipeline(spec: EngineSpec, args, device: str):
    from paddleocr import PaddleOCRVL

    if not args.verbose:  # must run after the import: PaddleX configures its logger on import
        from paddlex.utils import logging as pdx_logging

        pdx_logging.setup_logging("WARNING")

    if spec.reader == "paddle":
        reader_args = {"vl_rec_model_dir": str(args.models_dir / VL_MODEL_DIR)}
    else:
        # The Reader is replaced by the worker (reader_bridge). Naming a server backend with a URL that is
        # never contacted keeps PaddleOCR-VL's weights, ~7 GB of GPU memory, from being loaded.
        reader_args = {"vl_rec_backend": "vllm-server", "vl_rec_server_url": "http://127.0.0.1:9/v1"}
    return PaddleOCRVL(
        pipeline_version="v1.6",
        layout_detection_model_dir=str(args.models_dir / LAYOUT_MODEL_DIR),
        **reader_args,
        device=device,
        layout_threshold=spec.layout_threshold,
        use_layout_detection=True,
        use_chart_recognition=spec.charts,
        use_seal_recognition=spec.seals,
        use_ocr_for_image_block=spec.ocr_images,
        merge_layout_blocks=spec.merge_blocks,
        markdown_ignore_labels=list(args.md_ignore_labels),
        format_block_content=False,  # keep the raw recognised text in the JSON
        # The orientation/unwarping models are not part of ./models, so they stay off.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )


def judge_setup_problems() -> list[str]:
    """What the Judge's renderer needs from the machine (node, Noto Sans CJK, the bundled font), checked before
    any model is loaded. Chromium is checked by the worker, which has Playwright."""
    sys.path.insert(0, str(WORKER_DIR))
    try:
        import render_env
    finally:
        sys.path.remove(str(WORKER_DIR))
    return render_env.setup_problems()


def attach_worker(spec: EngineSpec, models_dir: Path, pipe, stack, ink):
    """Start the torch worker and make it (MiniCPM) or the Judge-only bridge (--reader paddle) the pipeline's
    VL model. Returns (bridge, worker ready event); the worker and the crop folder are released when `stack` closes."""
    from paddlex.inference.pipelines.paddleocr_vl.uilts import convert_otsl_to_html

    from script_repair import ScriptRepair, StickyScriptRepair

    try:
        reader_bridge.check_hook_support(pipe)
    except reader_bridge.HookUnsupported as e:
        sys.exit(f"error: {e} The hook is used by --reader minicpm and --judge; {WORKER_FREE}.")
    tmp_dir = Path(tempfile.mkdtemp(prefix="pdf_ocr_"))
    stack.callback(shutil.rmtree, tmp_dir, ignore_errors=True)
    judge = spec.judge
    repair = StickyScriptRepair(ScriptRepair()) if judge and spec.script == "auto" else None
    paddle_model = reader_bridge.original_vl_model(pipe) if spec.reader == "paddle" else None
    if paddle_model is not None:
        free_paddle_cache()  # the pipeline's construction left several GB cached; the Judge needs the room
    t0 = time.time()
    print("starting the torch worker ...", file=sys.stderr)
    client, ready = start_worker(spec, models_dir)
    stack.callback(client.close)
    print(f"worker ready (torch {ready['versions'].get('torch')}, transformers "
          f"{ready['versions'].get('transformers')}): {time.time() - t0:.1f}s", file=sys.stderr)
    common = dict(to_html=convert_otsl_to_html, judge=judge, repair=repair,
                  **({"judge_threshold": spec.judge_threshold} if judge else {}))
    if paddle_model is not None:
        bridge = reader_bridge.PaddleBridge(
            paddle_model, client, tmp_dir, before_read=client.park if judge else None,
            after_read=free_paddle_cache, **common)
    else:
        bridge = reader_bridge.MiniCPMBridge(client, tmp_dir, **common)
        if spec.ink_reread:
            bridge.ink_reread = ink.reread_with(bridge, spec)
    reader_bridge.install_reader_hook(pipe, bridge)
    return bridge, ready


def free_paddle_cache() -> None:
    """Hand paddle's cached but unused GPU memory back (the allocator keeps it otherwise)."""
    import paddle

    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()


TEXT_LABELS = {"text", "paragraph_title", "doc_title", "abstract", "content", "reference",
               "reference_content", "figure_title", "footnote", "vision_footnote", "aside_text"}
SUSPECT_RATIO = 0.35     # recognised chars / ink-estimated glyphs below this => likely truncated output
MIN_GLYPHS = 8           # ignore tiny blocks, the estimate is too coarse there
RETRY_SCALES = (1.25, 0.8)


def count_chars(text: str) -> int:
    return len(re.sub(r"\s", "", re.sub(r"<[^>]+>", "", text)))


def glyph_estimate(crop) -> float:
    """Roughly how many CJK-sized glyphs the ink in a text crop can hold (sum of line width / line height).

    Latin text packs ~2x more characters per line, so a *short* result is only ever flagged for
    output that is far below even this conservative count.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if ink.mean() > 0.5:  # light text on a dark background
        ink = 1 - ink
    has_ink = ink.mean(axis=1) > 0.004
    runs, start = [], None
    for y, v in enumerate(has_ink):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append([start, y])
            start = None
    if start is not None:
        runs.append([start, len(has_ink)])
    if not runs:
        return 0.0
    ref = max(b - a for a, b in runs)  # tallest ink run ~ one text line
    lines = [runs[0]]
    for a, b in runs[1:]:
        if a - lines[-1][1] < 0.2 * ref:
            lines[-1][1] = b
        else:
            lines.append([a, b])
    ref = max(b - a for a, b in lines)
    total = 0.0
    for a, b in lines:
        if b - a < 0.5 * ref:  # slivers of neighbouring lines or rules
            continue
        cols = np.where(ink[a:b].any(axis=0))[0]
        if len(cols):
            total += (cols[-1] - cols[0] + 1) / (b - a)
    return total


def ink_reread_block(crop, text: str, reread_at_scale) -> str | None:
    """The ink-based re-read of one Block: if `text` is implausibly short for the ink in `crop`, read the Block
    again from differently scaled renders (`reread_at_scale(factor)` returns that Reading) and return the
    replacement if it is clearly longer and plausible, else None."""
    est, n = glyph_estimate(crop), count_chars(text)
    if est < MIN_GLYPHS or n >= SUSPECT_RATIO * est:
        return None
    best_text, best_n = None, n
    for factor in RETRY_SCALES:
        reread = reread_at_scale(factor)
        m = count_chars(reread)
        if best_n < m <= 1.6 * est:
            best_text, best_n = reread, m
    return best_text if best_text is not None and best_n > 1.5 * n else None


def recover_short_blocks(pipe, page, res, img, scale: float, args, page_no: int) -> list[dict]:
    """The VLM occasionally emits a few characters for a whole paragraph. Re-read such blocks from
    differently scaled renders and keep the result only if it is clearly longer and plausible. When the re-read of a
    Block fails (the GPU ran out of memory, say) the Block keeps its first Reading and the page goes on, as in the
    bridge (`page_no` is the page's number from 1, for the warning).
    (PaddleOCR-VL without --judge; with the worker the same re-read runs in the bridge, see InkReread.)"""
    notes: list[dict] = []
    renders = PageRenders(page, img, scale, args.dpi)
    for b in res["parsing_res_list"]:
        if b.label not in TEXT_LABELS:
            continue
        bbox = tuple(int(v) for v in b.bbox)
        crop = renders.original(bbox)
        if crop.size == 0:
            continue

        def reread_at_scale(factor, bbox=bbox):
            r = renders.region(bbox, factor)
            out = pipe.predict(r, use_layout_detection=False, prompt_label="ocr",
                               max_new_tokens=args.max_new_tokens, use_queues=False)[0]
            return "\n".join(blk.content for blk in out["parsing_res_list"]).strip()

        try:
            better = ink_reread_block(crop, b.content, reread_at_scale)
        except Exception as e:
            print(f"warning: page {page_no}: the re-read of block {bbox} failed, "
                  f"keeping its first reading: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if better is not None:
            notes.append({"bbox": list(bbox), "original": b.content})
            b.content = better
    return notes


class PageRenders:
    """The current page for the re-read of a Block at another scale: its own crops and crops of a re-render at
    `dpi * factor` (rendered once per factor and page)."""

    def __init__(self, page, img, scale: float, dpi: float):
        self.page, self.img, self.scale, self.dpi = page, img, scale, dpi
        self._renders: dict[float, tuple] = {}

    def original(self, bbox):
        x1, y1, x2, y2 = bbox
        return self.img[max(y1, 0):y2, max(x1, 0):x2]

    def _render(self, factor: float):
        """(the page rendered at `dpi * factor`, how many times larger than the page's own render that really is:
        the render size is capped, so this can be less than `factor`)."""
        if factor not in self._renders:
            self._renders[factor] = render_page(self.page, self.dpi * factor)
        img2, scale2 = self._renders[factor]
        return img2, scale2 / self.scale

    def crop(self, bbox, factor: float):
        """The Block's box cut from the page rendered `factor` times larger, or None if the render could not be made
        that much larger (the render size is capped)."""
        img2, k = self._render(factor)
        if abs(k - factor) > 0.01:
            return None
        x1, y1, x2, y2 = (round(v * k) for v in bbox)
        return img2[max(y1, 0):y2, max(x1, 0):x2]

    def region(self, bbox, factor: float, pad: int = 4):
        """The Block's box plus `pad` pixels of margin cut from the page rendered `factor` times larger (as large as
        the cap allows), for the ink-based re-read of a short Block, which reads whatever it gets."""
        img2, k = self._render(factor)
        x1, y1, x2, y2 = bbox
        return img2[max(int(y1 * k) - pad, 0):int(y2 * k) + pad, max(int(x1 * k) - pad, 0):int(x2 * k) + pad]


class InkReread:
    """The ink-based re-read inside the bridge (--reader minicpm without --judge): the bridge hands over each Reading;
    the page it belongs to is set here before the pipeline runs. Re-reads go straight to the Reader."""

    def __init__(self):
        self.renders: PageRenders | None = None

    def set_page(self, renders: PageRenders) -> None:
        self.renders = renders

    def reset(self) -> None:
        self.renders = None

    def reread_with(self, bridge, spec: EngineSpec):
        def ink_reread(request, text):
            if self.renders is None or request.label not in TEXT_LABELS or request.bbox is None:
                return None
            renders, bbox = self.renders, tuple(request.bbox)
            crop = renders.original(bbox)
            if crop.size == 0:
                return None

            def reread_at_scale(factor):
                return bridge.read_crop(renders.region(bbox, factor), "text", spec.max_new_tokens).strip()

            return ink_reread_block(crop, text, reread_at_scale)

        return ink_reread


def fix_table_line_breaks(results: list[tuple[int, object]]) -> int:
    """The model writes an in-cell line break as a literal backslash + n; turn it into <br>."""
    fixed = 0
    for _, r in results:
        for b in r["parsing_res_list"]:
            if b.label == "table" and "\\n" in b.content:
                b.content = b.content.replace("\\n", "<br>")
                fixed += 1
    return fixed


def unwrap_block_symbols(results: list[tuple[int, object]]) -> int:
    """Write the lone LaTeX symbols of every Block as their character (minicpm_reader.unwrap_symbols); returns how
    many were written. Like fix_table_line_breaks, this is the clean-up step the pipeline's own post-processing
    lacks: without a bridge (PaddleOCR-VL without --judge, the default) the Readings only get it here, re-reads
    included. A bridge
    has already unwrapped its text, so with one this finds nothing."""
    written = 0
    for _, r in results:
        for b in r["parsing_res_list"]:
            text, n = unwrap_symbols(b.content, b.label)
            if n:
                b.content = text
                written += n
    return written


def contiguous_groups(results: list[tuple[int, object]]):
    """Split [(page_idx, result)] into runs of consecutive pages (tables only merge within a run)."""
    group: list = []
    for item in results:
        if group and item[0] != group[-1][0] + 1:
            yield group
            group = []
        group.append(item)
    if group:
        yield group


def page_json(idx: int, res, meta: dict, raw: dict, ignored: list[str], outcomes: dict | None = None) -> dict:
    d = res.json["res"]
    outcomes = outcomes or {}
    reread = {tuple(n["bbox"]): n["original"] for n in meta["recovered"]}
    blocks = []
    for b in d["parsing_res_list"]:
        block = {
            "id": b["block_id"],
            "order": b["block_order"],
            "label": b["block_label"],
            "content": b["block_content"],
            "bbox": b["block_bbox"],
            "polygon": b.get("block_polygon_points"),
            "in_markdown": b["block_label"] not in ignored,
        }
        if (idx, b["block_id"]) in raw:
            block["content_raw"] = raw[(idx, b["block_id"])]
        bbox = tuple(int(v) for v in b["block_bbox"])
        if bbox in reread:
            block["content_before_reread"] = reread[bbox]
        outcome = outcomes.get((idx, bbox))
        if outcome is not None:
            add_judge_fields(block, outcome)
        blocks.append(block)
    detections = [
        {"label": x["label"], "score": round(float(x["score"]), 4), "bbox": x["coordinate"]}
        for x in d.get("layout_det_res", {}).get("boxes", [])
    ]
    return {
        "page": idx + 1,
        "width": d["width"],
        "height": d["height"],
        "scale_px_per_pt": round(meta["scale"], 6),
        "blocks": blocks,
        "layout_detections": detections,
    }


def blocks_without_outcome(idx: int, res, outcomes: dict) -> list[tuple[str, tuple]]:
    """(label, box) of the Blocks of a page that should have been judged but have no outcome, which the outcomes
    (keyed by page and box) cannot match to the page's Blocks. Expected to be empty."""
    missing = []
    for b in res["parsing_res_list"]:
        bbox = tuple(int(v) for v in b.bbox)
        if block_reader.judge_category(b.label) is not None and (idx, bbox) not in outcomes:
            missing.append((b.label, bbox))
    return missing


def add_judge_fields(block: dict, outcome) -> None:
    """The Judge's verdict on a Block (nothing for a Block that is never judged: charts, seals, figures)."""
    if outcome.text_before_reread is not None:
        block["content_before_reread"] = outcome.text_before_reread
    if not outcome.judged:
        return
    block["score"] = None if outcome.score is None else round(outcome.score, 6)
    block["verdict"] = outcome.verdict
    block["flagged"] = outcome.flagged
    if outcome.verdict_detail and outcome.verdict != "scored":
        block["verdict_detail"] = outcome.verdict_detail
    if outcome.judged_before_script_decision:
        block["judged_before_script_decision"] = True  # judged before the document was known to be Traditional
    if outcome.text != block["content"]:
        block["judged_text"] = outcome.text  # the Score describes this text, not the one written above
    if outcome.readings is not None:  # the Block was re-read: content/score/verdict above are the winning Reading's
        block["stop_reason"] = outcome.stop_reason
        if outcome.stop_detail:
            block["stop_detail"] = outcome.stop_detail
        block["readings"] = [reading_json(r) for r in outcome.readings]
        if outcome.skipped_rounds:
            block["skipped_rounds"] = outcome.skipped_rounds


def reading_json(r) -> dict:
    """One Reading of a re-read Block: its text, Score and verdict, and how it was produced."""
    d = {"content": r.text, "score": None if r.score is None else round(r.score, 6), "verdict": r.verdict,
         "round": r.round, "decode": r.decode, "temperature": r.temperature, "scale": r.scale}
    if r.scale_source:
        d["scale_source"] = r.scale_source  # a re-render of the page, or the Block's crop resampled
    if r.detail and r.verdict != "scored":
        d["verdict_detail"] = r.detail
    if r.duplicates:
        d["duplicates"] = r.duplicates  # later Readings with this very text, not scored again
    return d


def undo_sticky_repair(ok: list[tuple[int, object]], outcomes: dict) -> int:
    """Put back the text the sticky repair changed while reading, so that repair_script (the document-level
    pass) decides on the whole document alone and the output text stays what it was before the Judge existed.
    A Block the pipeline changed after the bridge returned it is left alone."""
    undone = 0
    for idx, r in ok:
        for b in r["parsing_res_list"]:
            o = outcomes.get((idx, tuple(int(v) for v in b.bbox)))
            if o is not None and o.text_before_repair is not None and b.content == o.text:
                b.content = o.text_before_repair
                undone += 1
    return undone


def judge_summary(outcomes: dict) -> dict:
    """The JSON `postprocessing.judge` section: how many Blocks were judged, re-read and flagged, and with which
    verdicts."""
    judged = [o for o in outcomes.values() if o.judged]
    verdicts: dict[str, int] = {}
    for o in judged:
        verdicts[o.verdict] = verdicts.get(o.verdict, 0) + 1
    reread = [o for o in judged if o.readings is not None]
    stops: dict[str, int] = {}
    for o in reread:
        stops[o.stop_reason] = stops.get(o.stop_reason, 0) + 1
    return {"blocks_judged": len(judged),
            "blocks_reread": len(reread),
            "blocks_flagged": sum(1 for o in judged if o.flagged),
            "reread_stop_reasons": dict(sorted(stops.items())),
            "readings_beyond_the_first": sum(sum(1 + r.duplicates for r in o.readings) - 1 for o in reread),
            "verdicts": dict(sorted(verdicts.items())),
            "blocks_judged_before_script_decision": sum(1 for o in judged if o.judged_before_script_decision)}


def memory_hint(args, errors: dict, outcomes: dict) -> str | None:
    """One line of advice if a page failed, or the Judge could not score a Block, because the GPU ran out of memory."""
    texts = [*errors.values(), *(f"{o.verdict_detail or ''} {o.stop_detail or ''}" for o in outcomes.values())]
    if not any("out of memory" in t.lower() for t in texts):
        return None
    return f"hint: the GPU ran out of memory: {MEMORY_WAYS_OUT[args.reader, args.judge]}"


def page_line(n: int, total: int, page_no: int, n_blocks: int, seconds: float, *, summary: dict | None,
              recovered: int) -> str:
    """One progress line on stderr."""
    line = f"[{n:>3}/{total}] page {page_no}: {n_blocks} blocks, {seconds:.1f}s"
    if summary is not None:
        line += (f", {summary['flagged']} of {summary['judged']} judged flagged, "
                 f"{summary['reread']} re-read")
        if summary["not_scored"]:
            line += f" ({summary['not_scored']} not scored)"
    if recovered:
        line += f", re-read {recovered} short block(s)"
    return line


def page_markdown(idx: int, res, out_dir: Path, markers: bool) -> str:
    md = res._to_markdown(pretty=False, show_formula_number=True)
    text = md["markdown_texts"]
    # Image names are derived from bbox only, so they collide across pages: prefix with the page.
    for rel, image in (md.get("markdown_images") or {}).items():
        if image is None:
            continue
        new_rel = f"{document_record.IMGS}/p{idx + 1:03d}_{Path(rel).name}"
        (out_dir / document_record.IMGS).mkdir(exist_ok=True)
        image.save(out_dir / new_rel)
        text = text.replace(rel, new_rel)
    return (f"<!-- page {idx + 1} -->\n\n" if markers else "") + text.strip()


# --------------------------------------------------------------------------- CSV export

CSV_NAME = re.compile(r"p\d{3}_table\d{2}\.csv")
MAX_SPAN = 1000  # a larger colspan/rowspan is a recognition error, not a real table


class _TableParser(HTMLParser):
    """Collect the cells of one HTML table as rows of (text, rowspan, colspan); raise ValueError if malformed."""

    def __init__(self):
        super().__init__(convert_charrefs=True)  # also decodes entities such as &#x27;
        self.rows: list[list[tuple[str, int, int]]] = []
        self.depth = 0
        self.tables = 0
        self.cell: list[str] | None = None
        self.spans = (1, 1)

    def _close_cell(self):
        if self.cell is not None:
            lines = (" ".join(line.split()) for line in "".join(self.cell).split("\0"))  # \0 marks <br>
            self.rows[-1].append(("\n".join(lines).strip("\n"), *self.spans))
            self.cell = None

    def _span(self, attrs: dict, name: str) -> int:
        v = attrs.get(name) or "1"
        if not v.strip().isdigit() or not 1 <= int(v) <= MAX_SPAN:
            raise ValueError(f"bad {name}={v!r}")
        return int(v)

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.depth or self.tables:
                raise ValueError("nested or multiple tables")
            self.depth, self.tables = 1, 1
        elif not self.depth:
            raise ValueError(f"<{tag}> outside the table")
        elif tag == "tr":
            self._close_cell()  # end tags may be omitted, as in browsers
            self.rows.append([])
        elif tag in ("td", "th"):
            self._close_cell()
            if not self.rows:
                self.rows.append([])
            a = dict(attrs)
            self.spans = (self._span(a, "rowspan"), self._span(a, "colspan"))
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append("\0")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._close_cell()
        elif tag == "tr":
            self._close_cell()
        elif tag == "table":
            self._close_cell()
            self.depth = 0

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        elif data.strip():
            raise ValueError(f"text outside a cell: {data.strip()[:30]!r}" if self.tables
                             else "not an HTML table")


def table_to_grid(html: str) -> list[list[str]]:
    """Turn an HTML table into a rectangular grid, expanding row/col spans by repeating the cell text.

    Raises ValueError when the markup does not describe a valid table (no table, text outside cells,
    rows of different widths, spans running past the table edge, ...).
    """
    p = _TableParser()
    p.feed(html)
    p.close()
    if not p.tables:
        raise ValueError("no <table> element")
    if p.depth:
        raise ValueError("unclosed <table>")
    if not any(p.rows):
        raise ValueError("table has no cells")
    grid: dict[tuple[int, int], str] = {}
    n_rows = len(p.rows)
    for r, row in enumerate(p.rows):
        c = 0
        for text, rs, cs in row:
            while (r, c) in grid:
                c += 1
            if r + rs > n_rows:
                raise ValueError(f"rowspan in row {r + 1} runs past the last row")
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = text
            c += cs
    widths = [max((c + 1 for rr, c in grid if rr == r), default=0) for r in range(n_rows)]
    if len(set(widths)) != 1:
        raise ValueError(f"rows have different numbers of columns: {sorted(set(widths))}")
    if len(grid) != n_rows * widths[0]:
        raise ValueError("table has holes (cells missing inside a row)")
    return [[grid[(r, c)] for c in range(widths[0])] for r in range(n_rows)]


FILLER_CELLS = {"", "-", "--", "—", "–"}


def unwrap_panels(grid: list[list[str]]) -> tuple[list[list[str]], int]:
    """Stack a long table that was typeset as side-by-side panels back into one table.

    Such a page repeats its header (年度 | 係數 | 年度 | 係數 | 年度 | 係數). As a CSV that header has
    duplicate column names, and readers that key rows by name (csv.DictReader, most converters and
    importers) keep only the last panel. Returns (grid, panel count); the grid is returned unchanged
    unless the first row is an exact repeat of a header whose cells are all different. Filler rows
    (empty or dashes) that pad the end of a panel are dropped.
    """
    header, n = grid[0], len(grid[0])
    for k in range(2, n // 2 + 1):
        panel = header[:k]
        if n % k == 0 and header == panel * (n // k) and len(set(panel)) == k and any(panel):
            break
    else:
        return grid, 1
    stacked = [panel]
    for p in range(n // k):
        rows = [row[p * k:(p + 1) * k] for row in grid[1:]]
        while rows and all(c in FILLER_CELLS for c in rows[-1]):
            rows.pop()
        stacked.extend(rows)
    return stacked, n // k


def export_csv_tables(restructured: list[tuple[int, object]], pages: dict, csv_dir: Path) -> dict:
    """Write each valid table block as csv/pNNN_tableKK.csv (UTF-8 with BOM, so Excel reads CJK text).

    Uses the same results as the Markdown, so a table merged across a page break becomes one CSV, and a
    table printed as side-by-side panels is stacked into one (see unwrap_panels).
    Returns the summary stored in the JSON.
    """
    if csv_dir.is_dir():  # drop CSVs of an earlier run into the same folder; other files are left alone
        for f in csv_dir.iterdir():
            if CSV_NAME.fullmatch(f.name):
                f.unlink()
    exported, skipped = [], []
    last = None  # entry of the latest table, which a continuation on the next page was merged into
    for idx, res in restructured:
        k = 0
        for i, b in enumerate(res["parsing_res_list"]):
            if b.label != "table":
                continue
            if not b.content.strip():
                page_blocks = pages.get(idx, {}).get("blocks", [])
                if last is not None and i < len(page_blocks) and page_blocks[i]["content"].strip():
                    last["pages"].append(idx + 1)  # emptied by the cross-page merge
                continue
            k += 1
            entry = {"page": idx + 1, "block_id": i, "pages": [idx + 1]}
            last = entry
            try:
                grid = table_to_grid(b.content)
            except ValueError as e:
                skipped.append({**entry, "reason": str(e)})
                last = skipped[-1]
                continue
            grid, panels = unwrap_panels(grid)
            name = f"p{idx + 1:03d}_table{k:02d}.csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            with open(csv_dir / name, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerows(grid)
            entry.update(file=f"{csv_dir.name}/{name}", rows=len(grid), columns=len(grid[0]))
            if panels > 1:
                entry["panels"] = panels
            exported.append(entry)
    for s in skipped:
        print(f"warning: table on page {s['page']} (block {s['block_id']}) not exported to CSV: {s['reason']}",
              file=sys.stderr)
    return {"exported": exported, "skipped": skipped}


@dataclass(frozen=True)
class PlannedDocument:
    """One Document of a folder Run: the PDF as found under the folder (a symlink stays a symlink), its path relative
    to the folder, and the folder its files go to. `done` says whether it is Already done
    (`document_record.why_not_done` has the rules); if not, `reason` says why it has to be processed."""
    path: Path
    relative: Path
    out_dir: Path
    done: bool = False
    reason: str = ""

    @property
    def relative_posix(self) -> str:
        return self.relative.as_posix()


class PlanError(Exception):
    """A folder Run that cannot start: it has no Documents, a folder cannot be read, or Documents would write over
    each other."""


def find_pdfs(folder: Path) -> list[Path]:
    """The paths, relative to `folder`, of every Document under it, in order: a file whose name ends in .pdf in any
    case. Names starting with "." are skipped at any depth. A symlinked file counts and a symlinked folder is not
    entered, so a link cannot make the search endless or reach outside the tree."""
    def unreadable(e: OSError):
        raise PlanError(f"cannot read {e.filename}: {e.strerror}")

    found = []
    for dirpath, dirnames, filenames in os.walk(folder, onerror=unreadable):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        found += [Path(dirpath, f).relative_to(folder) for f in filenames
                  if f.lower().endswith(".pdf") and not f.startswith(".")]
    return sorted(found, key=lambda rel: rel.parts)


def refuse_collisions(documents: list[PlannedDocument]) -> None:
    """Raise a PlanError naming the Documents that would write over each other: two with one output folder, or one
    whose output folder lies inside another's imgs/ or csv/."""
    by_dir: dict[Path, list[PlannedDocument]] = {}
    for d in documents:
        by_dir.setdefault(d.out_dir, []).append(d)
    problems = [f"{', '.join(d.relative_posix for d in same)} all write to {out_dir}"
                for out_dir, same in by_dir.items() if len(same) > 1]
    for d in documents:
        for folder in (d.out_dir, *d.out_dir.parents):
            if folder.name in (document_record.IMGS, document_record.CSV):
                problems += [f"{d.relative_posix} writes to {d.out_dir}, "
                             f"inside the {folder.name}/ of {outer.relative_posix}"
                             for outer in by_dir.get(folder.parent, [])]
    if problems:
        raise PlanError("the output folders of these Documents collide (nothing was written):\n  "
                        + "\n  ".join(problems))


def plan_run(folder: Path, output_root: Path | None, args) -> list[PlannedDocument]:
    """Plan a folder Run without loading, opening or writing anything: the Documents under `folder` in order, each
    with the folder it writes to and whether it is Already done (`document_record.why_not_done`, under one
    Fingerprint for the whole Run; never with --force). `output_root` is `-o`, or None for output/<name of the
    resolved folder>. Raises PlanError for a folder without Documents and for colliding output folders, those of
    Documents that are done included."""
    root = output_root if output_root is not None else Path("output") / folder.resolve().name
    documents = [PlannedDocument(folder / rel, rel, root / rel.parent / rel.stem) for rel in find_pdfs(folder)]
    if not documents:
        raise PlanError(f"no PDF found under {folder}")
    refuse_collisions(documents)
    if args.force:
        return [replace(d, reason="--force") for d in documents]
    fingerprint = run_fingerprint(args)
    reasons = [document_record.why_not_done(d.path, d.out_dir, fingerprint) for d in documents]
    return [replace(d, done=reason is None, reason=reason or "") for d, reason in zip(documents, reasons)]


class DocumentError(Exception):
    """A Document that cannot be transcribed at all: it does not open, has no pages, or --pages does not fit it."""


class Document:
    """An input PDF, opened: its pdfium document, its page count and the page indices to transcribe."""

    def __init__(self, path: Path, pdf, indices: list[int]):
        self.path, self.pdf, self.indices, self.total = path, pdf, indices, len(pdf)

    def close(self) -> None:
        self.pdf.close()


def open_document(path: Path, pages: str | None) -> Document:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        raise DocumentError(f"cannot open {path}: {e}") from None
    except FileNotFoundError:  # a symlink to nothing, in a folder Run
        raise DocumentError(f"cannot open {path}: no such file") from None
    except OSError as e:
        raise DocumentError(f"cannot open {path}: {e.strerror}") from None
    if len(pdf) == 0:
        raise DocumentError(f"{path} has no pages")
    try:
        return Document(path, pdf, parse_pages(pages, len(pdf)))
    except ValueError as e:
        raise DocumentError(str(e)) from None


def check_setup(spec: EngineSpec, models_dir: Path) -> None:
    """Exit with an error unless the models, opencc, the worker environment and the Judge's renderer that the Run
    needs are in place. Done once per Run, before anything is loaded."""
    if msg := missing_models_message(spec, models_dir):
        sys.exit(f"error: {msg}")
    if spec.script == "auto":
        try:
            importlib.import_module("opencc")
        except ImportError:
            sys.exit("error: --script auto (the default) repairs Simplified drift with opencc, which cannot be "
                     "imported. Install it with:  uv sync   (or run with --script off, which keeps the raw model "
                     "output)")
    if spec.needs_worker and (problem := worker_env_problem()):
        sys.exit(f"error: {problem}")
    if spec.judge and (problems := judge_setup_problems()):
        sys.exit("error: --judge needs the Judge's renderer, which is not set up (or run without --judge):\n  - "
                 + "\n  - ".join(problems))


class Engine:
    """What a Run loads once and every Document of it shares: the `spec` it was loaded from, the Layout model and the
    Reader (`pipe`, with the Reader hook installed), the `device`, and, when the spec needs the worker, its `ready`
    event and the `bridge`. `load_started` is when the loading began. The worker is released when the Run's ExitStack
    closes."""

    def __init__(self, spec: EngineSpec, pipe, device: str, bridge, ink: InkReread, ready: dict | None,
                 load_started: float | None = None):
        self.spec, self.pipe, self.device, self.bridge, self.ink, self.ready = spec, pipe, device, bridge, ink, ready
        self.load_started = load_started

    def begin_document(self, doc_hash: str) -> None:
        """Start a Document: forget everything that belongs to the previous one. `doc_hash` seeds the sampling of
        the re-read Rounds."""
        if self.bridge is not None:
            self.bridge.begin_document(doc_hash)  # Readings, Scores, Verdicts, counters, page, sticky script repair
        self.ink.reset()


def load_engine(spec: EngineSpec, args, stack) -> Engine:
    """Load the models once per Run (after check_setup): the Layout model, the worker, and the bridge with the
    Reader hook installed. The worker and the crop folder are released when `stack` closes."""
    device = pick_device(args.device)
    t_start = time.time()
    print(f"loading models from {args.models_dir} on {device} ...", file=sys.stderr)
    pipe = build_pipeline(spec, args, device)
    bridge = ready = None
    ink = InkReread()
    if spec.needs_worker:
        bridge, ready = attach_worker(spec, args.models_dir, pipe, stack, ink)
    print(f"model load: {time.time() - t_start:.1f}s", file=sys.stderr)
    return Engine(spec, pipe, device, bridge, ink, ready, t_start)


def transcribe_document(engine: Engine, args, document: Document, out_dir: Path, started: float | None = None) -> dict:
    """Transcribe one Document with a loaded Engine and write its Markdown, JSON and CSVs into `out_dir`.
    Returns the pages that failed ({page index: error}). `elapsed_seconds` counts from `started`: the Document's own
    start, or, for a single-PDF Run, the start of the model load."""
    t_start = time.time() if started is None else started
    pdf, indices, total = document.pdf, document.indices, document.total
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_hash = document_record.sha256(document.path)
    engine.begin_document(doc_hash)
    pipe, bridge, ink, device = engine.pipe, engine.bridge, engine.ink, engine.device

    ok: list[tuple[int, object]] = []
    meta: dict[int, dict] = {}
    errors: dict[int, str] = {}
    for n, idx in enumerate(indices, 1):
        t0 = time.time()
        page = pdf[idx]
        try:
            img, scale = render_page(page, args.dpi)
            if bridge is not None:
                bridge.current_page = idx
                bridge.page_source = PageRenders(page, img, scale, args.dpi)
                ink.set_page(bridge.page_source)
            res = pipe.predict(img, max_new_tokens=args.max_new_tokens, use_queues=False)[0]
            recovered = []
            if not args.judge and not args.no_recover and args.reader == "paddle":
                recovered = recover_short_blocks(pipe, page, res, img, scale, args, idx + 1)  # PaddleOCR-VL, as before
            n_reread = len(recovered) + (sum(1 for (p, _), o in bridge.outcomes.items()
                                             if p == idx and o.text_before_reread is not None) if bridge else 0)
            if bridge is not None and args.judge:
                for label, bbox in blocks_without_outcome(idx, res, bridge.outcomes):
                    print(f"warning: page {idx + 1}: the {label} Block at {bbox} has no Score (it was not read by the "
                          "Reader hook, or its box changed after reading)", file=sys.stderr)
            ok.append((idx, res))
            meta[idx] = {"scale": scale, "recovered": recovered}
            print(page_line(n, len(indices), idx + 1, len(res["parsing_res_list"]), time.time() - t0,
                            summary=bridge.page_summary(idx) if bridge and args.judge else None,
                            recovered=n_reread), file=sys.stderr)
        except Exception as e:  # keep going; the failure is recorded in the JSON and the exit code
            errors[idx] = f"{type(e).__name__}: {e}"
            print(f"[{n:>3}/{len(indices)}] page {idx + 1}: FAILED {errors[idx]}", file=sys.stderr)
        finally:
            page.close()

    outcomes = bridge.outcomes if bridge else {}

    # Block-level clean-up, then the JSON, which stays faithful to each page's own layout.
    undo_sticky_repair(ok, outcomes)  # the document-level repair below decides alone, as it always did
    tables_fixed = fix_table_line_breaks(ok)
    unwrapped = unwrap_block_symbols(ok)
    script_info, raw = repair_script(ok, args.script)
    pages = {idx: page_json(idx, r, meta[idx], raw, args.md_ignore_labels, outcomes) for idx, r in ok}

    # Cross-page restructuring is for the Markdown only: it moves the content of a table that
    # continues over a page break into one table and re-levels the headings.
    restructured: list[tuple[int, object]] = []
    restructuring_failed = None
    try:
        for group in contiguous_groups(ok):
            out = pipe.restructure_pages(
                [r for _, r in group], merge_tables=not args.no_merge_tables, relevel_titles=True,
                concatenate_pages=False,
            )
            restructured.extend(zip((i for i, _ in group), out))
    except Exception as e:  # the OCR result is too expensive to lose over a formatting step
        restructuring_failed = f"{type(e).__name__}: {e}"
        print(f"warning: cross-page restructuring failed ({restructuring_failed}); "
              "Markdown is written per page without merged tables or re-levelled titles", file=sys.stderr)
        restructured = ok
    md_parts = {idx: page_markdown(idx, r, out_dir, not args.no_page_markers) for idx, r in restructured}
    csv_info = None if args.no_csv else export_csv_tables(restructured, pages, out_dir / document_record.CSV)
    for idx, msg in errors.items():
        pages[idx] = {"page": idx + 1, "error": msg, "blocks": []}
        md_parts[idx] = f"<!-- page {idx + 1} failed: {msg} -->"

    md_path = document_record.markdown_path(document.path, out_dir)
    md_path.write_text("\n\n".join(md_parts[i] for i in sorted(md_parts)) + "\n", encoding="utf-8")
    tally = document_record.EngineTally()
    if bridge is not None:
        judge = ({**judge_summary(outcomes), "scaled_images_by_source": dict(sorted(bridge.scale_sources.items()))}
                 if args.judge else None)
        tally = document_record.EngineTally(bridge.repetition_cuts, bridge.symbols_unwrapped, judge)
    transcription = document_record.Transcription(
        sha256=doc_hash, page_count=total, pages_processed=[i + 1 for i in indices],
        pages_failed=[i + 1 for i in sorted(errors)], tables_with_line_breaks_fixed=tables_fixed,
        latex_symbols_unwrapped=unwrapped, script_repair=script_info, restructuring_failed=restructuring_failed,
        csv_tables=csv_info, elapsed_seconds=time.time() - t_start, pages=[pages[i] for i in sorted(pages)])
    json_path = document_record.write(document.path, out_dir, run_fingerprint(args),
                                      engine_info(args, device, engine.ready), tally, transcription)
    if hint := memory_hint(args, errors, outcomes):
        print(hint, file=sys.stderr)
    n_csv = f", {len(csv_info['exported'])} CSV table(s)" if csv_info else ""
    print(f"wrote {md_path} and {json_path}{n_csv} in {time.time() - t_start:.0f}s", file=sys.stderr)
    return errors


def transcribe(spec: EngineSpec, args) -> int:
    """A Run on one PDF: load the Engine, then transcribe the Document."""
    import contextlib

    if not args.pdf.is_file():
        sys.exit(f"error: no such file: {args.pdf}")
    check_setup(spec, args.models_dir)
    try:
        document = open_document(args.pdf, args.pages)
    except DocumentError as e:
        sys.exit(f"error: {e}")
    with contextlib.ExitStack() as stack:  # releases the worker and its crop folder when the Run ends, however it ends
        stack.callback(document.close)
        engine = load_engine(spec, args, stack)
        errors = transcribe_document(engine, args, document, args.output_dir or Path("output") / args.pdf.stem,
                                     started=engine.load_started)
    return 1 if errors else 0


def transcribe_folder(spec: EngineSpec, args) -> int:
    """A Run on every Document under a folder: plan it, and if any Document is not already done, check the setup, load
    the Engine once and transcribe those Documents one at a time. A Document that fails is reported and the Run goes
    on. A Run with nothing left to do touches no model. Documents are numbered by their place in the whole Run."""
    import contextlib
    import traceback

    try:
        plan = plan_run(args.pdf, args.output_dir, args)
    except PlanError as e:
        sys.exit(f"error: {e}")
    labels = [f"[{i}/{len(plan)}] {planned.relative_posix}" for i, planned in enumerate(plan, 1)]
    todo = [(label, planned) for label, planned in zip(labels, plan) if not planned.done]
    skipped = len(plan) - len(todo)
    print(f"found {len(plan)} Documents in {args.pdf} ({skipped} already done)", file=sys.stderr)
    for label, planned in zip(labels, plan):
        if planned.done:
            print(f"{label} skipped: already done", file=sys.stderr)
    written, failed, partly_failed = 0, [], []  # partly_failed: written, but with failed pages

    def fail(label: str, planned: PlannedDocument, reason: str) -> None:
        print(f"{label} FAILED: {reason}", file=sys.stderr)
        failed.append(f"{planned.relative_posix}: {reason}")

    with contextlib.ExitStack() as stack:
        if todo:
            check_setup(spec, args.models_dir)
            engine = load_engine(spec, args, stack)
        for label, planned in todo:
            document = None
            try:
                document = open_document(planned.path, None)
                print(f"{label} ({document.total} pages)", file=sys.stderr)
                errors = transcribe_document(engine, args, document, planned.out_dir)
            except DocumentError as e:  # the Run goes on; the next Run retries the Document, which has no JSON
                fail(label, planned, str(e))
            except Exception as e:
                traceback.print_exc()
                fail(label, planned, f"{type(e).__name__}: {e}")
            else:
                written += 1
                if errors:
                    pages = ", ".join(str(idx + 1) for idx in sorted(errors))
                    partly_failed.append(f"{planned.relative_posix}: {'page' if len(errors) == 1 else 'pages'} "
                                         f"{pages} failed (recorded in its JSON)")
            finally:
                if document is not None:
                    document.close()
    print(f"{len(plan)} Documents: {written} written, {skipped} skipped, {len(failed)} failed", file=sys.stderr)
    for line in failed + partly_failed:
        print(f"  {line}", file=sys.stderr)
    return 1 if failed or partly_failed else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = EngineSpec.from_options(args)
    # Skip model-hoster connectivity checks and quiet noisy third-party output before importing paddle.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if not args.verbose:
        warnings.filterwarnings("ignore", message=".*ccache.*")
        warnings.filterwarnings("ignore", message=".*copy construct from a tensor.*")
    return transcribe_folder(spec, args) if args.pdf.is_dir() else transcribe(spec, args)


if __name__ == "__main__":
    sys.exit(main())
