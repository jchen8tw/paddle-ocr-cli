"""A Document's record: where its files go, the JSON that describes it, and whether a later Run finds it Already done.

The writer of the JSON and the check that reads it back share one Fingerprint per Run (the `settings` section and the
model names this Run writes), so they cannot disagree. Plain Python: nothing from paddle, the bridge or the Engine.
The model names, the JSON `engine` section and the Engine's part of `postprocessing` come in as plain data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from block_reader import MAX_READINGS, SCHEDULE, ReadStep

IMGS = "imgs"  # a Document's figures, charts and seals
CSV = "csv"    # a Document's tables, one CSV each
MODEL_KEYS = ("layout_model", "reader", "judge")  # the model names in the JSON `engine` section


def markdown_path(pdf: Path, folder: Path) -> Path:
    return folder / f"{pdf.stem}.md"


def json_path(pdf: Path, folder: Path) -> Path:
    return folder / f"{pdf.stem}.json"


def sha256(pdf: Path) -> str:
    """The PDF's sha256: the JSON records it, the check compares it, and it seeds the sampling of the Re-read Rounds."""
    with open(pdf, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@dataclass(frozen=True)
class Fingerprint:
    """What a Run writes into every Document's JSON and compares an earlier JSON with: the `settings` section and the
    Engine's model names (`MODEL_KEYS`)."""
    settings: dict
    models: dict


def fingerprint(options, models: Mapping[str, str], supports: Callable[[ReadStep], bool]) -> Fingerprint:
    """The Fingerprint of a Run with these options and models. `settings` comes from the options alone; a re-read
    Round is `supported` when `supports`, the rule of the Reader that --reader names, says so."""
    settings = {
        "reader": options.reader,
        "judge": options.judge,
        "judge_threshold": options.judge_threshold,
        **({"reread_schedule": [{"round": s.round, "decode": s.decode, "temperature": s.temperature, "scale": s.scale,
                                 "samples": s.samples, "supported": bool(supports(s))} for s in SCHEDULE],
            "max_readings_per_block": MAX_READINGS} if options.judge else {}),
        "dpi": options.dpi,
        "layout_threshold": options.layout_threshold,
        "max_new_tokens": options.max_new_tokens,
        "chart_recognition": options.charts,
        "seal_recognition": options.seals,
        "ocr_for_image_blocks": options.ocr_images,
        "merge_layout_blocks": not options.no_merge_blocks,
        "reread_short_blocks": not options.judge and not options.no_recover,  # the ink-based one: never with the Judge
        "markdown_merge_tables": not options.no_merge_tables,
        "markdown_ignore_labels": list(options.md_ignore_labels),
        "script": options.script,
        "page_markers": not options.no_page_markers,
        "csv_export": not options.no_csv,
    }
    return Fingerprint(settings, {k: models[k] for k in MODEL_KEYS if k in models})


@dataclass(frozen=True)
class EngineTally:
    """The Engine's part of a Document's `postprocessing`: the repetition cuts (None without the worker), the lone
    LaTeX symbols it unwrapped, and the Judge's summary (None without the Judge)."""
    repetition_cuts: int | None = None
    latex_symbols_unwrapped: int = 0
    judge: dict | None = None


@dataclass(frozen=True)
class Transcription:
    """What transcribing one Document gave: the source facts (pages from 1), the Document's own post-processing,
    the CSV tables (None with --no-csv), the elapsed time and the JSON of each page."""
    sha256: str
    page_count: int
    pages_processed: list[int]
    pages_failed: list[int]
    tables_with_line_breaks_fixed: int
    latex_symbols_unwrapped: int
    script_repair: dict
    restructuring_failed: str | None
    csv_tables: dict | None
    elapsed_seconds: float
    pages: list[dict]


def write(pdf: Path, folder: Path, fingerprint: Fingerprint, engine: Mapping, tally: EngineTally,
          transcription: Transcription) -> Path:
    """Write the JSON of the Document `pdf` into `folder` and return its path. `engine` is the JSON `engine` section;
    its model names are the Fingerprint's, in the section's own order."""
    models = fingerprint.models
    record = {
        "source": {
            "file": pdf.name,
            "sha256": transcription.sha256,
            "page_count": transcription.page_count,
            "pages_processed": transcription.pages_processed,
            "pages_failed": transcription.pages_failed,
        },
        "engine": {k: v for k, v in engine.items() if k not in MODEL_KEYS or k in models} | models,
        "settings": fingerprint.settings,
        "postprocessing": {
            "tables_with_line_breaks_fixed": transcription.tables_with_line_breaks_fixed,
            **({"repetition_cuts": tally.repetition_cuts} if tally.repetition_cuts is not None else {}),
            "latex_symbols_unwrapped": transcription.latex_symbols_unwrapped + tally.latex_symbols_unwrapped,
            "script_repair": transcription.script_repair,
            **({"judge": tally.judge} if tally.judge is not None else {}),
            **({"restructuring_failed": transcription.restructuring_failed}
               if transcription.restructuring_failed else {}),
        },
        "csv_tables": transcription.csv_tables,
        "elapsed_seconds": round(transcription.elapsed_seconds, 1),
        "pages": transcription.pages,
    }
    path = json_path(pdf, folder)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def why_not_done(pdf: Path, folder: Path, fingerprint: Fingerprint) -> str | None:
    """None when the JSON an earlier Run left in `folder` for `pdf` reads cleanly and shows it is Already done: the PDF
    is the same (sha256), every page from 1 to `source.page_count` was transcribed and none failed, the script repair
    was not skipped (`postprocessing.script_repair.skipped`), and the `settings` and the Engine's model names
    (`MODEL_KEYS`) are the Fingerprint's. Otherwise the reason it has to be processed. A JSON that is missing, cut off
    or lacks a field is never an error. Library versions and the device are not compared, and a failed cross-page
    restructuring (`postprocessing.restructuring_failed`) is not read: the Document stays done, and --force redoes
    it."""
    try:
        existing = json.loads(json_path(pdf, folder).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "no output yet"
    except (OSError, ValueError) as e:
        return f"its JSON cannot be read ({type(e).__name__})"
    if not isinstance(existing, dict):
        return "its JSON is not an object"
    source, engine, written = existing.get("source"), existing.get("engine"), existing.get("settings")
    postprocessing = existing.get("postprocessing")
    repair = postprocessing.get("script_repair") if isinstance(postprocessing, dict) else None
    if not (isinstance(source, dict) and isinstance(source.get("sha256"), str)
            and isinstance(source.get("page_count"), int) and isinstance(source.get("pages_processed"), list)
            and isinstance(source.get("pages_failed"), list) and isinstance(engine, dict) and isinstance(written, dict)
            and isinstance(repair, dict)):
        return ("its JSON lacks source.sha256, source.page_count, source.pages_processed, source.pages_failed, engine, "
                "settings or postprocessing.script_repair")
    if source["pages_processed"] != list(range(1, source["page_count"] + 1)):
        return (f"only page(s) {', '.join(map(str, source['pages_processed']))} of {source['page_count']} were "
                "transcribed")
    if source["pages_failed"]:
        return f"page(s) {', '.join(map(str, source['pages_failed']))} failed last time"
    if "skipped" in repair:
        return f"its script repair was skipped last time ({repair['skipped']})"
    settings = fingerprint.settings
    if written != settings:
        keys = sorted(k for k in written.keys() | settings.keys() if written.get(k) != settings.get(k))
        return f"settings differ: {', '.join(keys)}"
    if differing := [k for k in MODEL_KEYS if engine.get(k) != fingerprint.models.get(k)]:
        return f"models differ: {', '.join(differing)}"
    try:
        if sha256(pdf) != source["sha256"]:
            return "the PDF changed"
    except OSError as e:
        return f"cannot read the PDF ({e.strerror})"
    return None
