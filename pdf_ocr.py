#!/usr/bin/env python3
"""Transcribe a PDF to Markdown and JSON with PaddleOCR-VL-1.6, using the local models in ./models.

    python pdf_ocr.py input.pdf                  # -> output/<stem>/<stem>.md, <stem>.json, imgs/, csv/
    python pdf_ocr.py input.pdf -o out --pages 1-5,9

Pipeline: pypdfium2 renders each page -> PP-DocLayoutV3 finds and orders layout blocks ->
PaddleOCR-VL-1.6 recognises every block (text / table / formula; charts and seals on request) ->
block-level clean-up (Simplified/Traditional repair, truncated-block re-read) -> JSON per page, and
Markdown with cross-page tables merged and title levels fixed, plus one CSV per well-formed table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import warnings
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAYOUT_MODEL_DIR = "PP-DocLayoutV3"
VL_MODEL_DIR = "PaddleOCR-VL-1.6"

# Page furniture that is kept in the JSON but left out of the Markdown by default.
# (PaddleOCR's own default also drops `footnote`; that is real content, so it is kept here.)
DEFAULT_MD_IGNORE = ["header", "header_image", "footer", "footer_image", "number", "aside_text"]
MAX_RENDER_PIXELS = 40_000_000  # guard against huge pages exhausting memory at high DPI


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf_ocr",
        description="Transcribe a PDF to Markdown + JSON with PaddleOCR-VL-1.6 (local models).",
    )
    p.add_argument("pdf", type=Path, help="input PDF")
    p.add_argument("-o", "--output-dir", type=Path, help="output directory (default: output/<pdf stem>)")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models",
                   help=f"folder holding {LAYOUT_MODEL_DIR}/ and {VL_MODEL_DIR}/ (default: ./models)")
    p.add_argument("--pages", help="1-based pages to process, e.g. '1-3,7,10-' (default: all)")
    p.add_argument("--device", help="paddle device, e.g. gpu:0 or cpu (default: gpu:0 if available)")

    g = p.add_argument_group("accuracy")
    # Defaults below were tuned against the text layer of a 21-page Traditional Chinese PDF:
    # 216 vs 288 DPI were equivalent (216 is faster); layout threshold 0.3 -> 0.2 cut text lost to
    # missed layout blocks from ~85 to ~20 characters, while 0.1 starts producing false boxes.
    g.add_argument("--dpi", type=float, default=216.0, help="page render resolution (default: %(default)s)")
    g.add_argument("--layout-threshold", type=float, default=0.2,
                   help="layout detection score threshold; lower = fewer missed blocks, "
                        "too low = false boxes (default: %(default)s)")
    g.add_argument("--max-new-tokens", type=int, default=8192,
                   help="decoding limit per block; raise for very large tables (default: %(default)s)")
    g.add_argument("--charts", action="store_true",
                   help="convert charts to data tables. Off by default: values are estimated and the model "
                        "can invent data for figures it mistakes for charts; otherwise the chart is saved as an image")
    g.add_argument("--seals", action="store_true",
                   help="read text on seals/stamps (off by default; otherwise saved as an image)")
    g.add_argument("--ocr-images", action="store_true", help="also OCR text inside figure/image blocks")
    g.add_argument("--no-merge-blocks", action="store_true", help="do not merge adjacent text blocks before OCR")
    g.add_argument("--no-recover", action="store_true",
                   help="do not re-read blocks whose text is implausibly short for the ink they contain")
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
    p.add_argument("-v", "--verbose", action="store_true", help="show PaddleOCR/PaddleX logs")
    return p


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


# --------------------------------------------------------------------------- script repair

class ScriptRepair:
    """Repair Simplified-Chinese characters inside a Traditional-Chinese document.

    PaddleOCR-VL tends to drift to Simplified forms (银行, 医院, 细胞) when reading Traditional text.
    A character is treated as drift only if it cannot be encoded in Big5 (so it is not valid
    Traditional) and OpenCC maps it to something else. Valid Traditional characters that merely also
    exist in Simplified (后, 干, 里, 台, 表) are never touched.
    """

    def __init__(self):
        from opencc import OpenCC  # optional dependency

        def make(name):
            try:
                return OpenCC(name)
            except Exception:
                return OpenCC(name + ".json")

        self.s2t, self.t2s = make("s2t"), make("t2s")
        self._simp: dict[str, bool] = {}
        self._trad: dict[str, bool] = {}

    def is_simplified_only(self, c: str) -> bool:
        r = self._simp.get(c)
        if r is None:
            r = False
            if "一" <= c <= "鿿":
                try:
                    c.encode("big5")
                except UnicodeEncodeError:
                    r = self.s2t.convert(c) != c
            self._simp[c] = r
        return r

    def is_traditional_only(self, c: str) -> bool:
        r = self._trad.get(c)
        if r is None:
            r = "一" <= c <= "鿿" and not self.is_simplified_only(c) and self.t2s.convert(c) != c
            self._trad[c] = r
        return r

    def survey(self, texts) -> tuple[int, int]:
        simp = trad = 0
        for t in texts:
            for c in t:
                if self.is_simplified_only(c):
                    simp += 1
                elif self.is_traditional_only(c):
                    trad += 1
        return simp, trad

    def fix(self, text: str) -> str:
        if not any(self.is_simplified_only(c) for c in text):
            return text
        conv = self.s2t.convert(text)  # phrase-aware (e.g. 头发 -> 頭髮)
        if len(conv) == len(text):
            return "".join(conv[i] if self.is_simplified_only(c) else c for i, c in enumerate(text))
        return "".join(self.s2t.convert(c) if self.is_simplified_only(c) else c for c in text)


NO_REPAIR_LABELS = {"display_formula", "inline_formula", "formula_number", "algorithm"}


def repair_script(results: list[tuple[int, object]], mode: str) -> tuple[dict, dict]:
    """Fix Simplified drift in the block objects of `results` in place.

    Returns (summary for the JSON, {(page_idx, block_idx): original_text} for changed blocks).
    """
    info: dict = {"mode": mode, "applied": False}
    raw: dict = {}
    if mode == "off":
        return info, raw
    try:
        rep = ScriptRepair()
    except ImportError:
        print("note: opencc not installed, skipping Simplified->Traditional repair "
              "(uv add opencc-python-reimplemented)", file=sys.stderr)
        info["skipped"] = "opencc not installed"
        return info, raw

    blocks = [(idx, i, b) for idx, r in results for i, b in enumerate(r["parsing_res_list"])
              if b.label not in NO_REPAIR_LABELS]
    simp, trad = rep.survey(b.content for _, _, b in blocks)
    share = simp / (simp + trad) if simp + trad else 0.0
    # Only act on a clearly Traditional document; Simplified or mixed ones are left alone.
    is_trad = trad >= 30 and share <= 0.15
    info.update(simplified_chars=simp, traditional_chars=trad,
                document_script="traditional" if is_trad else "not traditional")
    if not is_trad or simp == 0:
        return info, raw
    for idx, i, b in blocks:
        fixed = rep.fix(b.content)
        if fixed != b.content:
            raw[(idx, i)] = b.content
            b.content = fixed
    info.update(applied=True, blocks_changed=len(raw))
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


def build_pipeline(args, device: str):
    from paddleocr import PaddleOCRVL

    if not args.verbose:  # must run after the import: PaddleX configures its logger on import
        from paddlex.utils import logging as pdx_logging

        pdx_logging.setup_logging("WARNING")

    return PaddleOCRVL(
        pipeline_version="v1.6",
        layout_detection_model_dir=str(args.models_dir / LAYOUT_MODEL_DIR),
        vl_rec_model_dir=str(args.models_dir / VL_MODEL_DIR),
        device=device,
        layout_threshold=args.layout_threshold,
        use_layout_detection=True,
        use_chart_recognition=args.charts,
        use_seal_recognition=args.seals,
        use_ocr_for_image_block=args.ocr_images,
        merge_layout_blocks=not args.no_merge_blocks,
        markdown_ignore_labels=list(args.md_ignore_labels),
        format_block_content=False,  # keep the raw recognised text in the JSON
        # The orientation/unwarping models are not part of ./models, so they stay off.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )


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


def recover_short_blocks(pipe, page, res, img, scale: float, args) -> list[dict]:
    """The VLM occasionally emits a few characters for a whole paragraph. Re-read such blocks from
    differently scaled renders and keep the result only if it is clearly longer and plausible."""
    notes: list[dict] = []
    renders: dict[float, tuple] = {}
    h_img, w_img = img.shape[:2]
    for b in res["parsing_res_list"]:
        if b.label not in TEXT_LABELS:
            continue
        x1, y1, x2, y2 = (int(v) for v in b.bbox)
        crop = img[max(y1, 0):min(y2, h_img), max(x1, 0):min(x2, w_img)]
        if crop.size == 0:
            continue
        est, n = glyph_estimate(crop), count_chars(b.content)
        if est < MIN_GLYPHS or n >= SUSPECT_RATIO * est:
            continue
        best_text, best_n = None, n
        for factor in RETRY_SCALES:
            if factor not in renders:
                renders[factor] = render_page(page, args.dpi * factor)
            img2, scale2 = renders[factor]
            k, pad = scale2 / scale, 4
            r = img2[max(int(y1 * k) - pad, 0):int(y2 * k) + pad, max(int(x1 * k) - pad, 0):int(x2 * k) + pad]
            out = pipe.predict(r, use_layout_detection=False, prompt_label="ocr",
                               max_new_tokens=args.max_new_tokens, use_queues=False)[0]
            text = "\n".join(blk.content for blk in out["parsing_res_list"]).strip()
            m = count_chars(text)
            if best_n < m <= 1.6 * est:
                best_text, best_n = text, m
        if best_text is not None and best_n > 1.5 * n:
            notes.append({"bbox": [x1, y1, x2, y2], "original": b.content})
            b.content = best_text
    return notes


def fix_table_line_breaks(results: list[tuple[int, object]]) -> int:
    """The model writes an in-cell line break as a literal backslash + n; turn it into <br>."""
    fixed = 0
    for _, r in results:
        for b in r["parsing_res_list"]:
            if b.label == "table" and "\\n" in b.content:
                b.content = b.content.replace("\\n", "<br>")
                fixed += 1
    return fixed


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


def page_json(idx: int, res, meta: dict, raw: dict, ignored: list[str]) -> dict:
    d = res.json["res"]
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
        if tuple(int(v) for v in b["block_bbox"]) in reread:
            block["content_before_reread"] = reread[tuple(int(v) for v in b["block_bbox"])]
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


def page_markdown(idx: int, res, out_dir: Path, markers: bool) -> str:
    md = res._to_markdown(pretty=False, show_formula_number=True)
    text = md["markdown_texts"]
    # Image names are derived from bbox only, so they collide across pages: prefix with the page.
    for rel, image in (md.get("markdown_images") or {}).items():
        if image is None:
            continue
        new_rel = f"imgs/p{idx + 1:03d}_{Path(rel).name}"
        (out_dir / "imgs").mkdir(exist_ok=True)
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


def export_csv_tables(restructured: list[tuple[int, object]], pages: dict, csv_dir: Path) -> dict:
    """Write each valid table block as csv/pNNN_tableKK.csv (UTF-8 with BOM, so Excel reads CJK text).

    Uses the same results as the Markdown, so a table merged across a page break becomes one CSV.
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
            name = f"p{idx + 1:03d}_table{k:02d}.csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            with open(csv_dir / name, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerows(grid)
            entry.update(file=f"{csv_dir.name}/{name}", rows=len(grid), columns=len(grid[0]))
            exported.append(entry)
    for s in skipped:
        print(f"warning: table on page {s['page']} (block {s['block_id']}) not exported to CSV: {s['reason']}",
              file=sys.stderr)
    return {"exported": exported, "skipped": skipped}


def transcribe(args) -> int:
    import pypdfium2 as pdfium

    if not args.pdf.is_file():
        sys.exit(f"error: no such file: {args.pdf}")
    for name, what in ((LAYOUT_MODEL_DIR, "layout model"), (VL_MODEL_DIR, "VL model")):
        if not (args.models_dir / name).is_dir():
            sys.exit(f"error: {what} not found at {args.models_dir / name}")
    try:
        pdf = pdfium.PdfDocument(str(args.pdf))
    except pdfium.PdfiumError as e:
        sys.exit(f"error: cannot open {args.pdf}: {e}")
    total = len(pdf)
    if total == 0:
        sys.exit(f"error: {args.pdf} has no pages")
    try:
        indices = parse_pages(args.pages, total)
    except ValueError as e:
        sys.exit(f"error: {e}")

    out_dir = args.output_dir or Path("output") / args.pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    t_start = time.time()
    print(f"loading models from {args.models_dir} on {device} ...", file=sys.stderr)
    pipe = build_pipeline(args, device)
    print(f"model load: {time.time() - t_start:.1f}s", file=sys.stderr)

    ok: list[tuple[int, object]] = []
    meta: dict[int, dict] = {}
    errors: dict[int, str] = {}
    for n, idx in enumerate(indices, 1):
        t0 = time.time()
        page = pdf[idx]
        try:
            img, scale = render_page(page, args.dpi)
            res = pipe.predict(img, max_new_tokens=args.max_new_tokens, use_queues=False)[0]
            recovered = [] if args.no_recover else recover_short_blocks(pipe, page, res, img, scale, args)
            ok.append((idx, res))
            meta[idx] = {"scale": scale, "recovered": recovered}
            print(f"[{n:>3}/{len(indices)}] page {idx + 1}: {len(res['parsing_res_list'])} blocks, "
                  f"{time.time() - t0:.1f}s" + (f", re-read {len(recovered)} short block(s)" if recovered else ""),
                  file=sys.stderr)
        except Exception as e:  # keep going; the failure is recorded in the JSON and the exit code
            errors[idx] = f"{type(e).__name__}: {e}"
            print(f"[{n:>3}/{len(indices)}] page {idx + 1}: FAILED {errors[idx]}", file=sys.stderr)
        finally:
            page.close()

    # Block-level clean-up, then the JSON, which stays faithful to each page's own layout.
    tables_fixed = fix_table_line_breaks(ok)
    script_info, raw = repair_script(ok, args.script)
    pages = {idx: page_json(idx, r, meta[idx], raw, args.md_ignore_labels) for idx, r in ok}

    # Cross-page restructuring is for the Markdown only: it moves the content of a table that
    # continues over a page break into one table and re-levels the headings.
    restructured: list[tuple[int, object]] = []
    try:
        for group in contiguous_groups(ok):
            out = pipe.restructure_pages(
                [r for _, r in group], merge_tables=not args.no_merge_tables, relevel_titles=True,
                concatenate_pages=False,
            )
            restructured.extend(zip((i for i, _ in group), out))
    except Exception as e:  # the OCR result is too expensive to lose over a formatting step
        print(f"warning: cross-page restructuring failed ({type(e).__name__}: {e}); "
              "Markdown is written per page without merged tables or re-levelled titles", file=sys.stderr)
        restructured = ok
    md_parts = {idx: page_markdown(idx, r, out_dir, not args.no_page_markers) for idx, r in restructured}
    csv_info = None if args.no_csv else export_csv_tables(restructured, pages, out_dir / "csv")
    for idx, msg in errors.items():
        pages[idx] = {"page": idx + 1, "error": msg, "blocks": []}
        md_parts[idx] = f"<!-- page {idx + 1} failed: {msg} -->"

    stem = args.pdf.stem
    md_path, json_path = out_dir / f"{stem}.md", out_dir / f"{stem}.json"
    md_path.write_text("\n\n".join(md_parts[i] for i in sorted(md_parts)) + "\n", encoding="utf-8")
    doc = {
        "source": {
            "file": args.pdf.name,
            "sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
            "page_count": total,
            "pages_processed": [i + 1 for i in indices],
            "pages_failed": [i + 1 for i in sorted(errors)],
        },
        "engine": {
            "pipeline": "PaddleOCR-VL-1.6",
            "layout_model": LAYOUT_MODEL_DIR,
            "vl_model": VL_MODEL_DIR,
            "paddleocr": importlib.metadata.version("paddleocr"),
            "paddlex": importlib.metadata.version("paddlex"),
            "device": device,
        },
        "settings": {
            "dpi": args.dpi,
            "layout_threshold": args.layout_threshold,
            "max_new_tokens": args.max_new_tokens,
            "chart_recognition": args.charts,
            "seal_recognition": args.seals,
            "ocr_for_image_blocks": args.ocr_images,
            "merge_layout_blocks": not args.no_merge_blocks,
            "reread_short_blocks": not args.no_recover,
            "markdown_merge_tables": not args.no_merge_tables,
            "markdown_ignore_labels": list(args.md_ignore_labels),
            "csv_export": not args.no_csv,
        },
        "postprocessing": {"tables_with_line_breaks_fixed": tables_fixed, "script_repair": script_info},
        "csv_tables": csv_info,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "pages": [pages[i] for i in sorted(pages)],
    }
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_csv = f", {len(csv_info['exported'])} CSV table(s)" if csv_info else ""
    print(f"wrote {md_path} and {json_path}{n_csv} in {time.time() - t_start:.0f}s", file=sys.stderr)
    return 1 if errors else 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Skip model-hoster connectivity checks and quiet noisy third-party output before importing paddle.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if not args.verbose:
        warnings.filterwarnings("ignore", message=".*ccache.*")
        warnings.filterwarnings("ignore", message=".*copy construct from a tensor.*")
    return transcribe(args)


if __name__ == "__main__":
    sys.exit(main())
