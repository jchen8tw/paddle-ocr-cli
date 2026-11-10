"""Character-error-rate harness for pdf_ocr.py output, scored against the PDF's own text layer.

The PDF's characters are assigned to the layout Blocks by position (a character belongs to the
smallest Block whose box contains its centre), so the order of the Blocks never matters. Each Block's
Reading is then compared with the characters assigned to it by edit distance. Characters the layout
model never covered count as errors ("missed"). Blocks that are not read by default (figures, charts,
seals) are left out of the score ("skipped").

Standalone use, on a finished run (the PDF is needed again for its text layer):

    python ocr_eval.py input.pdf [output/<stem>/<stem>.json] [--report scores.json]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

# Blocks whose content is not comparable with the text layer, or not read unless asked for.
UNREAD_LABELS = frozenset({"image", "chart", "seal", "header_image", "footer_image"})
_TAG = re.compile(r"<[^>]*>")
_LEADER = re.compile(r"\.{3,}")  # table-of-contents dot leaders; their length is typesetting, not text


class TextChar(NamedTuple):
    ch: str
    x: float  # centre of the glyph box, PDF points, origin top-left
    y: float


@dataclass
class Tally:
    errors: int = 0
    ref_chars: int = 0

    @property
    def cer(self) -> float:
        return self.errors / self.ref_chars if self.ref_chars else 0.0

    def add(self, other: "Tally") -> None:
        self.errors += other.errors
        self.ref_chars += other.ref_chars


@dataclass
class PageScore(Tally):
    missed_chars: int = 0   # text-layer characters inside no Block (counted as errors)
    skipped_chars: int = 0  # text-layer characters inside unread Blocks (not scored)
    by_label: dict[str, Tally] = field(default_factory=dict)


@dataclass
class DocScore:
    total: Tally = field(default_factory=Tally)
    pages: dict[int, PageScore] = field(default_factory=dict)
    by_label: dict[str, Tally] = field(default_factory=dict)
    missed_chars: int = 0
    skipped_chars: int = 0
    skipped_pages: list[int] = field(default_factory=list)  # failed, or no text layer to compare with


def normalize(text: str, *, html_markup: bool = False) -> str:
    """Drop what the comparison should not depend on: markup, whitespace, full/half-width forms,
    dot leaders (a run of 3+ dots)."""
    if html_markup:
        text = html.unescape(_TAG.sub("", text))
    text = _LEADER.sub("", unicodedata.normalize("NFKC", text))
    return "".join(c for c in text if unicodedata.category(c)[0] not in "ZC")


def edit_distance(a: str, b: str) -> int:
    while a and b and a[0] == b[0]:
        a, b = a[1:], b[1:]
    while a and b and a[-1] == b[-1]:
        a, b = a[:-1], b[:-1]
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_page(blocks: list[dict], text_chars: list[TextChar], scale: float) -> PageScore:
    """Score one page. `blocks` are the JSON blocks (bbox in px); `scale` converts px to points."""
    boxes = []
    for i, b in enumerate(blocks):
        x0, y0, x1, y1 = (v / scale for v in b["bbox"])
        boxes.append((max(0.0, x1 - x0) * max(0.0, y1 - y0), (x0, y0, x1, y1), i))
    boxes.sort()  # smallest first, so the first box containing a character is the tightest one
    assigned: dict[int, list[str]] = {i: [] for i in range(len(blocks))}
    outside: list[str] = []
    for c in text_chars:
        for _, (x0, y0, x1, y1), i in boxes:
            if x0 <= c.x <= x1 and y0 <= c.y <= y1:
                assigned[i].append(c.ch)
                break
        else:
            outside.append(c.ch)

    page = PageScore()
    for i, b in enumerate(blocks):
        label = b["label"]
        ref = normalize("".join(assigned[i]))
        if label in UNREAD_LABELS:
            page.skipped_chars += len(ref)
            continue
        hyp = normalize(b["content"], html_markup=label == "table")
        t = Tally(edit_distance(ref, hyp), len(ref))
        page.add(t)
        page.by_label.setdefault(label, Tally()).add(t)
    # Characters outside every Block: the layout model never saw them.
    page.missed_chars = len(normalize("".join(outside)))
    page.errors += page.missed_chars
    page.ref_chars += page.missed_chars
    return page


def score_document(doc: dict, layers: dict[int, list[TextChar]]) -> DocScore:
    """Score the CLI's JSON (`doc`) against `layers` (page number -> text-layer characters)."""
    out = DocScore()
    for p in doc["pages"]:
        n = p["page"]
        if "error" in p or n not in layers:
            out.skipped_pages.append(n)
            continue
        s = score_page(p["blocks"], layers[n], p["scale_px_per_pt"])
        out.pages[n] = s
        out.total.add(s)
        out.missed_chars += s.missed_chars
        out.skipped_chars += s.skipped_chars
        for label, t in s.by_label.items():
            out.by_label.setdefault(label, Tally()).add(t)
    return out


def extract_text_layers(pdf_path: Path, pages: list[int]) -> dict[int, list[TextChar]]:
    """Text-layer characters of the given 1-based pages: glyph centres in points, origin top-left."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    layers: dict[int, list[TextChar]] = {}
    try:
        for n in pages:
            page = pdf[n - 1]
            height = page.get_height()
            tp = page.get_textpage()
            chars = []
            for i, ch in enumerate(tp.get_text_range(0, tp.count_chars())):
                left, bottom, right, top = tp.get_charbox(i)
                chars.append(TextChar(ch, (left + right) / 2, height - (bottom + top) / 2))
            layers[n] = chars
            tp.close()
            page.close()
    finally:
        pdf.close()
    return layers


def format_report(score: DocScore) -> str:
    def row(name: str, t: Tally) -> str:
        return f"  {name:<22}{t.cer:>9.2%}{t.errors:>9}{t.ref_chars:>10}"

    head = f"  {'':<22}{'CER':>9}{'errors':>9}{'ref chars':>10}"
    lines = ["per page", head]
    lines += [row(f"page {n}", t) for n, t in sorted(score.pages.items())]
    lines += ["", "per label", head]
    lines += [row(label, t) for label, t in sorted(score.by_label.items())]
    lines += ["", row("total", score.total),
              f"  of which {score.missed_chars} text-layer chars lie in no Block (layout misses); "
              f"{score.skipped_chars} chars inside unread Blocks (figures, charts, seals) are not scored"]
    if score.skipped_pages:
        lines.append(f"  pages not scored (failed or no text layer): {score.skipped_pages}")
    return "\n".join(lines)


def to_dict(score: DocScore) -> dict:
    def t(x: Tally) -> dict:
        return {"cer": round(x.cer, 6), "errors": x.errors, "ref_chars": x.ref_chars}

    return {
        "total": t(score.total),
        "missed_chars": score.missed_chars,
        "skipped_chars": score.skipped_chars,
        "skipped_pages": score.skipped_pages,
        "pages": {str(n): t(s) for n, s in sorted(score.pages.items())},
        "by_label": {k: t(v) for k, v in sorted(score.by_label.items())},
    }


def evaluate(pdf: Path, json_path: Path) -> DocScore:
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    return score_document(doc, extract_text_layers(pdf, [p["page"] for p in doc["pages"]]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Character error rate of a pdf_ocr.py run against the PDF's own text layer.")
    ap.add_argument("pdf", type=Path, help="the input PDF (for its text layer)")
    ap.add_argument("json", type=Path, nargs="?",
                    help="the run's JSON (default: output/<pdf stem>/<pdf stem>.json)")
    ap.add_argument("--report", type=Path, metavar="FILE", help="also write the scores as JSON")
    args = ap.parse_args(argv)
    score = evaluate(args.pdf, args.json or Path("output") / args.pdf.stem / f"{args.pdf.stem}.json")
    print(format_report(score))
    if args.report:
        args.report.write_text(json.dumps(to_dict(score), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
