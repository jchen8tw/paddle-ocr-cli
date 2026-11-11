"""MiniCPM-OCR-1.0 Reader logic that needs neither torch nor a GPU: the prompt and token limit of each
Block label, and the clean-up of a raw Reading. The model itself runs in the worker (worker/)."""
from __future__ import annotations

import re
from types import MappingProxyType
from typing import Final, Mapping, NamedTuple

from block_reader import judge_category
from script_repair import NO_REPAIR_LABELS

TEXT_PROMPT = "Text Recognition:"
TABLE_PROMPT = "Table Recognition:"
FORMULA_PROMPT = "Formula Recognition:"
CHART_PROMPT = "Chart to Table:"
SEAL_PROMPT = "Seal Recognition:"

# Token limits per Block label. The spec's numbers (text 4096, titles 512, formula 2048, seal 256,
# chart 4096, table = --max-new-tokens); the few labels the official SDK limits more tightly keep the SDK's
# value (sdk/minicpmvocr/prompts.py).
_LIMITS = {
    "doc_title": 512, "paragraph_title": 512, "figure_title": 512, "header": 512, "footer": 512,
    "formula_number": 256, "number": 256,
    "vision_footnote": 1024, "footnote": 1024, "aside_text": 1024,
    "reference": 2048, "reference_content": 2048, "vertical_text": 2048,
    "display_formula": 2048, "inline_formula": 2048,
    "chart": 4096, "seal": 256,
}
DEFAULT_LIMIT = 4096
_PROMPTS = {
    "table": TABLE_PROMPT,
    "display_formula": FORMULA_PROMPT,
    "chart": CHART_PROMPT,
    "seal": SEAL_PROMPT,
}


def prompt_and_limit(label: str, *, table_limit: int) -> tuple[str, int]:
    """(official prompt, max new tokens) for one Block label. Anything not listed is read as text."""
    if label == "table":
        return TABLE_PROMPT, table_limit
    return _PROMPTS.get(label, TEXT_PROMPT), _LIMITS.get(label, DEFAULT_LIMIT)


# ---------------------------------------------------------------------------- clean-up of a raw Reading

def truncate_repetitive_content(content: str, min_count: int = 50) -> str:
    """Cut a repetition loop out of a Reading.

    Ported from the official MiniCPM-OCR SDK (Apache-2.0, sdk/minicpmvocr/postprocess/markdown.py).
    `min_count` is really a minimum length: shorter content is never touched.
    """
    from collections import Counter

    if not content or len(content) < min_count:
        return content
    stripped = content.strip()
    if not stripped:
        return content

    if "\n" not in stripped and len(stripped) > 100:  # a repeated tail: keep the text before it
        for unit_len in range(8, len(stripped) // 5 + 1):
            unit = stripped[-unit_len:]
            count = 0
            pos = len(stripped) - unit_len
            while pos >= 0 and stripped[pos:pos + unit_len] == unit:
                count += 1
                pos -= unit_len
            if count >= 5:
                start_index = len(stripped) - (count * unit_len)
                if len(unit) * count > len(stripped) * 0.5:
                    return stripped[:start_index]

    if "\n" not in stripped and len(stripped) > 10:  # one unit repeated over the whole text
        for unit_len in range(1, len(stripped) // 2 + 1):
            unit = stripped[:unit_len]
            if unit * (len(stripped) // unit_len) == stripped[:unit_len * (len(stripped) // unit_len)]:
                if len(stripped) // unit_len >= 10:
                    return unit
                break

    lines = [line.strip() for line in stripped.split("\n") if line.strip()]  # one line repeated
    if len(lines) >= 10:
        most_common, count = Counter(lines).most_common(1)[0]
        if count >= 10 and count / len(lines) >= 0.8:
            return most_common
    return content


def rewrite_latex_delimiters(text: str, label: str) -> str:
    """PaddleX's rewrite of `\\( \\)` and `\\[ \\]` into ` $ ... $ ` and ` $$ ... $$ `, done by the pipeline after
    the Reader returns. Done here too, before judging, so that the Score describes the text that is written;
    the pipeline's own pass then finds nothing left to do."""
    if not (("\\(" in text and "\\)" in text) or ("\\[" in text and "\\]" in text)):
        return text
    text = text.replace("$", "")
    text = (text.replace("\\(", " $ ").replace("\\)", " $").replace("\\[\\[", "\\[").replace("\\]\\]", "\\]")
            .replace("\\[", " $$ ").replace("\\]", " $$ "))
    return text.replace("$", "") if label == "formula_number" else text


LATEX_SYMBOLS: Final[Mapping[str, str]] = MappingProxyType({
    r"\geq": "≥", r"\ge": "≥", r"\geqslant": "≥",
    r"\leq": "≤", r"\le": "≤", r"\leqslant": "≤",
    r"\geqq": "≧", r"\leqq": "≦",
    r"\neq": "≠", r"\ne": "≠",
    r"\times": "×", r"\div": "÷", r"\pm": "±", r"\approx": "≈", r"\%": "%",
    r"^\circ": "°", r"^{\circ}": "°", r"\degree": "°",
    r"\rightarrow": "→", r"\to": "→",
})


def _symbol_pattern(command: str) -> str:
    # A command that ends in a letter must not run on into another letter: \top is not \to followed by p.
    return re.escape(command) + ("(?![A-Za-z])" if command[-1].isascii() and command[-1].isalpha() else "")


# Left to right, one token at a time: a display span (kept), an inline span with the one space before it, a symbol
# command (longest first), and any other command or backslash pair, which is consumed whole so that no symbol is
# found inside it (`\left`, `\\%`).
_TOKEN = re.compile(
    r"(?P<display> ?\$\$.*?\$\$)"
    r"|(?P<pad> ?)\$(?P<body>(?:\\.|[^$\\])*)\$"
    r"|(?P<symbol>" + "|".join(_symbol_pattern(c) for c in sorted(LATEX_SYMBOLS, key=len, reverse=True)) + ")"
    r"|\\[A-Za-z]+|\\.",
    re.S)


def unwrap_symbols(text: str, label: str) -> tuple[str, int]:
    """Write each lone symbol command of LATEX_SYMBOLS as its character. Returns (text, how many were written).

    A lone symbol is a bare command outside every math span (`歲\\geq者`), or an inline `$` span holding nothing but
    the command and whitespace (`$\\geq$`, `$ \\geq $`). Such a span is replaced by the character, delimiters and
    inner whitespace included. When its content starts with whitespace, the one space before the span goes too: that
    is the space the delimiter rewrite puts before `\\(`, so `歲\\(\\geq\\)者` ends up as `歲≥者`. Display spans, spans holding
    anything more (`$x \\geq 5$`) and an unpaired `$` are left as they are. Only Blocks the Judge rates as text and
    the script repair also touches are changed: formulas, `formula_number`, `algorithm`, tables, charts, seals and
    images are returned unchanged. The text is expected after rewrite_latex_delimiters, ours or the pipeline's, so
    a run without a bridge gets the same result from the pipeline's output.
    """
    if judge_category(label) != "text" or label in NO_REPAIR_LABELS:
        return text, 0
    written = 0

    def replace(m: re.Match) -> str:
        nonlocal written
        if m["symbol"]:
            written += 1
            return LATEX_SYMBOLS[m["symbol"]]
        body = m["body"]
        if body is not None and body.strip() in LATEX_SYMBOLS:
            written += 1
            return ("" if body[:1].isspace() else m["pad"]) + LATEX_SYMBOLS[body.strip()]
        return m[0]

    return _TOKEN.sub(replace, text), written


class Cleaned(NamedTuple):
    text: str
    repetition_cut: bool
    symbols_unwrapped: int


def clean_reading(raw: str, label: str, *, to_html, strip: bool = True) -> Cleaned:
    """Clean one raw Reading (MiniCPM's, or PaddleOCR-VL's with strip=False).

    Order: repetition cut, math delimiters (the pipeline does both next anyway), lone symbols written as
    their character (unwrap_symbols; text Blocks only), OTSL to HTML (tables), in-cell line breaks to <br>
    (tables). The Simplified to Traditional repair is not here: it needs the document (script_repair.py).
    `to_html` is PaddleX's convert_otsl_to_html, which returns "" for text that is not OTSL.
    """
    text = raw.strip() if strip else raw
    cut = truncate_repetitive_content(text, min_count=5000 if label == "table" else 50)
    was_cut = cut != text
    text, unwrapped = unwrap_symbols(rewrite_latex_delimiters(cut, label), label)
    if label == "table":
        html = to_html(text)
        if html:
            text = html
        # MiniCPM writes an in-cell line break as a real newline; PaddleOCR-VL as a literal backslash + n.
        # The HTML has no newlines between its tags, so every newline is inside a cell.
        text = text.replace("\\n", "<br>").replace("\n", "<br>")
    return Cleaned(text, was_cut, unwrapped)
