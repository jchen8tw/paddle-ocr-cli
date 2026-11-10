"""MiniCPM-OCR-1.0 Reader logic that needs neither torch nor a GPU: the prompt and token limit of each
Block label, and the clean-up of a raw Reading. The model itself runs in the worker (worker/)."""
from __future__ import annotations

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


def clean_reading(raw: str, label: str, *, to_html, strip: bool = True) -> tuple[str, bool]:
    """Clean one raw Reading (MiniCPM's, or PaddleOCR-VL's with strip=False). Returns (text, whether a
    repetition loop was cut).

    Order: repetition cut, math delimiters (the pipeline does both next anyway), OTSL to HTML (tables),
    in-cell line breaks to <br> (tables). The Simplified to Traditional repair is not here: it needs the
    document (script_repair.py). `to_html` is PaddleX's convert_otsl_to_html, which returns "" for text that
    is not OTSL.
    """
    text = raw.strip() if strip else raw
    cut = truncate_repetitive_content(text, min_count=5000 if label == "table" else 50)
    was_cut = cut != text
    text = rewrite_latex_delimiters(cut, label)
    if label == "table":
        html = to_html(text)
        if html:
            text = html
        # MiniCPM writes an in-cell line break as a real newline; PaddleOCR-VL as a literal backslash + n.
        # The HTML has no newlines between its tags, so every newline is inside a cell.
        text = text.replace("\\n", "<br>").replace("\n", "<br>")
    return text, was_cut
