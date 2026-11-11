"""Simplified to Traditional Chinese repair: the character-level repair, the document-level decision rule,
and the sticky per-Reading decision the Judge path uses (no GPU, no paddle; opencc is imported lazily)."""
from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

# Symbols that cannot be encoded in Big5, and the Big5 glyph a Traditional document gets for each instead.
BIG5_LOOKALIKES: Final[Mapping[str, str]] = MappingProxyType({"≥": "≧", "≤": "≦"})
_TO_BIG5_GLYPHS = str.maketrans(dict(BIG5_LOOKALIKES))


class ScriptRepair:
    """Repair Simplified-Chinese characters inside a Traditional-Chinese document.

    PaddleOCR-VL tends to drift to Simplified forms (银行, 医院, 细胞) when reading Traditional text.
    A character is treated as drift only if it cannot be encoded in Big5 (so it is not valid
    Traditional) and OpenCC maps it to something else. Valid Traditional characters that merely also
    exist in Simplified (后, 干, 里, 台, 表) are never touched. The same rule picks the glyph of `≥` and
    `≤`: they are not in Big5, so they are written as `≧` and `≦` (BIG5_LOOKALIKES).
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
        """The text repaired, for a document already decided Traditional: BIG5_LOOKALIKES as their Big5 glyph, and
        each Simplified-only character as its Traditional form."""
        text = text.translate(_TO_BIG5_GLYPHS)
        if not any(self.is_simplified_only(c) for c in text):
            return text
        conv = self.s2t.convert(text)  # phrase-aware (e.g. 头发 -> 頭髮)
        if len(conv) == len(text):
            return "".join(conv[i] if self.is_simplified_only(c) else c for i, c in enumerate(text))
        return "".join(self.s2t.convert(c) if self.is_simplified_only(c) else c for c in text)


NO_REPAIR_LABELS = {"display_formula", "inline_formula", "formula_number", "algorithm"}


def is_traditional(simp: int, trad: int) -> bool:
    """The document-level rule: act only on a clearly Traditional document; Simplified or mixed ones are left alone."""
    share = simp / (simp + trad) if simp + trad else 0.0
    return trad >= 30 and share <= 0.15


class StickyScriptRepair:
    """Decide "this document is Traditional" while reading, and repair every Reading from then on.

    The document-level repair (pdf_ocr.repair_script) surveys all pages first. The Judge needs the repaired
    text of each Block as it is read, so this keeps a running survey of the Readings in reading order and,
    as soon as the rule holds, keeps it decided (sticky, never undone). The Reading that completes the
    evidence is itself repaired; Readings before that point are not (the caller records that they were
    judged unrepaired). Same Readings in the same order always give the same decisions.
    `rep` is a ScriptRepair, or None when opencc is not installed (then nothing is ever repaired).
    """

    def __init__(self, rep: "ScriptRepair | None"):
        self.rep = rep
        self.simplified = self.traditional = 0
        self.is_traditional = False

    def apply(self, text: str, label: str, *, survey: bool = True) -> tuple[str, bool]:
        """(text to judge, whether the document was already decided Traditional when this Reading came in).

        `survey=False` is for a re-read of a Block whose first Reading was already counted: the text is repaired
        if the document is decided, but the running survey is left alone (it is the document's, not the re-reads')."""
        if self.rep is None or label in NO_REPAIR_LABELS:
            return text, self.is_traditional
        if survey and not self.is_traditional:
            simp, trad = self.rep.survey([text])
            self.simplified += simp
            self.traditional += trad
            self.is_traditional = is_traditional(self.simplified, self.traditional)
        return (self.rep.fix(text) if self.is_traditional else text), self.is_traditional
