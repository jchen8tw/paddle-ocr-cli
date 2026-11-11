"""The Block-level clean-up after the pipeline has read every page (pdf_ocr): the lone symbols that a run without a
bridge (--reader paddle --no-judge) only gets here, and the document-level script repair, which writes `≥`/`≤` as the
Big5 glyphs `≧`/`≦` in a Traditional document (no GPU, no models)."""
import types

import pytest

import minicpm_reader as mr
import pdf_ocr
from test_reading_cleanup import CORPUS

TRAD = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"  # 39 Traditional-only characters


def blk(label, content):
    return types.SimpleNamespace(label=label, content=content)


def pages(*blocks_per_page):
    return [(i, {"parsing_res_list": list(blocks)}) for i, blocks in enumerate(blocks_per_page)]


def test_without_a_bridge_the_pipelines_text_gets_its_lone_symbols_unwrapped():
    blocks = [blk("text", r"年滿15足歲 $ \geq $者"),  # what the pipeline's own pass makes of 年滿15足歲\(\geq\)者
              blk("text", r"年滿15足歲$\geq$者"), blk("paragraph_title", r"年滿15足歲\geq者"),
              blk("display_formula", r"$\geq$"), blk("table", r"<td>$\geq$</td>")]
    assert pdf_ocr.unwrap_block_symbols(pages(blocks[:3], blocks[3:])) == 3
    assert [b.content for b in blocks] == ["年滿15足歲≥者"] * 3 + [r"$\geq$", r"<td>$\geq$</td>"]


@pytest.mark.parametrize("label", ["text", "footnote", "display_formula", "formula_number"])
def test_without_a_bridge_a_block_ends_up_with_the_text_a_bridge_gives_it(label):
    for raw in CORPUS:
        b = blk(label, mr.rewrite_latex_delimiters(mr.truncate_repetitive_content(raw), label))  # the pipeline's pass
        pdf_ocr.unwrap_block_symbols(pages([b]))
        assert b.content == mr.clean_reading(raw, label, to_html=lambda s: "", strip=False).text, raw


@pytest.mark.parametrize("strip", [True, False])
def test_on_text_a_bridge_has_cleaned_the_pass_changes_and_counts_nothing(strip):
    cleaned = [mr.clean_reading(raw, "text", to_html=lambda s: "", strip=strip).text for raw in CORPUS]
    blocks = [blk("text", t) for t in cleaned]
    assert pdf_ocr.unwrap_block_symbols(pages(blocks)) == 0
    assert [b.content for b in blocks] == cleaned


# --- the document-level script repair: the Big5 glyph of >= and <= ----------------------------------------

def test_a_document_decided_traditional_writes_the_big5_glyphs_even_with_no_simplified_character():
    blocks = [blk("text", TRAD), blk("text", "年滿15足歲≥者"), blk("footnote", "≤65歲"), blk("display_formula", "x ≥ 1")]
    info, raw = pdf_ocr.repair_script(pages(blocks), "auto")
    assert [b.content for b in blocks] == [TRAD, "年滿15足歲≧者", "≦65歲", "x ≥ 1"]
    assert (info["document_script"], info["simplified_chars"], info["applied"]) == ("traditional", 0, True)
    assert (info["blocks_changed"], info["big5_symbols_changed"]) == (2, 2)
    assert raw == {(0, 1): "年滿15足歲≥者", (0, 2): "≤65歲"}


def test_blocks_changed_counts_either_fix_and_big5_symbols_changed_counts_each_symbol():
    blocks = [blk("text", TRAD), blk("text", "≥≥與≤"), blk("text", "银行"), blk("text", "银行≥")]
    info, _ = pdf_ocr.repair_script(pages(blocks), "auto")
    assert [b.content for b in blocks[1:]] == ["≧≧與≦", "銀行", "銀行≧"]
    assert (info["blocks_changed"], info["big5_symbols_changed"]) == (3, 4)


def test_an_undecided_document_keeps_ge_and_le():
    blocks = [blk("text", TRAD[:20]), blk("text", "年滿15足歲≥者")]
    info, raw = pdf_ocr.repair_script(pages(blocks), "auto")
    assert blocks[1].content == "年滿15足歲≥者" and raw == {}
    assert info["applied"] is False and info["document_script"] == "not traditional"
    assert "big5_symbols_changed" not in info


def test_script_off_keeps_ge_and_le():
    blocks = [blk("text", TRAD), blk("text", "年滿15足歲≥者")]
    assert pdf_ocr.repair_script(pages(blocks), "off") == ({"mode": "off", "applied": False}, {})
    assert blocks[1].content == "年滿15足歲≥者"
