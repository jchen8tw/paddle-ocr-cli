"""The Block-level clean-up after the pipeline has read every page (pdf_ocr): the lone symbols that a run without a
bridge (--reader paddle --no-judge) only gets here (no GPU, no models)."""
import types

import pytest

import minicpm_reader as mr
import pdf_ocr
from test_reading_cleanup import CORPUS


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
