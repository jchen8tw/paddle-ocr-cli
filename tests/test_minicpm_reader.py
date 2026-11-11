"""MiniCPM Reader logic that needs no GPU: which prompt and token limit a Block gets, and how a raw
Reading is cleaned up before it is handed back to the pipeline."""
import pytest

import minicpm_reader as mr


@pytest.mark.parametrize("label,prompt,limit", [
    ("text", "Text Recognition:", 4096),
    ("abstract", "Text Recognition:", 4096),
    ("paragraph_title", "Text Recognition:", 512),
    ("doc_title", "Text Recognition:", 512),
    ("figure_title", "Text Recognition:", 512),
    ("inline_formula", "Text Recognition:", 2048),
    ("display_formula", "Formula Recognition:", 2048),
    ("table", "Table Recognition:", 7777),      # the table limit is --max-new-tokens
    ("seal", "Seal Recognition:", 256),
    ("chart", "Chart to Table:", 4096),
])
def test_each_block_label_gets_the_official_prompt_and_its_token_limit(label, prompt, limit):
    assert mr.prompt_and_limit(label, table_limit=7777) == (prompt, limit)


def test_the_layout_less_reread_label_and_unknown_labels_are_read_as_text():
    assert mr.prompt_and_limit("ocr", table_limit=8192) == ("Text Recognition:", 4096)
    assert mr.prompt_and_limit("something_new", table_limit=8192) == ("Text Recognition:", 4096)


def test_a_table_limit_below_the_text_limits_is_still_the_table_limit():
    assert mr.prompt_and_limit("table", table_limit=300)[1] == 300


# ---------------------------------------------------------------------------- clean-up of a raw Reading

def fake_to_html(otsl: str) -> str:
    """Stands in for PaddleX's converter: HTML for text that looks like OTSL, '' otherwise (as the real one does)."""
    if "<fcel>" not in otsl:
        return ""
    return "<table><tr><td>" + otsl.replace("<otsl>", "").replace("</otsl>", "").replace("<fcel>", "").replace(
        "<nl>", "</td></tr><tr><td>") + "</td></tr></table>"


def clean(raw, label="text"):
    cleaned = mr.clean_reading(raw, label, to_html=fake_to_html)
    return cleaned.text, cleaned.repetition_cut


def test_ordinary_text_is_returned_stripped_and_not_counted_as_a_cut():
    assert clean("  第一條 保險範圍\n第二行  \n") == ("第一條 保險範圍\n第二行", False)


def test_a_repeated_tail_is_cut_back_to_the_text_before_the_loop():
    raw = "前言" + "ABCDEFGHIJ" * 20
    assert clean(raw) == ("前言", True)


def test_a_text_that_is_one_unit_repeated_collapses_to_the_unit():
    assert clean("ab" * 30) == ("ab", True)


def test_a_line_repeated_for_most_of_the_text_collapses_to_that_line():
    assert clean("same line\n" * 12) == ("same line", True)


def test_short_text_is_never_cut_even_if_repetitive():
    assert clean("哈" * 20) == ("哈" * 20, False)


def test_tables_are_only_cut_when_very_long():
    rows = "<fcel>x<nl>" * 12  # repeated rows are legitimate in a table
    html, cut = clean("<otsl>" + rows + "</otsl>", "table")
    assert not cut and html.count("<tr>") == 13


def test_a_table_is_converted_from_otsl_to_html():
    html, _ = clean("<otsl><fcel>甲<nl></otsl>", "table")
    assert html.startswith("<table>") and "甲" in html and "<otsl>" not in html


def test_a_table_that_is_already_html_is_left_alone():
    html = "<table><tr><td>甲</td></tr></table>"
    assert clean(html, "table") == (html, False)


def test_real_newlines_and_literal_backslash_n_inside_a_table_become_br():
    html, _ = clean("<otsl><fcel>第一行\n第二行<fcel>甲\\n乙<nl></otsl>", "table")
    assert "第一行<br>第二行" in html and "甲<br>乙" in html
    assert "\n" not in html and "\\n" not in html


def test_newlines_in_text_blocks_are_kept():
    assert clean("推動自動化及 AI 導入\n廠商企劃書", "header") == ("推動自動化及 AI 導入\n廠商企劃書", False)


def test_with_the_real_paddlex_converter_a_minicpm_table_keeps_its_cell_line_breaks():
    """MiniCPM's raw OTSL (real newline inside a cell) through the converter the pipeline itself uses."""
    uilts = pytest.importorskip("paddlex.inference.pipelines.paddleocr_vl.uilts")
    raw = "<otsl><fcel>項目<fcel>內容<nl><fcel>蒐集\n轉換<fcel>需求1～3<nl></otsl>"
    html = mr.clean_reading(raw, "table", to_html=uilts.convert_otsl_to_html).text
    assert html == ("<table><tr><td>項目</td><td>內容</td></tr>"
                    "<tr><td>蒐集<br>轉換</td><td>需求1～3</td></tr></table>")
