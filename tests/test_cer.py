"""Fast, no-GPU tests for the character-error-rate harness (ocr_eval.py).

Coordinates: a block's bbox is in rendered-page pixels (top-left origin, as in the CLI's JSON);
the text layer's characters are in PDF points, also top-left origin. `scale` is px per pt.
"""
import json

import ocr_eval
from ocr_eval import TextChar


def block(label, content, bbox, bid=0):
    return {"id": bid, "label": label, "content": content, "bbox": list(bbox)}


def chars(text, x0, y, step=10.0):
    """A line of characters, one every `step` points, whose centres sit at (x0 + i*step, y)."""
    return [TextChar(c, x0 + i * step, y) for i, c in enumerate(text)]


def test_perfect_reading_has_zero_error():
    page = ocr_eval.score_page(
        blocks=[block("text", "你好嗎", (0, 0, 100, 20))],
        text_chars=chars("你好嗎", 5, 10),
        scale=1.0,
    )
    assert page.errors == 0
    assert page.ref_chars == 3
    assert page.cer == 0.0


def test_substitution_counts_as_one_error():
    page = ocr_eval.score_page(
        blocks=[block("text", "你好嗎", (0, 0, 100, 20))],
        text_chars=chars("你好馬", 5, 10),
        scale=1.0,
    )
    assert page.errors == 1
    assert page.cer == 1 / 3


def test_text_layer_chars_outside_every_block_count_as_missed():
    # The layout model never found the second line: its 3 characters are all errors.
    page = ocr_eval.score_page(
        blocks=[block("text", "你好嗎", (0, 0, 100, 20))],
        text_chars=chars("你好嗎", 5, 10) + chars("再見啦", 5, 50),
        scale=1.0,
    )
    assert page.ref_chars == 6
    assert page.errors == 3
    assert page.missed_chars == 3


def test_block_order_in_the_json_does_not_matter():
    a = block("text", "甲乙丙", (0, 0, 100, 20), bid=0)
    b = block("text", "丁戊己", (0, 40, 100, 60), bid=1)
    layer = chars("甲乙丙", 5, 10) + chars("丁戊己", 5, 50)
    forward = ocr_eval.score_page([a, b], layer, 1.0)
    swapped = ocr_eval.score_page([b, a], chars("丁戊己", 5, 50) + chars("甲乙丙", 5, 10), 1.0)
    assert forward.errors == swapped.errors == 0


def test_char_in_overlapping_blocks_goes_to_the_smaller_one():
    big = block("text", "", (0, 0, 200, 200), bid=0)  # e.g. a falsely wide box that read nothing
    small = block("text", "你好", (0, 0, 40, 20), bid=1)
    page = ocr_eval.score_page([big, small], chars("你好", 5, 10), 1.0)
    assert page.errors == 0


def test_bbox_is_converted_from_pixels_to_points_with_scale():
    # 3 px per point: a 300 px wide box covers 100 pt.
    page = ocr_eval.score_page(
        [block("text", "你好嗎", (0, 0, 300, 60))], chars("你好嗎", 5, 10), 3.0
    )
    assert page.errors == 0 and page.missed_chars == 0


def test_html_tags_whitespace_and_width_variants_are_ignored():
    table = "<table><tr><td>保 額</td><td>１０</td></tr></table>"
    page = ocr_eval.score_page(
        [block("table", table, (0, 0, 100, 20))], chars("保額10", 5, 10), 1.0
    )
    assert page.errors == 0


def test_text_layer_whitespace_and_control_chars_are_not_reference_text():
    layer = chars("你 好\r\n", 5, 10)
    page = ocr_eval.score_page([block("text", "你好", (0, 0, 100, 20))], layer, 1.0)
    assert page.ref_chars == 2 and page.errors == 0


def test_unread_labels_are_excluded_not_counted_as_errors():
    # Figures, charts and seals are not read by default; their text-layer chars are not scored.
    fig = block("image", "", (0, 40, 100, 60), bid=1)
    page = ocr_eval.score_page(
        [block("text", "你好", (0, 0, 100, 20)), fig],
        chars("你好", 5, 10) + chars("圖內字", 5, 50),
        1.0,
    )
    assert page.errors == 0
    assert page.ref_chars == 2
    assert page.skipped_chars == 3


def test_dot_leaders_of_a_table_of_contents_are_ignored():
    # The text layer has 30 leader dots, the Reader wrote 5: neither is a reading error.
    ref = "第一章" + "." * 30 + "3"
    page = ocr_eval.score_page(
        [block("content", "第一章.....3", (0, 0, 400, 20))], chars(ref, 5, 10, step=1.0), 1.0
    )
    assert page.errors == 0
    assert page.ref_chars == 4
