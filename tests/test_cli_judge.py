"""The Judge's command line, the JSON it adds and the stderr line (no GPU, no models)."""
import types

import pytest

import pdf_ocr
from block_reader import BlockResult


def args(*argv):
    return pdf_ocr.build_parser().parse_args(["in.pdf", *argv])


def test_the_judge_is_on_by_default_with_the_official_threshold():
    a = args()
    assert a.no_judge is False and a.judge_threshold == 0.98
    assert args("--no-judge").no_judge is True
    assert args("--judge-threshold", "0.9").judge_threshold == 0.9


def test_the_judge_needs_its_model_folder_unless_it_is_off(tmp_path):
    assert pdf_ocr.required_model_dirs(args()) == ["PP-DocLayoutV3", "MiniCPM-OCR-1.0", "MiniCPM-OCR-Jev"]
    assert pdf_ocr.required_model_dirs(args("--no-judge")) == ["PP-DocLayoutV3", "MiniCPM-OCR-1.0"]
    assert pdf_ocr.required_model_dirs(args("--reader", "paddle")) == ["PP-DocLayoutV3", "PaddleOCR-VL-1.6", "MiniCPM-OCR-Jev"]
    assert pdf_ocr.required_model_dirs(args("--reader", "paddle", "--no-judge")) == ["PP-DocLayoutV3", "PaddleOCR-VL-1.6"]
    assert "MiniCPM-OCR-Jev" in pdf_ocr.missing_models_message(args("--models-dir", str(tmp_path)))


def test_the_worker_is_needed_for_minicpm_or_for_the_judge_and_loads_only_what_is_used():
    assert pdf_ocr.needs_worker(args()) and pdf_ocr.needs_worker(args("--no-judge"))
    assert pdf_ocr.needs_worker(args("--reader", "paddle"))
    assert not pdf_ocr.needs_worker(args("--reader", "paddle", "--no-judge"))
    cmd = lambda *a: pdf_ocr.worker_command(args("--models-dir", "/m", *a), uv="/usr/bin/uv")
    assert "--no-reader" not in cmd() and "--no-judge" not in cmd()
    assert "--no-judge" in cmd("--no-judge") and "--no-reader" not in cmd("--no-judge")
    assert "--no-reader" in cmd("--reader", "paddle") and "--no-judge" not in cmd("--reader", "paddle")


READY = {"reader": "MiniCPM-OCR-1.0", "judge": "MiniCPM-OCR-Jev",
         "versions": {"torch": "2.15.0+cu132", "transformers": "5.22.2", "playwright": "1.62.0"}}


def test_the_engine_names_the_judge_when_it_is_on():
    e = pdf_ocr.engine_info(args(), "gpu:0", READY)
    assert e["judge"] == "MiniCPM-OCR-Jev" and e["playwright"] == "1.62.0"
    assert "judge" not in pdf_ocr.engine_info(args("--no-judge"), "gpu:0", {**READY, "judge": None})


def test_the_paddle_reader_with_the_judge_has_the_workers_versions_too():
    e = pdf_ocr.engine_info(args("--reader", "paddle"), "gpu:0", {**READY, "reader": None})
    assert e["reader"] == "PaddleOCR-VL-1.6" and e["judge"] == "MiniCPM-OCR-Jev" and e["torch"] == "2.15.0+cu132"


def test_the_settings_record_whether_the_judge_was_on_and_its_threshold():
    s = pdf_ocr.settings_info(args("--judge-threshold", "0.9"))
    assert s["judge"] is True and s["judge_threshold"] == 0.9 and s["reader"] == "minicpm"
    s = pdf_ocr.settings_info(args("--no-judge"))
    assert s["judge"] is False and s["judge_threshold"] is None
    assert s["reread_short_blocks"] is True  # the ink-based re-read is what runs without the Judge


def test_with_the_judge_on_the_ink_based_reread_is_not_run_and_the_settings_say_so():
    assert pdf_ocr.settings_info(args())["reread_short_blocks"] is False
    assert pdf_ocr.settings_info(args("--no-recover", "--no-judge"))["reread_short_blocks"] is False


# --- the JSON of a judged Block -------------------------------------------------------------------------

def block(**kw):
    return {"block_id": 0, "block_order": 1, "block_label": "text", "block_content": "你好", "block_bbox": [1, 2, 30, 40],
            "block_polygon_points": None, **kw}


def res_for(*blocks):
    return types.SimpleNamespace(json={"res": {"width": 100, "height": 200, "parsing_res_list": list(blocks)}})


def outcome(**kw):
    kw.setdefault("text", "你好")
    return BlockResult(**kw)


def page(blocks, outcomes, raw=None, idx=0):
    return pdf_ocr.page_json(idx, res_for(*blocks), {"scale": 3.0, "recovered": []}, raw or {}, [], outcomes)


def test_a_judged_block_carries_score_verdict_and_flag():
    o = outcome(judged=True, score=0.91, verdict="scored", below_threshold=True)
    (b,) = page([block()], {(0, (1, 2, 30, 40)): o})["blocks"]
    assert (b["score"], b["verdict"], b["below_threshold"]) == (0.91, "scored", True)


def test_a_block_the_judge_could_not_score_has_a_null_score_and_the_reason():
    o = outcome(judged=True, score=None, verdict="render_error", verdict_detail="render_failed", below_threshold=True)
    (b,) = page([block()], {(0, (1, 2, 30, 40)): o})["blocks"]
    assert b["score"] is None and b["verdict"] == "render_error" and b["below_threshold"] is True
    assert b["verdict_detail"] == "render_failed"


def test_charts_seals_and_figures_have_none_of_the_judge_fields():
    o = outcome(judged=False)
    (b,) = page([block(block_label="chart")], {(0, (1, 2, 30, 40)): o})["blocks"]
    assert not {"score", "verdict", "below_threshold", "verdict_detail"} & b.keys()


def test_without_outcomes_the_json_is_what_it_was_before_the_judge():
    (b,) = page([block()], {})["blocks"]
    assert set(b) == {"id", "order", "label", "content", "bbox", "polygon", "in_markdown"}


def test_outcomes_are_matched_by_page_and_bbox_not_by_block_id():
    o = outcome(judged=True, score=0.5, verdict="scored", below_threshold=True)
    assert "score" in page([block(block_id=7)], {(3, (1, 2, 30, 40)): o}, idx=3)["blocks"][0]
    assert "score" not in page([block(block_id=7)], {(2, (1, 2, 30, 40)): o}, idx=3)["blocks"][0]


def test_a_block_judged_before_the_document_was_known_to_be_traditional_is_marked():
    o = outcome(judged=True, score=0.9, verdict="scored", below_threshold=True, judged_before_script_decision=True)
    (b,) = page([block()], {(0, (1, 2, 30, 40)): o})["blocks"]
    assert b["judged_before_script_decision"] is True


def test_the_text_that_was_judged_is_recorded_when_it_differs_from_the_text_that_was_written():
    o = outcome(text="银行", judged=True, score=0.9, verdict="scored", below_threshold=True)
    (b,) = page([block(block_content="銀行")], {(0, (1, 2, 30, 40)): o})["blocks"]
    assert b["judged_text"] == "银行"
    (b,) = page([block(block_content="银行")], {(0, (1, 2, 30, 40)): o})["blocks"]
    assert "judged_text" not in b


def test_the_ink_reread_of_a_bridge_block_is_recorded_like_the_post_pass_one():
    o = outcome(text_before_reread="x")
    (b,) = page([block()], {(0, (1, 2, 30, 40)): o})["blocks"]
    assert b["content_before_reread"] == "x"


# --- document-level repair after a sticky decision -------------------------------------------------------

class Blk:
    def __init__(self, label, bbox, content):
        self.label, self.bbox, self.content = label, bbox, content


def test_text_the_sticky_repair_changed_is_put_back_so_the_document_level_repair_decides_alone():
    b1, b2 = Blk("text", (0, 0, 1, 1), "銀行"), Blk("text", (0, 2, 1, 3), "plain")
    outcomes = {(0, (0, 0, 1, 1)): outcome(text="銀行", text_before_repair="银行", judged=True),
                (0, (0, 2, 1, 3)): outcome(text="plain", judged=True)}
    pdf_ocr.undo_sticky_repair([(0, {"parsing_res_list": [b1, b2]})], outcomes)
    assert (b1.content, b2.content) == ("银行", "plain")


def test_text_the_pipeline_changed_afterwards_is_left_alone():
    b1 = Blk("text", (0, 0, 1, 1), "something else entirely")
    outcomes = {(0, (0, 0, 1, 1)): outcome(text="銀行", text_before_repair="银行", judged=True)}
    pdf_ocr.undo_sticky_repair([(0, {"parsing_res_list": [b1]})], outcomes)
    assert b1.content == "something else entirely"


# --- the summary, stderr and the postprocessing section ------------------------------------------------

def test_the_postprocessing_summary_counts_blocks_verdicts_and_early_judgements():
    outcomes = {
        (0, (0, 0, 1, 1)): outcome(judged=True, score=0.99, verdict="scored", below_threshold=False),
        (0, (0, 2, 1, 3)): outcome(judged=True, score=0.5, verdict="scored", below_threshold=True, judged_before_script_decision=True),
        (1, (0, 0, 1, 1)): outcome(judged=True, verdict="render_error", below_threshold=True),
        (1, (0, 5, 1, 6)): outcome(judged=False),
    }
    assert pdf_ocr.judge_summary(outcomes) == {
        "blocks_judged": 3, "blocks_below_threshold": 2,
        "verdicts": {"render_error": 1, "scored": 2}, "blocks_judged_before_script_decision": 1}


def test_the_page_line_on_stderr_says_how_many_blocks_are_below_the_threshold():
    line = pdf_ocr.page_line(2, 21, 4, 13, 7.04, summary={"judged": 9, "below_threshold": 3, "not_scored": 1}, recovered=0)
    assert line == "[  2/21] page 4: 13 blocks, 7.0s, 3 of 9 judged below the threshold (1 not scored)"
    plain = pdf_ocr.page_line(1, 2, 1, 5, 1.0, summary=None, recovered=2)
    assert plain == "[  1/2] page 1: 5 blocks, 1.0s, re-read 2 short block(s)"
    assert "below" in pdf_ocr.page_line(1, 2, 1, 5, 1.0, summary={"judged": 5, "below_threshold": 0, "not_scored": 0}, recovered=0)
