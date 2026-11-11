"""The Judge's command line, the JSON it adds and the stderr line (no GPU, no models)."""
import types

import pytest

import pdf_ocr
from block_reader import BlockResult, Reading


def args(*argv):
    return pdf_ocr.parse_args(["in.pdf", *argv])


def test_the_judge_is_off_by_default_and_judge_turns_it_on_with_the_official_threshold():
    a = args()
    assert a.reader == "paddle" and a.judge is False and a.judge_threshold is None
    assert args("--judge").judge is True and args("--judge").judge_threshold == 0.5
    assert args("--no-judge").judge is False and args("--no-judge").judge_threshold is None
    assert args("--reader", "minicpm").judge is False
    assert args("--judge", "--judge-threshold", "0.7").judge_threshold == 0.7
    assert args("--judge", "--no-judge").judge is False and args("--no-judge", "--judge").judge is True


@pytest.mark.parametrize("argv", [["--judge-threshold", "0.7"], ["--no-judge", "--judge-threshold", "0.7"],
                                  ["--judge-threshold", "0.7", "--no-judge"], ["--judge-thr", "0.7"]])
def test_a_judge_threshold_without_the_judge_is_a_usage_error_that_says_to_add_judge(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        args(*argv)
    assert exit_info.value.code == 2
    assert "add --judge" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["nan", "-0.1", "1.5", "inf", "abc"])
def test_a_judge_threshold_outside_0_to_1_is_rejected_at_parse_time(bad, capsys):
    with pytest.raises(SystemExit):
        args("--judge", "--judge-threshold", bad)
    assert "--judge-threshold" in capsys.readouterr().err


@pytest.mark.parametrize("ok", ["0", "1", "0.5"])
def test_the_ends_of_the_judge_threshold_range_are_accepted(ok):
    assert args("--judge", "--judge-threshold", ok).judge_threshold == float(ok)


def test_the_judge_needs_its_model_folder_only_when_it_is_on(tmp_path):
    assert pdf_ocr.required_model_dirs(args("--reader", "minicpm", "--judge")) == [
        "PP-DocLayoutV3", "MiniCPM-OCR-1.0", "MiniCPM-OCR-Jev"]
    assert pdf_ocr.required_model_dirs(args("--reader", "minicpm")) == ["PP-DocLayoutV3", "MiniCPM-OCR-1.0"]
    assert pdf_ocr.required_model_dirs(args("--judge")) == ["PP-DocLayoutV3", "PaddleOCR-VL-1.6", "MiniCPM-OCR-Jev"]
    assert pdf_ocr.required_model_dirs(args()) == ["PP-DocLayoutV3", "PaddleOCR-VL-1.6"]
    assert "MiniCPM-OCR-Jev" in pdf_ocr.missing_models_message(args("--judge", "--models-dir", str(tmp_path)))


def test_the_worker_is_needed_for_minicpm_or_for_the_judge_and_loads_only_what_is_used():
    assert not pdf_ocr.needs_worker(args()) and not pdf_ocr.needs_worker(args("--no-judge"))
    assert pdf_ocr.needs_worker(args("--reader", "minicpm")) and pdf_ocr.needs_worker(args("--judge"))
    assert pdf_ocr.needs_worker(args("--reader", "minicpm", "--judge"))
    cmd = lambda *a: pdf_ocr.worker_command(args("--models-dir", "/m", *a), uv="/usr/bin/uv")
    both = cmd("--reader", "minicpm", "--judge")
    assert "--no-reader" not in both and "--no-judge" not in both
    assert "--no-judge" in cmd("--reader", "minicpm") and "--no-reader" not in cmd("--reader", "minicpm")
    assert "--no-reader" in cmd("--judge") and "--no-judge" not in cmd("--judge")


READY = {"reader": "MiniCPM-OCR-1.0", "judge": "MiniCPM-OCR-Jev",
         "versions": {"torch": "2.15.0+cu132", "transformers": "5.22.2", "playwright": "1.62.0"}}


def test_the_engine_names_the_judge_when_it_is_on():
    e = pdf_ocr.engine_info(args("--reader", "minicpm", "--judge"), "gpu:0", READY)
    assert e["judge"] == "MiniCPM-OCR-Jev" and e["playwright"] == "1.62.0"
    assert "judge" not in pdf_ocr.engine_info(args("--reader", "minicpm"), "gpu:0", {**READY, "judge": None})


def test_the_paddle_reader_with_the_judge_has_the_workers_versions_too():
    e = pdf_ocr.engine_info(args("--judge"), "gpu:0", {**READY, "reader": None})
    assert e["reader"] == "PaddleOCR-VL-1.6" and e["judge"] == "MiniCPM-OCR-Jev" and e["torch"] == "2.15.0+cu132"


def test_the_settings_record_whether_the_judge_was_on_and_its_threshold():
    s = pdf_ocr.settings_info(args("--reader", "minicpm", "--judge", "--judge-threshold", "0.9"))
    assert s["judge"] is True and s["judge_threshold"] == 0.9 and s["reader"] == "minicpm"
    s = pdf_ocr.settings_info(args("--no-judge"))
    assert s["judge"] is False and s["judge_threshold"] is None
    assert s["reread_short_blocks"] is True  # the ink-based re-read is what runs without the Judge


def test_a_run_with_no_options_records_paddle_without_the_judge_and_with_the_ink_based_reread():
    s = pdf_ocr.settings_info(args())
    assert (s["reader"], s["judge"], s["judge_threshold"], s["reread_short_blocks"]) == ("paddle", False, None, True)
    assert "reread_schedule" not in s


def test_with_the_judge_on_the_ink_based_reread_is_not_run_and_the_settings_say_so():
    assert pdf_ocr.settings_info(args("--judge"))["reread_short_blocks"] is False
    assert pdf_ocr.settings_info(args("--no-recover"))["reread_short_blocks"] is False


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
        "blocks_judged": 3, "blocks_reread": 0, "blocks_below_threshold": 2, "reread_stop_reasons": {},
        "readings_beyond_the_first": 0,
        "verdicts": {"render_error": 1, "scored": 2}, "blocks_judged_before_script_decision": 1}


def test_the_postprocessing_summary_reports_re_read_blocks_and_those_still_below_the_threshold():
    r = lambda text, rnd, score, dup=0: Reading(text, rnd, "greedy", None, 1.0, verdict="scored", score=score, duplicates=dup)
    outcomes = {
        (0, (0, 0, 1, 1)): outcome(judged=True, score=0.99, verdict="scored", below_threshold=False),     # passed at once
        (0, (0, 2, 1, 3)): outcome(judged=True, score=0.99, verdict="scored", below_threshold=False,
                                   readings=[r("a", 1, 0.5), r("b", 2, 0.99)], stop_reason="passed"),
        (1, (0, 0, 1, 1)): outcome(judged=True, score=0.7, verdict="scored", below_threshold=True, stop_reason="exhausted",
                                   readings=[r("a", 1, 0.5, dup=2), r("b", 3, 0.7)]),
    }
    summary = pdf_ocr.judge_summary(outcomes)
    assert (summary["blocks_judged"], summary["blocks_reread"], summary["blocks_below_threshold"]) == (3, 2, 1)
    assert summary["reread_stop_reasons"] == {"exhausted": 1, "passed": 1}
    assert summary["readings_beyond_the_first"] == 1 + 3


def test_the_page_line_on_stderr_reports_blocks_below_the_threshold_and_blocks_re_read():
    line = pdf_ocr.page_line(2, 21, 4, 13, 7.04, summary={"judged": 9, "reread": 5, "below_threshold": 3, "not_scored": 1}, recovered=0)
    assert line == "[  2/21] page 4: 13 blocks, 7.0s, 3 of 9 judged below the threshold, 5 re-read (1 not scored)"
    plain = pdf_ocr.page_line(1, 2, 1, 5, 1.0, summary=None, recovered=2)
    assert plain == "[  1/2] page 1: 5 blocks, 1.0s, re-read 2 short block(s)"
    assert "0 re-read" in pdf_ocr.page_line(1, 2, 1, 5, 1.0, summary={"judged": 5, "reread": 0, "below_threshold": 0, "not_scored": 0}, recovered=0)


# --- a re-read Block in the JSON ----------------------------------------------------------------------------

def test_a_re_read_block_lists_its_readings_how_each_was_made_and_why_the_re_reading_stopped():
    readings = [Reading("你好", 1, "greedy", None, 1.0, verdict="scored", score=0.4, duplicates=1),
                Reading("你號", 3, "temperature", 0.3, 1.0, verdict="scored", score=0.93),
                Reading("x", 2, "greedy", None, 1.25, "page_render", "render_error", None, "render_failed")]
    o = outcome(text="你號", judged=True, score=0.93, verdict="scored", below_threshold=True, readings=readings,
                stop_reason="exhausted", skipped_rounds=[{"round": 2, "reason": "r"}])
    (b,) = page([block(block_content="你號")], {(0, (1, 2, 30, 40)): o})["blocks"]
    assert (b["content"], b["score"], b["stop_reason"]) == ("你號", 0.93, "exhausted")
    assert b["readings"] == [
        {"content": "你好", "score": 0.4, "verdict": "scored", "round": 1, "decode": "greedy", "temperature": None,
         "scale": 1.0, "duplicates": 1},
        {"content": "你號", "score": 0.93, "verdict": "scored", "round": 3, "decode": "temperature", "temperature": 0.3,
         "scale": 1.0},
        {"content": "x", "score": None, "verdict": "render_error", "round": 2, "decode": "greedy",
         "temperature": None, "scale": 1.25, "scale_source": "page_render", "verdict_detail": "render_failed"}]
    assert b["skipped_rounds"] == [{"round": 2, "reason": "r"}]


def test_a_block_that_was_not_re_read_has_no_readings_stop_reason_or_skipped_rounds():
    o = outcome(judged=True, score=0.99, verdict="scored", below_threshold=False)
    (b,) = page([block()], {(0, (1, 2, 30, 40)): o})["blocks"]
    assert not {"readings", "stop_reason", "stop_detail", "skipped_rounds"} & b.keys()


def test_the_settings_record_the_schedule_and_which_rounds_the_reader_can_do():
    s = pdf_ocr.settings_info(args("--judge"))
    assert [(r["round"], r["scale"], r["temperature"]) for r in s["reread_schedule"]] == [
        (1, 1.0, None), (2, 1.25, None), (3, 1.0, 0.3), (4, 1.0, 0.7)]
    assert "supported" not in s["reread_schedule"][0] and s["max_readings_per_block"] == 10
    bridge = types.SimpleNamespace(supports=lambda step: step.decode == "greedy")
    assert [r["supported"] for r in pdf_ocr.settings_info(args("--judge"), bridge)["reread_schedule"]] == [
        True, True, False, False]
    assert "reread_schedule" not in pdf_ocr.settings_info(args())


# --- a run that ran out of GPU memory says what to do ---------------------------------------------------

OOM_PAGE = {3: "RuntimeError: CUDA out of memory. Tried to allocate"}


@pytest.mark.parametrize("argv, ways_out", [
    ((), ["close other GPU programs", "--reader minicpm"]),
    (("--reader", "minicpm"), ["close other GPU programs", "smallest"]),
    (("--reader", "minicpm", "--judge"), ["close other GPU programs", "without --judge"]),
    (("--judge",), ["close other GPU programs", "without --judge", "--reader minicpm --judge", "tightest"]),
])
def test_the_memory_hint_names_the_ways_out_of_the_configuration_that_ran(argv, ways_out):
    hint = pdf_ocr.memory_hint(args(*argv), OOM_PAGE, {})
    assert hint.startswith("hint: the GPU ran out of memory")
    assert all(w in hint for w in ways_out)


def test_a_verdict_detail_about_memory_gives_the_hint_too():
    oom = outcome(judged=True, verdict="forward_error", verdict_detail="OutOfMemoryError: CUDA out of memory", below_threshold=True)
    assert "GPU" in pdf_ocr.memory_hint(args("--judge"), {}, {(0, (0, 0, 1, 1)): oom})


def test_a_re_read_that_ran_out_of_memory_also_gives_the_hint():
    oom = outcome(judged=True, verdict="scored", score=0.5, below_threshold=True, readings=[], stop_reason="reader_error",
                  stop_detail="round 3: worker error: OutOfMemoryError: CUDA out of memory")
    assert "GPU" in pdf_ocr.memory_hint(args("--judge"), {}, {(0, (0, 0, 1, 1)): oom})


def test_other_failures_give_no_memory_hint():
    other = outcome(judged=True, verdict="render_error", verdict_detail="render_failed", below_threshold=True)
    assert pdf_ocr.memory_hint(args("--judge"), {1: "ValueError: bad page"}, {(0, (0, 0, 1, 1)): other}) is None
    assert pdf_ocr.memory_hint(args(), {}, {}) is None


def hook_unsupported_messages():
    import reader_bridge

    class Renamed:
        vl_rec_model = None

    class Wide:
        vl_rec_model = None

        def _paddleocr_vl_collect_page_vlm_entries_core(self, page_idx, blocks_for_img, imgs_in_doc, cfg):
            return [(0, 0, "img", "OCR:", (1, 2), None, "new field")], [], []

    texts = []
    wide = types.SimpleNamespace(paddlex_pipeline=types.SimpleNamespace(_pipeline=Wide()))
    reader_bridge.install_reader_hook(wide, object())
    for attempt in (lambda: reader_bridge.check_hook_support(types.SimpleNamespace()),
                    lambda: reader_bridge.check_hook_support(
                        types.SimpleNamespace(paddlex_pipeline=types.SimpleNamespace(_pipeline=Renamed()))),
                    lambda: wide.paddlex_pipeline._pipeline._paddleocr_vl_collect_page_vlm_entries_core(
                        0, [{"label": "text", "box": [0, 0, 1, 1]}], [], {})):
        with pytest.raises(reader_bridge.HookUnsupported) as e:
            attempt()
        texts.append(str(e.value))
    return texts


def test_no_message_suggests_a_flag_that_the_new_defaults_made_meaningless(monkeypatch, tmp_path):
    monkeypatch.setenv("PDF_OCR_WORKER_ENV", str(tmp_path / "missing"))
    monkeypatch.setattr(pdf_ocr.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as no_uv:
        pdf_ocr.start_worker(args("--reader", "minicpm"))
    messages = [pdf_ocr.worker_env_problem(), str(no_uv.value.code), *hook_unsupported_messages(),
                *(pdf_ocr.memory_hint(args(*a), OOM_PAGE, {}) for a in
                  ((), ("--reader", "minicpm"), ("--judge",), ("--reader", "minicpm", "--judge")))]
    for msg in messages:
        assert "--no-judge" not in msg and "--reader paddle" not in msg, msg


def test_judged_blocks_without_an_outcome_are_found_by_page_and_box():
    block = lambda label, bbox: types.SimpleNamespace(label=label, bbox=bbox)  # noqa: E731
    res = {"parsing_res_list": [block("text", [1, 2, 3, 4]), block("text", [5, 6, 7, 8]), block("chart", [9, 9, 9, 9]),
                                block("table", [0.4, 0.6, 20.2, 30.9])]}
    outcomes = {(2, (1, 2, 3, 4)): outcome(judged=True)}
    assert pdf_ocr.blocks_without_outcome(2, res, outcomes) == [("text", (5, 6, 7, 8)), ("table", (0, 0, 20, 30))]
    assert len(pdf_ocr.blocks_without_outcome(3, res, outcomes)) == 3  # another page's outcomes do not count
