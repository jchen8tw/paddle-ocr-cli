"""The Judge's command line, the JSON it adds and the stderr line (no GPU, no models)."""
import types
from pathlib import Path

import pytest

import pdf_ocr
from block_reader import BlockResult, Reading


def args(*argv):
    return pdf_ocr.parse_args(["in.pdf", *argv])


def spec(*argv):
    return pdf_ocr.EngineSpec.from_options(args(*argv))


def fingerprint(*argv):
    options = args(*argv)
    return pdf_ocr.run_fingerprint(pdf_ocr.EngineSpec.from_options(options), options)


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
    assert "MiniCPM-OCR-Jev" in pdf_ocr.missing_models_message(spec("--judge"), tmp_path)
    assert "MiniCPM-OCR-Jev" not in pdf_ocr.missing_models_message(spec(), tmp_path)


def test_the_worker_loads_only_what_is_used():
    cmd = lambda *a: pdf_ocr.worker_command(spec(*a), Path("/m"), uv="/usr/bin/uv")
    both = cmd("--reader", "minicpm", "--judge")
    assert "--no-reader" not in both and "--no-judge" not in both
    assert "--no-judge" in cmd("--reader", "minicpm") and "--no-reader" not in cmd("--reader", "minicpm")
    assert "--no-reader" in cmd("--judge") and "--no-judge" not in cmd("--judge")


READY = {"reader": "MiniCPM-OCR-1.0", "judge": "MiniCPM-OCR-Jev",
         "versions": {"torch": "2.15.0+cu132", "transformers": "5.22.2", "playwright": "1.62.0"}}


def test_the_engine_names_the_judge_when_it_is_on():
    e = pdf_ocr.engine_info(spec("--reader", "minicpm", "--judge"), "gpu:0", READY)
    assert e["judge"] == "MiniCPM-OCR-Jev" and e["playwright"] == "1.62.0"
    assert "judge" not in pdf_ocr.engine_info(spec("--reader", "minicpm"), "gpu:0", {**READY, "judge": None})


def test_the_paddle_reader_with_the_judge_has_the_workers_versions_too():
    e = pdf_ocr.engine_info(spec("--judge"), "gpu:0", {**READY, "reader": None})
    assert e["reader"] == "PaddleOCR-VL-1.6" and e["judge"] == "MiniCPM-OCR-Jev" and e["torch"] == "2.15.0+cu132"


def test_the_settings_record_whether_the_judge_was_on_and_its_threshold():
    s = fingerprint("--reader", "minicpm", "--judge", "--judge-threshold", "0.9").settings
    assert s["judge"] is True and s["judge_threshold"] == 0.9 and s["reader"] == "minicpm"
    s = fingerprint("--no-judge").settings
    assert s["judge"] is False and s["judge_threshold"] is None
    assert s["reread_short_blocks"] is True  # the ink-based re-read is what runs without the Judge


def test_a_run_with_no_options_records_paddle_without_the_judge_and_with_the_ink_based_reread():
    s = fingerprint().settings
    assert (s["reader"], s["judge"], s["judge_threshold"], s["reread_short_blocks"]) == ("paddle", False, None, True)
    assert "reread_schedule" not in s


@pytest.mark.parametrize("argv, expected", [([], "auto"), (["--script", "auto"], "auto"), (["--script", "off"], "off")])
@pytest.mark.parametrize("judge", [[], ["--judge"]])
def test_the_settings_record_the_script_option_asked_for(argv, expected, judge):
    assert fingerprint(*argv, *judge).settings["script"] == expected


@pytest.mark.parametrize("argv, expected", [([], True), (["--no-page-markers"], False)])
@pytest.mark.parametrize("judge", [[], ["--judge"]])
def test_the_settings_record_whether_page_markers_were_asked_for(argv, expected, judge):
    assert fingerprint(*argv, *judge).settings["page_markers"] is expected


def test_with_the_judge_on_the_ink_based_reread_is_not_run_and_the_settings_say_so():
    assert fingerprint("--judge").settings["reread_short_blocks"] is False
    assert fingerprint("--no-recover").settings["reread_short_blocks"] is False


# --- the JSON of a judged Block -------------------------------------------------------------------------

def block(**kw):
    return {"block_id": 0, "block_order": 1, "block_label": "text", "block_content": "你好", "block_bbox": [1, 2, 30, 40],
            "block_polygon_points": None, **kw}


def res_for(*blocks):
    return types.SimpleNamespace(json={"res": {"width": 100, "height": 200, "parsing_res_list": list(blocks)}})


def outcome(**kw):
    kw.setdefault("text", "你好")
    return BlockResult(**kw)


def page(blocks, notes):
    return pdf_ocr.page_json(0, pdf_ocr.ReadPage(res_for(*blocks), 3.0, notes, 0, None), {}, [])


def test_a_judged_block_carries_score_verdict_and_flag():
    o = outcome(judged=True, score=0.91, verdict="scored", flagged=True)
    (b,) = page([block()], {(1, 2, 30, 40): o})["blocks"]
    assert (b["score"], b["verdict"], b["flagged"]) == (0.91, "scored", True)


def test_a_block_the_judge_could_not_score_has_a_null_score_and_the_reason():
    o = outcome(judged=True, score=None, verdict="render_error", verdict_detail="render_failed", flagged=True)
    (b,) = page([block()], {(1, 2, 30, 40): o})["blocks"]
    assert b["score"] is None and b["verdict"] == "render_error" and b["flagged"] is True
    assert b["verdict_detail"] == "render_failed"


def test_charts_seals_and_figures_have_none_of_the_judge_fields():
    o = outcome(judged=False)
    (b,) = page([block(block_label="chart")], {(1, 2, 30, 40): o})["blocks"]
    assert not {"score", "verdict", "flagged", "verdict_detail"} & b.keys()


def test_without_outcomes_the_json_is_what_it_was_before_the_judge():
    (b,) = page([block()], {})["blocks"]
    assert set(b) == {"id", "order", "label", "content", "bbox", "polygon", "in_markdown"}


def test_notes_are_matched_by_box_not_by_block_id():
    o = outcome(judged=True, score=0.5, verdict="scored", flagged=True)
    assert "score" in page([block(block_id=7, block_bbox=[1.4, 2.6, 30.2, 40.9])], {(1, 2, 30, 40): o})["blocks"][0]
    assert "score" not in page([block(block_id=7)], {(1, 2, 30, 41): o})["blocks"][0]


def test_a_block_judged_before_the_document_was_known_to_be_traditional_is_marked():
    o = outcome(judged=True, score=0.9, verdict="scored", flagged=True, judged_before_script_decision=True)
    (b,) = page([block()], {(1, 2, 30, 40): o})["blocks"]
    assert b["judged_before_script_decision"] is True


def test_the_text_that_was_judged_is_recorded_when_it_differs_from_the_text_that_was_written():
    o = outcome(text="银行", judged=True, score=0.9, verdict="scored", flagged=True)
    (b,) = page([block(block_content="銀行")], {(1, 2, 30, 40): o})["blocks"]
    assert b["judged_text"] == "银行"
    (b,) = page([block(block_content="银行")], {(1, 2, 30, 40): o})["blocks"]
    assert "judged_text" not in b


def test_the_ink_reread_of_a_bridge_block_is_recorded_like_the_post_pass_one():
    o = outcome(text_before_reread="x")
    (b,) = page([block()], {(1, 2, 30, 40): o})["blocks"]
    assert b["content_before_reread"] == "x"


# --- the summary, stderr and the postprocessing section ------------------------------------------------

def test_the_postprocessing_summary_counts_blocks_verdicts_and_early_judgements():
    outcomes = {
        (0, (0, 0, 1, 1)): outcome(judged=True, score=0.99, verdict="scored", flagged=False),
        (0, (0, 2, 1, 3)): outcome(judged=True, score=0.5, verdict="scored", flagged=True, judged_before_script_decision=True),
        (1, (0, 0, 1, 1)): outcome(judged=True, verdict="render_error", flagged=True),
        (1, (0, 5, 1, 6)): outcome(judged=False),
    }
    assert pdf_ocr.judge_summary(outcomes) == {
        "blocks_judged": 3, "blocks_reread": 0, "blocks_flagged": 2, "reread_stop_reasons": {},
        "readings_beyond_the_first": 0,
        "verdicts": {"render_error": 1, "scored": 2}, "blocks_judged_before_script_decision": 1}


def test_the_postprocessing_summary_reports_re_read_blocks_and_flagged_blocks():
    r = lambda text, rnd, score, dup=0: Reading(text, rnd, "greedy", None, 1.0, verdict="scored", score=score, duplicates=dup)
    outcomes = {
        (0, (0, 0, 1, 1)): outcome(judged=True, score=0.99, verdict="scored", flagged=False),     # passed at once
        (0, (0, 2, 1, 3)): outcome(judged=True, score=0.99, verdict="scored", flagged=True,
                                   readings=[r("a", 1, 0.5), r("b", 2, 0.99)], stop_reason="passed"),
        (1, (0, 0, 1, 1)): outcome(judged=True, score=0.7, verdict="scored", flagged=True, stop_reason="exhausted",
                                   readings=[r("a", 1, 0.5, dup=2), r("b", 3, 0.7)]),
    }
    summary = pdf_ocr.judge_summary(outcomes)
    assert (summary["blocks_judged"], summary["blocks_reread"], summary["blocks_flagged"]) == (3, 2, 2)
    assert summary["reread_stop_reasons"] == {"exhausted": 1, "passed": 1}
    assert summary["readings_beyond_the_first"] == 1 + 3


def test_the_page_line_on_stderr_reports_flagged_blocks_and_blocks_re_read():
    line = pdf_ocr.page_line(2, 21, 4, 13, 7.04, summary={"judged": 9, "reread": 5, "flagged": 3, "not_scored": 1}, recovered=0)
    assert line == "[  2/21] page 4: 13 blocks, 7.0s, 3 of 9 judged flagged, 5 re-read (1 not scored)"
    plain = pdf_ocr.page_line(1, 2, 1, 5, 1.0, summary=None, recovered=2)
    assert plain == "[  1/2] page 1: 5 blocks, 1.0s, re-read 2 short block(s)"
    assert "0 re-read" in pdf_ocr.page_line(1, 2, 1, 5, 1.0, summary={"judged": 5, "reread": 0, "flagged": 0, "not_scored": 0}, recovered=0)


# --- a re-read Block in the JSON ----------------------------------------------------------------------------

def test_a_re_read_block_lists_its_readings_how_each_was_made_and_why_the_re_reading_stopped():
    readings = [Reading("你好", 1, "greedy", None, 1.0, verdict="scored", score=0.4, duplicates=1),
                Reading("你號", 3, "temperature", 0.3, 1.0, verdict="scored", score=0.93),
                Reading("x", 2, "greedy", None, 1.25, "page_render", "render_error", None, "render_failed")]
    o = outcome(text="你號", judged=True, score=0.93, verdict="scored", flagged=True, readings=readings,
                stop_reason="exhausted", skipped_rounds=[{"round": 2, "reason": "r"}])
    (b,) = page([block(block_content="你號")], {(1, 2, 30, 40): o})["blocks"]
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
    o = outcome(judged=True, score=0.99, verdict="scored", flagged=False)
    (b,) = page([block()], {(1, 2, 30, 40): o})["blocks"]
    assert not {"readings", "stop_reason", "stop_detail", "skipped_rounds"} & b.keys()


def test_the_settings_record_the_schedule_and_which_rounds_the_reader_can_do():
    s = fingerprint("--judge").settings
    assert [(r["round"], r["scale"], r["temperature"]) for r in s["reread_schedule"]] == [
        (1, 1.0, None), (2, 1.25, None), (3, 1.0, 0.3), (4, 1.0, 0.7)]
    assert [r["supported"] for r in s["reread_schedule"]] == [True, True, False, False]
    assert s["max_readings_per_block"] == 10
    minicpm = fingerprint("--reader", "minicpm", "--judge").settings["reread_schedule"]
    assert [r["supported"] for r in minicpm] == [True] * 4
    assert "reread_schedule" not in fingerprint().settings


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
        pdf_ocr.start_worker(spec("--reader", "minicpm"), tmp_path)
    messages = [pdf_ocr.worker_env_problem(), str(no_uv.value.code), *hook_unsupported_messages(),
                *(spec(*a).memory_advice for a in
                  ((), ("--reader", "minicpm"), ("--judge",), ("--reader", "minicpm", "--judge")))]
    for msg in messages:
        assert "--no-judge" not in msg and "--reader paddle" not in msg, msg
