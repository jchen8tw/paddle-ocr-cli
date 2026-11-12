"""A whole Document in each of the four Reader x Judge setups, without a GPU: the safety net under the Engine.

The Engine is built from an option list by `fakes.engine_for`, the way a Run builds it. Only the PaddleX pipe and the
worker are fakes; the pages, the post-processing and the files are the real ones. The tests check what a user sees:
the JSON, the Markdown and stderr.
"""
import json
import re
from dataclasses import dataclass

import pytest

import pdf_ocr
from fakes import INK_BOX, FakeClient, blank_pdf, engine_for, inked_pdf
from worker_client import WorkerError

# (options, Reader folder in the JSON, Judge folder in the JSON or None, whether the setup starts the worker)
SETUPS = [
    pytest.param([], "PaddleOCR-VL-1.6", None, False, id="paddleocr-vl"),
    pytest.param(["--judge"], "PaddleOCR-VL-1.6", "MiniCPM-OCR-Jev", True, id="paddleocr-vl+judge"),
    pytest.param(["--reader", "minicpm"], "MiniCPM-OCR-1.0", None, True, id="minicpm"),
    pytest.param(["--reader", "minicpm", "--judge"], "MiniCPM-OCR-1.0", "MiniCPM-OCR-Jev", True, id="minicpm+judge"),
]
JUDGED = [s for s in SETUPS if s.values[2]]


@dataclass
class Written:
    errors: dict
    json: dict
    markdown: str
    stderr: str

    @property
    def blocks(self):
        return [b for page in self.json["pages"] for b in page["blocks"]]


def transcribe(engine, options, pdf, out, capsys) -> Written:
    """Transcribe one Document with `options` and read back what a user gets."""
    args = pdf_ocr.parse_args([str(pdf), "--no-csv", *options])
    capsys.readouterr()
    document = pdf_ocr.open_document(pdf, None)
    try:
        errors = pdf_ocr.transcribe_document(engine, args, document, out)
    finally:
        document.close()
    stderr = capsys.readouterr().err
    return Written(errors, json.loads((out / f"{pdf.stem}.json").read_text(encoding="utf-8")),
                   (out / f"{pdf.stem}.md").read_text(encoding="utf-8"), stderr)


@pytest.mark.parametrize("options, reader, judge, worker", SETUPS)
def test_a_document_runs_end_to_end_and_its_json_records_the_reader_and_the_judge_of_the_setup(
        tmp_path, capsys, options, reader, judge, worker):
    engine, fakes = engine_for(options, tmp_path / "engine")
    fakes.read_as("some words")
    done = transcribe(engine, options, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    assert done.errors == {} and done.json["source"]["pages_failed"] == []
    assert done.json["engine"]["reader"] == reader
    assert done.json["engine"].get("judge") == judge
    assert done.json["settings"]["reader"] == ("minicpm" if "minicpm" in options else "paddle")
    assert done.json["settings"]["judge"] == bool(judge)
    assert [b["content"] for b in done.blocks] == ["some words", "some words"]
    assert done.markdown.count("some words") == 2


@pytest.mark.parametrize("options, reader, judge, worker", SETUPS)
def test_the_worker_and_the_judge_leave_their_marks_in_the_json_only_when_the_setup_has_them(
        tmp_path, capsys, options, reader, judge, worker):
    engine, fakes = engine_for(options, tmp_path / "engine")
    fakes.read_as("some words")
    done = transcribe(engine, options, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    assert ("torch" in done.json["engine"]) == worker  # the library versions the worker reports
    post = done.json["postprocessing"]
    assert ("repetition_cuts" in post) == worker
    assert ("judge" in post) == bool(judge)
    assert all(("score" in b) == bool(judge) for b in done.blocks)
    assert ("reread_schedule" in done.json["settings"]) == bool(judge)
    if judge:
        assert post["judge"]["blocks_judged"] == 2 and {b["verdict"] for b in done.blocks} == {"scored"}


@pytest.mark.parametrize("options, reader, judge, worker", SETUPS)
def test_the_progress_line_names_the_judge_only_when_the_setup_has_it(tmp_path, capsys, options, reader, judge, worker):
    engine, fakes = engine_for(options, tmp_path / "engine")
    done = transcribe(engine, options, blank_pdf(tmp_path / "doc.pdf", pages=2), tmp_path / "out", capsys)
    lines = [ln for ln in done.stderr.splitlines() if ln.startswith("[")]
    assert [ln.split(" blocks")[0] for ln in lines] == ["[  1/2] page 1: 2", "[  2/2] page 2: 2"]
    assert all(("judged flagged" in ln) == bool(judge) for ln in lines)
    assert done.stderr.splitlines()[-1].startswith("wrote ")


@pytest.mark.parametrize("options, reader, judge, worker", JUDGED)
@pytest.mark.parametrize("threshold, flagged", [("0.4", False), ("0.98", True)])
def test_the_judge_flags_the_blocks_that_score_below_the_threshold_of_the_options(
        tmp_path, capsys, options, reader, judge, worker, threshold, flagged):
    options = [*options, "--judge-threshold", threshold]
    engine, fakes = engine_for(options, tmp_path / "engine")  # every Reading scores 0.5
    done = transcribe(engine, options, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    assert done.json["settings"]["judge_threshold"] == float(threshold)
    assert [b["flagged"] for b in done.blocks] == [flagged, flagged]
    assert all(("readings" in b) == flagged for b in done.blocks)  # a low Score is re-read, a high one is not


# --- the sticky script repair under the Judge ---------------------------------------------------------------------------

TRAD = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"
SIMPLIFIED, REPAIRED = "银行医院", "銀行醫院"
SCORES = {REPAIRED: ("scored", 0.99), SIMPLIFIED: ("scored", 0.2)}  # the Judge likes the Traditional text, not the other


@pytest.mark.parametrize("options, reader, judge, worker", JUDGED)
def test_the_judge_scores_the_repaired_text_of_a_block_read_after_the_document_proved_traditional(
        tmp_path, capsys, options, reader, judge, worker):
    engine, fakes = engine_for(options, tmp_path / "engine")
    fakes.worker.scores = SCORES
    fakes.read_as(TRAD, SIMPLIFIED)  # the first Block makes the Document Traditional; the second is Simplified
    done = transcribe(engine, options, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    _, second = done.blocks
    assert (second["score"], second["flagged"]) == (0.99, False)  # judged as 銀行醫院
    assert (second["content"], second["content_raw"]) == (REPAIRED, SIMPLIFIED)  # the Document's own repair decided alone


@pytest.mark.parametrize("options, reader, judge, worker", JUDGED)
def test_with_the_script_repair_off_the_judge_scores_the_text_as_read(tmp_path, capsys, options, reader, judge, worker):
    options = [*options, "--script", "off"]
    engine, fakes = engine_for(options, tmp_path / "engine")
    fakes.worker.scores = SCORES
    fakes.read_as(TRAD, SIMPLIFIED)
    done = transcribe(engine, options, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    _, second = done.blocks
    assert (second["score"], second["flagged"], second["content"]) == (0.2, True, SIMPLIFIED)


# --- the ink-based Re-read of a Block whose text is implausibly short for the ink it covers ---------------------------

LONGER = "twelve words"  # 11 characters: clearly longer than "a", and plausible for the ink (about 10 glyphs)


def short_block_over_ink(fakes):
    fakes.pipe.blocks = [("text", INK_BOX)]
    fakes.read_as("a")


def test_without_options_a_short_block_over_a_line_of_ink_is_re_read_through_the_pipe(tmp_path, capsys):
    engine, fakes = engine_for([], tmp_path / "engine")
    short_block_over_ink(fakes)
    fakes.pipe.rereads = LONGER
    done = transcribe(engine, [], inked_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    (block,) = done.blocks
    assert (block["content_before_reread"], block["content"]) == ("a", LONGER)
    assert LONGER in done.markdown
    assert "re-read 1 short block(s)" in done.stderr
    assert done.json["settings"]["reread_short_blocks"] is True


def test_without_options_a_short_block_over_no_ink_is_left_alone(tmp_path, capsys):
    engine, fakes = engine_for([], tmp_path / "engine")
    short_block_over_ink(fakes)
    fakes.pipe.rereads = LONGER
    done = transcribe(engine, [], blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    (block,) = done.blocks
    assert block["content"] == "a" and "content_before_reread" not in block
    assert "short block" not in done.stderr


def test_no_recover_turns_the_re_read_of_a_short_block_off(tmp_path, capsys):
    engine, fakes = engine_for(["--no-recover"], tmp_path / "engine")
    short_block_over_ink(fakes)
    fakes.pipe.rereads = LONGER
    done = transcribe(engine, ["--no-recover"], inked_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    (block,) = done.blocks
    assert block["content"] == "a" and "content_before_reread" not in block
    assert done.json["settings"]["reread_short_blocks"] is False


def test_with_minicpm_alone_a_short_block_over_a_line_of_ink_is_re_read_by_the_worker(tmp_path, capsys):
    options = ["--reader", "minicpm"]
    engine, fakes = engine_for(options, tmp_path / "engine")
    fakes.pipe.blocks = [("text", INK_BOX)]
    fakes.worker.texts = {"Text Recognition:": ["a", LONGER]}  # the first Reading, then the re-reads (the last repeats)
    done = transcribe(engine, options, inked_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    (block,) = done.blocks
    assert (block["content_before_reread"], block["content"]) == ("a", LONGER)
    assert re.search(r"re-read 1 short block\(s\)", done.stderr)


# --- a Re-read that fails keeps the first Reading; a first read that fails fails the page --------------------------------

OUT_OF_MEMORY = "CUDA error: out of memory"


class WorkerThatFailsReReads(FakeClient):
    """The first Reading of a Block succeeds; every read after it (the Re-reads) runs out of memory."""

    def read(self, image, prompt, max_new_tokens, **kw):
        if self.calls:
            raise WorkerError(f"worker error: {OUT_OF_MEMORY}")
        return super().read(image, prompt, max_new_tokens, **kw)


def engine_whose_reread_runs_out_of_memory(options, tmp_path):
    """An Engine for `options` whose one Block reads as "a" over a line of ink, and whose Re-read of it fails."""
    if "minicpm" in options:
        engine, fakes = engine_for(options, tmp_path, worker=WorkerThatFailsReReads())
    else:
        engine, fakes = engine_for(options, tmp_path)
        fakes.pipe.reread_fails = RuntimeError(OUT_OF_MEMORY)
    short_block_over_ink(fakes)
    return engine


@pytest.mark.parametrize("options", [pytest.param([], id="paddleocr-vl"),
                                     pytest.param(["--reader", "minicpm"], id="minicpm")])
def test_a_re_read_that_runs_out_of_memory_keeps_the_first_reading_and_does_not_fail_the_page(
        tmp_path, capsys, options):
    engine = engine_whose_reread_runs_out_of_memory(options, tmp_path / "engine")
    done = transcribe(engine, options, inked_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    assert done.errors == {} and done.json["source"]["pages_failed"] == []
    (block,) = done.blocks
    assert block["content"] == "a" and "content_before_reread" not in block
    assert "a" in done.markdown.splitlines() and "failed" not in done.markdown
    warnings = [ln for ln in done.stderr.splitlines() if ln.startswith("warning:")]
    assert len(warnings) == 1
    assert "page 1" in warnings[0] and str(INK_BOX) in warnings[0] and OUT_OF_MEMORY in warnings[0]
    assert "keeping its first reading" in warnings[0]
    assert "hint:" not in done.stderr  # not a page failure, so no advice on what to do about one
    assert "short block" not in done.stderr and done.stderr.splitlines()[-1].startswith("wrote ")


def test_a_re_read_that_fails_does_not_stop_the_re_read_of_the_next_block(tmp_path, capsys):
    engine, fakes = engine_for([], tmp_path / "engine")
    fakes.pipe.blocks = [("text", INK_BOX), ("text", INK_BOX)]  # both over the one line of ink
    fakes.pipe.reread_fails = RuntimeError(OUT_OF_MEMORY)
    fakes.read_as("a")
    done = transcribe(engine, [], inked_pdf(tmp_path / "doc.pdf"), tmp_path / "out", capsys)
    assert done.errors == {} and [b["content"] for b in done.blocks] == ["a", "a"]
    assert done.stderr.count("warning:") == 2  # each Block's Re-read was tried and each failure reported


def test_without_options_a_page_whose_first_read_fails_is_a_failed_page_and_the_rest_of_the_document_is_still_written(
        tmp_path, capsys):
    engine, fakes = engine_for([], tmp_path / "engine")
    fakes.read_as("some words")
    fakes.pipe.read_fails = {1: RuntimeError(OUT_OF_MEMORY)}
    done = transcribe(engine, [], blank_pdf(tmp_path / "doc.pdf", pages=2), tmp_path / "out", capsys)
    assert list(done.errors) == [0] and done.json["source"]["pages_failed"] == [1]
    assert f"[  1/2] page 1: FAILED RuntimeError: {OUT_OF_MEMORY}" in done.stderr
    assert "hint: the GPU ran out of memory" in done.stderr
    first, second = done.json["pages"]
    assert first["blocks"] == [] and OUT_OF_MEMORY in first["error"]
    assert [b["content"] for b in second["blocks"]] == ["some words", "some words"]
    assert "<!-- page 1 failed:" in done.markdown and done.markdown.count("some words") == 2
