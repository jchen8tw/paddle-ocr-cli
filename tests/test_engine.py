"""One loaded Engine reads the pages of one Document after another: what it hands back is written as the Document's
JSON and stderr, and nothing of a Document may reach the next.

The Engine is built from the options by `fakes.engine_for`, with the PaddleX pipe and the worker faked; the pages, the
post-processing and the files are the real ones. No GPU, no models.
"""
import json

import pytest

import pdf_ocr
from fakes import INK_BOX, SETUPS, blank_pdf, engine_for, inked_pdf
from worker_client import WorkerError

OPTIONS = ["--reader", "minicpm", "--judge", "--judge-threshold", "0.98"]  # these tests need the Judge's counters
TRAD = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"
LOOP = "前言" + "ABCDEFGHIJ" * 20  # a repetition loop, cut back to "前言" and counted

every_setup = pytest.mark.parametrize("setup", SETUPS, ids=lambda setup: setup.id)
judged_setups = pytest.mark.parametrize("setup", [s for s in SETUPS if s.judge], ids=lambda setup: setup.id)


def transcribe(engine, pdf, out, options=OPTIONS):
    args = pdf_ocr.parse_args([str(pdf), "--no-csv", *options])
    assert pdf_ocr.transcribe_document(engine, args, pdf_ocr.open_document(pdf, None), out) == {}
    doc = json.loads((out / f"{pdf.stem}.json").read_text(encoding="utf-8"))
    del doc["elapsed_seconds"]
    return doc, (out / f"{pdf.stem}.md").read_text(encoding="utf-8")


def test_a_document_after_another_in_the_same_engine_comes_out_as_if_it_were_alone(tmp_path):
    first, second = blank_pdf(tmp_path / "first.pdf", pages=2), blank_pdf(tmp_path / "second.pdf", height=301)
    engine, fakes = engine_for(OPTIONS, tmp_path)
    client = fakes.worker
    client.texts = {"Text Recognition:": TRAD + LOOP}  # a Traditional document, with repetition loops
    a, _ = transcribe(engine, first, tmp_path / "a")
    post = a["postprocessing"]
    assert post["repetition_cuts"] > 0 and post["judge"]["blocks_judged"] == 4
    assert post["judge"]["scaled_images_by_source"]  # the 1.25x Round made scaled images

    client.texts = {"Text Recognition:": "银行医院"}  # Simplified: the sticky script repair must not have been decided
    in_the_same_engine = transcribe(engine, second, tmp_path / "b1")

    alone_engine, alone_fakes = engine_for(OPTIONS, tmp_path / "alone")
    alone_client = alone_fakes.worker
    alone_client.texts = {"Text Recognition:": "银行医院"}
    assert in_the_same_engine == transcribe(alone_engine, second, tmp_path / "b2")


def test_with_paddleocr_vl_alone_a_document_after_another_in_the_same_engine_comes_out_as_if_it_were_alone(tmp_path):
    first, second = inked_pdf(tmp_path / "first.pdf"), blank_pdf(tmp_path / "second.pdf", height=301)
    engine, fakes = engine_for([], tmp_path)
    fakes.pipe.blocks, fakes.pipe.rereads = [("text", INK_BOX)], "twelve words"
    fakes.read_as("a")  # short for the line of ink on the first Document's page, and for nothing on the second's
    a, _ = transcribe(engine, first, tmp_path / "a", [])
    assert a["pages"][0]["blocks"][0]["content_before_reread"] == "a"

    in_the_same_engine = transcribe(engine, second, tmp_path / "b1", [])
    assert in_the_same_engine[0]["pages"][0]["blocks"][0]["content"] == "a"  # the first Document's ink is gone

    alone_engine, alone_fakes = engine_for([], tmp_path / "alone")
    alone_fakes.pipe.blocks, alone_fakes.pipe.rereads = [("text", INK_BOX)], "twelve words"
    alone_fakes.read_as("a")
    assert in_the_same_engine == transcribe(alone_engine, second, tmp_path / "b2", [])


def test_the_lone_symbols_one_document_unwrapped_are_not_counted_for_the_next(tmp_path):
    first, second = blank_pdf(tmp_path / "first.pdf"), blank_pdf(tmp_path / "second.pdf", height=301)
    engine, fakes = engine_for(OPTIONS, tmp_path)
    client = fakes.worker
    client.texts = {"Text Recognition:": r"年滿15足歲\geq者"}
    a, _ = transcribe(engine, first, tmp_path / "a")
    assert a["postprocessing"]["latex_symbols_unwrapped"] > 0

    client.texts = {"Text Recognition:": "no symbol here"}
    in_the_same_engine = transcribe(engine, second, tmp_path / "b1")
    assert in_the_same_engine[0]["postprocessing"]["latex_symbols_unwrapped"] == 0

    alone_engine, alone_fakes = engine_for(OPTIONS, tmp_path / "alone")
    alone_client = alone_fakes.worker
    alone_client.texts = {"Text Recognition:": "no symbol here"}
    assert in_the_same_engine == transcribe(alone_engine, second, tmp_path / "b2")


def test_with_paddleocr_vl_alone_the_lone_symbols_one_document_unwrapped_are_not_counted_for_the_next(tmp_path):
    first, second = blank_pdf(tmp_path / "first.pdf"), blank_pdf(tmp_path / "second.pdf", height=301)
    engine, fakes = engine_for([], tmp_path)
    fakes.read_as(r"年滿15足歲\geq者")
    a, _ = transcribe(engine, first, tmp_path / "a", [])
    assert a["postprocessing"]["latex_symbols_unwrapped"] > 0

    fakes.read_as("no symbol here")
    in_the_same_engine = transcribe(engine, second, tmp_path / "b1", [])
    assert in_the_same_engine[0]["postprocessing"]["latex_symbols_unwrapped"] == 0

    alone_engine, alone_fakes = engine_for([], tmp_path / "alone")
    alone_fakes.read_as("no symbol here")
    assert in_the_same_engine == transcribe(alone_engine, second, tmp_path / "b2", [])


@judged_setups
def test_each_page_of_a_document_has_the_scores_of_its_own_blocks(tmp_path, setup):
    engine, fakes = engine_for(setup.options, tmp_path / "engine")
    texts = {"one": 0.61, "two": 0.62, "three": 0.63, "four": 0.64}  # the Blocks have the same boxes on every page
    fakes.worker.scores = {text: ("scored", score) for text, score in texts.items()}
    fakes.read_as(*texts)
    doc, _ = transcribe(engine, blank_pdf(tmp_path / "doc.pdf", pages=2), tmp_path / "out", setup.options)
    blocks = [b for page in doc["pages"] for b in page["blocks"]]
    assert [(b["content"], b["score"]) for b in blocks] == list(texts.items())
    assert not any("judged_text" in b for b in blocks)


# --- what the pipeline does to a Block after the Reader read it -------------------------------------------------------

SIMPLIFIED = "银行医院" * 5  # enough Simplified to keep the whole Document from being Traditional
REPAIRED = "銀行醫院" * 5


def after_reading(pipe, change):
    """The pipeline's own post-processing calls `change` on the Blocks of every page after the Reader read them."""
    read = pipe.predict

    def predict(*args, **kwargs):
        results = read(*args, **kwargs)
        change(results[0]["parsing_res_list"])
        return results

    pipe.predict = predict


@judged_setups
def test_a_block_the_sticky_repair_changed_is_written_as_the_reader_read_it_when_the_document_is_not_traditional(
        tmp_path, setup):
    engine, fakes = engine_for(setup.options, tmp_path / "engine")
    fakes.read_as(TRAD, SIMPLIFIED)  # the first Block decides the sticky repair; the whole Document is not Traditional
    doc, markdown = transcribe(engine, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", setup.options)
    _, second = doc["pages"][0]["blocks"]
    assert doc["postprocessing"]["script_repair"]["applied"] is False
    assert (second["content"], second["judged_text"]) == (SIMPLIFIED, REPAIRED)  # the Score describes the repaired text
    assert SIMPLIFIED in markdown and REPAIRED not in markdown


@judged_setups
def test_a_block_the_pipeline_changed_after_the_reader_read_it_keeps_the_pipelines_text(tmp_path, setup):
    engine, fakes = engine_for(setup.options, tmp_path / "engine")
    fakes.read_as(TRAD, SIMPLIFIED)
    after_reading(fakes.pipe, lambda blocks: setattr(blocks[1], "content", "rewritten by the pipeline"))
    doc, markdown = transcribe(engine, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", setup.options)
    _, second = doc["pages"][0]["blocks"]
    assert (second["content"], second["judged_text"]) == ("rewritten by the pipeline", REPAIRED)
    assert SIMPLIFIED not in markdown


def page_by_page(pipe, *pages):
    """The n-th page the pipe reads has the Blocks of `pages[n]`, each (label, the box it is read at, the box the
    pipeline moves it to after reading)."""
    read, pages = pipe.predict, iter(pages)

    def predict(*args, **kwargs):
        blocks = next(pages)
        pipe.blocks = [(label, box) for label, box, _ in blocks]
        results = read(*args, **kwargs)
        for b, (_, _, moved_to) in zip(results[0]["parsing_res_list"], blocks):
            b.bbox = moved_to
        return results

    pipe.predict = predict


@every_setup
def test_a_judged_block_whose_box_the_pipeline_moved_after_reading_has_no_score_and_only_the_judge_warns(
        tmp_path, capsys, setup):
    engine, fakes = engine_for(setup.options, tmp_path / "engine")
    page_by_page(fakes.pipe,
                 [("text", (20, 20, 36, 30), (20, 20, 36, 30))],  # page 2's text is moved onto the box of this Score
                 [("chart", (0, 0, 16, 8), (0.4, 0.6, 16.2, 10.9)),
                  ("text", (20, 20, 36, 28), (20.4, 20.6, 36.2, 30.9))])
    capsys.readouterr()
    doc, _ = transcribe(engine, blank_pdf(tmp_path / "doc.pdf", pages=2), tmp_path / "out", setup.options)
    first, second = doc["pages"]
    assert not any("score" in b for b in second["blocks"])
    err = capsys.readouterr().err.splitlines()
    warnings = [ln for ln in err if ln.startswith("warning:")]
    if not setup.judge:
        assert warnings == []
        return
    assert "score" in first["blocks"][0]
    (warning,) = warnings  # a chart is never judged, so it has no Score to miss
    assert warning.startswith("warning: page 2: the text Block at (20, 20, 36, 30) has no Score")
    assert err.index(warning) < next(i for i, ln in enumerate(err) if ln.startswith("[  2/2] page 2:"))


# --- what the Engine tells about a whole Document: the memory hint and the `engine` section ---------------------------

OUT_OF_MEMORY = "CUDA error: out of memory"
WAYS_OUT = {
    "paddleocr-vl": "try --reader minicpm, which needs less GPU memory",
    "paddleocr-vl+judge": "PaddleOCR-VL with the Judge is the tightest combination",
    "minicpm": "MiniCPM-OCR-1.0 without the Judge is the smallest configuration",
    "minicpm+judge": "must fit side by side; close other GPU programs, or run without --judge",
}


def run(engine, pdf, out, options, capsys):
    """Transcribe a Document. Returns its JSON and its lines on stderr."""
    capsys.readouterr()
    document = pdf_ocr.open_document(pdf, None)
    try:
        pdf_ocr.transcribe_document(engine, pdf_ocr.parse_args([str(pdf), "--no-csv", *options]), document, out)
    finally:
        document.close()
    return json.loads((out / f"{pdf.stem}.json").read_text(encoding="utf-8")), capsys.readouterr().err.splitlines()


def hint(err):
    """The memory hint, or None."""
    assert err[-1].startswith("wrote ")
    hints = [ln for ln in err if ln.startswith("hint:")]
    assert hints in ([], [err[-2]])
    return hints[0] if hints else None


def the_first_page_fails(pipe, error):
    read, calls = pipe.predict, []

    def predict(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise error
        return read(*args, **kwargs)

    pipe.predict = predict


def every_read_after_the_first_runs_out_of_memory(fakes):
    reader, name = (fakes.paddle, "predict") if fakes.paddle is not None else (fakes.worker, "read")
    read, calls = getattr(reader, name), []

    def read_or_fail(*args, **kwargs):
        calls.append(args)
        if len(calls) > 1:
            raise RuntimeError(OUT_OF_MEMORY)
        return read(*args, **kwargs)

    setattr(reader, name, read_or_fail)


@every_setup
def test_a_page_that_runs_out_of_memory_prints_the_memory_hint_of_the_setup_before_the_wrote_line(
        tmp_path, capsys, setup):
    engine, fakes = engine_for(setup.options, tmp_path / "engine")
    the_first_page_fails(fakes.pipe, RuntimeError(OUT_OF_MEMORY))
    doc, err = run(engine, blank_pdf(tmp_path / "doc.pdf", pages=2), tmp_path / "out", setup.options, capsys)
    assert doc["source"]["pages_failed"] == [1]
    assert hint(err).startswith("hint: the GPU ran out of memory: ") and WAYS_OUT[setup.id] in hint(err)


@judged_setups
def test_a_verdict_that_says_the_gpu_ran_out_of_memory_prints_the_memory_hint_of_the_setup(tmp_path, capsys, setup):
    engine, fakes = engine_for(setup.options, tmp_path / "engine")
    fakes.worker.fail_score = WorkerError(f"worker error: {OUT_OF_MEMORY}")
    doc, err = run(engine, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", setup.options, capsys)
    assert doc["source"]["pages_failed"] == []
    blocks = doc["pages"][0]["blocks"]
    assert all(b["verdict"] == "forward_error" and OUT_OF_MEMORY in b["verdict_detail"] for b in blocks)
    assert WAYS_OUT[setup.id] in hint(err)


@judged_setups
def test_a_re_read_round_that_runs_out_of_memory_prints_the_memory_hint_of_the_setup(tmp_path, capsys, setup):
    options = [*setup.options, "--judge-threshold", "0.98"]
    engine, fakes = engine_for(options, tmp_path / "engine")
    fakes.pipe.blocks = [("text", (0, 0, 16, 8))]
    every_read_after_the_first_runs_out_of_memory(fakes)
    doc, err = run(engine, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", options, capsys)
    (block,) = doc["pages"][0]["blocks"]
    assert block["stop_reason"] == "reader_error" and OUT_OF_MEMORY in block["stop_detail"]
    assert WAYS_OUT[setup.id] in hint(err)


@every_setup
def test_a_page_and_verdicts_that_fail_for_another_reason_print_no_memory_hint(tmp_path, capsys, setup):
    engine, fakes = engine_for(setup.options, tmp_path / "engine")
    fakes.read_as("some words")
    if setup.judge:
        fakes.worker.scores = {"some words": ("render_error", None)}
    the_first_page_fails(fakes.pipe, ValueError("bad page"))
    doc, err = run(engine, blank_pdf(tmp_path / "doc.pdf", pages=2), tmp_path / "out", setup.options, capsys)
    assert doc["source"]["pages_failed"] == [1]
    assert {b.get("verdict") for b in doc["pages"][1]["blocks"]} == ({"render_error"} if setup.judge else {None})
    assert hint(err) is None


@every_setup
def test_a_document_after_one_that_ran_out_of_memory_in_the_same_engine_gets_no_memory_hint(tmp_path, capsys, setup):
    engine, fakes = engine_for(setup.options, tmp_path / "engine")
    the_first_page_fails(fakes.pipe, RuntimeError(OUT_OF_MEMORY))
    _, first = run(engine, blank_pdf(tmp_path / "first.pdf"), tmp_path / "a", setup.options, capsys)
    _, second = run(engine, blank_pdf(tmp_path / "second.pdf"), tmp_path / "b", setup.options, capsys)
    assert hint(first) is not None and hint(second) is None


@every_setup
def test_the_engine_section_names_the_models_and_the_libraries_and_the_device_in_their_order(tmp_path, setup):
    engine, _ = engine_for(setup.options, tmp_path / "engine")
    doc, _ = transcribe(engine, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", setup.options)
    paddleocr_vl = "minicpm" not in setup.options
    assert list(doc["engine"]) == [
        "layout_model", "reader", *(["pipeline", "vl_model"] if paddleocr_vl else []), "paddleocr", "paddlex",
        *(["judge"] if setup.judge else []), *(["torch"] if setup.worker else []), "device"]
    assert (doc["engine"]["reader"], doc["engine"].get("judge"), doc["engine"]["device"]) == (
        setup.reader, setup.judge, "cpu")
