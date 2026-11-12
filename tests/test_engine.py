"""One loaded Engine reads the pages of one Document after another: what it hands back is written as the Document's
JSON and stderr, and nothing of a Document may reach the next.

The Engine is built from the options by `fakes.engine_for`, with the PaddleX pipe and the worker faked; the pages, the
post-processing and the files are the real ones. No GPU, no models.
"""
import json
import types

import numpy as np
import pytest

import pdf_ocr
from fakes import INK_BOX, SETUPS, blank_pdf, engine_for, inked_pdf
from reader_bridge import BlockQuery

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


class PreviousDocumentsPage:
    """A page of the previous Document: anything that touches it is a leak."""

    def original(self, bbox):
        raise AssertionError("the previous Document's page was used")

    crop = region = original


def test_a_new_document_starts_on_page_one_without_the_previous_documents_page(tmp_path):
    engine, _ = engine_for(OPTIONS, tmp_path)
    engine.bridge.current_page, engine.bridge.page_source = 7, PreviousDocumentsPage()
    engine.begin_document("ab" * 32)
    q = BlockQuery("OCR:")
    q.label, q.bbox = "text", (0, 0, 16, 8)
    list(engine.bridge.predict([{"image": np.full((8, 16, 3), 255, np.uint8), "query": q}], max_new_tokens=8192))
    assert list(engine.bridge.outcomes) == [(0, (0, 0, 16, 8))]  # not page 8
    assert list(engine.bridge.scale_sources) == ["crop_resample"]  # the 1.25x Round never saw the old page


def test_a_new_document_starts_without_the_ink_rereads_page(tmp_path):
    engine, _ = engine_for(OPTIONS, tmp_path)
    engine.ink.set_page(PreviousDocumentsPage())
    engine.begin_document("ab" * 32)
    reread = engine.ink.reread_with(None, types.SimpleNamespace(dpi=216, max_new_tokens=512))
    assert reread(types.SimpleNamespace(label="text", bbox=(0, 0, 16, 8)), "a") is None


# --- what the pipeline does to a Block after the Reader read it --------------------------------------------------------

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


def move_both_blocks(blocks):
    chart, text = blocks
    chart.bbox, text.bbox = (0.4, 0.6, 16.2, 10.9), (20.4, 20.6, 36.2, 30.9)


@every_setup
def test_a_judged_block_whose_box_the_pipeline_moved_after_reading_has_no_score_and_only_the_judge_warns(
        tmp_path, capsys, setup):
    engine, fakes = engine_for(setup.options, tmp_path / "engine")
    fakes.pipe.blocks = [("chart", (0, 0, 16, 8)), ("text", (20, 20, 36, 28))]
    after_reading(fakes.pipe, move_both_blocks)
    capsys.readouterr()
    doc, _ = transcribe(engine, blank_pdf(tmp_path / "doc.pdf"), tmp_path / "out", setup.options)
    assert not any("score" in b for b in doc["pages"][0]["blocks"])
    err = capsys.readouterr().err.splitlines()
    warnings = [ln for ln in err if ln.startswith("warning:")]
    if not setup.judge:
        assert warnings == []
        return
    (warning,) = warnings  # a chart is never judged, so it has no Score to miss
    assert warning.startswith("warning: page 1: the text Block at (20, 20, 36, 30) has no Score")
    assert err.index(warning) < next(i for i, ln in enumerate(err) if ln.startswith("[  1/1] page 1:"))
