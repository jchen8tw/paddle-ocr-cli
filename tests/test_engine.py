"""One loaded Engine transcribes one Document after another: nothing of a Document may reach the next.

The pipeline is faked (it reads each page's Blocks through the Engine's bridge, as the Reader hook does) and so is
the worker; the pages, the post-processing and the files are the real ones. No GPU, no models.
"""
import json
import types

import numpy as np

import pdf_ocr
from fakes import blank_pdf, engine_with
from reader_bridge import BlockQuery

TRAD = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"
LOOP = "前言" + "ABCDEFGHIJ" * 20  # a repetition loop, cut back to "前言" and counted


def transcribe(engine, pdf, out):
    args = pdf_ocr.build_parser().parse_args([str(pdf), "--no-csv", "--judge-threshold", "0.98"])
    assert pdf_ocr.transcribe_document(engine, args, pdf_ocr.open_document(pdf, None), out) == {}
    doc = json.loads((out / f"{pdf.stem}.json").read_text(encoding="utf-8"))
    del doc["elapsed_seconds"]
    return doc, (out / f"{pdf.stem}.md").read_text(encoding="utf-8")


def test_a_document_after_another_in_the_same_engine_comes_out_as_if_it_were_alone(tmp_path):
    first, second = blank_pdf(tmp_path / "first.pdf", pages=2), blank_pdf(tmp_path / "second.pdf", height=301)
    engine, client = engine_with(tmp_path)
    client.texts = {"Text Recognition:": TRAD + LOOP}  # a Traditional document, with repetition loops
    a, _ = transcribe(engine, first, tmp_path / "a")
    post = a["postprocessing"]
    assert post["repetition_cuts"] > 0 and post["judge"]["blocks_judged"] == 4
    assert post["judge"]["scaled_images_by_source"]  # the 1.25x Round made scaled images

    client.texts = {"Text Recognition:": "银行医院"}  # Simplified: the sticky script repair must not have been decided
    in_the_same_engine = transcribe(engine, second, tmp_path / "b1")

    alone_engine, alone_client = engine_with(tmp_path / "alone")
    alone_client.texts = {"Text Recognition:": "银行医院"}
    assert in_the_same_engine == transcribe(alone_engine, second, tmp_path / "b2")


def test_the_lone_symbols_one_document_unwrapped_are_not_counted_for_the_next(tmp_path):
    first, second = blank_pdf(tmp_path / "first.pdf"), blank_pdf(tmp_path / "second.pdf", height=301)
    engine, client = engine_with(tmp_path)
    client.texts = {"Text Recognition:": r"年滿15足歲\geq者"}
    a, _ = transcribe(engine, first, tmp_path / "a")
    assert a["postprocessing"]["latex_symbols_unwrapped"] > 0

    client.texts = {"Text Recognition:": "no symbol here"}
    in_the_same_engine = transcribe(engine, second, tmp_path / "b1")
    assert in_the_same_engine[0]["postprocessing"]["latex_symbols_unwrapped"] == 0

    alone_engine, alone_client = engine_with(tmp_path / "alone")
    alone_client.texts = {"Text Recognition:": "no symbol here"}
    assert in_the_same_engine == transcribe(alone_engine, second, tmp_path / "b2")


class PreviousDocumentsPage:
    """A page of the previous Document: anything that touches it is a leak."""

    def original(self, bbox):
        raise AssertionError("the previous Document's page was used")

    crop = region = original


def test_a_new_document_starts_on_page_one_without_the_previous_documents_page(tmp_path):
    engine, client = engine_with(tmp_path)
    engine.bridge.current_page, engine.bridge.page_source = 7, PreviousDocumentsPage()
    engine.begin_document("ab" * 32)
    q = BlockQuery("OCR:")
    q.label, q.bbox = "text", (0, 0, 16, 8)
    list(engine.bridge.predict([{"image": np.full((8, 16, 3), 255, np.uint8), "query": q}], max_new_tokens=8192))
    assert list(engine.bridge.outcomes) == [(0, (0, 0, 16, 8))]  # not page 8
    assert list(engine.bridge.scale_sources) == ["crop_resample"]  # the 1.25x Round never saw the old page


def test_a_new_document_starts_without_the_ink_rereads_page(tmp_path):
    engine, _ = engine_with(tmp_path)
    engine.ink.set_page(PreviousDocumentsPage())
    engine.begin_document("ab" * 32)
    reread = engine.ink.reread_with(None, types.SimpleNamespace(dpi=216, max_new_tokens=512))
    assert reread(types.SimpleNamespace(label="text", bbox=(0, 0, 16, 8)), "a") is None
