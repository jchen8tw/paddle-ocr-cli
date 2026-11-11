"""One loaded Engine transcribes one Document after another: nothing of a Document may reach the next.

The pipeline is faked (it reads each page's Blocks through the Engine's bridge, as the Reader hook does) and so is
the worker; the pages, the post-processing and the files are the real ones. No GPU, no models.
"""
import json
import types

import numpy as np
import pypdfium2 as pdfium
import pytest

import pdf_ocr
from fakes import FakeClient
from reader_bridge import BlockQuery, MiniCPMBridge
from script_repair import ScriptRepair, StickyScriptRepair

TRAD = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"
LOOP = "前言" + "ABCDEFGHIJ" * 20  # a repetition loop, cut back to "前言" and counted
BLOCKS = [("text", (0, 0, 16, 8)), ("text", (20, 20, 36, 28))]


class FakeBlock:
    def __init__(self, label, bbox):
        self.label, self.bbox, self.content = label, bbox, ""


class FakeResult:
    def __init__(self, blocks):
        self.blocks = blocks

    def __getitem__(self, key):
        assert key == "parsing_res_list"
        return self.blocks

    @property
    def json(self):
        return {"res": {"width": 200, "height": 300, "layout_det_res": {"boxes": []}, "parsing_res_list": [
            {"block_id": i, "block_order": i, "block_label": b.label, "block_content": b.content,
             "block_bbox": list(b.bbox), "block_polygon_points": None} for i, b in enumerate(self.blocks)]}}

    def _to_markdown(self, pretty, show_formula_number):
        return {"markdown_texts": "\n\n".join(b.content for b in self.blocks)}


class FakePipe:
    """PaddleOCRVL with the Reader hook installed: finds the same Blocks on every page and reads them."""

    def __init__(self, bridge):
        self.bridge = bridge

    def predict(self, img, max_new_tokens, use_queues):
        blocks = [FakeBlock(label, bbox) for label, bbox in BLOCKS]
        items = []
        for b in blocks:
            q = BlockQuery("OCR:")
            q.label, q.bbox = b.label, b.bbox
            x1, y1, x2, y2 = b.bbox
            items.append({"image": img[y1:y2, x1:x2], "query": q})
        for b, r in zip(blocks, self.bridge.predict(items, max_new_tokens=max_new_tokens)):
            b.content = r["result"]
        return [FakeResult(blocks)]

    def restructure_pages(self, results, merge_tables, relevel_titles, concatenate_pages):
        return results


def engine_with(tmp_path):
    """A loaded Engine (what `load_engine` returns) whose worker answers from `client.texts`."""
    client = FakeClient(default_score=0.5)  # every Reading scores below the threshold: all re-read Rounds run
    crops = tmp_path / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    bridge = MiniCPMBridge(client, crops, to_html=lambda s: "", judge=True, judge_threshold=0.98,
                           repair=StickyScriptRepair(ScriptRepair()))
    ready = {"judge": "fake-judge", "versions": {"torch": "0.0"}}
    return pdf_ocr.Engine(FakePipe(bridge), "cpu", bridge, pdf_ocr.InkReread(), ready), client


def pdf_of_size(path, height, pages=1):
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(200, height)
    pdf.save(str(path))
    return path


def transcribe(engine, pdf, out):
    args = pdf_ocr.build_parser().parse_args([str(pdf), "--no-csv", "--judge-threshold", "0.98"])
    assert pdf_ocr.transcribe_document(engine, args, pdf_ocr.open_document(pdf, None), out) == {}
    doc = json.loads((out / f"{pdf.stem}.json").read_text(encoding="utf-8"))
    del doc["elapsed_seconds"]
    return doc, (out / f"{pdf.stem}.md").read_text(encoding="utf-8")


def test_a_document_after_another_in_the_same_engine_comes_out_as_if_it_were_alone(tmp_path):
    first, second = pdf_of_size(tmp_path / "first.pdf", 300, pages=2), pdf_of_size(tmp_path / "second.pdf", 301)
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
