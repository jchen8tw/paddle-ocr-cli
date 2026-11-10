"""The bridge that makes the worker the pipeline's Reader, and the hook that installs it (no GPU, no weights).

The hook touches private PaddleX names, so tests/test_hook_regression.py also runs it against the real
pipeline; the tests here use a stand-in with the same shape.
"""
import types

import numpy as np
import pytest

from reader_bridge import BlockQuery, HookUnsupported, MiniCPMBridge, check_hook_support, install_reader_hook
from fakes import FakeClient
from worker_client import WorkerError


def query(text, label):
    q = BlockQuery(text)
    q.label, q.bbox = label, (0, 0, 10, 10)
    return q


def item(text, label):
    return {"image": np.full((8, 16, 3), 255, np.uint8), "query": query(text, label)}


def bridge(client, tmp_path, **kw):
    return MiniCPMBridge(client, tmp_path, to_html=lambda s: "", **kw)


def test_each_block_is_read_with_its_labels_prompt_and_limit_in_order(tmp_path):
    client = FakeClient()
    out = list(bridge(client, tmp_path).predict(
        [item("OCR:", "paragraph_title"), item("Table Recognition:", "table"), item("Formula Recognition:", "inline_formula")],
        max_new_tokens=6000, min_pixels=1, max_pixels=2))
    assert client.calls == [("Text Recognition:", 512), ("Table Recognition:", 6000), ("Text Recognition:", 2048)]
    assert [o["result"] for o in out] == ["text for Text Recognition:", "text for Table Recognition:",
                                           "text for Text Recognition:"]


def test_readings_are_cleaned_before_the_pipeline_sees_them_and_cuts_are_counted(tmp_path):
    loop = "前言" + "ABCDEFGHIJ" * 20
    client = FakeClient({"Text Recognition:": loop, "Table Recognition:": "<otsl><fcel>甲\n乙<nl></otsl>"})
    b = MiniCPMBridge(client, tmp_path, to_html=lambda s: "<table><tr><td>甲\n乙</td></tr></table>")
    out = list(b.predict([item("OCR:", "text"), item("Table Recognition:", "table")], max_new_tokens=8192))
    assert out[0]["result"] == "前言"
    assert out[1]["result"] == "<table><tr><td>甲<br>乙</td></tr></table>"
    assert b.repetition_cuts == 1


def test_a_query_without_a_label_falls_back_to_its_prompt_text(tmp_path):
    client = FakeClient()
    list(bridge(client, tmp_path).predict(
        [{"image": np.zeros((4, 4, 3), np.uint8), "query": "Seal Recognition:"}], max_new_tokens=8192))
    assert client.calls == [("Seal Recognition:", 256)]


def test_a_worker_failure_surfaces_so_the_page_fails_and_no_crops_are_left_behind(tmp_path):
    client = FakeClient(fail_on="Table Recognition:")
    with pytest.raises(WorkerError, match="exited with code 3"):
        list(bridge(client, tmp_path).predict(
            [item("OCR:", "text"), item("Table Recognition:", "table")], max_new_tokens=8192))
    assert list(tmp_path.iterdir()) == []


def test_a_dead_worker_is_restarted_before_the_next_page(tmp_path):
    client = FakeClient()
    list(bridge(client, tmp_path).predict([item("OCR:", "text")], max_new_tokens=8192))
    assert client.started == 1


# ---------------------------------------------------------------------------- the hook

class FakeInner:
    """The shape of paddlex's _PaddleOCRVLPipeline that the hook relies on."""

    def __init__(self):
        self.vl_rec_model = "the real PaddleOCR-VL model"

    def _paddleocr_vl_collect_page_vlm_entries_core(self, page_idx, blocks_for_img, imgs_in_doc, cfg):
        entries = [(0, j, f"img{j}", "OCR:", (1, 2), None) for j, _ in enumerate(blocks_for_img)]
        return entries, ["spotting"], ["drop"]


def fake_pipe(inner=None):
    return types.SimpleNamespace(paddlex_pipeline=types.SimpleNamespace(_pipeline=inner or FakeInner()))


def test_the_hook_swaps_the_reader_and_tags_every_query_with_its_block(tmp_path):
    pipe = fake_pipe()
    inner = pipe.paddlex_pipeline._pipeline
    reader = object()
    undo = install_reader_hook(pipe, reader)
    assert inner.vl_rec_model is reader
    blocks = [{"label": "text", "box": [1.2, 2.0, 30.9, 40.0]}, {"label": "table", "box": [0, 0, 5, 5]}]
    entries, spotting, drop = inner._paddleocr_vl_collect_page_vlm_entries_core(0, blocks, [], {})
    assert (spotting, drop) == (["spotting"], ["drop"])
    assert [(e[3], e[3].label, e[3].bbox) for e in entries] == [
        ("OCR:", "text", (1, 2, 30, 40)), ("OCR:", "table", (0, 0, 5, 5))]
    assert entries[0][:3] == (0, 0, "img0") and entries[0][4:] == ((1, 2), None)

    undo()
    assert inner.vl_rec_model == "the real PaddleOCR-VL model"
    assert "_paddleocr_vl_collect_page_vlm_entries_core" not in vars(inner)


def test_the_start_up_guard_names_the_installed_versions_when_a_private_name_moved(monkeypatch):
    class Renamed:
        vl_rec_model = None

    with pytest.raises(HookUnsupported, match=r"_paddleocr_vl_collect_page_vlm_entries_core.*paddlex"):
        check_hook_support(fake_pipe(Renamed()))
    with pytest.raises(HookUnsupported, match="paddlex_pipeline"):
        check_hook_support(types.SimpleNamespace())


def test_the_guard_notices_a_changed_signature():
    class Changed(FakeInner):
        def _paddleocr_vl_collect_page_vlm_entries_core(self, page_idx, blocks, cfg):
            return [], [], []

    with pytest.raises(HookUnsupported, match="signature"):
        check_hook_support(fake_pipe(Changed()))
    check_hook_support(fake_pipe())  # the expected shape passes


def test_entries_of_a_different_layout_fail_loudly_instead_of_misreading_blocks():
    class Wide(FakeInner):
        def _paddleocr_vl_collect_page_vlm_entries_core(self, page_idx, blocks_for_img, imgs_in_doc, cfg):
            return [(0, 0, "img", "OCR:", (1, 2), None, "new field")], [], []

    pipe = fake_pipe(Wide())
    install_reader_hook(pipe, object())
    with pytest.raises(HookUnsupported, match="entries"):
        pipe.paddlex_pipeline._pipeline._paddleocr_vl_collect_page_vlm_entries_core(0, [{"label": "text", "box": [0, 0, 1, 1]}], [], {})
