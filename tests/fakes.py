"""Test doubles shared by the bridge and the Run tests."""
from pathlib import Path

import pdf_ocr
from reader_bridge import BlockQuery, MiniCPMBridge
from script_repair import ScriptRepair, StickyScriptRepair
from worker_client import WorkerError


class FakeClient:
    """Stands in for WorkerClient: records what the bridge asks the worker for."""

    def __init__(self, texts=None, fail_on=None, scores=None, fail_score=None, default_score=0.99):
        self.calls, self.texts, self.fail_on = [], texts or {}, fail_on
        self.scores, self.fail_score, self.default_score = scores or {}, fail_score, default_score  # scores: text -> (verdict, score)
        self.score_calls, self.events = [], []
        self.decodes, self.images = [], []
        self.started = 0

    def ensure_started(self):
        self.started += 1

    def read(self, image, prompt, max_new_tokens, *, decode=None, timeout=None):
        assert Path(image).is_file(), "the crop must exist on disk while the worker reads it"
        self.calls.append((prompt, max_new_tokens))
        self.decodes.append(decode)
        self.images.append(Path(image).name)
        self.events.append("read")
        if self.fail_on == prompt:
            raise WorkerError("the worker stopped (exited with code 3)")
        text = self.texts.get(prompt, f"text for {prompt}")
        if decode is None:
            return [{"text": text, "tokens": 5, "finish": "stop"}]
        return [{"text": f"{text} sample {decode['seed']}.{k}", "tokens": 5, "finish": "stop"} for k in range(decode["n"])]

    def score(self, items, *, timeout=None):
        for i in items:
            assert Path(i["image"]).is_file(), "the crop must exist on disk while the worker scores"
        self.score_calls.append(items)
        self.events.append("score")
        if self.fail_score:
            raise self.fail_score
        out = []
        for i in items:
            verdict, score = self.scores.get(i["text"], ("scored", self.default_score))
            out.append({"key": i["key"], "verdict": verdict, "score": score, "detail": None if verdict == "scored" else "why"})
        return out

    def park(self):
        self.events.append("park")


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
