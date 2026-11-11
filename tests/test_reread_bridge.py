"""Re-reading through the bridges: what each Reader is asked for in rounds 2-4, with the worker and PaddleOCR-VL faked.

The schedule itself is tested at the seam (test_reread.py); here: the decoding settings and seeds that reach the
worker, the scaled crop of round 2, the paddle Reader doing only what it honestly can, and a re-read going wrong
without costing the page.
"""
import cv2
import numpy as np

from block_reader import derive_seed
from fakes import FakeClient
from reader_bridge import BlockQuery, MiniCPMBridge, PaddleBridge, scaled_image
from worker_client import WorkerError

DOC = "ab" * 32
CROP = (8, 16)


def item(prompt, label, bbox=(0, 0, 10, 10), index=0, shape=CROP):
    q = BlockQuery(prompt)
    q.label, q.bbox, q.block_index = label, bbox, index
    return {"image": np.full((*shape, 3), 255, np.uint8), "query": q}


def minicpm(client, tmp_path, **kw):
    return MiniCPMBridge(client, tmp_path, to_html=lambda s: "", judge=True, judge_threshold=0.98, doc_hash=DOC, **kw)


def read_page(bridge, items, page=0):
    bridge.current_page = page
    return [r["result"] for r in bridge.predict(items, max_new_tokens=8192)]


class ScaleSensitiveClient(FakeClient):
    def __init__(self, by_shape, **kw):
        super().__init__(**kw)
        self.by_shape = by_shape

    def read(self, image, prompt, max_new_tokens, *, decode=None, timeout=None):
        out = super().read(image, prompt, max_new_tokens, decode=decode, timeout=timeout)
        text = self.by_shape.get(cv2.imread(str(image)).shape[:2])
        if decode is None and text is not None:
            out[0]["text"] = text
        return out


AT_1_25X = {scaled_image(np.zeros((*CROP, 3), np.uint8), 1.25).shape[:2]: "read again at 1.25x"}


def test_a_low_scoring_block_is_read_greedily_at_1x_then_at_1_25x_then_sampled_twice_with_seeds(tmp_path):
    client = ScaleSensitiveClient(AT_1_25X, default_score=0.5)
    b = minicpm(client, tmp_path)
    read_page(b, [item("OCR:", "text", index=7)], page=2)
    assert client.decodes == [
        None, None,
        {"mode": "sample", "temperature": 0.3, "top_k": 20, "top_p": 0.8, "n": 4, "seed": derive_seed(DOC, 2, "7", 3)},
        {"mode": "sample", "temperature": 0.7, "top_k": 20, "top_p": 0.8, "n": 4, "seed": derive_seed(DOC, 2, "7", 4)}]
    assert client.images[0] != client.images[1] and client.images[2] == client.images[0]  # only round 2 sends a scaled crop


def test_the_scaled_crop_is_the_blocks_own_crop_made_1_25_times_larger_and_the_judge_never_sees_it(tmp_path):
    img = np.arange(8 * 16 * 3, dtype=np.uint8).reshape(8, 16, 3)
    assert scaled_image(img, 1.0) is img
    assert scaled_image(img, 1.25).shape == (10, 20, 3)
    sizes = []

    class Spy(ScaleSensitiveClient):
        def read(self, image, prompt, max_new_tokens, **kw):
            sizes.append(cv2.imread(str(image)).shape[:2])
            return super().read(image, prompt, max_new_tokens, **kw)

        def score(self, items, **kw):
            sizes.append(("score", [cv2.imread(i["image"]).shape[:2] for i in items]))
            return super().score(items, **kw)

    b = minicpm(Spy({(50, 100): "read again at 1.25x"}, default_score=0.5), tmp_path)
    read_page(b, [item("OCR:", "text", shape=(40, 80))])
    assert sizes[0] == (40, 80) and sizes[2] == (50, 100)
    assert sizes[3] == ("score", [(40, 80)])
    assert all(s[1] == [(40, 80)] * len(s[1]) for s in sizes if s[0] == "score")


def test_a_block_that_passes_is_read_and_scored_once(tmp_path):
    client = FakeClient(default_score=0.99)
    read_page(minicpm(client, tmp_path), [item("OCR:", "text")])
    assert client.decodes == [None] and len(client.score_calls) == 1


def test_the_whole_page_is_re_read_in_batches_of_the_blocks_still_low(tmp_path):
    client = ScaleSensitiveClient(AT_1_25X, texts={"Text Recognition:": "low", "Table Recognition:": "high"},
                                  scores={"low": ("scored", 0.5), "high": ("scored", 0.99)})
    b = minicpm(client, tmp_path)
    read_page(b, [item("OCR:", "text", (0, 0, 1, 1), 0), item("Table Recognition:", "table", (2, 2, 3, 3), 1)])
    assert [d is None for d in client.decodes].count(True) == 3      # text: rounds 1, 2; table: round 1
    assert [len(c) for c in client.score_calls] == [2, 1]


def test_samples_are_judged_as_separate_readings_with_unique_keys_but_one_crop_file(tmp_path):
    client = ScaleSensitiveClient(AT_1_25X, default_score=0.5)
    b = minicpm(client, tmp_path)
    read_page(b, [item("OCR:", "text")])
    keys = [i["key"] for call in client.score_calls for i in call]
    assert len(set(keys)) == len(keys) == 1 + 1 + 4 + 4
    assert list(tmp_path.iterdir()) == []


def test_the_seed_belongs_to_the_block_not_to_its_position_in_the_pipelines_call(tmp_path):
    seeds = []
    for first_block in (False, True):
        client = ScaleSensitiveClient(AT_1_25X, default_score=0.5)
        b = minicpm(client, tmp_path)
        items = [item("OCR:", "text", (9, 9, 19, 19), index=5)]
        if first_block:
            items.insert(0, item("OCR:", "text", (0, 0, 5, 5), index=2))
        read_page(b, items)
        seeds.append([d["seed"] for d in client.decodes if d and d["temperature"] == 0.3][-1])
    assert seeds[0] == seeds[1] == derive_seed(DOC, 0, "5", 3)


def test_a_worker_error_in_a_sampled_round_keeps_the_best_reading_and_does_not_fail_the_page(tmp_path):
    class OOM(ScaleSensitiveClient):
        def read(self, image, prompt, max_new_tokens, *, decode=None, timeout=None):
            if decode is not None:
                raise WorkerError("worker error: OutOfMemoryError: CUDA out of memory")
            return super().read(image, prompt, max_new_tokens, decode=decode, timeout=timeout)

    b = minicpm(OOM(AT_1_25X, default_score=0.5), tmp_path)
    assert read_page(b, [item("OCR:", "text")]) == ["text for Text Recognition:"]
    out = b.outcomes[(0, (0, 0, 10, 10))]
    assert out.stop_reason == "reader_error" and "out of memory" in out.stop_detail
    assert out.score == 0.5


def test_a_non_worker_error_while_preparing_a_re_read_image_keeps_the_first_reading_and_does_not_fail_the_page(tmp_path):
    class BrokenRender:
        def original(self, bbox):
            return np.full((8, 16, 3), 255, np.uint8)

        def crop(self, bbox, factor):
            raise cv2.error("the re-render of the page failed")

    b = minicpm(FakeClient(default_score=0.5), tmp_path)
    b.page_source = BrokenRender()
    assert read_page(b, [item("OCR:", "text"), item("Table Recognition:", "table", (20, 20, 30, 30), 1)]) == [
        "text for Text Recognition:", "text for Table Recognition:"]
    out = b.outcomes[(0, (0, 0, 10, 10))]
    assert out.stop_reason == "reader_error" and "re-render" in out.stop_detail
    assert out.score == 0.5


def test_a_crop_that_cannot_be_written_for_the_judge_is_an_encode_error_not_a_failed_page(tmp_path):
    degenerate = {"image": np.zeros((0, 0, 3), np.uint8), "query": item("OCR:", "text")["query"]}
    b, _ = paddle(FakeClient(), tmp_path, {"OCR:": "words"})
    assert read_page(b, [degenerate]) == ["words"]
    out = b.outcomes[(0, (0, 0, 10, 10))]
    assert (out.verdict, out.score, out.flagged) == ("encode_error", None, True)


# ----------------------------------------------------------------------------- --reader paddle

class FakePaddle:
    def __init__(self, texts, fail_on_round2=False, by_shape=None):
        self.texts, self.calls, self.fail, self.by_shape = texts, [], fail_on_round2, by_shape or {}

    def predict(self, inputs, **kw):
        inputs = list(inputs)
        self.calls.append([i["image"].shape[:2] for i in inputs])
        if self.fail and len(self.calls) == 2:
            raise RuntimeError("paddle blew up")
        for i in inputs:
            yield {"result": self.by_shape.get(i["image"].shape[:2], self.texts[str(i["query"])])}


def paddle(client, tmp_path, texts, **kw):
    real = FakePaddle(texts, **kw)
    return PaddleBridge(real, client, tmp_path, to_html=lambda s: "", judge=True, judge_threshold=0.98, doc_hash=DOC), real


def test_paddle_cannot_sample_so_it_only_gets_the_1_25x_round_and_the_sampled_rounds_are_recorded_as_skipped(tmp_path):
    client = FakeClient(default_score=0.5)
    b, real = paddle(client, tmp_path, {"OCR:": "words"}, by_shape={(50, 100): "words, read again"})
    assert read_page(b, [item("OCR:", "text", shape=(40, 80))]) == ["words"]
    assert real.calls == [[(40, 80)], [(50, 100)]]
    out = b.outcomes[(0, (0, 0, 10, 10))]
    assert [s["round"] for s in out.skipped_rounds] == [3, 4]
    assert out.stop_reason == "exhausted" and [r.round for r in out.readings] == [1, 2]
    assert [b.supports(s) for s in __import__("block_reader").SCHEDULE] == [True, True, False, False]


def test_paddle_reading_the_same_text_at_1_25x_stops_the_re_reading_before_the_rounds_it_cannot_do(tmp_path):
    b, real = paddle(FakeClient(default_score=0.5), tmp_path, {"OCR:": "words"})
    assert read_page(b, [item("OCR:", "text")]) == ["words"]
    assert len(real.calls) == 2
    out = b.outcomes[(0, (0, 0, 10, 10))]
    assert out.stop_reason == "no_new_text" and out.skipped_rounds == []
    assert [(r.round, r.duplicates) for r in out.readings] == [(1, 1)]


def test_a_paddle_failure_in_round_2_keeps_the_first_reading(tmp_path):
    b, _ = paddle(FakeClient(default_score=0.5), tmp_path, {"OCR:": "words"}, fail_on_round2=True)
    assert read_page(b, [item("OCR:", "text")]) == ["words"]
    assert b.outcomes[(0, (0, 0, 10, 10))].stop_reason == "reader_error"


def test_minicpm_bridge_supports_every_round(tmp_path):
    from block_reader import SCHEDULE

    assert all(minicpm(FakeClient(), tmp_path).supports(s) for s in SCHEDULE)
