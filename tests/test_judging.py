"""The worker's Judge: render each Reading, score it, and report an official verdict per item (no GPU).

A fake renderer and a fake model stand in for Chromium and Jev.
"""
import types

import pytest

from judging import EncodeError, RenderedJudge


def rendered(case_id, ok=True, reason=None, detail=None):
    return types.SimpleNamespace(case_id=case_id, ok=ok, render_image_path=f"/render/{case_id}.png" if ok else None,
                                 reason=reason, detail=detail)


class FakeRenderer:
    def __init__(self, failing=None, boom=None):
        self.failing, self.boom, self.batches = failing or {}, boom, []

    def __call__(self, candidates, work_dir):
        self.batches.append([(c.case_id, c.image_path, c.text, c.category) for c in candidates])
        if self.boom:
            raise self.boom
        return [rendered(c.case_id, **self.failing[c.case_id]) if c.case_id in self.failing else rendered(c.case_id)
                for c in candidates]


class FakeModel:
    def __init__(self, scores=None, raises=None):
        self.scores, self.raises, self.calls = scores or {}, raises or {}, []

    def score(self, original, render, text):
        self.calls.append((original, render, text))
        if text in self.raises:
            raise self.raises[text]
        return self.scores.get(text, 0.97)


def item(key, text="t", category="text"):
    return {"key": key, "image": f"/crops/{key}.png", "text": text, "category": category}


def judge(model=None, renderer=None):
    return RenderedJudge(model or FakeModel(), renderer or FakeRenderer())


def test_items_are_rendered_in_one_batch_and_scored_with_the_original_and_the_render():
    r, m = FakeRenderer(), FakeModel({"b": 0.4})
    out = RenderedJudge(m, r).score_many([item("k1", "a"), item("k2", "b", "table")])
    assert r.batches == [[("k1", "/crops/k1.png", "a", "text"), ("k2", "/crops/k2.png", "b", "table")]]
    assert m.calls == [("/crops/k1.png", "/render/k1.png", "a"), ("/crops/k2.png", "/render/k2.png", "b")]
    assert out == [{"key": "k1", "verdict": "scored", "score": 0.97, "detail": None},
                   {"key": "k2", "verdict": "scored", "score": 0.4, "detail": None}]


@pytest.mark.parametrize("reason", ["precheck_reject", "render_error", "missing_original_image"])
def test_a_render_failure_is_its_official_verdict_without_a_score_and_is_not_scored(reason):
    m = FakeModel()
    out = judge(m, FakeRenderer({"k1": dict(ok=False, reason=reason, detail="KaTeX parse error")})).score_many(
        [item("k1"), item("k2")])
    assert (out[0]["verdict"], out[0]["score"], out[0]["detail"]) == (reason, None, "KaTeX parse error")
    assert out[1]["verdict"] == "scored"
    assert [c[0] for c in m.calls] == ["/crops/k2.png"]


def test_a_render_failure_with_an_unknown_reason_is_a_render_error():
    out = judge(renderer=FakeRenderer({"k1": dict(ok=False, reason=None)})).score_many([item("k1")])
    assert out[0]["verdict"] == "render_error"


def test_a_renderer_that_cannot_run_at_all_gives_every_item_a_render_error_and_the_worker_goes_on():
    out = judge(renderer=FakeRenderer(boom=RuntimeError("chromium crashed"))).score_many([item("k1"), item("k2")])
    assert [o["verdict"] for o in out] == ["render_error", "render_error"]
    assert "chromium crashed" in out[0]["detail"]


def test_an_image_that_cannot_be_encoded_is_an_encode_error_and_only_that_item():
    m = FakeModel(raises={"bad": EncodeError("cannot identify image file")})
    out = judge(m).score_many([item("k1", "bad"), item("k2", "ok")])
    assert [o["verdict"] for o in out] == ["encode_error", "scored"]
    assert out[0]["score"] is None and "cannot identify" in out[0]["detail"]


def test_a_failing_forward_is_a_forward_error_and_only_that_item():
    m = FakeModel(raises={"bad": RuntimeError("CUDA out of memory")})
    recovered = []
    j = RenderedJudge(m, FakeRenderer(), on_forward_error=lambda e: recovered.append(str(e)))
    out = j.score_many([item("k1", "bad"), item("k2", "ok")])
    assert [o["verdict"] for o in out] == ["forward_error", "scored"]
    assert recovered == ["CUDA out of memory"]  # the model gets a chance to free memory


def test_a_score_that_is_not_a_probability_is_a_forward_error():
    out = judge(FakeModel({"x": float("nan")})).score_many([item("k1", "x")])
    assert out[0]["verdict"] == "forward_error" and out[0]["score"] is None


def test_an_empty_request_does_not_start_the_renderer():
    r = FakeRenderer()
    assert RenderedJudge(FakeModel(), r).score_many([]) == [] and r.batches == []
