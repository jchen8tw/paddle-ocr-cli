"""The bridges that make a Reader (and the Judge) the pipeline's VL model, with the worker and PaddleOCR-VL faked.

What the pipeline sees is the final text of each Block; what the caller of the run sees afterwards, per
page and Block, is the Score, verdict and flag. No GPU, no weights.
"""
import numpy as np
import pytest

from fakes import FakeClient
from reader_bridge import BlockQuery, MiniCPMBridge, PaddleBridge
from script_repair import ScriptRepair, StickyScriptRepair
from worker_client import WorkerError

TRAD = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"


def item(prompt, label, bbox=(0, 0, 10, 10)):
    q = BlockQuery(prompt)
    q.label, q.bbox = label, bbox
    return {"image": np.full((8, 16, 3), 255, np.uint8), "query": q}


def minicpm(client, tmp_path, **kw):
    kw.setdefault("to_html", lambda s: "")
    kw.setdefault("judge_threshold", 0.98)
    kw.setdefault("judge", True)
    return MiniCPMBridge(client, tmp_path, **kw)


def read_page(bridge, items, page=0, **kw):
    bridge.current_page = page
    return [r["result"] for r in bridge.predict(items, max_new_tokens=8192, **kw)]


def test_every_judged_block_gets_a_score_verdict_and_flag_keyed_by_page_and_bbox(tmp_path):
    client = FakeClient({"Text Recognition:": "words"}, scores={"words": ("scored", 0.9)}, default_score=0.5)
    b = minicpm(client, tmp_path)
    assert read_page(b, [item("OCR:", "text", (1, 2, 3, 4))], page=4) == ["words"]
    out = b.outcomes[(4, (1, 2, 3, 4))]
    assert (out.score, out.verdict, out.flagged) == (0.9, "scored", True)


def test_the_judge_gets_the_cleaned_text_the_pipeline_will_write_and_a_crop_that_exists(tmp_path):
    client = FakeClient({"Text Recognition:": r"if \(x\) then"})
    b = minicpm(client, tmp_path)
    (written,) = read_page(b, [item("OCR:", "text")])
    assert written == "if  $ x $ then"
    assert client.score_calls[0][0]["text"] == written and client.score_calls[0][0]["category"] == "text"


def test_a_page_is_one_score_request_and_no_crops_are_left_behind(tmp_path):
    client = FakeClient()
    read_page(minicpm(client, tmp_path), [item("OCR:", "text", (0, 0, 1, 1)), item("OCR:", "text", (0, 2, 1, 3)),
                                          item("Table Recognition:", "table", (5, 5, 9, 9))])
    assert len(client.score_calls) == 1 and len(client.score_calls[0]) == 3
    assert [i["category"] for i in client.score_calls[0]] == ["text", "text", "table"]
    assert list(tmp_path.iterdir()) == []


def test_tables_are_judged_as_html_not_otsl(tmp_path):
    client = FakeClient({"Table Recognition:": "<otsl><fcel>甲\n乙<nl></otsl>"})
    b = minicpm(client, tmp_path, to_html=lambda s: "<table><tr><td>甲\n乙</td></tr></table>")
    (written,) = read_page(b, [item("Table Recognition:", "table")])
    assert written == "<table><tr><td>甲<br>乙</td></tr></table>"
    assert client.score_calls[0][0]["text"] == written


def test_charts_and_seals_are_read_but_never_scored_and_have_no_outcome_fields(tmp_path):
    client = FakeClient()
    b = minicpm(client, tmp_path)
    read_page(b, [item("Chart Recognition:", "chart", (0, 0, 1, 1)), item("Seal Recognition:", "seal", (2, 2, 3, 3))])
    assert client.score_calls == []
    assert all(not o.judged and o.score is None and o.verdict is None for o in b.outcomes.values())


def test_the_judge_sees_repaired_text_once_the_document_is_known_to_be_traditional(tmp_path):
    client = FakeClient({"Text Recognition:": TRAD})
    repair = StickyScriptRepair(ScriptRepair())
    b = minicpm(client, tmp_path, repair=repair)
    read_page(b, [item("OCR:", "text", (0, 0, 1, 1))], page=0)
    client.texts["Text Recognition:"] = "银行"
    (written,) = read_page(b, [item("OCR:", "text", (0, 0, 1, 1))], page=1)
    assert written == "銀行" and client.score_calls[1][0]["text"] == "銀行"
    assert b.outcomes[(1, (0, 0, 1, 1))].text_before_repair == "银行"


def test_the_judge_sees_a_lone_symbol_unwrapped_and_as_its_big5_glyph_once_the_document_is_traditional(tmp_path):
    client = FakeClient()
    b = minicpm(client, tmp_path, repair=StickyScriptRepair(ScriptRepair()))
    for page, reading in enumerate([r"年滿15足歲$\geq$者", TRAD, r"年滿15足歲$\geq$者"]):
        client.texts["Text Recognition:"] = reading
        read_page(b, [item("OCR:", "text", (0, 0, 1, 1))], page=page)
    assert [call[0]["text"] for call in client.score_calls] == ["年滿15足歲≥者", TRAD, "年滿15足歲≧者"]
    assert b.outcomes[(2, (0, 0, 1, 1))].text_before_repair == "年滿15足歲≥者"


def test_a_worker_that_fails_to_score_is_a_forward_error_not_a_failed_page(tmp_path):
    client = FakeClient({"Text Recognition:": "words"}, fail_score=WorkerError("the worker stopped (exited with code 3)"))
    b = minicpm(client, tmp_path)
    assert read_page(b, [item("OCR:", "text")]) == ["words"]
    out = b.outcomes[(0, (0, 0, 10, 10))]
    assert (out.verdict, out.score, out.flagged) == ("forward_error", None, True)
    assert "exited with code 3" in out.verdict_detail


def test_without_a_judge_nothing_is_scored_and_the_worker_is_only_asked_to_read(tmp_path):
    client = FakeClient()
    b = minicpm(client, tmp_path, judge=False)
    read_page(b, [item("OCR:", "text")])
    assert client.score_calls == [] and not b.outcomes[(0, (0, 0, 10, 10))].judged


def test_the_page_summary_counts_flagged_blocks_and_those_not_scored(tmp_path):
    client = FakeClient({"Text Recognition:": "words"}, scores={"words": ("scored", 0.5)}, default_score=0.5)
    b = minicpm(client, tmp_path)
    read_page(b, [item("OCR:", "text", (0, 0, 1, 1)), item("OCR:", "text", (0, 2, 1, 3))], page=2)
    client.scores, client.default_score = {}, 0.99
    read_page(b, [item("OCR:", "text", (5, 5, 6, 6))], page=3)
    assert b.page_summary(2) == {"judged": 2, "reread": 2, "flagged": 2, "not_scored": 0}
    assert b.page_summary(3) == {"judged": 1, "reread": 0, "flagged": 0, "not_scored": 0}
    assert b.page_summary(9) == {"judged": 0, "reread": 0, "flagged": 0, "not_scored": 0}


# --- --reader paddle: PaddleOCR-VL reads, the worker only judges ---------------------------------------

class FakePaddle:
    def __init__(self, texts):
        self.texts, self.calls = texts, []
        self.events = None

    def predict(self, inputs, **kw):
        inputs = list(inputs)
        self.calls.append(([(str(i["query"]), i["image"].shape) for i in inputs], kw))
        if self.events is not None:
            self.events.append("paddle")
        for i in inputs:
            yield {"result": self.texts[str(i["query"])]}

    def close(self):
        pass


def paddle(client, tmp_path, texts, **kw):
    kw.setdefault("to_html", lambda s: "<table><tr><td>a\\nb</td></tr></table>" if "<fcel>" in s else "")
    kw.setdefault("judge_threshold", 0.98)
    kw.setdefault("judge", True)
    real = FakePaddle(texts)
    real.events = client.events
    return PaddleBridge(real, client, tmp_path, **kw), real


def test_paddle_blocks_are_read_by_paddle_in_one_batch_with_the_pipelines_own_arguments(tmp_path):
    client = FakeClient()
    b, real = paddle(client, tmp_path, {"OCR:": "words", "Table Recognition:": "<fcel>x"})
    out = read_page(b, [item("OCR:", "text"), item("Table Recognition:", "table", (1, 1, 2, 2))],
                    min_pixels=1, max_pixels=2)
    assert out == ["words", "<table><tr><td>a<br>b</td></tr></table>"]
    assert len(real.calls) == 1
    assert [q for q, _ in real.calls[0][0]] == ["OCR:", "Table Recognition:"]
    assert real.calls[0][1]["min_pixels"] == 1 and real.calls[0][1]["max_new_tokens"] == 8192


def test_paddle_readings_are_cleaned_like_the_pipeline_does_before_they_are_judged(tmp_path):
    client = FakeClient()
    b, _ = paddle(client, tmp_path, {"OCR:": r"if \(x\)", "Table Recognition:": "<fcel>x"})
    read_page(b, [item("OCR:", "text"), item("Table Recognition:", "table", (1, 1, 2, 2))])
    assert [i["text"] for i in client.score_calls[0]] == ["if  $ x $", "<table><tr><td>a<br>b</td></tr></table>"]


def test_paddle_text_is_not_stripped_so_it_stays_what_the_pipeline_would_have_written(tmp_path):
    client = FakeClient()
    b, _ = paddle(client, tmp_path, {"OCR:": "  padded\n"})
    assert read_page(b, [item("OCR:", "text")]) == ["  padded\n"]


def test_the_judge_leaves_the_gpu_while_paddle_reads_and_paddle_frees_its_cache_before_scoring(tmp_path):
    client = FakeClient()
    freed = []
    b, _ = paddle(client, tmp_path, {"OCR:": "words"}, before_read=client.park,
                  after_read=lambda: (freed.append(1), client.events.append("free")))
    read_page(b, [item("OCR:", "text")])
    assert client.events == ["park", "paddle", "free", "score"]


def test_paddle_without_a_judge_makes_no_worker_calls(tmp_path):
    client = FakeClient()
    b, _ = paddle(client, tmp_path, {"OCR:": "words"}, judge=False)
    assert read_page(b, [item("OCR:", "text")]) == ["words"]
    assert client.events == ["paddle"]


def test_two_blocks_with_the_same_box_on_a_page_are_reported_instead_of_silently_losing_a_score(tmp_path, capsys):
    b = minicpm(FakeClient(), tmp_path)
    read_page(b, [item("OCR:", "text", (1, 2, 3, 4)), item("Table Recognition:", "table", (1, 2, 3, 4))], page=6)
    err = capsys.readouterr().err
    assert "page 7" in err and "(1, 2, 3, 4)" in err and "same box" in err
    assert len(b.outcomes) == 1


def test_a_block_without_a_box_is_reported(tmp_path, capsys):
    plain = {"image": np.full((8, 16, 3), 255, np.uint8), "query": "OCR:"}  # not a BlockQuery: the hook did not tag it
    read_page(minicpm(FakeClient(), tmp_path), [plain])
    assert "without a box" in capsys.readouterr().err


# --- the ink-based re-read, used without a Judge -------------------------------------------------------

def test_without_a_judge_the_ink_reread_gets_the_block_with_its_bbox_and_may_replace_the_reading(tmp_path):
    seen = []

    def ink(request, text):
        seen.append((request.bbox, request.label, text))
        return "a much longer reading"

    client = FakeClient({"Text Recognition:": "x"})
    b = minicpm(client, tmp_path, judge=False, ink_reread=ink)
    assert read_page(b, [item("OCR:", "text", (3, 4, 5, 6))]) == ["a much longer reading"]
    assert seen == [((3, 4, 5, 6), "text", "x")]
    assert b.outcomes[(0, (3, 4, 5, 6))].text_before_reread == "x"


def test_an_ink_reread_that_fails_keeps_the_first_reading_warns_and_does_not_fail_the_page(tmp_path, capsys):
    class FailsOnSecondRead(FakeClient):
        def read(self, image, prompt, max_new_tokens, **kw):
            if len(self.calls) >= 2:  # the two first Readings succeed, the re-reads do not
                raise WorkerError("worker error: CUDA out of memory")
            return super().read(image, prompt, max_new_tokens, **kw)

    b = minicpm(FailsOnSecondRead({"Text Recognition:": "short"}), tmp_path, judge=False)
    b.ink_reread = lambda request, text: b.read_crop(request.image, "text")
    assert read_page(b, [item("OCR:", "text", (1, 2, 3, 4)), item("OCR:", "text", (5, 6, 7, 8))]) == ["short", "short"]
    err = capsys.readouterr().err
    assert "keeping its first reading" in err and "out of memory" in err and "(5, 6, 7, 8)" in err
    assert b.outcomes[(0, (5, 6, 7, 8))].text_before_reread is None


def test_with_a_judge_the_ink_reread_is_not_called(tmp_path):
    def ink(request, text):
        raise AssertionError("must not run")

    b = minicpm(FakeClient(), tmp_path, ink_reread=ink)
    read_page(b, [item("OCR:", "text")])


# --- a worker that only judges must never cost a page that the Reader itself could read ------------------

def test_with_paddle_a_worker_that_cannot_be_restarted_gives_forward_errors_not_a_failed_page(tmp_path):
    class Dead(FakeClient):
        def ensure_started(self):
            raise WorkerError("the worker has crashed 4 times; not restarting it again")

    b, _ = paddle(Dead(), tmp_path, {"OCR:": "words"})
    assert read_page(b, [item("OCR:", "text")]) == ["words"]
    out = b.outcomes[(0, (0, 0, 10, 10))]
    assert (out.verdict, out.score, out.flagged) == ("forward_error", None, True)
    assert "not restarting" in out.verdict_detail


def test_with_paddle_a_failing_park_does_not_fail_the_page(tmp_path):
    def broken_park():
        raise WorkerError("no answer within 120s")

    client = FakeClient(fail_score=WorkerError("the worker is not running"))
    b, _ = paddle(client, tmp_path, {"OCR:": "words"}, before_read=broken_park)
    assert read_page(b, [item("OCR:", "text")]) == ["words"]
    assert b.outcomes[(0, (0, 0, 10, 10))].verdict == "forward_error"


def test_with_minicpm_the_worker_is_the_reader_so_a_dead_worker_still_fails_the_page(tmp_path):
    class Dead(FakeClient):
        def ensure_started(self):
            raise WorkerError("the worker has crashed 4 times; not restarting it again")

    with pytest.raises(WorkerError, match="not restarting"):
        read_page(minicpm(Dead(), tmp_path), [item("OCR:", "text")])
