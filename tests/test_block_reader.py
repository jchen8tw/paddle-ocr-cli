"""The "read a Block" seam: Block images in, final Readings with their Scores out (no GPU, no models).

A fake Reader and a fake Judge stand in for the torch worker. The behaviour under test is what a caller
can observe: the final text, the Score, the verdict, whether the Block is flagged, and what the Judge was
shown.
"""
import pytest

from block_reader import (BlockRequest, ReadSettings, judge_category, read_block, read_blocks)
from script_repair import ScriptRepair, StickyScriptRepair

TRAD = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"


class FakeReader:
    def __init__(self, texts):
        self.texts, self.batches = texts, []

    def read(self, requests):
        self.batches.append([r.key for r in requests])
        return [self.texts[r.key] for r in requests]


class FakeJudge:
    """Scores by a table of text -> (verdict, score); records every item it is shown."""

    def __init__(self, scores=None, default=("scored", 0.99), fail=None):
        self.scores, self.default, self.fail = scores or {}, default, fail
        self.calls = []

    def score(self, items):
        self.calls.append(list(items))
        if self.fail:
            raise self.fail
        return [self.scores.get(i.text, self.default) for i in items]


def req(key, label="text", **kw):
    return BlockRequest(key=key, image=f"<crop {key}>", label=label, **kw)


def settings(**kw):
    kw.setdefault("threshold", 0.98)
    kw.setdefault("clean", lambda raw, label: raw.strip())
    return ReadSettings(**kw)


def test_a_scored_block_returns_its_reading_score_and_verdict():
    out = read_block(req("a"), FakeReader({"a": " hello "}), FakeJudge(default=("scored", 0.995)), settings())
    assert (out.text, out.verdict, out.score, out.below_threshold) == ("hello", "scored", 0.995, False)


def test_a_score_below_the_threshold_flags_the_block_and_the_threshold_is_the_setting():
    judge = FakeJudge(default=("scored", 0.90))
    assert read_block(req("a"), FakeReader({"a": "x"}), judge, settings(threshold=0.98)).below_threshold is True
    assert read_block(req("a"), FakeReader({"a": "x"}), judge, settings(threshold=0.85)).below_threshold is False


@pytest.mark.parametrize("label,category", [
    ("text", "text"), ("paragraph_title", "text"), ("doc_title", "text"), ("footnote", "text"),
    ("inline_formula", "formula"), ("display_formula", "formula"), ("table", "table"),
    ("chart", None), ("seal", None), ("image", None), ("header_image", None), ("footer_image", None),
])
def test_labels_map_to_the_judges_categories_and_charts_seals_figures_are_never_judged(label, category):
    assert judge_category(label) == category


def test_the_judge_is_shown_the_crop_the_reading_and_the_category():
    judge = FakeJudge()
    read_blocks([req("f", "display_formula"), req("t", "table")], FakeReader({"f": "E=mc^2", "t": "<table></table>"}),
                judge, settings())
    (items,) = judge.calls
    assert [(i.key, i.image, i.text, i.category) for i in items] == [
        ("f", "<crop f>", "E=mc^2", "formula"), ("t", "<crop t>", "<table></table>", "table")]


def test_unjudged_labels_are_read_but_never_shown_to_the_judge_and_carry_no_score():
    judge = FakeJudge()
    out = read_blocks([req("c", "chart"), req("t")], FakeReader({"c": "| a |", "t": "words"}), judge, settings())
    assert [i.key for i in judge.calls[0]] == ["t"]
    chart = out[0]
    assert (chart.text, chart.score, chart.verdict, chart.below_threshold, chart.judged) == ("| a |", None, None, None, False)


def test_a_page_of_blocks_is_one_read_and_one_judge_call():
    reader, judge = FakeReader({"a": "1", "b": "2", "c": "3"}), FakeJudge()
    read_blocks([req("a"), req("b"), req("c")], reader, judge, settings())
    assert reader.batches == [["a", "b", "c"]] and len(judge.calls) == 1


def test_a_page_where_nothing_is_judged_makes_no_judge_call():
    judge = FakeJudge()
    read_blocks([req("c", "chart")], FakeReader({"c": "x"}), judge, settings())
    assert judge.calls == []


def test_the_reading_is_cleaned_before_it_is_judged():
    judge = FakeJudge()
    s = settings(clean=lambda raw, label: raw.replace("\\n", "<br>") if label == "table" else raw)
    out = read_block(req("t", "table"), FakeReader({"t": "<td>a\\nb</td>"}), judge, s)
    assert judge.calls[0][0].text == "<td>a<br>b</td>" and out.text == "<td>a<br>b</td>"


def test_repair_happens_before_judging_so_the_judge_sees_the_repaired_text():
    judge = FakeJudge()
    s = settings(repair=StickyScriptRepair(ScriptRepair()))
    out = read_blocks([req("a"), req("b")], FakeReader({"a": TRAD, "b": "银行"}), judge, s)
    assert [i.text for i in judge.calls[0]] == [TRAD, "銀行"]
    assert out[1].text == "銀行"
    assert out[1].text_before_repair == "银行"


def test_blocks_judged_before_the_document_was_known_to_be_traditional_are_marked():
    s = settings(repair=StickyScriptRepair(ScriptRepair()))
    out = read_blocks([req("a"), req("b"), req("c")], FakeReader({"a": "银行", "b": TRAD, "c": "银行"}), FakeJudge(), s)
    assert [o.judged_before_script_decision for o in out] == [True, False, False]
    assert out[0].text == "银行"          # judged as read; the document-level repair fixes the output later


def test_an_unrepaired_reading_has_no_text_before_repair():
    out = read_block(req("a"), FakeReader({"a": "plain"}), FakeJudge(), settings(repair=StickyScriptRepair(ScriptRepair())))
    assert out.text_before_repair is None


def test_the_repair_state_carries_over_from_one_page_to_the_next():
    repair = StickyScriptRepair(ScriptRepair())
    s = settings(repair=repair)
    read_blocks([req("a")], FakeReader({"a": TRAD}), FakeJudge(), s)
    judge = FakeJudge()
    read_blocks([req("b")], FakeReader({"b": "银行"}), judge, s)
    assert judge.calls[0][0].text == "銀行"


@pytest.mark.parametrize("verdict", ["precheck_reject", "render_error", "missing_original_image",
                                     "encode_error", "forward_error"])
def test_a_failure_verdict_is_recorded_without_a_score_the_block_is_flagged_and_the_text_is_kept(verdict):
    judge = FakeJudge(scores={"bad": (verdict, None)})
    out = read_blocks([req("a"), req("b")], FakeReader({"a": "bad", "b": "good"}), judge, settings())
    assert (out[0].text, out[0].score, out[0].verdict, out[0].below_threshold) == ("bad", None, verdict, True)
    assert (out[1].verdict, out[1].below_threshold) == ("scored", False)


def test_a_sentinel_zero_from_the_judge_is_never_taken_for_a_score():
    judge = FakeJudge(default=("render_error", 0.0))
    out = read_block(req("a"), FakeReader({"a": "x"}), judge, settings())
    assert out.score is None


def test_a_judge_that_fails_outright_marks_every_judged_block_forward_error_and_the_run_goes_on():
    judge = FakeJudge(fail=RuntimeError("the worker stopped"))
    out = read_blocks([req("a"), req("c", "chart"), req("b")], FakeReader({"a": "1", "c": "2", "b": "3"}), judge, settings())
    assert [o.verdict for o in out] == ["forward_error", None, "forward_error"]
    assert [o.text for o in out] == ["1", "2", "3"]
    assert all(o.below_threshold for o in (out[0], out[2]))
    assert "the worker stopped" in out[0].verdict_detail


def test_a_judge_that_answers_for_the_wrong_number_of_items_is_a_forward_error():
    class Short(FakeJudge):
        def score(self, items):
            return [("scored", 0.99)]

    out = read_blocks([req("a"), req("b")], FakeReader({"a": "1", "b": "2"}), Short(), settings())
    assert [o.verdict for o in out] == ["forward_error", "forward_error"]


def test_a_reader_failure_is_not_swallowed_the_page_fails():
    class Broken:
        def read(self, requests):
            raise RuntimeError("worker crashed")

    with pytest.raises(RuntimeError, match="worker crashed"):
        read_blocks([req("a")], Broken(), FakeJudge(), settings())


# --- without a Judge: no Scores, and the ink-based re-read of implausibly short Blocks -------------------

def test_without_a_judge_there_are_no_scores_and_no_flags():
    out = read_block(req("a"), FakeReader({"a": "x"}), None, settings())
    assert (out.text, out.score, out.verdict, out.below_threshold, out.judged) == ("x", None, None, None, False)


def test_without_a_judge_the_ink_reread_may_replace_a_short_reading():
    asked = []

    def ink(request, text):
        asked.append((request.key, text))
        return "a much longer reading of the block" if text == "x" else None

    out = read_blocks([req("a"), req("b")], FakeReader({"a": "x", "b": "fine"}), None, settings(ink_reread=ink))
    assert asked == [("a", "x"), ("b", "fine")]
    assert (out[0].text, out[0].text_before_reread) == ("a much longer reading of the block", "x")
    assert (out[1].text, out[1].text_before_reread) == ("fine", None)


def test_with_a_judge_the_ink_reread_does_not_run():
    def ink(request, text):
        raise AssertionError("the ink-based re-read must not run when the Judge is on")

    out = read_block(req("a"), FakeReader({"a": "x"}), FakeJudge(), settings(ink_reread=ink))
    assert out.text == "x"


# --- what the Judge is shown is not always byte-for-byte the written text -----------------------------------
# Measured with the real Judge on the SDK's formula example: the pipeline's rewrite of `\[ x \]` is `$$  x  $$`
# (two spaces inside the delimiters, because the model already spaces them), and that scored 0.73 against 0.97 for
# `$$ x $$`, which renders the same in Markdown.

def test_the_judge_sees_display_formulas_with_single_spaces_inside_the_delimiters_the_written_text_is_untouched():
    judge = FakeJudge()
    out = read_block(req("f", "display_formula"), FakeReader({"f": " $$  x^2  $$ "}), judge, settings())
    assert judge.calls[0][0].text == "$$ x^2 $$"
    assert out.text == "$$  x^2  $$"


def test_inline_formulas_get_the_same_treatment_but_text_and_tables_are_shown_as_written():
    judge = FakeJudge()
    read_blocks([req("a", "inline_formula"), req("b", "text"), req("c", "table")],
                FakeReader({"a": " $  x  $", "b": "see  $  x  $  and  more", "c": "<td>a  b</td>"}), judge, settings())
    assert [i.text for i in judge.calls[0]] == ["$ x $", "see  $  x  $  and  more", "<td>a  b</td>"]
