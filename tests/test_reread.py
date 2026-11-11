"""Re-reading low-Score Blocks (ticket #5), at the "read a Block" seam: a fake Reader and a fake Judge.

The schedule (each step runs only while the best Score so far is below the threshold):
    round 1  greedy, 1x          (the first Reading)
    round 2  greedy, 1.25x
    round 3  4 samples, temperature 0.3
    round 4  4 samples, temperature 0.7
"""
import pytest

from block_reader import (SCHEDULE, BlockRequest, ReadSettings, ReadStep, derive_seed, read_block, read_blocks)


class ScriptedReader:
    """Answers per (block key, round) from a table; everything else is "<key> r<round>.<sample>".
    Records every read call: (round, keys, seeds)."""

    def __init__(self, texts=None, unsupported=(), fail_rounds=()):
        self.texts, self.unsupported, self.fail_rounds = texts or {}, set(unsupported), set(fail_rounds)
        self.calls = []

    def supports(self, step):
        return step.round not in self.unsupported

    def read(self, requests, step, seeds):
        self.calls.append((step.round, [r.key for r in requests], list(seeds)))
        if step.round in self.fail_rounds:
            return [RuntimeError("worker error: out of memory")] * len(requests)
        out = []
        for r in requests:
            texts = self.texts.get((r.key, step.round))
            if texts is None:
                texts = [f"{r.key} r{step.round}.{k}" for k in range(step.samples)]
            out.append([texts] if isinstance(texts, str) else list(texts))
        return out

    @property
    def rounds(self):
        return [c[0] for c in self.calls]


class ScriptedJudge:
    """Scores by text (default 0.5); `raise_on_call` is the 1-based call number that raises."""

    def __init__(self, scores=None, default=("scored", 0.5), raise_on_call=None):
        self.scores, self.default, self.raise_on_call = scores or {}, default, raise_on_call
        self.calls = []

    def score(self, items):
        self.calls.append(list(items))
        if self.raise_on_call == len(self.calls):
            raise RuntimeError("renderer exploded")
        return [(("scored", self.scores[i.text]) if isinstance(self.scores.get(i.text), float)
                 else self.scores.get(i.text, self.default)) for i in items]

    @property
    def shown(self):
        return [i.text for call in self.calls for i in call]


def req(key, label="text", **kw):
    return BlockRequest(key=key, image=f"<crop {key}>", label=label, **kw)


def settings(**kw):
    kw.setdefault("threshold", 0.9)
    kw.setdefault("clean", lambda raw, label: raw.strip())
    kw.setdefault("doc_hash", "d" * 64)
    kw.setdefault("page", 3)
    return ReadSettings(**kw)


R1 = "a r1.0"  # the first Reading of block "a" from the default ScriptedReader


def first(key="a"):
    return f"{key} r1.0"


# ------------------------------------------------------------------ the schedule

def test_the_schedule_is_greedy_1x_greedy_1_25x_four_samples_at_0_3_four_samples_at_0_7():
    assert [(s.round, s.decode, s.temperature, s.scale, s.samples) for s in SCHEDULE] == [
        (1, "greedy", None, 1.0, 1), (2, "greedy", None, 1.25, 1), (3, "temperature", 0.3, 1.0, 4),
        (4, "temperature", 0.7, 1.0, 4)]


def test_a_block_that_passes_on_the_first_reading_is_read_once_and_has_no_readings():
    reader, judge = ScriptedReader(), ScriptedJudge(default=("scored", 0.95))
    out = read_block(req("a"), reader, judge, settings())
    assert reader.rounds == [1] and len(judge.calls) == 1
    assert out.readings is None and out.stop_reason is None
    assert (out.text, out.score, out.flagged) == (first(), 0.95, False)


def test_it_stops_as_soon_as_a_reading_passes():
    reader = ScriptedReader({("a", 2): "good"})
    judge = ScriptedJudge({"good": 0.97})
    out = read_block(req("a"), reader, judge, settings())
    assert reader.rounds == [1, 2]
    assert (out.text, out.score, out.flagged, out.stop_reason) == ("good", 0.97, False, "passed")
    assert [r.text for r in out.readings] == [first(), "good"]


def test_a_sampled_round_that_passes_stops_the_schedule_before_the_next_round():
    reader = ScriptedReader({("a", 3): ["s0", "s1", "s2", "s3"]})
    judge = ScriptedJudge({"s2": 0.95})
    out = read_block(req("a"), reader, judge, settings())
    assert reader.rounds == [1, 2, 3]
    assert (out.text, out.stop_reason) == ("s2", "passed")


def test_a_block_below_the_threshold_goes_through_every_round_in_order_and_is_exhausted_at_ten_readings():
    reader, judge = ScriptedReader(), ScriptedJudge(default=("scored", 0.5))
    out = read_block(req("a"), reader, judge, settings())
    assert reader.rounds == [1, 2, 3, 4]
    assert len(out.readings) == 10  # 1 + 1 + 4 + 4, all distinct
    assert out.stop_reason == "exhausted" and out.flagged is True
    assert [(r.round, r.decode, r.temperature, r.scale) for r in out.readings] == (
        [(1, "greedy", None, 1.0), (2, "greedy", None, 1.25)] + [(3, "temperature", 0.3, 1.0)] * 4
        + [(4, "temperature", 0.7, 1.0)] * 4)


def test_no_block_gets_more_than_ten_readings_even_with_a_longer_schedule():
    steps = (*SCHEDULE, ReadStep(5, "temperature", 0.9, 1.0, 4))
    reader = ScriptedReader()
    out = read_block(req("a"), reader, ScriptedJudge(), settings(schedule=steps))
    assert len(out.readings) == 10 and 5 not in reader.rounds


def test_the_cap_cuts_the_samples_of_the_last_round_it_still_has_room_for():
    steps = (SCHEDULE[0], ReadStep(2, "temperature", 0.3, 1.0, 4), ReadStep(3, "temperature", 0.7, 1.0, 4),
             ReadStep(4, "temperature", 0.9, 1.0, 4))
    reader = ScriptedReader()
    out = read_block(req("a"), reader, ScriptedJudge(), settings(schedule=steps))
    assert len(out.readings) == 10
    assert [r.round for r in out.readings].count(4) == 1


# ------------------------------------------------------------------ choosing the winner

def test_the_highest_score_wins_over_the_most_recent_reading():
    reader = ScriptedReader({("a", 2): "better", ("a", 3): ["worse"] * 4, ("a", 4): ["worst"] * 4})
    judge = ScriptedJudge({first(): 0.4, "better": 0.8, "worse": 0.6, "worst": 0.2})
    out = read_block(req("a"), reader, judge, settings())
    assert (out.text, out.score, out.verdict) == ("better", 0.8, "scored")
    assert out.flagged is True and out.stop_reason == "exhausted"


def test_a_tie_goes_to_the_earliest_reading():
    reader = ScriptedReader({("a", 2): "second"})
    out = read_block(req("a"), reader, ScriptedJudge({first(): 0.7, "second": 0.7}), settings())
    assert out.text == first()


def test_unscorable_verdicts_never_win_and_the_first_reading_is_kept_when_nothing_was_scored():
    bad = ("precheck_reject", None)
    reader = ScriptedReader({("a", 2): "x2", ("a", 3): ["x3"] * 4, ("a", 4): ["x4"] * 4})
    judge = ScriptedJudge({first(): ("render_error", None), "x2": bad, "x3": bad, "x4": ("missing_original_image", None)})
    out = read_block(req("a"), reader, judge, settings())
    assert (out.text, out.score, out.verdict, out.flagged) == (first(), None, "render_error", True)
    assert out.stop_reason == "no_scored"
    assert len(out.readings) == 4  # one distinct text per round: x3 and x4 are four identical samples each


def test_a_sentinel_score_on_an_unscorable_verdict_cannot_win_over_a_real_score():
    reader = ScriptedReader({("a", 2): "x2"})
    judge = ScriptedJudge({first(): ("scored", 0.1), "x2": ("precheck_reject", 1.0)}, default=("render_error", None))
    out = read_block(req("a"), reader, judge, settings())
    assert (out.text, out.score) == (first(), 0.1)
    assert out.stop_reason == "exhausted"


def test_the_winner_that_is_not_the_first_reading_replaces_the_blocks_text_score_and_verdict():
    reader = ScriptedReader({("a", 2): "x2"})
    judge = ScriptedJudge({first(): ("render_error", None), "x2": 0.6})
    out = read_block(req("a"), reader, judge, settings())
    assert (out.text, out.score, out.verdict) == ("x2", 0.6, "scored")


# ------------------------------------------------------------------ duplicates

def test_duplicate_texts_are_scored_once_and_listed_once():
    reader = ScriptedReader({("a", 2): first(), ("a", 3): ["s", "s", "s", first()], ("a", 4): ["s"] * 4})
    judge = ScriptedJudge()
    out = read_block(req("a"), reader, judge, settings())
    assert sorted(judge.shown) == sorted([first(), "s"])
    assert [r.text for r in out.readings] == [first(), "s"]
    assert [r.duplicates for r in out.readings] == [2, 6]  # the Readings are counted, not hidden
    assert out.stop_reason == "exhausted"


def test_a_round_whose_readings_are_all_duplicates_makes_no_judge_call():
    reader = ScriptedReader({("a", 2): first(), ("a", 3): [first()] * 4, ("a", 4): [first()] * 4})
    judge = ScriptedJudge()
    read_block(req("a"), reader, judge, settings())
    assert len(judge.calls) == 1


def test_duplicates_count_against_the_ten_reading_cap():
    reader = ScriptedReader({("a", 2): first(), ("a", 3): [first()] * 4, ("a", 4): [first()] * 4})
    out = read_block(req("a"), reader, ScriptedJudge(), settings())
    assert out.readings[0].duplicates == 9


# ------------------------------------------------------------------ batching

def test_re_reading_is_one_read_and_one_score_call_per_round_for_the_blocks_still_below_the_threshold():
    reader = ScriptedReader({("c", 2): "c ok"})
    judge = ScriptedJudge({"a r1.0": 0.99, "b r1.0": 0.5, "c r1.0": 0.5, "c ok": 0.99, "b r2.0": 0.99})
    out = read_blocks([req("a"), req("b"), req("c")], reader, judge, settings())
    assert [(c[0], c[1]) for c in reader.calls] == [(1, ["a", "b", "c"]), (2, ["b", "c"])]
    assert [len(c) for c in judge.calls] == [3, 2]
    assert [o.stop_reason for o in out] == [None, "passed", "passed"]


def test_unjudged_labels_are_never_re_read():
    reader = ScriptedReader()
    out = read_blocks([req("c", "chart")], reader, ScriptedJudge(), settings())
    assert reader.rounds == [1] and out[0].readings is None


def test_without_a_judge_there_is_no_re_read():
    reader = ScriptedReader()
    out = read_blocks([req("a")], reader, None, settings())
    assert reader.rounds == [1] and out[0].readings is None


# ------------------------------------------------------------------ the Judge failing

def test_a_judge_failure_mid_schedule_keeps_the_best_reading_so_far():
    reader = ScriptedReader({("a", 2): "second"})
    judge = ScriptedJudge({first(): 0.4, "second": 0.7}, raise_on_call=3)  # round 3's batch raises
    out = read_block(req("a"), reader, judge, settings())
    assert reader.rounds == [1, 2, 3]
    assert (out.text, out.score, out.stop_reason) == ("second", 0.7, "judge_error")
    assert out.flagged is True
    assert all(r.verdict == "forward_error" for r in out.readings if r.round == 3)


def test_a_per_item_forward_error_also_stops_that_block_with_judge_error():
    reader = ScriptedReader({("a", 2): "second"})
    judge = ScriptedJudge({first(): 0.4, "second": ("forward_error", None)})
    out = read_block(req("a"), reader, judge, settings())
    assert reader.rounds == [1, 2]
    assert (out.text, out.score, out.stop_reason) == (first(), 0.4, "judge_error")


def test_a_failing_judge_on_the_first_reading_means_no_re_read_at_all():
    reader = ScriptedReader()
    out = read_block(req("a"), reader, ScriptedJudge(raise_on_call=1), settings())
    assert reader.rounds == [1]
    assert (out.verdict, out.readings, out.stop_reason) == ("forward_error", None, None)


def test_one_blocks_judge_error_does_not_stop_the_others():
    reader = ScriptedReader({("a", 2): "a2", ("b", 2): "b2"})
    judge = ScriptedJudge({"a r1.0": 0.4, "b r1.0": 0.4, "a2": ("encode_error", None), "b2": 0.95})
    a, b = read_blocks([req("a"), req("b")], reader, judge, settings())
    assert (a.stop_reason, b.stop_reason, b.text) == ("judge_error", "passed", "b2")


# ------------------------------------------------------------------ a Reader that cannot do a round

def test_rounds_the_reader_does_not_support_are_skipped_and_recorded_not_faked():
    reader = ScriptedReader(unsupported={3, 4})
    out = read_block(req("a"), reader, ScriptedJudge(), settings())
    assert reader.rounds == [1, 2]
    assert [s["round"] for s in out.skipped_rounds] == [3, 4]
    assert all(s["reason"] for s in out.skipped_rounds)
    assert len(out.readings) == 2 and out.stop_reason == "exhausted"


def test_a_reader_error_in_a_re_read_round_keeps_the_best_reading_so_far():
    reader = ScriptedReader({("a", 2): "second"}, fail_rounds={3})
    judge = ScriptedJudge({first(): 0.4, "second": 0.7})
    out = read_block(req("a"), reader, judge, settings())
    assert (out.text, out.score, out.stop_reason) == ("second", 0.7, "reader_error")
    assert "out of memory" in out.stop_detail


# ------------------------------------------------------------------ seeds

def test_the_same_inputs_give_the_same_seeds():
    r1, r2 = ScriptedReader(), ScriptedReader()
    read_block(req("a"), r1, ScriptedJudge(), settings())
    read_block(req("a"), r2, ScriptedJudge(), settings())
    assert r1.calls == r2.calls
    assert r1.calls[2][2][0] is not None and r1.calls[0][2] == [None]  # greedy rounds have no seed


def test_a_seed_depends_on_document_page_block_and_round():
    base = derive_seed("d" * 64, 3, "7", 3)
    assert derive_seed("d" * 64, 3, "7", 3) == base
    assert len({base, derive_seed("e" * 64, 3, "7", 3), derive_seed("d" * 64, 4, "7", 3),
                derive_seed("d" * 64, 3, "8", 3), derive_seed("d" * 64, 3, "7", 4)}) == 5


def test_a_seed_is_a_fixed_number_independent_of_pythons_hash_randomisation():
    # blake2b of "<hash>|<page>|<block>|<round>", first 8 bytes, 31 bits: pinned so a process-dependent hash() cannot sneak in
    assert derive_seed("d" * 64, 3, "7", 3) == 310125698
    assert 0 <= derive_seed("d" * 64, 3, "7", 3) < 2 ** 31


def test_seeds_use_the_block_id_not_its_position_in_the_batch():
    reader = ScriptedReader()
    read_blocks([req("x", block_id="7"), req("y", block_id="8")], reader, ScriptedJudge(), settings())
    seeds = reader.calls[2][2]
    assert seeds == [derive_seed("d" * 64, 3, "7", 3), derive_seed("d" * 64, 3, "8", 3)]
    reader2 = ScriptedReader()
    read_blocks([req("y", block_id="8")], reader2, ScriptedJudge(), settings())  # alone in its batch: same seed
    assert reader2.calls[2][2] == [seeds[1]]


# ------------------------------------------------------------------ every Reading is repaired and cleaned before judging

def test_every_reading_is_cleaned_and_repaired_before_it_is_judged_and_the_judge_sees_the_judge_view():
    from script_repair import ScriptRepair, StickyScriptRepair

    trad = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"
    reader = ScriptedReader({("a", 1): trad, ("a", 2): " 银行 ", ("a", 3): ["\\[ x \\]"] * 4})
    judge = ScriptedJudge()
    s = settings(repair=StickyScriptRepair(ScriptRepair()),
                 clean=lambda raw, label: raw.strip().replace("\\[ x \\]", "$$  x  $$"))
    read_block(req("a", "display_formula"), reader, judge, s)
    # formulas are not repaired (NO_REPAIR_LABELS) but are shown with single spaces inside the delimiters
    assert "$$ x $$" in judge.shown
    out = read_block(req("b"), ScriptedReader({("b", 1): trad, ("b", 2): " 银行 "}), ScriptedJudge(), s)
    assert [r.text for r in out.readings][:2] == [trad, "銀行"]
    assert out.readings[1].text_before_repair == "银行"


def test_re_reads_do_not_count_again_in_the_scripts_running_survey():
    from script_repair import ScriptRepair, StickyScriptRepair

    once, many = StickyScriptRepair(ScriptRepair()), StickyScriptRepair(ScriptRepair())
    text = "銀行國際"
    read_block(req("a"), ScriptedReader({("a", 1): text}), ScriptedJudge(default=("scored", 0.99)), settings(repair=once))
    read_block(req("a"), ScriptedReader({("a", 1): text}), ScriptedJudge(default=("scored", 0.1)), settings(repair=many))
    assert (once.simplified, once.traditional) == (many.simplified, many.traditional) != (0, 0)


def test_the_winners_repair_record_is_the_winners_own():
    from script_repair import ScriptRepair, StickyScriptRepair

    trad = "國際學習體驗電腦關係開發設計價值義務織組條約證據隨時變數與認為對應條款歷計劃書據點"
    repair = StickyScriptRepair(ScriptRepair())
    reader = ScriptedReader({("a", 1): trad, ("a", 2): "银行"})
    judge = ScriptedJudge({trad: 0.5, "銀行": 0.8})
    out = read_block(req("a"), reader, judge, settings(repair=repair))
    assert (out.text, out.text_before_repair) == ("銀行", "银行")
    reader = ScriptedReader({("a", 1): "银行", ("a", 2): trad + "x"})
    out = read_block(req("a"), reader, ScriptedJudge({"銀行": 0.5, trad + "x": 0.3}), settings(repair=repair))
    assert (out.text, out.text_before_repair) == ("銀行", "银行")


# ------------------------------------------------------------------ threshold 0 turns re-reading off

@pytest.mark.parametrize("verdict", ["precheck_reject", "render_error", "missing_original_image"])
def test_threshold_zero_never_re_reads_even_when_the_first_reading_is_unusable(verdict):
    reader, judge = ScriptedReader(), ScriptedJudge(default=(verdict, None))
    out = read_block(req("a"), reader, judge, settings(threshold=0.0))
    assert reader.rounds == [1] and len(judge.calls) == 1
    assert out.readings is None and out.stop_reason is None
    assert (out.text, out.score, out.verdict) == (first(), None, verdict)


# ------------------------------------------------------------------ a Reader that answers for too few Blocks

class ShortReader(ScriptedReader):
    """Drops the last Block's answer in the given round."""

    def __init__(self, short_round, **kw):
        super().__init__(**kw)
        self.short_round = short_round

    def read(self, requests, step, seeds):
        out = super().read(requests, step, seeds)
        return out[:-1] if step.round == self.short_round else out


def test_a_first_round_with_fewer_answers_than_blocks_fails_instead_of_dropping_blocks():
    with pytest.raises(RuntimeError, match="1 of 2"):
        read_blocks([req("a"), req("b")], ShortReader(1), ScriptedJudge(), settings())


def test_a_re_read_round_with_fewer_answers_than_blocks_is_a_reader_error_for_those_blocks_only():
    a, b = read_blocks([req("a"), req("b")], ShortReader(2), ScriptedJudge(), settings())
    assert (a.stop_reason, b.stop_reason) == ("reader_error", "reader_error")
    assert "1 of 2" in a.stop_detail
    assert (a.text, b.text) == (first("a"), first("b"))  # the first Readings are kept
