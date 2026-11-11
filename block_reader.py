"""Read a Block: Block images in, final Readings with their Scores out.

This is the seam between the pipeline's Blocks and the two models: a Reader that turns a Block image
into text and a Judge that rates a Reading against the image. It is plain Python (no torch, no paddle,
no GPU), so its behaviour is tested with a fake Reader and a fake Judge.

One call takes a page's Blocks, not one Block: the Judge renders every Reading with a headless Chromium
that takes ~0.8 s to launch, so the Readings of a page are rendered and scored in one batch. `read_block`
is the same thing for a single Block.

Re-reading (ticket #5): after the first Reading, the Blocks whose best Score is still below the threshold are
read again, round by round (SCHEDULE), each round being one batch for the Reader and one batch for the Judge;
the highest Score wins. A Block that passes on its first Reading costs nothing extra.

Per Reading: Reader -> `clean` (repetition cut, OTSL to HTML, table line breaks) -> sticky Simplified to
Traditional repair -> Judge. With no Judge the repair is left to the document-level pass and the ink-based
re-read of implausibly short Blocks runs instead.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

from script_repair import NO_REPAIR_LABELS

SCORED = "scored"
# The official verdicts. Only SCORED carries a real Score. The first three mean the Reading is unusable,
# the last two mean scoring never happened.
UNUSABLE_VERDICTS = ("precheck_reject", "render_error", "missing_original_image")
NOT_SCORED_VERDICTS = ("encode_error", "forward_error")
VERDICTS = (SCORED, *UNUSABLE_VERDICTS, *NOT_SCORED_VERDICTS)

# The Judge was not trained on these: they are read (when enabled) but never judged.
UNJUDGED_LABELS = frozenset({"chart", "seal", "image", "header_image", "footer_image"})
FORMULA_LABELS = frozenset({"inline_formula", "display_formula"})


def judge_category(label: str) -> str | None:
    """The Judge's category of a Block label, or None for a Block that is never judged."""
    if label in UNJUDGED_LABELS:
        return None
    if label in FORMULA_LABELS:
        return "formula"
    return "table" if label == "table" else "text"


def judge_view(text: str, category: str) -> str:
    """The text the Judge is shown for a Reading. The same as the written text except for one whitespace detail
    of formulas: runs of spaces just inside `$` / `$$` are made single. The pipeline rewrites the model's
    `\\[ x \\]` into `$$  x  $$`; the real Judge rates that 0.73 where `$$ x $$` (which renders identically in
    Markdown) gets 0.97, so it is shown the latter. The written text is not changed."""
    if category != "formula":
        return text
    text = re.sub(r"(\$+)[ \t]{2,}", r"\1 ", text)
    return re.sub(r"[ \t]{2,}(\$+)", r" \1", text)


@dataclass(frozen=True)
class ReadStep:
    """How one round of the re-read schedule reads a Block."""

    round: int
    decode: str                # "greedy" or "temperature"
    temperature: float | None  # None for greedy
    scale: float               # 1.0 = the Block's own image
    samples: int               # Readings produced for each Block in this round


# Spec #1: each step runs only while the best Score so far is below the threshold.
SCHEDULE = (
    ReadStep(1, "greedy", None, 1.0, 1),        # the first Reading
    ReadStep(2, "greedy", None, 1.25, 1),
    ReadStep(3, "temperature", 0.3, 1.0, 4),
    ReadStep(4, "temperature", 0.7, 1.0, 4),
)
MAX_READINGS = 10  # per Block, duplicates included


def derive_seed(doc_hash: str, page: int, block_id: str, round: int) -> int:
    """The sampling seed of one Block in one round: a fixed function of the document hash, the page, the Block's id
    and the round. blake2b rather than hash(), which differs from one process to the next."""
    digest = hashlib.blake2b(f"{doc_hash}|{page}|{block_id}|{round}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


@dataclass
class BlockRequest:
    """One Block to read: `image` is whatever the Reader and Judge adapters understand (a BGR crop);
    `key` is unique among the Blocks of one call; `bbox` lets an adapter find the Block on its page.
    `block_id` identifies the Block on its page for the sampling seeds (default: `key`)."""

    key: str
    image: Any
    label: str
    bbox: tuple | None = None
    payload: Any = None  # whatever the Reader adapter needs besides the above (the pipeline's query string)
    block_id: str | None = None
    scaler: Any = None   # offers `source(scale)`: where the Block's image at another scale comes from (a re-render, a resample)

    @property
    def seed_id(self) -> str:
        return self.block_id if self.block_id is not None else self.key


@dataclass
class JudgeItem:
    key: str                       # unique among the items of one call
    image: Any
    text: str
    category: str
    image_key: str | None = None   # the request key of the Block (several items can show the same image)


@dataclass
class JudgeVerdict:
    verdict: str
    score: float | None = None
    detail: str | None = None


class Reader(Protocol):
    def supports(self, step: ReadStep) -> bool:
        """Whether this Reader can honestly do `step` (a Reader that ignores temperature cannot do a sampled round)."""

    def read(self, requests: list[BlockRequest], step: ReadStep, seeds: list[int | None]) -> list:
        """For each Block, in order, the raw texts of the `step.samples` Readings (a list of str), or an Exception
        for a Block whose Reading failed. `seeds[i]` seeds the sampling of Block i (None for greedy rounds).
        Raises if the Reader itself fails; in round 1 that fails the page."""


class Judge(Protocol):
    def score(self, items: list[JudgeItem]) -> list[JudgeVerdict | tuple]:
        """One verdict per item, in order. May raise: every Block of the batch then gets `forward_error`."""


@dataclass
class ReadSettings:
    threshold: float
    clean: Callable[[str, str], str]  # (raw Reading, label) -> text; the Reader's own clean-up
    repair: Any = None                # a StickyScriptRepair, shared by all pages of a run, or None
    ink_reread: Callable[[BlockRequest, str], str | None] | None = None  # used only without a Judge
    doc_hash: str = ""                # with the page and the Block id, what the sampling seeds are made of
    page: int = 0
    schedule: tuple = SCHEDULE
    max_readings: int = MAX_READINGS


@dataclass
class Reading:
    """One distinct text a Block was read as, with how it was produced and how the Judge rated it."""

    text: str                                  # as it would be written (cleaned, and repaired when the document is Traditional)
    round: int
    decode: str
    temperature: float | None
    scale: float
    scale_source: str | None = None            # for a scaled Reading: "page_render" or "crop_resample"
    verdict: str | None = None
    score: float | None = None                 # a real Score, only with verdict "scored"
    detail: str | None = None
    duplicates: int = 0                        # later Readings with this very text, which were not scored again
    text_before_repair: str | None = None
    judged_before_script_decision: bool = False


@dataclass
class BlockResult:
    text: str                                  # the final Reading (the winner's)
    judged: bool = False
    score: float | None = None                 # a real Score, only with verdict "scored"
    verdict: str | None = None                 # None for a Block that was not judged
    verdict_detail: str | None = None
    below_threshold: bool | None = None        # None for a Block that was not judged
    text_before_repair: str | None = None      # set when the sticky repair changed the Reading before judging
    judged_before_script_decision: bool = False  # judged before the document was known to be Traditional
    text_before_reread: str | None = None      # set when the ink-based re-read replaced the Reading
    readings: list[Reading] | None = None      # the distinct Readings, only for a Block that was re-read
    stop_reason: str | None = None             # passed / exhausted / no_scored / judge_error / reader_error
    stop_detail: str | None = None
    skipped_rounds: list[dict] = field(default_factory=list)  # rounds the Reader cannot do, with the reason


def _as_verdict(v) -> JudgeVerdict:
    return v if isinstance(v, JudgeVerdict) else JudgeVerdict(*v)


def _failure(verdict: str, detail: str) -> JudgeVerdict:
    return JudgeVerdict(verdict, None, detail)


class _Block:
    """A Block's progress through the schedule."""

    def __init__(self, request: BlockRequest, result: BlockResult):
        self.request, self.result = request, result
        self.readings: list[Reading] = []
        self.by_text: dict[str, Reading] = {}
        self.produced = 0                       # Readings made, duplicates included (the cap counts these)
        self.entered = False                    # went into the re-read schedule
        self.stop: tuple[str, str | None] | None = None

    @property
    def best(self) -> Reading | None:
        best = None
        for r in self.readings:
            if r.verdict == SCORED and r.score is not None and (best is None or r.score > best.score):
                best = r  # strict: the earliest of equal Scores wins
        return best


def read_blocks(requests: list[BlockRequest], reader: Reader, judge: Judge | None,
                settings: ReadSettings) -> list[BlockResult]:
    """Read and judge a page's Blocks, re-reading the ones below the threshold. Results come back in the order of
    `requests`."""
    first_step = settings.schedule[0]
    raws = reader.read(requests, first_step, [None] * len(requests))
    if len(raws) != len(requests):  # zip() would silently drop the Blocks without an answer
        raise RuntimeError(f"the Reader answered for {len(raws)} of {len(requests)} Blocks")
    results: list[BlockResult] = []
    blocks: list[_Block] = []
    items: list[JudgeItem] = []
    owners: list[tuple[_Block, Reading]] = []
    for request, texts in zip(requests, raws):
        if isinstance(texts, Exception):
            raise texts
        text = settings.clean(texts[0] if texts else "", request.label)
        result = BlockResult(text=text)
        results.append(result)
        category = judge_category(request.label) if judge is not None else None
        if judge is None:
            if settings.ink_reread is not None:
                better = settings.ink_reread(request, text)
                if better is not None and better != text:
                    result.text, result.text_before_reread = better, text
        elif category is not None:
            block = _Block(request, result)
            blocks.append(block)
            result.judged = True
            reading = _new_reading(block, text, first_step, settings, survey=True)
            items.append(JudgeItem(request.key, request.image, judge_view(reading.text, category), category, request.key))
            owners.append((block, reading))

    if blocks:
        _score(judge, items, owners)
        _reread(blocks, reader, judge, settings)
        for block in blocks:
            _finish(block, settings)
    return results


def _new_reading(block: _Block, text: str, step: ReadStep, settings: ReadSettings, *, survey: bool) -> Reading:
    reading = Reading(text, step.round, step.decode, step.temperature, step.scale)
    scaler = block.request.scaler
    if step.scale != 1.0 and scaler is not None:
        reading.scale_source = scaler.source(step.scale)
    if settings.repair is not None:
        repaired, decided = settings.repair.apply(text, block.request.label, survey=survey)
        reading.judged_before_script_decision = not decided and block.request.label not in NO_REPAIR_LABELS
        if repaired != text:
            reading.text, reading.text_before_repair = repaired, text
    block.readings.append(reading)
    block.by_text[reading.text] = reading
    block.produced += 1
    return reading


def _score(judge: Judge, items: list[JudgeItem], owners: list[tuple[_Block, Reading]]) -> None:
    if not items:
        return
    for (_, reading), v in zip(owners, _judge(judge, items)):
        reading.verdict, reading.detail = v.verdict, v.detail
        reading.score = v.score if v.verdict == SCORED else None  # a sentinel 0.0 or -1.0 is not a Score


def _passed(block: _Block, settings: ReadSettings) -> bool:
    best = block.best
    return best is not None and best.score >= settings.threshold


def _reread(blocks: list[_Block], reader: Reader, judge: Judge, settings: ReadSettings) -> None:
    """Rounds 2.. of the schedule, round by round over the Blocks still below the threshold: one Reader batch and one
    Judge batch per round. A threshold of 0 (or less) turns re-reading off: nothing is below it, and a Block whose
    first Reading is unusable has no Score to improve on either."""
    if settings.threshold <= 0:
        return
    for block in blocks:
        if block.readings[0].verdict in NOT_SCORED_VERDICTS:
            continue  # the Judge is not working: there is no Score to improve on, so no re-read
        block.entered = True
        if _passed(block, settings):
            block.stop = ("passed", None)
    for step in settings.schedule[1:]:
        active = [b for b in blocks if b.entered and b.stop is None]
        if not active:
            return
        if not reader.supports(step):
            for b in active:
                b.result.skipped_rounds.append({
                    "round": step.round, "decode": step.decode, "temperature": step.temperature, "scale": step.scale,
                    "reason": "the Reader cannot do this round"
                              + (" (it ignores the sampling temperature)" if step.decode == "temperature" else "")})
            continue
        room = min(settings.max_readings - b.produced for b in active)
        if room <= 0:
            return
        step = replace(step, samples=min(step.samples, room))
        seeds = [derive_seed(settings.doc_hash, settings.page, b.request.seed_id, step.round)
                 if step.decode == "temperature" else None for b in active]
        answers = reader.read([b.request for b in active], step, seeds)
        if len(answers) != len(active):  # no telling which Block an answer belongs to: none of them is used
            for b in active:
                b.stop = ("reader_error", f"round {step.round}: the Reader answered for {len(answers)} of {len(active)} Blocks")
            continue
        items: list[JudgeItem] = []
        owners: list[tuple[_Block, Reading]] = []
        for b, texts in zip(active, answers):
            if isinstance(texts, Exception):
                b.stop = ("reader_error", f"round {step.round}: {texts}")
                continue
            category = judge_category(b.request.label)
            for k, raw in enumerate(texts[:step.samples]):
                text = settings.clean(raw, b.request.label)
                judged_text = text
                if settings.repair is not None:
                    judged_text = settings.repair.apply(text, b.request.label, survey=False)[0]
                twin = b.by_text.get(judged_text)
                if twin is not None:
                    twin.duplicates += 1
                    b.produced += 1
                    continue
                reading = _new_reading(b, text, step, settings, survey=False)
                items.append(JudgeItem(f"{b.request.key}/r{step.round}.{k}", b.request.image,
                                       judge_view(reading.text, category), category, b.request.key))
                owners.append((b, reading))
        _score(judge, items, owners)
        for b in active:
            if b.stop is not None:
                continue
            fresh = [r for r in b.readings if r.round == step.round]
            if _passed(b, settings):
                b.stop = ("passed", None)
            elif any(r.verdict in NOT_SCORED_VERDICTS for r in fresh):
                detail = next(r.detail for r in fresh if r.verdict in NOT_SCORED_VERDICTS)
                b.stop = ("judge_error", f"round {step.round}: {detail}")


def _finish(block: _Block, settings: ReadSettings) -> None:
    """The Block's result is its winning Reading: the highest `scored` one, else the first."""
    result, best = block.result, block.best
    winner = best or block.readings[0]
    result.text, result.text_before_repair = winner.text, winner.text_before_repair
    result.judged_before_script_decision = winner.judged_before_script_decision
    result.verdict, result.verdict_detail, result.score = winner.verdict, winner.detail, winner.score
    result.below_threshold = not (best is not None and best.score >= settings.threshold)
    if not block.entered or block.produced == 1 and block.stop == ("passed", None):
        return  # not re-read: the first Reading's own verdict says it all
    stop = block.stop or ("exhausted" if best is not None else "no_scored", None)
    result.readings = block.readings
    result.stop_reason, result.stop_detail = stop


def _judge(judge: Judge, items: list[JudgeItem]) -> list[JudgeVerdict]:
    try:
        answers = [_as_verdict(v) for v in judge.score(items)]
    except Exception as e:  # one failing batch must not cost the page (or the document)
        return [_failure("forward_error", f"{type(e).__name__}: {e}")] * len(items)
    if len(answers) != len(items):
        return [_failure("forward_error", f"the Judge answered for {len(answers)} of {len(items)} Readings")] * len(items)
    return [v if v.verdict in VERDICTS else _failure("forward_error", f"unknown verdict {v.verdict!r}") for v in answers]


def read_block(request: BlockRequest, reader: Reader, judge: Judge | None, settings: ReadSettings) -> BlockResult:
    """`read_blocks` for a single Block."""
    return read_blocks([request], reader, judge, settings)[0]
