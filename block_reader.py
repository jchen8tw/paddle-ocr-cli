"""Read a Block: Block images in, final Readings with their Scores out.

This is the seam between the pipeline's Blocks and the two models: a Reader that turns a Block image
into text and a Judge that rates a Reading against the image. It is plain Python (no torch, no paddle,
no GPU), so its behaviour is tested with a fake Reader and a fake Judge.

One call takes a page's Blocks, not one Block: the Judge renders every Reading with a headless Chromium
that takes ~0.8 s to launch, so the Readings of a page are rendered and scored in one batch. `read_block`
is the same thing for a single Block.

One Reading per Block today (ticket #4). The re-read schedule (ticket #5) extends `read_blocks` after the
first round: the Blocks whose best Score is still below the threshold are read again, as another batch of
BlockRequests with different decoding settings, and scored again; the loop keeps the highest Score.

Per Reading: Reader -> `clean` (repetition cut, OTSL to HTML, table line breaks) -> sticky Simplified to
Traditional repair -> Judge. With no Judge the repair is left to the document-level pass and the ink-based
re-read of implausibly short Blocks runs instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

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


@dataclass
class BlockRequest:
    """One Block to read: `image` is whatever the Reader and Judge adapters understand (a BGR crop);
    `key` is unique among the Blocks of one call; `bbox` lets an adapter find the Block on its page."""

    key: str
    image: Any
    label: str
    bbox: tuple | None = None
    payload: Any = None  # whatever the Reader adapter needs besides the above (the pipeline's query string)


@dataclass
class JudgeItem:
    key: str
    image: Any
    text: str
    category: str


@dataclass
class JudgeVerdict:
    verdict: str
    score: float | None = None
    detail: str | None = None


class Reader(Protocol):
    def read(self, requests: list[BlockRequest]) -> list[str]:
        """The raw text of each Block, in order. Raises if the Reader itself fails (the page fails)."""


class Judge(Protocol):
    def score(self, items: list[JudgeItem]) -> list[JudgeVerdict | tuple]:
        """One verdict per item, in order. May raise: every Block of the batch then gets `forward_error`."""


@dataclass
class ReadSettings:
    threshold: float
    clean: Callable[[str, str], str]  # (raw Reading, label) -> text; the Reader's own clean-up
    repair: Any = None                # a StickyScriptRepair, shared by all pages of a run, or None
    ink_reread: Callable[[BlockRequest, str], str | None] | None = None  # used only without a Judge


@dataclass
class BlockResult:
    text: str                                  # the final Reading
    judged: bool = False
    score: float | None = None                 # a real Score, only with verdict "scored"
    verdict: str | None = None                 # None for a Block that was not judged
    verdict_detail: str | None = None
    below_threshold: bool | None = None        # None for a Block that was not judged
    text_before_repair: str | None = None      # set when the sticky repair changed the Reading before judging
    judged_before_script_decision: bool = False  # judged before the document was known to be Traditional
    text_before_reread: str | None = None      # set when the ink-based re-read replaced the Reading
    extra: dict = field(default_factory=dict)  # room for the re-read history (ticket #5)


def _as_verdict(v) -> JudgeVerdict:
    return v if isinstance(v, JudgeVerdict) else JudgeVerdict(*v)


def _failure(verdict: str, detail: str) -> JudgeVerdict:
    return JudgeVerdict(verdict, None, detail)


def read_blocks(requests: list[BlockRequest], reader: Reader, judge: Judge | None,
                settings: ReadSettings) -> list[BlockResult]:
    """Read and judge a page's Blocks. Results come back in the order of `requests`."""
    raws = reader.read(requests)
    results: list[BlockResult] = []
    items: list[JudgeItem] = []
    slots: list[int] = []
    for i, (request, raw) in enumerate(zip(requests, raws)):
        text = settings.clean(raw, request.label)
        result = BlockResult(text=text)
        category = judge_category(request.label) if judge is not None else None
        if judge is None:
            if settings.ink_reread is not None:
                better = settings.ink_reread(request, text)
                if better is not None and better != text:
                    result.text, result.text_before_reread = better, text
        elif category is not None:
            if settings.repair is not None:
                repaired, decided = settings.repair.apply(text, request.label)
                result.judged_before_script_decision = not decided
                if repaired != text:
                    result.text, result.text_before_repair = repaired, text
            result.judged = True
            items.append(JudgeItem(request.key, request.image, judge_view(result.text, category), category))
            slots.append(i)
        results.append(result)

    if items:
        verdicts = _judge(judge, items)
        for i, v in zip(slots, verdicts):
            r = results[i]
            r.verdict, r.verdict_detail = v.verdict, v.detail
            r.score = v.score if v.verdict == SCORED else None  # a sentinel 0.0 or -1.0 is not a Score
            r.below_threshold = not (r.verdict == SCORED and r.score is not None and r.score >= settings.threshold)
    return results


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
