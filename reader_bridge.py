"""Make the torch worker the Reader of the PaddleOCR-VL pipeline.

Spike outcome (ticket #3): the tool does NOT own a Block loop. The pipeline already crops, merges and
orders the Blocks, and hands every Block to `vl_rec_model.predict(...)`. This module replaces that
object with a bridge that sends each crop to the worker, and shadows one private method of the
pipeline so every query also carries its Block's label (the query string alone cannot tell an inline
formula or a title from other text). Everything around the Reader (block merging, reading order,
cross-page restructuring, Markdown) is the pipeline's own code, so it behaves as before.

Three private PaddleX names are relied on; check_hook_support() verifies them at start-up and
install_reader_hook() verifies the entry layout on every call, so a PaddleX upgrade that moves them
fails with a clear message (and tests/test_hook_regression.py fails in CI).
"""
from __future__ import annotations

import inspect
import itertools
import sys
import types
from pathlib import Path

import block_reader
import minicpm_reader
from block_reader import JudgeVerdict
from worker_client import WorkerError

COLLECT = "_paddleocr_vl_collect_page_vlm_entries_core"
COLLECT_ARITY = 4  # (page_idx, blocks_for_img, imgs_in_doc_for_img, layout_prep_cfg); the hook passes them positionally
ENTRY_LEN = 6  # (page_in_batch, block_index, image, query, pixel_budget, figure_info)

# The pipeline's query strings, for a query that arrives without its label.
_LABEL_BY_QUERY = {"OCR:": "text", "Table Recognition:": "table", "Formula Recognition:": "display_formula",
                   "Chart Recognition:": "chart", "Seal Recognition:": "seal"}


class HookUnsupported(RuntimeError):
    """The installed PaddleX no longer has the private names the Reader hook is built on."""


class BlockQuery(str):
    """The prompt string the pipeline passes to its VL model, plus the Block it belongs to."""

    label: str
    bbox: tuple
    page_in_batch: int
    block_index: int


def _versions() -> str:
    import importlib.metadata as md

    try:
        return f"paddleocr {md.version('paddleocr')}, paddlex {md.version('paddlex')}"
    except md.PackageNotFoundError:
        return "paddleocr/paddlex versions unknown"


def _inner(pipe):
    try:
        return pipe.paddlex_pipeline._pipeline
    except AttributeError:
        raise HookUnsupported(
            f"the installed PaddleOCR ({_versions()}) has no `paddlex_pipeline._pipeline` on PaddleOCRVL; "
            "the MiniCPM Reader hook was written for paddleocr 3.7 / paddlex 3.7 (pinned in uv.lock). "
            "Use --reader paddle, or `uv sync` to restore the pinned versions.") from None


def original_vl_model(pipe):
    """The pipeline's own VL model (PaddleOCR-VL), before a bridge replaces it."""
    return _inner(pipe).vl_rec_model


def check_hook_support(pipe) -> None:
    """Raise HookUnsupported unless the private names the hook uses are where it expects them."""
    inner = _inner(pipe)
    problems = []
    if not hasattr(inner, "vl_rec_model"):
        problems.append("`vl_rec_model` is missing")
    method = getattr(inner, COLLECT, None)
    if method is None:
        problems.append(f"`{COLLECT}` is missing")
    else:
        names = list(inspect.signature(method).parameters)
        if len(names) != COLLECT_ARITY:
            problems.append(f"`{COLLECT}` has signature {names}, expected {COLLECT_ARITY} parameters")
    if problems:
        raise HookUnsupported(
            f"the installed PaddleX ({_versions()}) is not compatible with the MiniCPM Reader hook: "
            + "; ".join(problems) + ". The hook was written for paddleocr 3.7 / paddlex 3.7 (pinned in uv.lock). "
            "Use --reader paddle, or `uv sync` to restore the pinned versions.")


def install_reader_hook(pipe, reader):
    """Make `reader` the pipeline's VL model; returns a callable that undoes it.

    `reader.predict(inputs, **kw)` must yield one {"result": str} per input {"image": BGR array,
    "query": BlockQuery}, in order.
    """
    inner = _inner(pipe)
    original_model = inner.vl_rec_model
    original_collect = getattr(inner, COLLECT)

    def collect(page_idx, blocks_for_img, imgs_in_doc, cfg):
        entries, spotting, drop = original_collect(page_idx, blocks_for_img, imgs_in_doc, cfg)
        wrapped = []
        for entry in entries:
            if len(entry) != ENTRY_LEN:
                raise HookUnsupported(
                    f"the installed PaddleX ({_versions()}) builds VL entries of {len(entry)} fields, the hook "
                    f"expects {ENTRY_LEN}; use --reader paddle, or `uv sync` to restore the pinned versions.")
            i, j, image, prompt, pixels, figures = entry
            q = BlockQuery(prompt)
            q.label = blocks_for_img[j]["label"]
            q.bbox = tuple(int(v) for v in blocks_for_img[j]["box"])
            q.page_in_batch, q.block_index = i, j
            wrapped.append((i, j, image, q, pixels, figures))
        return wrapped, spotting, drop

    setattr(inner, COLLECT, collect)  # an instance attribute shadows the class's method
    inner.vl_rec_model = reader

    def undo():
        vars(inner).pop(COLLECT, None)
        inner.vl_rec_model = original_model

    return undo


class CropFiles:
    """The crops of one predict call as PNG files (the worker reads paths), written once and removed together.

    The array is the pipeline's BGR crop; the file is a normal RGB PNG, lossless.
    """

    def __init__(self, tmp_dir: Path):
        self.tmp_dir, self.paths = Path(tmp_dir), {}
        self._counter = itertools.count()

    def path(self, key: str, image) -> Path:
        if key not in self.paths:
            import cv2

            path = self.tmp_dir / f"crop{next(self._counter):06d}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"cannot write {path}")
            self.paths[key] = path
        return self.paths[key]

    def cleanup(self) -> None:
        for path in self.paths.values():
            path.unlink(missing_ok=True)
        self.paths.clear()


class WorkerJudge:
    """The Judge in the torch worker, as block_reader's Judge: one score request for a page's Readings."""

    def __init__(self, client, crops: CropFiles):
        self.client, self.crops = client, crops

    def score(self, items):
        payload, results = [], [None] * len(items)
        for k, item in enumerate(items):
            try:
                payload.append((k, {"key": item.key, "image": str(self.crops.path(item.key, item.image)),
                                    "text": item.text, "category": item.category}))
            except OSError as e:
                results[k] = JudgeVerdict("encode_error", None, str(e))
        if payload:
            answers = self.client.score([p for _, p in payload])
            if len(answers) != len(payload):
                raise RuntimeError(f"the worker answered for {len(answers)} of {len(payload)} Readings")
            for (k, _), a in zip(payload, answers):
                results[k] = JudgeVerdict(a["verdict"], a.get("score"), a.get("detail"))
        return results


class _DownJudge:
    """The Judge while its worker cannot be started: every Block is reported as not scored."""

    def __init__(self, error: Exception):
        self.error = error

    def score(self, items):
        raise self.error


class _MiniCPMRead:
    def __init__(self, client, crops: CropFiles, table_limit: int):
        self.client, self.crops, self.table_limit = client, crops, table_limit

    def read(self, requests):
        texts = []
        for r in requests:
            prompt, limit = minicpm_reader.prompt_and_limit(r.label, table_limit=self.table_limit)
            texts.append(self.client.read(self.crops.path(r.key, r.image), prompt, limit)[0]["text"])
        return texts


class _PaddleRead:
    """PaddleOCR-VL's own model, called once for the page with the pipeline's own arguments. `before` and
    `after` bracket the read (the Judge steps off the GPU, paddle's cache is released)."""

    def __init__(self, model, kwargs: dict, before, after):
        self.model, self.kwargs, self.before, self.after = model, kwargs, before, after

    def read(self, requests):
        if self.before:
            try:
                self.before()
            except WorkerError as e:  # the Judge's worker, not the Reader: scoring will report it, the read goes on
                print(f"warning: could not move the Judge off the GPU: {e}", file=sys.stderr)
        try:
            results = list(self.model.predict(
                [{"image": r.image, "query": r.payload} for r in requests], **self.kwargs))
        finally:
            if self.after:
                self.after()
        return [r.get("result") or "" for r in results]


class ReaderBridge:
    """The pipeline's `vl_rec_model`: reads a page's Blocks with a Reader, and (with `judge`) scores them.

    All the work is block_reader.read_blocks; this class is the glue to the pipeline and the worker. Each
    Reading is cleaned here (repetition cut, math delimiters, OTSL to HTML, table line breaks) so the
    pipeline's own post-processing, which repeats those steps, finds nothing left to do, and the Judge
    scores exactly the text that is written. `outcomes` maps (page index, bbox) to the Block's BlockResult;
    set `current_page` before each `pipe.predict`.
    """

    batch_sampler = types.SimpleNamespace(batch_size=1)  # only read when the pipeline uses queues
    strip = True

    def __init__(self, client, tmp_dir: Path, *, to_html, judge: bool = False, judge_threshold: float = 0.98,
                 repair=None, ink_reread=None):
        self.client, self.tmp_dir, self.to_html = client, Path(tmp_dir), to_html
        self.judge, self.threshold, self.repair, self.ink_reread = judge, judge_threshold, repair, ink_reread
        self.repetition_cuts = 0
        self.current_page = 0
        self.outcomes: dict[tuple, block_reader.BlockResult] = {}

    def _reader(self, kwargs: dict, crops: CropFiles):  # a block_reader.Reader
        raise NotImplementedError

    def _clean(self, raw: str, label: str) -> str:
        text, cut = minicpm_reader.clean_reading(raw, label, to_html=self.to_html, strip=self.strip)
        self.repetition_cuts += cut
        return text

    def predict(self, inputs, **kwargs):
        inputs = list(inputs)
        judge_down = None
        if self.judge or self.uses_worker_reader:
            try:
                self.client.ensure_started()  # a worker that crashed on an earlier page is started again
            except WorkerError as e:
                if self.uses_worker_reader:
                    raise  # the worker is the Reader: this page cannot be read
                judge_down = e  # the worker only judges: the page is still read, its Blocks get forward_error
        crops = CropFiles(self.tmp_dir)
        try:
            requests = []
            for n, inp in enumerate(inputs):
                query = inp["query"]
                label = getattr(query, "label", None) or _LABEL_BY_QUERY.get(str(query), "text")
                bbox = getattr(query, "bbox", None)
                requests.append(block_reader.BlockRequest(f"p{self.current_page}/b{n}", inp["image"], label, bbox, str(query)))
            settings = block_reader.ReadSettings(
                self.threshold, self._clean, self.repair if self.judge else None,
                (lambda r, t: self.ink_reread(r, t)) if self.ink_reread and not self.judge else None)
            judge = (_DownJudge(judge_down) if judge_down else WorkerJudge(self.client, crops)) if self.judge else None
            results = block_reader.read_blocks(requests, self._reader(kwargs, crops), judge, settings)
        finally:
            crops.cleanup()
        for request, result in zip(requests, results):
            self.outcomes[(self.current_page, request.bbox)] = result
        for result in results:
            yield {"result": result.text}

    uses_worker_reader = False

    def page_summary(self, page: int) -> dict:
        """How many Blocks of a page were judged, how many are below the threshold, how many could not be scored."""
        judged = [o for (p, _), o in self.outcomes.items() if p == page and o.judged]
        return {"judged": len(judged),
                "below_threshold": sum(1 for o in judged if o.below_threshold),
                "not_scored": sum(1 for o in judged if o.verdict != block_reader.SCORED)}

    def close(self):  # the pipeline calls this when it is closed; the worker is stopped by its owner
        pass


class MiniCPMBridge(ReaderBridge):
    """MiniCPM-OCR-1.0 in the worker reads; the same worker's Jev judges."""

    uses_worker_reader = True

    def _reader(self, kwargs, crops):
        # The pipeline's min_pixels/max_pixels budgets belong to PaddleOCR-VL and are ignored; the table token
        # limit is the pipeline's `max_new_tokens` (= --max-new-tokens).
        return _MiniCPMRead(self.client, crops, int(kwargs.get("max_new_tokens") or 8192))

    def read_crop(self, image, label: str, max_new_tokens: int = 8192) -> str:
        """One cleaned Reading of a crop outside the pipeline (the ink-based re-read of a short Block)."""
        self.client.ensure_started()
        crops = CropFiles(self.tmp_dir)
        try:
            raw = _MiniCPMRead(self.client, crops, max_new_tokens).read([block_reader.BlockRequest("reread", image, label)])[0]
        finally:
            crops.cleanup()
        return self._clean(raw, label)


class PaddleBridge(ReaderBridge):
    """PaddleOCR-VL reads (its own model, as without this bridge); the worker's Jev judges."""

    strip = False  # the pipeline does not strip PaddleOCR-VL's text

    def __init__(self, model, client, tmp_dir: Path, *, before_read=None, after_read=None, **kw):
        super().__init__(client, tmp_dir, **kw)
        self.model, self.before_read, self.after_read = model, before_read, after_read

    def _reader(self, kwargs, crops):
        return _PaddleRead(self.model, kwargs, self.before_read, self.after_read)
