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

import collections
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
                payload.append((k, {"key": item.key, "image": str(self.crops.path(item.image_key or item.key, item.image)),
                                    "text": item.text, "category": item.category}))
            except Exception as e:  # OSError, or cv2.error on a degenerate crop: this Reading only, not the page's batch
                results[k] = JudgeVerdict("encode_error", None, f"{type(e).__name__}: {e}")
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


def scaled_image(image, scale: float):
    """A BGR crop resampled by `scale` (Lanczos). The Judge always sees the Block's own, unscaled crop."""
    if scale == 1.0:
        return image
    import cv2

    h, w = image.shape[:2]
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_LANCZOS4)


PAGE_RENDER, CROP_RESAMPLE = "page_render", "crop_resample"


class BlockScaler:
    """A Block's image at another scale (the greedy 1.25x round).

    The better source is a re-render of the page at that scale, cropped at the Block's box: new pixels, as the
    ticket says. It is only used when the image the pipeline gave the Reader is exactly the plain crop of the
    page's own render at that box (merged Blocks, formulas with their margins cut, tables with figure tokens
    painted on and polygon-masked Blocks are not). Otherwise the image is resampled. `page_source` offers
    `original(bbox)` (the page's own crop) and `crop(bbox, factor)` (the re-rendered one, or None).
    """

    def __init__(self, image, bbox, page_source, stats=None):
        self.image, self.bbox, self.page_source, self.stats = image, bbox, page_source, stats
        self._cache: dict[float, tuple] = {}

    def get(self, scale: float) -> tuple:
        """(image, source) for `scale`; source is None at 1x."""
        if scale == 1.0:
            return self.image, None
        if scale not in self._cache:
            self._cache[scale] = self._render(scale) or (scaled_image(self.image, scale), CROP_RESAMPLE)
            if self.stats is not None:
                self.stats[self._cache[scale][1]] += 1
        return self._cache[scale]

    def source(self, scale: float) -> str | None:
        return self.get(scale)[1]

    def _render(self, scale: float):
        import numpy as np

        if self.page_source is None or self.bbox is None:
            return None
        original = self.page_source.original(self.bbox)
        if original is None or original.shape != self.image.shape or not np.array_equal(original, self.image):
            return None
        crop = self.page_source.crop(self.bbox, scale)
        return None if crop is None or crop.size == 0 else (crop, PAGE_RENDER)


def block_image(request, scale: float):
    """The image of a request at `scale`: its scaler's, or a resample."""
    scaler = getattr(request, "scaler", None)
    return scaler.get(scale)[0] if scaler is not None else scaled_image(request.image, scale)


# Sampling parameters of the temperature rounds (the MiniCPM-OCR SDK's own sampling rounds use them).
SAMPLE_TOP_K = 20
SAMPLE_TOP_P = 0.8


class _MiniCPMRead:
    """MiniCPM-OCR-1.0 in the worker: every round of the schedule. A failure in a re-read round (the worker's, or
    preparing the image) is returned for that Block (the Block keeps its best Reading so far); in round 1 it
    propagates and fails the page."""

    def __init__(self, client, crops: CropFiles | None, table_limit: int):
        self.client, self.crops, self.table_limit = client, crops, table_limit

    def supports(self, step) -> bool:
        return True

    def read(self, requests, step, seeds):
        answers = []
        for r, seed in zip(requests, seeds):
            prompt, limit = minicpm_reader.prompt_and_limit(r.label, table_limit=self.table_limit)
            decode = None if step.decode == "greedy" else {
                "mode": "sample", "temperature": step.temperature, "top_k": SAMPLE_TOP_K, "top_p": SAMPLE_TOP_P,
                "n": step.samples, "seed": seed}
            try:
                path = self.crops.path(r.key if step.scale == 1.0 else f"{r.key}@{step.scale:g}", block_image(r, step.scale))
                answers.append([x["text"] for x in self.client.read(path, prompt, limit, decode=decode)])
            except Exception as e:  # WorkerError, or a failure preparing the image (cv2.error ...)
                if step.round == 1:
                    raise
                answers.append(e)
        return answers


class _PaddleRead:
    """PaddleOCR-VL's own model, called once for the page with the pipeline's own arguments. `before` and
    `after` bracket the read (the Judge steps off the GPU, paddle's cache is released).

    The local PaddleOCR-VL ignores temperature and top_p (paddlex warns that they are "not supported by the local
    model"): it decodes greedily, so the sampled rounds are not offered. The 1.25x round is a resampled input, which
    it can honestly do (its own min/max pixel budget may map it back to the same grid; the text is then a duplicate)."""

    def __init__(self, model, kwargs: dict, before, after):
        self.model, self.kwargs, self.before, self.after = model, kwargs, before, after

    def supports(self, step) -> bool:
        return step.decode == "greedy"

    def read(self, requests, step, seeds):
        try:
            if self.before:
                try:
                    self.before()
                except WorkerError as e:  # the Judge's worker, not the Reader: scoring will report it, the read goes on
                    print(f"warning: could not move the Judge off the GPU: {e}", file=sys.stderr)
            try:
                results = list(self.model.predict(
                    [{"image": block_image(r, step.scale), "query": r.payload} for r in requests],
                    **self.kwargs))
            finally:
                if self.after:
                    self.after()
        except Exception as e:
            if step.round == 1:
                raise
            return [e] * len(requests)
        return [[r.get("result") or ""] for r in results]


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

    def __init__(self, client, tmp_dir: Path, *, to_html, judge: bool = False, judge_threshold: float = 0.5,
                 repair=None, ink_reread=None, doc_hash: str = ""):
        self.client, self.tmp_dir, self.to_html = client, Path(tmp_dir), to_html
        self.judge, self.threshold, self.repair, self.ink_reread = judge, judge_threshold, repair, ink_reread
        self.begin_document(doc_hash)

    def begin_document(self, doc_hash: str = "") -> None:
        """Forget what the previous Document left here; the worker, the schedule and the options stay."""
        self.doc_hash = doc_hash  # with the page and the Block, what the sampling seeds of the re-read rounds are made of
        self.repetition_cuts = 0
        self.current_page = 0
        self.scale_sources: dict[str, int] = collections.Counter()  # scaled images made, by where they came from
        self.page_source = None  # set per page by the caller: lets round 2 re-render the page (see BlockScaler)
        self.outcomes: dict[tuple, block_reader.BlockResult] = {}
        if self.repair is not None:
            self.repair.reset()

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
                # `block_index` is the Block's index on its page. The pipeline reads a page in one call per pixel
                # budget, so the position within `inputs` repeats from one call to the next; the index does not.
                block_index = getattr(query, "block_index", None)
                requests.append(block_reader.BlockRequest(
                    f"p{self.current_page}/b{n}", inp["image"], label, bbox, str(query),
                    block_id=None if block_index is None else str(block_index),
                    scaler=BlockScaler(inp["image"], bbox, self.page_source, self.scale_sources)))
            settings = block_reader.ReadSettings(
                self.threshold, self._clean, self.repair if self.judge else None,
                self._best_effort_ink_reread if self.ink_reread and not self.judge else None,
                doc_hash=self.doc_hash, page=self.current_page)
            judge = (_DownJudge(judge_down) if judge_down else WorkerJudge(self.client, crops)) if self.judge else None
            results = block_reader.read_blocks(requests, self._reader(kwargs, crops), judge, settings)
        finally:
            crops.cleanup()
        for request, result in zip(requests, results):
            key = (self.current_page, request.bbox)
            if request.bbox is None:
                print(f"warning: page {self.current_page + 1}: a Block came without a box; its Score cannot be "
                      "matched to it in the JSON", file=sys.stderr)
            elif key in self.outcomes:
                print(f"warning: page {self.current_page + 1}: two Blocks have the same box {request.bbox}; only the "
                      "later one's Score reaches the JSON", file=sys.stderr)
            self.outcomes[key] = result
        for result in results:
            yield {"result": result.text}

    uses_worker_reader = False

    def _best_effort_ink_reread(self, request, text):
        """The ink-based re-read is an improvement attempt: when it fails (the worker ran out of memory or timed out,
        a crop cannot be prepared) the Block keeps its first Reading and the page goes on."""
        try:
            return self.ink_reread(request, text)
        except Exception as e:
            print(f"warning: page {self.current_page + 1}: the re-read of block {request.bbox} failed, "
                  f"keeping its first reading: {type(e).__name__}: {e}", file=sys.stderr)
            return None

    def page_summary(self, page: int) -> dict:
        """How many Blocks of a page were judged, how many were re-read, how many are (still) below the threshold,
        how many could not be scored."""
        judged = [o for (p, _), o in self.outcomes.items() if p == page and o.judged]
        return {"judged": len(judged),
                "reread": sum(1 for o in judged if o.readings is not None),
                "below_threshold": sum(1 for o in judged if o.below_threshold),
                "not_scored": sum(1 for o in judged if o.verdict != block_reader.SCORED)}

    def supports(self, step) -> bool:
        """Whether this bridge's Reader can do a round of the re-read schedule."""
        return self._reader({}, None).supports(step)

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
            raw = _MiniCPMRead(self.client, crops, max_new_tokens).read(
                [block_reader.BlockRequest("reread", image, label)], block_reader.SCHEDULE[0], [None])[0][0]
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
