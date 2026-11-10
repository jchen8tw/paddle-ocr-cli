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
import types
from pathlib import Path

import minicpm_reader

COLLECT = "_paddleocr_vl_collect_page_vlm_entries_core"
COLLECT_PARAMS = ["page_idx", "blocks_for_img", "imgs_in_doc", "cfg"]
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
        if names != COLLECT_PARAMS:
            problems.append(f"`{COLLECT}` has signature {names}, expected {COLLECT_PARAMS}")
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


class MiniCPMBridge:
    """The pipeline's `vl_rec_model`, backed by the worker.

    Each Reading is cleaned here (repetition cut, OTSL to HTML, table line breaks) so the pipeline's own
    post-processing, which repeats the cut and the conversion, finds nothing left to do.
    """

    batch_sampler = types.SimpleNamespace(batch_size=1)  # only read when the pipeline uses queues

    def __init__(self, client, tmp_dir: Path, *, to_html):
        self.client, self.tmp_dir, self.to_html = client, Path(tmp_dir), to_html
        self.repetition_cuts = 0
        self._counter = itertools.count()

    def predict(self, inputs, **kwargs):
        """The pipeline's min_pixels/max_pixels budgets belong to PaddleOCR-VL and are ignored; the
        table token limit is the pipeline's `max_new_tokens` (= --max-new-tokens)."""
        import cv2

        self.client.ensure_started()  # a worker that crashed on an earlier page is started again
        table_limit = int(kwargs.get("max_new_tokens") or 8192)
        for inp in inputs:
            query = inp["query"]
            label = getattr(query, "label", None) or _LABEL_BY_QUERY.get(str(query), "text")
            prompt, limit = minicpm_reader.prompt_and_limit(label, table_limit=table_limit)
            path = self.tmp_dir / f"crop{next(self._counter):06d}.png"
            try:
                if not cv2.imwrite(str(path), inp["image"]):  # BGR array -> a normal RGB PNG
                    raise OSError(f"cannot write {path}")
                readings = self.client.read(path, prompt, limit)
            finally:
                path.unlink(missing_ok=True)
            text, cut = minicpm_reader.clean_reading(readings[0]["text"], label, to_html=self.to_html)
            self.repetition_cuts += cut
            yield {"result": text}

    def close(self):  # the pipeline calls this when it is closed; the worker is stopped by its owner
        pass
