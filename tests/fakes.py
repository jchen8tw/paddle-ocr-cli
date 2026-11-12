"""Test doubles shared by the bridge and the Run tests: the worker, the PaddleX pipe and PaddleOCR-VL's own model, and
`engine_for`, which builds an Engine from an option list out of them, and `SETUPS`, the four Reader x Judge setups."""
import argparse
import collections
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

import pdf_ocr
from reader_bridge import BlockQuery, MiniCPMBridge, PaddleBridge
from script_repair import ScriptRepair, StickyScriptRepair
from worker_client import WorkerError


class FakeClient:
    """Stands in for WorkerClient: records what the bridge asks the worker for."""

    def __init__(self, texts=None, fail_on=None, scores=None, fail_score=None, default_score=0.99):
        # texts: prompt -> the text it reads; a list answers the n-th read of that prompt with its n-th item (the last repeats)
        self.calls, self.texts, self.fail_on = [], texts or {}, fail_on
        self.scores, self.fail_score, self.default_score = scores or {}, fail_score, default_score  # scores: text -> (verdict, score)
        self.score_calls, self.events = [], []
        self.decodes, self.images = [], []
        self.started = 0  # calls of ensure_started: one per page the bridge reads
        self.crash_on_page = None  # the page (from 1, over the whole Run) that loses its worker on its first read
        self.worker_alive, self.restarts = True, 0

    def ensure_started(self):
        self.started += 1
        if not self.worker_alive:  # a worker that crashed on an earlier page is started again
            self.worker_alive, self.restarts = True, self.restarts + 1

    def read(self, image, prompt, max_new_tokens, *, decode=None, timeout=None):
        assert Path(image).is_file(), "the crop must exist on disk while the worker reads it"
        if self.started == self.crash_on_page and self.worker_alive:
            self.worker_alive = False
            raise WorkerError("the worker stopped (exited with code 3)")
        if not self.worker_alive:
            raise WorkerError("the worker is not running")
        self.calls.append((prompt, max_new_tokens))
        self.decodes.append(decode)
        self.images.append(Path(image).name)
        self.events.append("read")
        if self.fail_on == prompt:
            raise WorkerError("the worker stopped (exited with code 3)")
        text = self.texts.get(prompt, f"text for {prompt}")
        if isinstance(text, list):
            text = text[min(sum(1 for p, _ in self.calls if p == prompt) - 1, len(text) - 1)]
        if decode is None:
            return [{"text": text, "tokens": 5, "finish": "stop"}]
        return [{"text": f"{text} sample {decode['seed']}.{k}", "tokens": 5, "finish": "stop"} for k in range(decode["n"])]

    def score(self, items, *, timeout=None):
        for i in items:
            assert Path(i["image"]).is_file(), "the crop must exist on disk while the worker scores"
        self.score_calls.append(items)
        self.events.append("score")
        if self.fail_score:
            raise self.fail_score
        out = []
        for i in items:
            verdict, score = self.scores.get(i["text"], ("scored", self.default_score))
            out.append({"key": i["key"], "verdict": verdict, "score": score, "detail": None if verdict == "scored" else "why"})
        return out

    def park(self):
        self.events.append("park")


BLOCKS = [("text", (0, 0, 16, 8)), ("text", (20, 20, 36, 28))]


class FakeBlock:
    def __init__(self, label, bbox, content=""):
        self.label, self.bbox, self.content = label, bbox, content


class FakeResult:
    def __init__(self, blocks):
        self.blocks = blocks

    def __getitem__(self, key):
        assert key == "parsing_res_list"
        return self.blocks

    @property
    def json(self):
        return {"res": {"width": 200, "height": 300, "layout_det_res": {"boxes": []}, "parsing_res_list": [
            {"block_id": i, "block_order": i, "block_label": b.label, "block_content": b.content,
             "block_bbox": list(b.bbox), "block_polygon_points": None} for i, b in enumerate(self.blocks)]}}

    def _to_markdown(self, pretty, show_formula_number):
        return {"markdown_texts": "\n\n".join(b.content for b in self.blocks)}


class FakePipe:
    """PaddleOCRVL with the Reader hook installed: finds the same Blocks on every page (`blocks`, as (label, box) in
    render pixels) and has the bridge read them."""

    def __init__(self, bridge):
        self.bridge, self.blocks = bridge, list(BLOCKS)

    def predict(self, img, max_new_tokens, use_queues):
        blocks = [FakeBlock(label, bbox) for label, bbox in self.blocks]
        items = []
        for b in blocks:
            q = BlockQuery("OCR:")
            q.label, q.bbox = b.label, b.bbox
            x1, y1, x2, y2 = b.bbox
            items.append({"image": img[y1:y2, x1:x2], "query": q})
        for b, r in zip(blocks, self.bridge.predict(items, max_new_tokens=max_new_tokens)):
            b.content = r["result"]
        return [FakeResult(blocks)]

    def restructure_pages(self, results, merge_tables, relevel_titles, concatenate_pages):
        return results


class FakeVLPipe:
    """PaddleOCRVL without a Reader hook, as the default setup runs it: finds the same Blocks on every page (`blocks`)
    and reads each as `reads`. Called the way the ink-based Re-read calls it (layout detection off, the "ocr" prompt),
    it reads the region it is given as `rereads`. `read_fails` ({the n-th read of a page, from 1: the exception it raises})
    and `reread_fails` (the exception every Re-read raises) make it fail."""

    def __init__(self):
        self.blocks, self.reads, self.rereads = list(BLOCKS), "text for OCR:", ""
        self.read_fails, self.reread_fails, self.pages_read = {}, None, 0

    def predict(self, img, *, max_new_tokens, use_queues, use_layout_detection=True, prompt_label=None):
        if not use_layout_detection:
            assert prompt_label == "ocr", "the Re-read asks for the plain text prompt"
            if self.reread_fails is not None:
                raise self.reread_fails
            return [FakeResult([FakeBlock("text", (0, 0, 0, 0), self.rereads)])]
        self.pages_read += 1
        if self.pages_read in self.read_fails:
            raise self.read_fails[self.pages_read]
        blocks = [FakeBlock(label, bbox) for label, bbox in self.blocks]
        for b in blocks:
            b.content = self.reads
        return [FakeResult(blocks)]

    def restructure_pages(self, results, merge_tables, relevel_titles, concatenate_pages):
        return results


class FakePaddle:
    """PaddleOCR-VL's own model, as the Judge's bridge calls it: answers each query from `texts` (query -> text; a list
    answers the n-th read of that query with its n-th item, the last repeating)."""

    def __init__(self, texts):
        self.texts, self.calls = texts, []
        self.events = None
        self.asked = collections.Counter()

    def predict(self, inputs, **kw):
        inputs = list(inputs)
        self.calls.append(([(str(i["query"]), i["image"].shape) for i in inputs], kw))
        if self.events is not None:
            self.events.append("paddle")
        for i in inputs:
            query = str(i["query"])
            text = self.texts[query]
            if isinstance(text, list):
                text = text[min(self.asked[query], len(text) - 1)]
                self.asked[query] += 1
            yield {"result": text}

    def close(self):
        pass


@dataclass(frozen=True)
class Setup:
    """One of the four Reader x Judge setups: the options that choose it and what a Run of it writes and starts."""

    id: str
    options: tuple[str, ...]
    reader: str          # the Reader folder in the JSON
    judge: str | None    # the Judge folder in the JSON; None without --judge
    worker: bool         # whether a Run of it starts the worker


SETUPS = [
    Setup("paddleocr-vl", (), "PaddleOCR-VL-1.6", None, False),
    Setup("paddleocr-vl+judge", ("--judge",), "PaddleOCR-VL-1.6", "MiniCPM-OCR-Jev", True),
    Setup("minicpm", ("--reader", "minicpm"), "MiniCPM-OCR-1.0", None, True),
    Setup("minicpm+judge", ("--reader", "minicpm", "--judge"), "MiniCPM-OCR-1.0", "MiniCPM-OCR-Jev", True),
]


@dataclass
class Fakes:
    """What a test steers in an Engine that `engine_for` built: only what is outside the Engine."""

    pipe: object                  # the PaddleX pipe: FakePipe (the Reader hook installed) or FakeVLPipe (none)
    worker: FakeClient | None     # None when the options start no worker (PaddleOCR-VL alone)
    paddle: FakePaddle | None     # PaddleOCR-VL's own model under the Judge's bridge (PaddleOCR-VL with the Judge)

    def read_as(self, *texts: str) -> None:
        """Every text Block now reads as `texts[0]`; with more texts, the n-th Block read reads as the n-th (the last
        repeats). The same for whichever of the worker and PaddleOCR-VL reads; the pipe alone reads one text for every
        Block, and a ValueError says so when given more. Several texts count from the Engine's first read, so give them
        before the first page is read."""
        if self.paddle is not None:
            self.paddle.texts, self.paddle.asked = {"OCR:": list(texts)}, collections.Counter()
        elif self.worker is not None:
            self.worker.texts = {"Text Recognition:": list(texts)}
        elif len(texts) == 1:
            self.pipe.reads = texts[0]
        else:
            raise ValueError(f"the pipe alone reads one text for every Block, but read_as was given {len(texts)}")


def engine_for(options, tmp_path, *, worker=None):
    """A loaded Engine (what `load_engine` returns) for an option list (or the parsed options of a Run), built the way
    a Run builds it: `load_engine` and `attach_worker` in pdf_ocr.py, with the PaddleX pipe and the worker faked.
    Returns the Engine and the `Fakes` that steer it. `worker` is the FakeClient to use, if the test made its own.

    It mirrors `attach_worker` and `load_engine` by hand, on purpose (the real functions need a worker), so a change
    to the Reader x Judge wiring there must be mirrored here until the Engine spec (#31) replaces this function.
    It differs from a Run in three places: the bridge has no `after_read` (nothing to free in paddle's GPU cache),
    `to_html` converts nothing, and the Engine has the device "cpu" and no `load_started`."""
    args = options if isinstance(options, argparse.Namespace) else pdf_ocr.parse_args(["in.pdf", *options])
    ink = pdf_ocr.InkReread()
    if not pdf_ocr.needs_worker(args):  # PaddleOCR-VL alone: no worker, no bridge, no Reader hook
        pipe = FakeVLPipe()
        return pdf_ocr.Engine(pipe, "cpu", None, ink, None), Fakes(pipe, None, None)
    client = worker or FakeClient(default_score=0.5)  # every Reading scores below a high threshold: all re-read Rounds run
    crops = tmp_path / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    repair = None
    if args.judge and args.script == "auto":
        try:
            repair = StickyScriptRepair(ScriptRepair())
        except ImportError:
            pass  # as in attach_worker: then the Judge judges the text as read
    common = dict(to_html=lambda s: "", judge=args.judge, repair=repair,
                  **({"judge_threshold": args.judge_threshold} if args.judge else {}))
    paddle = None
    if args.reader == "paddle":
        paddle = FakePaddle({"OCR:": "text for OCR:"})
        paddle.events = client.events
        bridge = PaddleBridge(paddle, client, crops, before_read=client.park if args.judge else None, **common)
    else:
        bridge = MiniCPMBridge(client, crops, **common)
        if not args.judge and not args.no_recover:
            bridge.ink_reread = ink.reread_with(bridge, args)
    ready = {"event": "ready", "reader": pdf_ocr.MINICPM_MODEL_DIR if args.reader == "minicpm" else None,
             "judge": pdf_ocr.JUDGE_MODEL_DIR if args.judge else None, "versions": {"torch": "0.0"}}
    pipe = FakePipe(bridge)
    return pdf_ocr.Engine(pipe, "cpu", bridge, ink, ready), Fakes(pipe, client, paddle)


# A page whose ink (see `inked_pdf`) fills INK_BOX of a 216 dpi render: the box of a Block a test puts there.
INK_BOX = (30, 60, 330, 150)


def blank_pdf(path, pages=1, height=300):
    """A PDF of `pages` empty pages, 200 points wide."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(200, height)
    pdf.save(str(path))
    return path


def inked_pdf(path):
    """A one-page PDF, 200 x 300 points, with a line of ink in INK_BOX (at 216 dpi: 240 x 24 pixels of black, about ten
    glyph widths) and nothing else."""
    from PIL import Image, ImageDraw  # a transitive dependency (paddleocr): only the tests that need ink import it

    image = Image.new("RGB", (200, 300), "white")
    ImageDraw.Draw(image).rectangle([20, 30, 99, 37], fill="black")
    image.save(path, "PDF", resolution=72.0)
    return path
