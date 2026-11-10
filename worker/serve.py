#!/usr/bin/env python3
"""The torch worker of pdf_ocr.py: holds MiniCPM-OCR-1.0 (Reader) and MiniCPM-OCR Jev (Judge).

Started by the main command (never by hand) as

    uv run --project worker --no-sync python worker/serve.py --models-dir models [--no-reader] [--no-judge]

Protocol: JSON Lines, one request in flight at a time. The worker writes ONLY protocol lines to the
real stdout (everything else, including library noise, is redirected to stderr):

    -> {"event": "ready", "reader": "MiniCPM-OCR-1.0"|null, "judge": "MiniCPM-OCR-Jev"|null, "versions": {...}}
    -> {"event": "error", "what": "worker_env"|"model"|"setup", "message": "..."}  then exit code 2
    <- {"id": 7, "op": "read", "image": "/tmp/run/crop.png", "prompt": "Text Recognition:", "max_new_tokens": 512}
    -> {"id": 7, "ok": true, "readings": [{"text": "...", "tokens": 13, "finish": "stop"|"length"}]}
    <- {"id": 7, "op": "read", ..., "decode": {"mode": "sample", "temperature": 0.3, "top_k": 20, "top_p": 0.8,
                                              "n": 4, "seed": 1234567}}   n Readings; the same seed gives the same texts
    <- {"id": 8, "op": "score", "items": [{"key": "p1/b3", "image": "/tmp/run/crop.png", "text": "...",
                                           "category": "text"|"formula"|"table"}]}
    -> {"id": 8, "ok": true, "results": [{"key": "p1/b3", "verdict": "scored", "score": 0.93, "detail": null}]}
    <- {"id": 9, "op": "park"}                 move the Judge's weights off the GPU (a Reader that needs the room)
    -> {"id": 7, "ok": false, "error": "..."}                                  the worker keeps serving
    <- {"op": "quit"}                                                           or end of input
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

READER_DIR = "MiniCPM-OCR-1.0"
JUDGE_DIR = "MiniCPM-OCR-Jev"


def _reply(out, obj: dict) -> None:
    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    out.flush()


def serve(reader, stdin, out, judge=None) -> None:
    """Answer requests from `stdin` on `out` until `quit` or end of input. `reader.read(image, prompt,
    max_new_tokens)` returns a list of {"text", "tokens", "finish"}; `judge.score_many(items)` returns one
    result per item; `judge.park()` / `judge.unpark()` move its weights off / onto the GPU."""
    for line in stdin:
        if not line.strip():
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id") if isinstance(req, dict) else None
            op = req.get("op") if isinstance(req, dict) else None
            if op == "quit":
                return
            if op == "read":
                if reader is None:
                    raise RuntimeError("this worker was started without the Reader")
                extra = {"decode": req["decode"]} if req.get("decode") else {}  # greedy when absent
                readings = reader.read(req["image"], req["prompt"], int(req["max_new_tokens"]), **extra)
                _reply(out, {"id": req_id, "ok": True, "readings": readings})
            elif op == "score":
                if judge is None:
                    raise RuntimeError("this worker was started without the Judge")
                judge.unpark()
                _reply(out, {"id": req_id, "ok": True, "results": judge.score_many(req["items"])})
            elif op == "park":
                if judge is not None:
                    judge.park()
                _reply(out, {"id": req_id, "ok": True})
            else:
                _reply(out, {"id": req_id, "ok": False, "error": f"unknown op {op!r}"})
        except Exception as e:  # a bad request or a failed read must not take the worker down
            _reply(out, {"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="torch worker of pdf_ocr.py (started by the main command)")
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--no-reader", action="store_true", help="do not load MiniCPM-OCR-1.0 (--reader paddle)")
    ap.add_argument("--no-judge", action="store_true", help="do not load Jev nor check the renderer's setup")
    args = ap.parse_args(argv)

    # Libraries print to stdout. Keep a private copy of the real stdout for the protocol.
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.stdin.reconfigure(encoding="utf-8")

    def fail(what: str, message: str) -> int:
        _reply(proto, {"event": "error", "what": what, "message": message})
        return 2

    try:
        import torch
        import transformers
    except ImportError as e:
        return fail("worker_env", f"the worker environment is missing or incomplete ({e}); "
                                  "create it with: uv sync --project worker")

    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    judge_name = reader_name = None

    if not args.no_judge:
        # Fail on a missing node / Noto / Droid / Chromium before spending time on model loading.
        import render_env

        problems = render_env.setup_problems()
        if problems:
            return fail("setup", "the Judge's renderer is not set up:\n  - " + "\n  - ".join(problems))
        render_env.prepare_environment()  # the private fontconfig, before the first render
        if problem := render_env.smoke_render_problem():  # Chromium, Playwright, node and the fonts, for real
            return fail("setup", "the Judge's renderer is not set up:\n  - " + problem)
        import importlib.metadata as md

        versions.update(playwright=md.version("playwright"), pillow=md.version("pillow"))
        model_dir = args.models_dir / JUDGE_DIR
        if not model_dir.is_dir():
            return fail("model", f"MiniCPM-OCR Jev not found at {model_dir}")

    reader = judge = None
    if not args.no_reader:
        model_dir = args.models_dir / READER_DIR
        if not model_dir.is_dir():
            return fail("model", f"MiniCPM-OCR-1.0 not found at {model_dir}")
        try:
            from minicpm_model import MiniCPMReader  # worker/minicpm_model.py, next to this script

            reader, reader_name = MiniCPMReader(model_dir), READER_DIR
        except Exception as e:
            return fail("model", f"cannot load MiniCPM-OCR-1.0 from {model_dir}: {type(e).__name__}: {e}")
    if not args.no_judge:
        try:
            import render_env
            from jev_model import JevJudge
            from judging import RenderedJudge

            model = JevJudge(args.models_dir / JUDGE_DIR)
            judge = RenderedJudge(model, render_env.render_with_official_renderer, on_forward_error=lambda e: model.free_cache())
            judge_name = JUDGE_DIR
        except Exception as e:
            return fail("model", f"cannot load MiniCPM-OCR Jev from {args.models_dir / JUDGE_DIR}: "
                                 f"{type(e).__name__}: {e}")

    _reply(proto, {"event": "ready", "reader": reader_name, "judge": judge_name, "versions": versions})
    serve(reader, sys.stdin, proto, judge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
