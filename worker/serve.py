#!/usr/bin/env python3
"""The torch worker of pdf_ocr.py: holds MiniCPM-OCR-1.0 and answers requests from the main process.

Started by the main command (never by hand) as

    uv run --project worker --no-sync python worker/serve.py --models-dir models

Protocol: JSON Lines, one request in flight at a time. The worker writes ONLY protocol lines to the
real stdout (everything else, including library noise, is redirected to stderr):

    -> {"event": "ready", "reader": "MiniCPM-OCR-1.0", "versions": {...}}     once, after loading
    -> {"event": "error", "what": "worker_env"|"model", "message": "..."}     then exit code 2
    <- {"id": 7, "op": "read", "image": "/tmp/run/p001_b3.png", "prompt": "Text Recognition:", "max_new_tokens": 512}
    -> {"id": 7, "ok": true, "readings": [{"text": "...", "tokens": 13, "finish": "stop"|"length"}]}
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


def _reply(out, obj: dict) -> None:
    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    out.flush()


def serve(reader, stdin, out) -> None:
    """Answer requests from `stdin` on `out` until `quit` or end of input. `reader.read(image, prompt,
    max_new_tokens)` returns a list of {"text", "tokens", "finish"}."""
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
                readings = reader.read(req["image"], req["prompt"], int(req["max_new_tokens"]))
                _reply(out, {"id": req_id, "ok": True, "readings": readings})
            else:
                _reply(out, {"id": req_id, "ok": False, "error": f"unknown op {op!r}"})
        except Exception as e:  # a bad request or a failed read must not take the worker down
            _reply(out, {"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="torch worker of pdf_ocr.py (started by the main command)")
    ap.add_argument("--models-dir", type=Path, required=True)
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

    model_dir = args.models_dir / READER_DIR
    if not model_dir.is_dir():
        return fail("model", f"MiniCPM-OCR-1.0 not found at {model_dir}")
    try:
        from minicpm_model import MiniCPMReader  # worker/minicpm_model.py, next to this script

        reader = MiniCPMReader(model_dir)
    except Exception as e:
        return fail("model", f"cannot load MiniCPM-OCR-1.0 from {model_dir}: {type(e).__name__}: {e}")

    _reply(proto, {"event": "ready", "reader": READER_DIR,
                   "versions": {"torch": torch.__version__, "transformers": transformers.__version__}})
    serve(reader, sys.stdin, proto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
