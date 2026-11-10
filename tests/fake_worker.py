"""A stand-in for worker/serve.py that needs no torch: the real request loop around a fake model.

    python fake_worker.py [--mode ready|env-error|die|silent]

The prompt of a read selects the behaviour: "CRASH" kills the process, "HANG" never answers, "FAIL"
raises inside the model, anything else is echoed.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("worker_serve", ROOT / "worker" / "serve.py")
serve_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_mod)


class FakeReader:
    def read(self, image, prompt, max_new_tokens):
        if prompt == "CRASH":
            print("fake worker: about to crash (CUDA error)", file=sys.stderr, flush=True)
            os._exit(3)
        if prompt == "HANG":
            time.sleep(120)
        if prompt == "FAIL":
            raise RuntimeError("CUDA out of memory")
        return [{"text": f"{Path(image).name}|{prompt}|{max_new_tokens}", "tokens": 1, "finish": "stop"}]


def main() -> int:
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "ready"
    out = sys.stdout
    if mode == "die":
        print("fake worker: cannot start", file=sys.stderr)
        return 1
    if mode == "env-error":
        out.write(json.dumps({"event": "error", "what": "worker_env",
                              "message": "run: uv sync --project worker"}) + "\n")
        out.flush()
        return 2
    if mode == "silent":
        time.sleep(120)
        return 0
    out.write(json.dumps({"event": "ready", "reader": "fake", "versions": {"torch": "0.0"}}) + "\n")
    out.flush()
    serve_mod.serve(FakeReader(), sys.stdin, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
