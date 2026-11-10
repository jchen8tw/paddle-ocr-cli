"""Shared fixtures for the slow end-to-end tests.

The sample PDF is gitignored, so it comes from outside:

    PDF_OCR_SAMPLE_PDF=/path/to/policy.pdf pytest -m gpu        # 21-page Traditional Chinese policy
    PDF_OCR_PAGES=1-3                                           # pages to run (default 1-3)
    PDF_OCR_MODELS_DIR=/path/to/models                          # default: ./models
    PDF_OCR_CER_BOUND=0.01                                      # overrides tests/baseline.json
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINE = json.loads((Path(__file__).parent / "baseline.json").read_text(encoding="utf-8"))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: slow end-to-end test that needs the GPU, the models and a sample PDF")


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    path = os.environ.get("PDF_OCR_SAMPLE_PDF")
    if not path:
        pytest.skip("set PDF_OCR_SAMPLE_PDF to a PDF with a real text layer to run the end-to-end tests")
    pdf = Path(path)
    if not pdf.is_file():
        pytest.skip(f"PDF_OCR_SAMPLE_PDF does not exist: {pdf}")
    return pdf


@pytest.fixture(scope="session")
def models_dir() -> Path:
    d = Path(os.environ.get("PDF_OCR_MODELS_DIR", ROOT / "models"))
    for name in ("PP-DocLayoutV3", "PaddleOCR-VL-1.6"):
        if not (d / name).is_dir():
            pytest.skip(f"model folder missing: {d / name}")
    if importlib.util.find_spec("paddle") is None:
        pytest.skip("paddle is not installed in this environment (run pytest from the project venv)")
    return d


@pytest.fixture(scope="session")
def pages() -> str:
    return os.environ.get("PDF_OCR_PAGES", "1-3")


@pytest.fixture(scope="session")
def cer_bound() -> float:
    if "PDF_OCR_CER_BOUND" in os.environ:
        return float(os.environ["PDF_OCR_CER_BOUND"])
    return float(BASELINE["cer_bound"])


def run_cli(pdf: Path, out_dir: Path, models_dir: Path, pages: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "pdf_ocr.py"), str(pdf), "-o", str(out_dir),
         "--models-dir", str(models_dir), "--pages", pages],
        capture_output=True, text=True, cwd=ROOT,
    )


@pytest.fixture(scope="session")
def cli_runs(sample_pdf, models_dir, pages, tmp_path_factory) -> list[Path]:
    """Output folders of two identical runs of the CLI on the sample (shared by all e2e tests)."""
    outs = []
    for i in (1, 2):
        out = tmp_path_factory.mktemp(f"run{i}")
        done = run_cli(sample_pdf, out, models_dir, pages)
        assert done.returncode == 0, f"CLI run {i} failed:\n{done.stderr[-2000:]}"
        outs.append(out)
    return outs
