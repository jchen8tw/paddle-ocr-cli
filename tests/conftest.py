"""Shared fixtures for the slow end-to-end tests.

The sample PDF is gitignored, so it comes from outside:

    PDF_OCR_SAMPLE_PDF=/path/to/policy.pdf pytest -m gpu        # 21-page Traditional Chinese policy
    PDF_OCR_PAGES=1-3                                           # pages to run (default: "pages" in tests/baseline.json)
    PDF_OCR_MODELS_DIR=/path/to/models                          # default: ./models
    PDF_OCR_CER_BOUND=0.01                                      # overrides tests/baseline.json
    PDF_OCR_READERS=minicpm                                     # readers to run (default: minicpm,paddle)
    PDF_OCR_WORKER_ENV=/path/to/worker-venv                     # worker environment (default: worker/.venv)

The end-to-end tests run once per Reader; a Reader whose model folder or environment is missing is skipped.
The Judge is on by default, so the worker environment, `playwright install chromium`, node and Noto Sans CJK are
needed for every run that does not pass --no-judge (see the README). Set PLAYWRIGHT_BROWSERS_PATH if Chromium is
not in the default place.
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


READERS = os.environ.get("PDF_OCR_READERS", "minicpm,paddle").split(",")
READER_DIRS = {"minicpm": "MiniCPM-OCR-1.0", "paddle": "PaddleOCR-VL-1.6"}


def _models_dir(*folders: str) -> Path:
    d = Path(os.environ.get("PDF_OCR_MODELS_DIR", ROOT / "models"))
    for name in folders:
        if not (d / name).is_dir():
            pytest.skip(f"model folder missing: {d / name}")
    if importlib.util.find_spec("paddle") is None:
        pytest.skip("paddle is not installed in this environment (run pytest from the project venv)")
    return d


@pytest.fixture(scope="session")
def layout_models_dir() -> Path:
    """For tests that need only the Layout model (the Reader is faked)."""
    return _models_dir("PP-DocLayoutV3")


@pytest.fixture(scope="session", params=READERS)
def reader(request) -> str:
    if request.param == "minicpm":
        import pdf_ocr

        if problem := pdf_ocr.worker_env_problem():
            pytest.skip(problem)
    return request.param


@pytest.fixture(scope="session")
def models_dir(reader) -> Path:
    return _models_dir("PP-DocLayoutV3", READER_DIRS[reader], "MiniCPM-OCR-Jev")


@pytest.fixture(scope="session")
def pages() -> str:
    return os.environ.get("PDF_OCR_PAGES", BASELINE["pages"])


@pytest.fixture(scope="session")
def cer_bound() -> float:
    if "PDF_OCR_CER_BOUND" in os.environ:
        return float(os.environ["PDF_OCR_CER_BOUND"])
    return float(BASELINE["cer_bound"])


def run_cli(pdf: Path, out_dir: Path, models_dir: Path, pages: str, reader: str = "minicpm",
            *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "pdf_ocr.py"), str(pdf), "-o", str(out_dir),
         "--models-dir", str(models_dir), "--pages", pages, "--reader", reader, *extra],
        capture_output=True, text=True, cwd=ROOT,
    )


def run_and_keep_stderr(pdf, out, models_dir, pages, reader, *extra) -> Path:
    done = run_cli(pdf, out, models_dir, pages, reader, *extra)
    (out / "stderr.txt").write_text(done.stderr, encoding="utf-8")
    assert done.returncode == 0, f"CLI run ({reader} {' '.join(extra)}) failed:\n{done.stderr[-2000:]}"
    return out


@pytest.fixture(scope="session")
def cli_runs(sample_pdf, models_dir, pages, reader, tmp_path_factory) -> list[Path]:
    """Output folders of two identical runs of the CLI on the sample, with the Judge on (its default); shared by
    all e2e tests of a Reader. The run's stderr is kept as stderr.txt next to the outputs."""
    return [run_and_keep_stderr(sample_pdf, tmp_path_factory.mktemp(f"{reader}_run{i}"), models_dir, pages, reader)
            for i in (1, 2)]


@pytest.fixture(scope="session")
def cli_run_no_judge(sample_pdf, models_dir, pages, reader, tmp_path_factory) -> Path:
    """One run with --no-judge."""
    return run_and_keep_stderr(sample_pdf, tmp_path_factory.mktemp(f"{reader}_nojudge"), models_dir, pages, reader,
                               "--no-judge")
