"""Shared fixtures for the slow end-to-end tests.

The sample PDF is gitignored, so it comes from outside:

    PDF_OCR_SAMPLE_PDF="/path/to/台灣人壽解約金 115.09.23.pdf" pytest -m gpu   # the document tests/baseline.json was measured on
    PDF_OCR_PAGES=42-45                                         # pages to run (default: "pages" in tests/baseline.json)
    PDF_OCR_MODELS_DIR=/path/to/models                          # default: ./models
    PDF_OCR_CER_BOUND=0.01                                      # overrides each configuration's bound in tests/baseline.json
    PDF_OCR_READERS=minicpm                                     # readers to run (default: minicpm,paddle)
    PDF_OCR_REREAD_PAGES=4-5                                    # pages of the forced re-read runs (default: 4-5)
    PDF_OCR_WORKER_ENV=/path/to/worker-venv                     # worker environment (default: worker/.venv)

The end-to-end tests run once per measured configuration of tests/baseline.json (a run with no options, and each
Reader with --judge), plus a few per-Reader runs; PDF_OCR_READERS keeps only the configurations and runs of the
Readers it names. A run whose model folders, worker environment or Judge renderer are missing is skipped, by the
CLI's own checks: the worker environment is needed for --reader minicpm and --judge, and `playwright install
chromium`, node and Noto Sans CJK for --judge (see the README). Set PLAYWRIGHT_BROWSERS_PATH if Chromium is not in
the default place.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import pdf_ocr

ROOT = Path(__file__).resolve().parent.parent
BASELINE = json.loads((Path(__file__).parent / "baseline.json").read_text(encoding="utf-8"))
READERS = os.environ.get("PDF_OCR_READERS", "minicpm,paddle").split(",")
REREAD_THRESHOLD = "0.9999"  # so high that nearly every Block is re-read


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: slow end-to-end test that needs the GPU, the models and a sample PDF")


@dataclass(frozen=True)
class Config:
    """A measured configuration: its name and its entry in tests/baseline.json."""
    name: str
    argv: tuple[str, ...]
    record: dict

    @property
    def options(self):
        return pdf_ocr.parse_args(["x.pdf", *self.argv])


CONFIGS = [c for name, rec in BASELINE["configurations"].items()
           if (c := Config(name, tuple(rec["argv"]), rec)).options.reader in READERS]
JUDGED = [c for c in CONFIGS if c.options.judge]


def models_root() -> Path:
    return Path(os.environ.get("PDF_OCR_MODELS_DIR", ROOT / "models"))


def _models_dir(*folders: str) -> Path:
    d = models_root()
    for name in folders:
        if not (d / name).is_dir():
            pytest.skip(f"model folder missing: {d / name}")
    if importlib.util.find_spec("paddle") is None:
        pytest.skip("paddle is not installed in this environment (run pytest from the project venv)")
    return d


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
def layout_models_dir() -> Path:
    """For tests that need only the Layout model (the Reader is faked)."""
    return _models_dir("PP-DocLayoutV3")


@pytest.fixture(scope="session")
def pages() -> str:
    return os.environ.get("PDF_OCR_PAGES", BASELINE["pages"])


@pytest.fixture(scope="session")
def reread_pages() -> str:
    return os.environ.get("PDF_OCR_REREAD_PAGES", "4-5")


class Cli:
    """Runs pdf_ocr.py on the sample in a subprocess, once per distinct meaning: a run is identified by its parsed
    options, so `--reader paddle` and no options are one run. Each run gets a models folder holding symlinks to
    exactly required_model_dirs(options), and, unless needs_worker(options), a PDF_OCR_WORKER_ENV that does not
    exist, so every run also proves that the CLI asks only for what its options use."""

    def __init__(self, pdf: Path, pages: str, tmp: pytest.TempPathFactory):
        self.pdf, self.pages, self.tmp = pdf, pages, tmp
        self.done: dict[tuple[str, int], Path] = {}

    def runs(self, *argv: str, n: int = 1, pages: str | None = None) -> list[Path]:
        pages = pages or self.pages
        options = pdf_ocr.parse_args([str(self.pdf), "--models-dir", str(models_root()), "--pages", pages, *argv])
        if msg := pdf_ocr.missing_models_message(options):
            pytest.skip(msg)
        if importlib.util.find_spec("paddle") is None:
            pytest.skip("paddle is not installed in this environment (run pytest from the project venv)")
        if pdf_ocr.needs_worker(options) and (problem := pdf_ocr.worker_env_problem()):
            pytest.skip(problem)
        if options.judge and (problems := pdf_ocr.judge_setup_problems()):
            pytest.skip("; ".join(problems))
        key = repr(sorted(vars(options).items()))
        for i in range(n):
            if (key, i) not in self.done:
                self.done[key, i] = self._run(options, argv, pages)
        return [self.done[key, i] for i in range(n)]

    def run(self, *argv: str, pages: str | None = None) -> Path:
        return self.runs(*argv, pages=pages)[0]

    def _run(self, options, argv: tuple[str, ...], pages: str) -> Path:
        label = " ".join(argv) or "(no options)"
        out = self.tmp.mktemp("run")
        models = self.tmp.mktemp("models")
        for name in pdf_ocr.required_model_dirs(options):
            (models / name).symlink_to(models_root() / name, target_is_directory=True)
        env = dict(os.environ)
        if not pdf_ocr.needs_worker(options):
            env["PDF_OCR_WORKER_ENV"] = str(self.tmp.mktemp("no_worker") / "missing")
        done = subprocess.run(
            [sys.executable, str(ROOT / "pdf_ocr.py"), str(self.pdf), "-o", str(out), "--models-dir", str(models),
             "--pages", pages, *argv],
            capture_output=True, text=True, cwd=ROOT, env=env,
        )
        (out / "stderr.txt").write_text(done.stderr, encoding="utf-8")
        assert done.returncode == 0, f"CLI run ({label}) failed:\n{done.stderr[-2000:]}"
        return out


@pytest.fixture(scope="session")
def cli(sample_pdf, pages, tmp_path_factory) -> Cli:
    return Cli(sample_pdf, pages, tmp_path_factory)


@pytest.fixture(scope="session", params=CONFIGS, ids=lambda c: c.name)
def config(request) -> Config:
    return request.param


@pytest.fixture(scope="session", params=JUDGED, ids=lambda c: c.name)
def judged_config(request) -> Config:
    return request.param


@pytest.fixture(scope="session", params=READERS)
def reader(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def config_runs(cli, config) -> list[Path]:
    """Output folders of two runs of a measured configuration, with stderr.txt next to the outputs."""
    return cli.runs(*config.argv, n=2)


@pytest.fixture(scope="session")
def judged_runs(cli, judged_config) -> list[Path]:
    """The runs of a configuration with the Judge on (the same runs as config_runs)."""
    return cli.runs(*judged_config.argv, n=2)


@pytest.fixture(scope="session")
def judge_off_run(cli, reader) -> Path:
    """One run of a Reader without the Judge (for PaddleOCR-VL, the default run)."""
    return cli.run("--reader", reader)


@pytest.fixture(scope="session")
def judge_without_rereads_run(cli, reader) -> Path:
    """One run with the Judge on and a threshold of 0, so that nothing is ever re-read."""
    return cli.run("--reader", reader, "--judge", "--judge-threshold", "0")


@pytest.fixture(scope="session")
def reread_runs(cli, reader, reread_pages) -> list[Path]:
    """Output folders of two identical runs with --judge-threshold 0.9999, which forces the re-reads."""
    return cli.runs("--reader", reader, "--judge", "--judge-threshold", REREAD_THRESHOLD, n=2, pages=reread_pages)


@pytest.fixture(scope="session")
def cer_bound(config, sample_pdf, pages) -> float:
    """The recorded bound for this configuration. It was measured on one document and one page range, so it is only
    applied to those; with another document or other pages set PDF_OCR_CER_BOUND (the check is skipped otherwise)."""
    if "PDF_OCR_CER_BOUND" in os.environ:
        return float(os.environ["PDF_OCR_CER_BOUND"])
    digest = hashlib.sha256(sample_pdf.read_bytes()).hexdigest()
    if digest != BASELINE["document"]["sha256"]:
        pytest.skip(f"tests/baseline.json was measured on {BASELINE['document']['name']} "
                    f"(sha256 {BASELINE['document']['sha256'][:12]}...), not on {sample_pdf.name}; "
                    "set PDF_OCR_CER_BOUND to check the error rate against your own bound")
    if pages != BASELINE["pages"]:
        pytest.skip(f"tests/baseline.json was measured on pages {BASELINE['pages']}, not {pages}; "
                    "set PDF_OCR_CER_BOUND to check the error rate against your own bound")
    return float(config.record["cer_bound"])
