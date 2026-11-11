"""A folder Run on the GPU, with the real models, the default Reader (MiniCPM-OCR-1.0) and the Judge at the
threshold 0.98 (so that most Blocks go through the seeded sampling Rounds): its Documents come out as if each had
been a Run of its own. Needs the models, the worker environment, the Judge's setup and PDF_OCR_SAMPLE_PDF (see conftest.py).

The folder holds two 2-page PDFs cut from the sample (one of them in a subfolder) and a corrupt PDF between them.
Every Run is a CLI process of its own, one after another, so that each starts with a free GPU.
"""
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pytest

import pdf_ocr
from conftest import BASELINE, ROOT, _models_dir

pytestmark = pytest.mark.gpu

JUDGE_THRESHOLD = "0.98"
GOOD = {"a.pdf": (0, 1), "sub/b.pdf": (2, 3)}  # relative path -> the sample's pages (0-based) it is cut from


@dataclass
class Run:
    code: int
    stderr: str
    seconds: float  # wall time of the process


def run_cli(target: Path, out: Path, models: Path, *extra: str) -> Run:
    """One CLI process: a single PDF or a folder, into `out`, with the options these tests share."""
    started = time.time()
    done = subprocess.run(
        [sys.executable, str(ROOT / "pdf_ocr.py"), str(target), "-o", str(out), "--models-dir", str(models),
         "--judge-threshold", JUDGE_THRESHOLD, *extra],
        capture_output=True, text=True, cwd=ROOT)
    return Run(done.returncode, done.stderr, time.time() - started)


@pytest.fixture(scope="module")
def models() -> Path:
    if problem := pdf_ocr.worker_env_problem():
        pytest.skip(problem)
    return _models_dir("PP-DocLayoutV3", "MiniCPM-OCR-1.0", "MiniCPM-OCR-Jev")


@pytest.fixture(scope="module")
def folder(sample_pdf, tmp_path_factory) -> Path:
    """docs/a.pdf, docs/bad.pdf (not a PDF) and docs/sub/b.pdf; the good ones are table pages of the sample."""
    first = int(BASELINE["pages"].split("-")[0]) - 1
    source = pdfium.PdfDocument(str(sample_pdf))
    if len(source) < first + 4:
        pytest.skip(f"{sample_pdf.name} has no pages {first + 1}-{first + 4}")
    docs = tmp_path_factory.mktemp("folder") / "docs"
    (docs / "sub").mkdir(parents=True)
    for name, pages in GOOD.items():
        cut = pdfium.PdfDocument.new()
        cut.import_pages(source, [first + p for p in pages])
        cut.save(str(docs / name))
    (docs / "bad.pdf").write_bytes(b"%PDF-1.7 this is not a PDF")
    return docs


@pytest.fixture(scope="module")
def alone(folder, models, tmp_path_factory) -> dict[str, Path]:
    """Each good Document as a Run of its own (the same file, in place): relative path -> output folder."""
    out = tmp_path_factory.mktemp("alone")
    outputs = {}
    for name in GOOD:
        outputs[name] = out / name.replace("/", "_")
        run = run_cli(folder / name, outputs[name], models)
        assert run.code == 0, run.stderr[-2000:]
    return outputs


@pytest.fixture(scope="module")
def first_run(folder, models, tmp_path_factory) -> tuple[Run, Path]:
    """The folder Run: its result and its output root. Later Runs of the same folder start from this output."""
    out = tmp_path_factory.mktemp("folder_out")
    return run_cli(folder, out, models), out


def output_folder(root: Path, name: str) -> Path:
    """Where a Document at this relative path writes below an output root."""
    return root / Path(name).with_suffix("")


def read_json(root: Path, name: str) -> dict:
    return json.loads((output_folder(root, name) / f"{Path(name).stem}.json").read_text(encoding="utf-8"))


def read_markdown(root: Path, name: str) -> str:
    return (output_folder(root, name) / f"{Path(name).stem}.md").read_text(encoding="utf-8")


def csvs(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted((folder / "csv").glob("*.csv"))}


@pytest.mark.parametrize("name", GOOD)
def test_each_good_document_equals_a_run_of_its_own(first_run, alone, name):
    _, out = first_run
    single = alone[name]
    assert read_markdown(out, name) == (single / f"{Path(name).stem}.md").read_text(encoding="utf-8")
    in_folder = read_json(out, name)
    on_its_own = json.loads((single / f"{Path(name).stem}.json").read_text(encoding="utf-8"))
    in_folder.pop("elapsed_seconds"), on_its_own.pop("elapsed_seconds")
    assert in_folder == on_its_own
    assert in_folder["source"]["file"] == Path(name).name  # the bare file name
    assert csvs(output_folder(out, name)) == csvs(single) and csvs(single), "the pages have no table to compare"
    assert sorted(p.name for p in (output_folder(out, name) / "imgs").glob("*")) == sorted(
        p.name for p in (single / "imgs").glob("*"))


def test_the_corrupt_file_is_reported_and_the_two_good_documents_are_written(first_run):
    run, out = first_run
    lines = run.stderr.splitlines()
    assert any(ln.startswith("found 3 Documents in ") for ln in lines)
    assert "[1/3] a.pdf (2 pages)" in lines
    assert any(ln.startswith("[2/3] bad.pdf FAILED: cannot open ") for ln in lines)
    assert "[3/3] sub/b.pdf (2 pages)" in lines
    summary = lines.index("3 Documents: 2 written, 1 failed")
    assert lines[summary + 1].startswith("  bad.pdf: cannot open ")
    assert sorted(p.relative_to(out).as_posix() for p in out.glob("**/*.json")) == ["a/a.json", "sub/b/b.json"]
    assert not (out / "bad").exists()


def test_a_run_with_a_failed_document_exits_1(first_run):
    assert first_run[0].code == 1


def test_the_models_are_loaded_once_and_a_documents_time_does_not_count_the_load(first_run):
    run, out = first_run
    loads = re.findall(r"^model load: ([0-9.]+)s$", run.stderr, re.M)
    assert len(loads) == 1
    own_times = sum(read_json(out, name)["elapsed_seconds"] for name in GOOD)
    assert own_times + float(loads[0]) <= run.seconds + 0.5  # the load and the Documents' own times don't overlap
