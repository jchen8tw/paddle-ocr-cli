"""The order of a single-PDF Run before any model loads: the checks and the opening of the Document fail fast, in
this order, with nothing loaded (no GPU, no models: a model folder is an empty directory here)."""
import pytest

import pdf_ocr
from fakes import blank_pdf


def models_dir(tmp_path):
    d = tmp_path / "models"
    for name in ("PP-DocLayoutV3", "PaddleOCR-VL-1.6"):
        (d / name).mkdir(parents=True)
    return d


def run(argv, capsys):
    with pytest.raises(SystemExit) as e:
        pdf_ocr.main(argv)
    return e.value.code, capsys.readouterr().err


def test_a_missing_pdf_is_the_first_error(tmp_path, capsys):
    code, err = run([str(tmp_path / "gone.pdf"), "--models-dir", str(tmp_path / "none")], capsys)
    assert code.startswith("error: no such file") and err == ""


def test_missing_models_are_reported_before_the_pdf_is_opened(tmp_path, capsys):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    code, err = run([str(bad), "--models-dir", str(tmp_path / "none"), "--reader", "paddle", "--no-judge"], capsys)
    assert code.startswith("error: model folder(s) not found") and err == ""


def test_a_pdf_that_cannot_be_opened_stops_the_run_before_any_model_is_loaded(tmp_path, capsys):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    code, err = run([str(bad), "-o", str(tmp_path / "out"), "--models-dir", str(models_dir(tmp_path)),
                     "--reader", "paddle", "--no-judge"], capsys)
    assert code.startswith(f"error: cannot open {bad}:") and err == ""  # no "loading models ..." line
    assert not (tmp_path / "out").exists()


def test_pages_outside_the_document_stop_the_run_before_any_model_is_loaded(tmp_path, capsys):
    pdf = blank_pdf(tmp_path / "one.pdf")
    code, err = run([str(pdf), "-o", str(tmp_path / "out"), "--models-dir", str(models_dir(tmp_path)),
                     "--reader", "paddle", "--no-judge", "--pages", "2"], capsys)
    assert code == "error: page range '2' outside 1..1" and err == ""
    assert not (tmp_path / "out").exists()
