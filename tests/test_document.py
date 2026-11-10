"""Document-level scoring, text-layer extraction and the standalone report (no GPU, no models)."""
import json

import ocr_eval
from ocr_eval import TextChar


def block(label, content, bbox, bid=0):
    return {"id": bid, "label": label, "content": content, "bbox": list(bbox)}


def chars(text, x0, y, step=10.0):
    return [TextChar(c, x0 + i * step, y) for i, c in enumerate(text)]


def page_json(n, blocks, scale=1.0):
    return {"page": n, "scale_px_per_pt": scale, "blocks": blocks}


def test_document_score_totals_and_breaks_down_by_page_and_label():
    doc = {"pages": [
        page_json(1, [block("text", "你好嗎", (0, 0, 100, 20)),
                      block("table", "<td>甲乙</td>", (0, 40, 100, 60), bid=1)]),
        page_json(2, [block("text", "丙丁", (0, 0, 100, 20))]),
    ]}
    layers = {
        1: chars("你好馬", 5, 10) + chars("甲乙", 5, 50),  # 1 error in the text block
        2: chars("丙丁", 5, 10),
    }
    score = ocr_eval.score_document(doc, layers)
    assert score.total.errors == 1 and score.total.ref_chars == 7
    assert score.pages[1].errors == 1 and score.pages[2].errors == 0
    assert score.by_label["text"].errors == 1 and score.by_label["text"].ref_chars == 5
    assert score.by_label["table"].cer == 0.0


def test_failed_pages_and_pages_without_a_text_layer_are_left_out():
    doc = {"pages": [
        {"page": 1, "error": "OutOfMemoryError: boom", "blocks": []},
        page_json(2, [block("text", "丙丁", (0, 0, 100, 20))]),
    ]}
    score = ocr_eval.score_document(doc, {1: chars("甲", 5, 10)})
    assert score.pages == {}
    assert score.skipped_pages == [1, 2]


def minimal_pdf(text: str, x: int, y: int, width=200, height=300) -> bytes:
    """One page, Helvetica `text` drawn with its baseline origin at (x, y) in PDF user space."""
    stream = f"BT /F1 20 Tf {x} {y} Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Contents 4 0 R "
        f"/Resources << /Font << /F1 5 0 R >> >> >>".encode(),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_text_layer_is_extracted_in_top_left_points(tmp_path):
    pdf = tmp_path / "t.pdf"
    pdf.write_bytes(minimal_pdf("AB", x=50, y=100, height=300))
    got = ocr_eval.extract_text_layers(pdf, [1])[1]
    assert [c.ch for c in got] == ["A", "B"]
    # The baseline is 100 pt above the bottom of a 300 pt page, so ~200 pt from the top;
    # a 20 pt glyph's centre sits a few points above its baseline.
    assert all(180 < c.y < 200 for c in got)
    assert 50 < got[0].x < got[1].x < 80


def test_standalone_report_prints_the_error_rate_per_page_and_per_label(tmp_path, capsys):
    pdf = tmp_path / "t.pdf"
    pdf.write_bytes(minimal_pdf("AB", x=50, y=100))
    out = tmp_path / "t.json"
    # bbox in px at 2 px/pt, covering the text (x 50-80 pt, y 180-200 pt from the top)
    out.write_text(json.dumps({"pages": [page_json(
        1, [block("text", "AX", (80, 340, 200, 420))], scale=2.0)]}))
    assert ocr_eval.main([str(pdf), str(out)]) == 0
    text = capsys.readouterr().out
    assert "page 1" in text and "text" in text and "total" in text
    assert "50.00%" in text  # 1 of 2 characters wrong
