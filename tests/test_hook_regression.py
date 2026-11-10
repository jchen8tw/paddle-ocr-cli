"""The Reader hook against the real PaddleX pipeline, without any Reader weights.

This is the regression test for the private PaddleX names the hook uses: when a paddlex upgrade moves
them, it fails here (and the start-up guard fails for users). The Layout model runs for real (a few
hundred MB on the GPU); the Reader is a fake worker client, and PaddleOCR-VL's weights are never loaded
because the pipeline is built with a server backend that is never contacted.

Needs PDF_OCR_SAMPLE_PDF (any PDF with text) and the PP-DocLayoutV3 folder.
"""
import pytest

import pdf_ocr
from fakes import FakeClient
from reader_bridge import MiniCPMBridge, check_hook_support, install_reader_hook

pytestmark = pytest.mark.gpu


def render_first_page(pdf_path, dpi=216.0):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[0]
    try:
        return pdf_ocr.render_page(page, dpi)
    finally:
        page.close()


def test_every_block_of_a_real_page_goes_through_the_bridge_with_its_label(sample_pdf, layout_models_dir, tmp_path):
    args = pdf_ocr.build_parser().parse_args([str(sample_pdf), "--models-dir", str(layout_models_dir)])
    assert args.reader == "minicpm"
    pipe = pdf_ocr.build_pipeline(args, pdf_ocr.pick_device(None))  # only the Layout model is loaded
    check_hook_support(pipe)

    client = FakeClient()
    bridge = MiniCPMBridge(client, tmp_path, to_html=lambda s: "")
    install_reader_hook(pipe, bridge)
    img, _ = render_first_page(sample_pdf)
    res = pipe.predict(img, max_new_tokens=args.max_new_tokens, use_queues=False)[0]

    blocks = res["parsing_res_list"]
    assert blocks and client.calls, "the page has no blocks the Reader should have read"
    assert all(prompt in ("Text Recognition:", "Table Recognition:", "Formula Recognition:") for prompt, _ in client.calls)
    read = [b for b in blocks if b.label not in ("image", "chart", "seal", "header_image", "footer_image")]
    assert read and all(b.content.startswith("text for ") for b in read)
    assert len(client.calls) >= len(read)
    import paddle

    if pdf_ocr.pick_device(None).startswith("gpu"):  # PaddleOCR-VL's weights alone would be ~7 GB
        assert paddle.device.cuda.max_memory_reserved() < 3 * 2**30
    # Everything around the Reader still works on what the bridge returned.
    out = pipe.restructure_pages([res], merge_tables=True, relevel_titles=True, concatenate_pages=False)
    assert out[0]._to_markdown(pretty=False, show_formula_number=True)["markdown_texts"].strip()
