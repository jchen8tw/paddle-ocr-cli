"""The ink-based re-read of implausibly short Blocks cuts its crops from re-renders of the page. Both users of that
(PaddleOCR-VL without --judge: pdf_ocr.recover_short_blocks; --reader minicpm without --judge: InkReread) must send the
Reader exactly the crops they always did: the Block's box scaled to the re-render, 4 px of margin, no check on
whether the render could really be made larger (no GPU, no models; the page and the renders are fakes).
"""
import types

import numpy as np
import pytest

import pdf_ocr

H, W, DPI, SCALE = 400, 800, 216.0, 3.0
BBOX = (100, 150, 500, 190)


def page_image():
    img = np.full((H, W, 3), 255, np.uint8)
    img[155:185, 100:500] = 0  # one line of ink: ~13 glyphs of height 30 across 400 px
    return img


def make_render(capped_to=None):
    """A stand-in for pdf_ocr.render_page: each pixel of a render holds its own position, so a crop identifies
    exactly which part of the render it is. `capped_to` is the factor the render really manages (the size cap)."""
    calls = []

    def render(page, dpi):
        factor = dpi / DPI
        if capped_to is not None:
            factor = min(factor, capped_to)
        calls.append(dpi)
        h, w = int(H * factor), int(W * factor)
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = np.arange(w)[None, :] % 251
        img[..., 1] = np.arange(h)[:, None] % 241
        return img, SCALE * factor

    return render, calls


def expect(render_img_shape_factor, rows, cols):
    factor = render_img_shape_factor
    h, w = int(H * factor), int(W * factor)
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.arange(w)[None, :] % 251
    img[..., 1] = np.arange(h)[:, None] % 241
    return img[rows[0]:rows[1], cols[0]:cols[1]]


# bbox (100, 150, 500, 190) with 4 px of margin: at 1.25x rows 183:241, cols 121:629; at 0.8x rows 116:156, cols 76:404
# (the box at 1.0, which a render too large for the cap becomes, gives rows 146:194, cols 96:504)
EXPECTED = {1.25: ((183, 241), (121, 629)), 0.8: ((116, 156), (76, 404))}


class FakePipe:
    def __init__(self):
        self.images = []

    def predict(self, image, **kw):
        self.images.append(image)
        block = types.SimpleNamespace(content="a much longer reading, now with all of the words " * 2)
        return [{"parsing_res_list": [block]}]


def paddle_run(monkeypatch, capped_to=None):
    render, calls = make_render(capped_to)
    monkeypatch.setattr(pdf_ocr, "render_page", render)
    img = page_image()
    block = types.SimpleNamespace(label="text", bbox=list(BBOX), content="a")
    res = {"parsing_res_list": [block]}
    pipe = FakePipe()
    args = types.SimpleNamespace(dpi=DPI, max_new_tokens=512)
    notes = pdf_ocr.recover_short_blocks(pipe, object(), res, img, SCALE, args, 1)
    return pipe, notes, calls


def test_paddle_without_the_judge_reads_the_scaled_box_with_four_pixels_of_margin_from_renders_at_1_25_and_0_8(monkeypatch):
    pipe, notes, calls = paddle_run(monkeypatch)
    assert calls == [DPI * 1.25, DPI * 0.8]
    for image, factor in zip(pipe.images, (1.25, 0.8)):
        assert np.array_equal(image, expect(factor, *EXPECTED[factor]))
    assert notes == [{"bbox": list(BBOX), "original": "a"}]


def test_paddle_without_the_judge_crops_at_the_size_the_render_really_has_when_the_cap_stopped_it_growing(monkeypatch):
    pipe, _, _ = paddle_run(monkeypatch, capped_to=1.0)
    assert np.array_equal(pipe.images[0], expect(1.0, (146, 194), (96, 504)))  # asked 1.25x, got 1x: still read


class FakeBridge:
    def __init__(self):
        self.crops = []

    def read_crop(self, image, label, max_new_tokens):
        self.crops.append((image, label, max_new_tokens))
        return " a much longer reading, now with all of the words " * 2


def set_page(ink, page, img):
    ink.set_page(pdf_ocr.PageRenders(page, img, SCALE, DPI))


def minicpm_run(monkeypatch, capped_to=None):
    render, calls = make_render(capped_to)
    monkeypatch.setattr(pdf_ocr, "render_page", render)
    ink, bridge = pdf_ocr.InkReread(), FakeBridge()
    set_page(ink, object(), page_image())
    reread = ink.reread_with(bridge, types.SimpleNamespace(dpi=DPI, max_new_tokens=512))
    request = types.SimpleNamespace(label="text", bbox=BBOX)
    return reread(request, "a"), bridge, calls


def test_minicpm_without_the_judge_reads_the_same_crops_as_the_paddle_path(monkeypatch):
    better, bridge, calls = minicpm_run(monkeypatch)
    assert better is not None and calls == [DPI * 1.25, DPI * 0.8]
    for (image, label, limit), factor in zip(bridge.crops, (1.25, 0.8)):
        assert np.array_equal(image, expect(factor, *EXPECTED[factor]))
        assert (label, limit) == ("text", 512)


def test_minicpm_without_the_judge_crops_at_the_size_the_render_really_has_when_the_cap_stopped_it_growing(monkeypatch):
    _, bridge, _ = minicpm_run(monkeypatch, capped_to=1.0)
    assert np.array_equal(bridge.crops[0][0], expect(1.0, (146, 194), (96, 504)))


def test_minicpm_without_the_judge_skips_a_block_that_is_not_text_or_has_no_box(monkeypatch):
    _, bridge, calls = minicpm_run(monkeypatch)
    ink = pdf_ocr.InkReread()
    set_page(ink, object(), page_image())
    reread = ink.reread_with(bridge, types.SimpleNamespace(dpi=DPI, max_new_tokens=512))
    assert reread(types.SimpleNamespace(label="table", bbox=BBOX), "a") is None
    assert reread(types.SimpleNamespace(label="text", bbox=None), "a") is None
