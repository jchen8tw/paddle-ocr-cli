"""Round 2 of the re-read schedule is read from a 1.25x render of the page when the Block's image is the plain crop
of the page, and from a resampled crop otherwise (no GPU, no models)."""
import cv2
import numpy as np

from fakes import FakeClient
from reader_bridge import BlockQuery, MiniCPMBridge

BBOX = (10, 20, 50, 40)


class FakePage:
    """The page as pdf_ocr.PageRenders offers it: `original(bbox)` is the 1x crop, `crop(bbox, factor)` a re-render's."""

    def __init__(self, w=200, h=100):
        rng = np.random.default_rng(0)
        self.page1 = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        self.page2 = rng.integers(0, 255, (round(h * 1.25), round(w * 1.25), 3), dtype=np.uint8)
        self.crops = []

    def original(self, bbox):
        x1, y1, x2, y2 = bbox
        return self.page1[y1:y2, x1:x2]

    def crop(self, bbox, factor):
        self.crops.append(factor)
        x1, y1, x2, y2 = (round(v * factor) for v in bbox)
        return self.page2[y1:y2, x1:x2]


class Round2Differs(FakeClient):
    """Records the crop of every read; the second read (round 2) answers with another text."""

    def __init__(self):
        super().__init__(default_score=0.5)
        self.sent = []

    def read(self, image, prompt, max_new_tokens, **kw):
        self.sent.append(cv2.imread(str(image)))
        out = super().read(image, prompt, max_new_tokens, **kw)
        if len(self.sent) == 2:  # the second read of the Block is round 2
            out[0]["text"] = "read from round 2"
        return out


def block(image):
    q = BlockQuery("OCR:")
    q.label, q.bbox, q.block_index = "text", BBOX, 0
    return {"image": image, "query": q}


def run(tmp_path, page, image, with_page=True):
    client = Round2Differs()
    bridge = MiniCPMBridge(client, tmp_path, to_html=lambda s: "", judge=True, judge_threshold=0.98, doc_hash="ab" * 32)
    if with_page:
        bridge.page_source = page
    bridge.current_page = 0
    list(bridge.predict([block(image)], max_new_tokens=8192))
    return client, bridge.outcomes[(0, BBOX)]


def test_round_2_reads_a_re_rendered_crop_of_the_page_when_the_image_is_the_plain_page_crop(tmp_path):
    page = FakePage()
    client, out = run(tmp_path, page, page.original(BBOX).copy())
    assert page.crops == [1.25]
    expected = page.crop(BBOX, 1.25)
    assert np.array_equal(client.sent[1], expected)
    assert [(r.round, r.scale_source) for r in out.readings if r.round == 2] == [(2, "page_render")]


def test_without_a_page_the_crop_is_resampled_and_the_reading_says_so(tmp_path):
    page = FakePage()
    client, out = run(tmp_path, page, page.original(BBOX).copy(), with_page=False)
    assert client.sent[1].shape[:2] == (25, 50)  # 20 x 40 enlarged by 1.25
    assert [(r.round, r.scale_source) for r in out.readings if r.round == 2] == [(2, "crop_resample")]


def test_an_image_that_is_not_the_plain_page_crop_is_resampled_not_replaced(tmp_path):
    page = FakePage()
    painted = page.original(BBOX).copy()
    painted[0, 0] = 0  # a figure token painted on, a polygon mask, a stacked block: anything but the plain crop
    painted[0, 0] ^= 255
    client, out = run(tmp_path, page, painted)
    assert page.crops == []
    assert [(r.round, r.scale_source) for r in out.readings if r.round == 2] == [(2, "crop_resample")]


def test_round_1_and_the_sampled_rounds_have_no_scale_source(tmp_path):
    page = FakePage()
    _, out = run(tmp_path, page, page.original(BBOX).copy())
    assert all(r.scale_source is None for r in out.readings if r.round != 2)
