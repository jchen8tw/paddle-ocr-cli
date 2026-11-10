# The Judge renders Readings with the official Chromium + KaTeX renderer and its reference font

The Judge compares a Block's image with an image rendered from a Reading, so it is sensitive to the
rendered pixels. The MiniCPM-OCR authors run their renderer (Playwright Chromium, bundled KaTeX) as a
subprocess and warn that any other renderer changes the pixels and therefore the Scores. We vendor
that renderer unchanged and bundle DroidSansFallbackFull, the CJK font it was calibrated with,
instead of the system's Noto Sans CJK. Do not "simplify" this into a Pillow or in-process renderer.
Chromium is installed once at setup (`playwright install chromium`); runs stay offline.
