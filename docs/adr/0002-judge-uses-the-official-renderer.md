# The Judge renders Readings with the official Chromium + KaTeX renderer and its reference font

The Judge compares a Block's image with an image rendered from a Reading, so it is sensitive to the
rendered pixels. The MiniCPM-OCR authors run their renderer (Playwright Chromium, bundled KaTeX) as a
subprocess and warn that any other renderer changes the pixels and therefore the Scores. We vendor
that renderer unchanged and bundle DroidSansFallbackFull, the CJK font it was calibrated with,
instead of the system's Noto Sans CJK. Do not "simplify" this into a Pillow or in-process renderer.
Chromium is installed once at setup (`playwright install chromium`); runs stay offline.

## Amendment (2026-11-10): Noto Sans CJK is a system requirement, Droid fills in behind it

The decision above said to bundle DroidSansFallbackFull *instead of* the system's Noto Sans CJK. Building the
Judge showed that cannot work, so this amendment replaces that sentence. The rest of the ADR (vendor the
renderer unchanged, run it as a subprocess, no Pillow or in-process renderer) stands.

**Evidence.**

- DroidSansFallbackFull has no Latin letters, digits, punctuation or Greek, only CJK and the space (checked with
  `fc-query` and a Pillow render). With a fontconfig that lists only Droid, the SDK's English text example
  renders as a row of empty boxes (`mismatched`, 774x169 px against the reference's 459x169).
- The authors' own setup scripts install `fonts-droid-fallback` *and* `fonts-noto-cjk`, and the renderer's CSS
  stack names `"Noto Sans CJK SC"` first. Latin letters and digits in their reference renders therefore come
  from Noto Sans CJK; Droid only fills CJK glyphs Noto lacks.

**Decision.**

- Vendor `DroidSansFallbackFull.ttf` (Apache-2.0, 4,033,420 bytes, sha256
  `acb6440a713d880a13a21b468ba7cd43f5a2b2934972e51be791c880730777b8`, from Debian's `fonts-droid-fallback`, with
  its `NOTICE` and `copyright`) in `worker/fonts/`. Vendor the renderer and KaTeX unchanged in `worker/render/`
  (upstream commit `c65b84d`, see `worker/render/VENDORED.md`).
- Require the system's Noto Sans CJK (not vendored: 40 MB, SIL OFL), and `node` (the KaTeX precheck behind the
  `precheck_reject` verdict) and Chromium (`playwright install chromium`). The worker checks all of them before
  it loads a model, with a real test render, and stops with a message naming what is missing and how to
  install it.
- The renderer runs under a private `FONTCONFIG_FILE` that lists only the vendored Droid folder and links to the
  system's Noto Sans CJK files, so whatever else is installed (this machine has 1,190 fonts) cannot change the
  face set. The font cache lives in `~/.cache/pdf_ocr/fontconfig`.
- The private file also carries the rendering rules the reference machine gets from its default fontconfig
  (slight hinting, antialiasing, default LCD filter). A bare file made of `<dir>` entries alone drops them.

**Open item resolved: bare fontconfig file against the system's rules.** The three reference examples of the
Jev SDK (6 candidates with the Scores the authors observed) were rendered and scored with the real Jev under
four configurations. Deviation is `|score - observed_prob_yes|`:

| configuration | mean deviation | max deviation |
|---|---|---|
| this machine's system fontconfig (1,190 fonts) | 0.0017 | 0.0087 |
| private file, bare (`<dir>` entries only) | 0.0028 | 0.0154 |
| private file + the system's three rendering snippets (`10-hinting-slight`, `10-yes-antialias`, `11-lcdfilter-default`) | 0.0017 | 0.0087 |
| private file + the same three rules written out inline (the one we ship) | 0.0017 | 0.0087 |

The bare file is measurably worse (the table example's ground-truth Score moved from 0.032 to 0.025 against the
authors' 0.041), and the inline-rules file reproduces the system configuration exactly, so it is what ships. It
does not depend on the distribution's `/etc/fonts/conf.d`. The authors expect about 5e-3; the largest
deviation (0.009) is the table example, whose ground truth the Judge rates low anyway (0.04). This is checked
by a slow test (`tests/test_judge_reference.py`) that runs the three examples through the real worker.

**What this does not settle.** The text example renders 449 px wide against the reference's 459 px, with
every configuration tried: the reference's Latin face is not exactly this Noto Sans CJK build. The Scores still
agree to about 1e-3 for that example, but they are not bit-reproducible against the authors' deployment, and
the authors say so too.
