# Vendored: the MiniCPM-OCR Jev renderer

These files are copied **unchanged** from the MiniCPM-OCR SDK
(<https://github.com/OpenBMB/MiniCPM-OCR>, commit `c65b84d7b5dc54d20ba16cafb918973532a9b616`, Apache-2.0, see
`LICENSE`), from `scripts/minicpm-ocr-jev/`. The Judge compares a Block with an image rendered from a Reading,
and the authors warn that any other renderer changes the pixels and therefore the Scores
([ADR-0002](../../docs/adr/0002-judge-uses-the-official-renderer.md)). Do not edit them.

| here | upstream (`scripts/minicpm-ocr-jev/`) |
|---|---|
| `rm_render.py` | `rm_render.py` |
| `renderer/` | `renderer/` (render driver, `render_module/render_label.py`, `precheck/screen_v225_node.js`, `check_renderer.py`, `setup_renderer.sh`, `assets/katex-0.16.47/`) |
| `examples/` | `examples/` (three Blocks with the Scores the authors observed; `tests/test_judge_reference.py` uses them) |
| `LICENSE` | the repository's `LICENSE` (Apache-2.0) |
| `renderer/assets/katex-0.16.47/LICENSE` | KaTeX's MIT licence (KaTeX 0.16.47 is bundled in that folder) |

Not vendored: `infer_vllm.py`, `serve/`, `sdk/`, `eval/`, `data/`.

The renderer's Python dependencies are pinned in `worker/pyproject.toml` (`playwright==1.62.0`, `pillow==12.3.0`,
`markdown==3.10.3`, the versions the authors pin). It runs in the worker's interpreter; the main environment's
Pillow never renders anything.

The font is not part of the SDK: `../fonts/DroidSansFallbackFull.ttf` comes from Debian's `fonts-droid-fallback`
(see the amendment in ADR-0002).

## Checksum of the font

`../fonts/DroidSansFallbackFull.ttf`: 4,033,420 bytes, sha256
`acb6440a713d880a13a21b468ba7cd43f5a2b2934972e51be791c880730777b8` (also in `../fonts/SHA256SUMS`; the file is
identical in Debian's `fonts-droid-fallback` 6.0.1r16-1.1 and 8.1.0r7-1; Apache-2.0, `../fonts/NOTICE` and
`../fonts/copyright`).
