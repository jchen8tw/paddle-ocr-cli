# MiniCPM-OCR and the Judge run in a separate torch venv, started by the main command

MiniCPM-OCR-1.0 (the default Reader until ADR-0003) and MiniCPM-OCR Jev (the Judge) need torch and transformers
≥ 5.21. paddlepaddle-gpu 3.3.1 (cu130) and every cu13 torch build pin exact, different versions of
the CUDA libraries (cudnn, nccl, cuda-runtime, cublas), so they cannot share one uv environment
without overrides nobody has verified. We keep the main paddle venv for the Layout model and the
whole tuned PaddleOCR pipeline (block merging, cross-page tables, Markdown). The main command starts
a worker in a second uv project for the torch models, so running the tool is still one command.

## Considered Options

- **All torch through transformers.** transformers supports `pp_doclayout_v3`, `paddleocr_vl`,
  `minicpm_ocr` and Jev natively, so paddle could be dropped entirely. Rejected for now: we would
  rebuild the pipeline steps paddleocr does for us, convert the layout weights, and re-measure every
  tuned default. It is still the likely end state.
- **One venv with `override-dependencies`.** Fragile, and it is unknown whether paddle runs on the
  newer CUDA libraries.
- **An external vLLM/SGLang server, as the official SDK uses.** Rejected: the tool must stay one
  self-contained command.

## Amendment (2026-11-11): the default needs no torch

[ADR-0003](0003-paddleocr-vl-is-the-default-reader.md) makes PaddleOCR-VL-1.6 the default Reader and turns the
Judge off unless `--judge` is given, so a run with no options starts no worker. The separate torch venv above
stands, for `--reader minicpm` and `--judge`. ADR-0003 also says why the all-transformers option is no longer the
likely end state.
