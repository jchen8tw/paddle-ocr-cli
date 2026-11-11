# pdf_ocr

Turns a PDF into Markdown, JSON and CSV with local models only: a layout model finds the regions of
each page, a reader transcribes each region, and a judge rates how trustworthy each transcription is.

## Language

### Pages and regions

**Layout model**:
The model that finds the Blocks on a page, labels them and orders them for reading.
_Avoid_: detector, segmenter

**Block**:
One labelled region of a page (paragraph, title, table, formula, figure, …), the unit that is read
and judged.
_Avoid_: region, box, element

### Reading and judging

**Reader**:
A model that turns one Block into text, HTML or LaTeX. MiniCPM-OCR is the default Reader;
PaddleOCR-VL is the alternative.
_Avoid_: recognizer, OCR model, engine

**Reading**:
One Reader's output for one Block. A Block can have more than one Reading when it is re-read.
_Avoid_: candidate, result, prediction

**Re-read**:
Reading a Block again, differently, because its first Reading looks untrustworthy.
_Avoid_: retry, re-OCR, recovery

**Round**:
One step of the re-read schedule: greedy at 1x (the first Reading), greedy at 1.25x, 4 samples at temperature 0.3,
4 samples at temperature 0.7. A round runs for a Block only while its best Score is below the threshold and the round
before it gave a text not read before; a Reader skips the rounds it cannot do.
_Avoid_: pass, attempt, stage

**Judge**:
The model (MiniCPM-OCR Jev) that rates a Reading against the Block's image. It does not transcribe.
_Avoid_: verifier, scorer, Jev (as a role name)

**Score**:
The Judge's 0–1 rating of one Reading; higher means more likely correct.
_Avoid_: confidence, probability

**Verdict**:
What happened when the Judge was asked to rate one Reading. Only `scored` carries a real Score; the other
verdicts say the Reading was unusable (`precheck_reject`, `render_error`, `missing_original_image`) or that
scoring never happened (`encode_error`, `forward_error`).
_Avoid_: status, result

**Flagged**:
A judged Block a person should check: its first Reading scored below the threshold or could not be scored.
Re-reading never clears it.
_Avoid_: below threshold (the final Score can be above it), low-confidence

### Runs

**Engine**:
The full set of models and library versions used in one run (Layout model, Reader, Judge), not any
single model.
_Avoid_: backend, pipeline
